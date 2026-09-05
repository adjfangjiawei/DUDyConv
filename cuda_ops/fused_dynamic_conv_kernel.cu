#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_map>
#include <mutex>
#include <limits>
#include <algorithm>
#include <string>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Backward plan enum
// ======================================================================================

enum class FDCBackwardPlan : int {
    MidN6K3Warp = 1,
    Large = 4,
    ManyNK3D256 = 7,

    // 新增 backward GEMM 化特化方案。
    //
    // 这些方案不替换现有 ManyNK3D256 / Large / MidN6K3Warp。
    // 只作为额外 candidate 参与 warmup。
    //
    // 目标 shape:
    //   B == 1
    //   D == 256
    //   K == 3
    //   dilation == 1
    //   dtype == float32
    //   N == 16 / 32 / 64 / 128
    //
    // 核心:
    //   make G[3,D,T]
    //   grad_kernel: 3x SGEMM
    //   grad_mix:    3x SGEMM
    //   grad_h:      direct CUDA kernel
    N16K3D256Gemm = 16,
    N32K3D256Gemm = 32,
    N64K3D256Gemm = 64,
    N128K3D256Gemm = 128
};

static inline const char* fdc_backward_plan_name_impl(FDCBackwardPlan p) {
    switch (p) {
        case FDCBackwardPlan::MidN6K3Warp:
            return "backward_mid_n6k3_warp";

        case FDCBackwardPlan::Large:
            return "backward_large";

        case FDCBackwardPlan::ManyNK3D256:
            return "backward_manyn_k3_d256";

        case FDCBackwardPlan::N16K3D256Gemm:
            return "backward_n16_k3_d256_gemm";

        case FDCBackwardPlan::N32K3D256Gemm:
            return "backward_n32_k3_d256_gemm";

        case FDCBackwardPlan::N64K3D256Gemm:
            return "backward_n64_k3_d256_gemm";

        case FDCBackwardPlan::N128K3D256Gemm:
            return "backward_n128_k3_d256_gemm";

        default:
            return "unknown";
    }
}

// ======================================================================================
// Cache key
// ======================================================================================

struct FDCBackwardCacheKey {
    int device;
    int sm;
    int dtype;
    int B;
    int D;
    int L;
    int T;
    int N;
    int K;
    int off;
    int dilation;

    bool operator==(const FDCBackwardCacheKey& o) const {
        return device == o.device &&
               sm == o.sm &&
               dtype == o.dtype &&
               B == o.B &&
               D == o.D &&
               L == o.L &&
               T == o.T &&
               N == o.N &&
               K == o.K &&
               off == o.off &&
               dilation == o.dilation;
    }
};

struct FDCBackwardCacheKeyHash {
    std::size_t operator()(const FDCBackwardCacheKey& k) const {
        std::size_t h = 1469598103934665603ull;

        auto mix = [&](int v) {
            h ^= static_cast<std::size_t>(v);
            h *= 1099511628211ull;
        };

        mix(k.device);
        mix(k.sm);
        mix(k.dtype);
        mix(k.B);
        mix(k.D);
        mix(k.L);
        mix(k.T);
        mix(k.N);
        mix(k.K);
        mix(k.off);
        mix(k.dilation);

        return h;
    }
};

static std::unordered_map<FDCBackwardCacheKey, FDCBackwardPlan, FDCBackwardCacheKeyHash>
    g_fdc_backward_plan_cache;

static std::mutex g_fdc_backward_plan_cache_mutex;

static inline int fdc_dtype_tag(at::ScalarType t) {
    if (t == at::ScalarType::Float) {
        return 0;
    }

    if (t == at::ScalarType::Half) {
        return 1;
    }

    if (t == at::ScalarType::BFloat16) {
        return 2;
    }

    return -1;
}

static inline int fdc_get_sm(torch::Tensor x) {
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(
        &prop,
        x.get_device()
    );

    TORCH_CHECK(err == cudaSuccess, "cudaGetDeviceProperties failed");

    return prop.major * 10 + prop.minor;
}

static inline FDCBackwardCacheKey make_fdc_backward_cache_key(
    torch::Tensor h,
    torch::Tensor kc,
    int64_t off,
    int64_t dilation
) {
    FDCBackwardCacheKey key;

    key.device = h.get_device();
    key.sm = fdc_get_sm(h);
    key.dtype = fdc_dtype_tag(h.scalar_type());
    key.B = static_cast<int>(h.size(0));
    key.D = static_cast<int>(h.size(1));
    key.L = static_cast<int>(h.size(2));
    key.K = static_cast<int>(kc.size(1));
    key.N = static_cast<int>(kc.size(2));
    key.T = static_cast<int>(kc.size(3));
    key.off = static_cast<int>(off);
    key.dilation = static_cast<int>(dilation);

    return key;
}

static inline bool get_cached_fdc_backward_plan(
    const FDCBackwardCacheKey& key,
    FDCBackwardPlan* plan
) {
    std::lock_guard<std::mutex> lock(g_fdc_backward_plan_cache_mutex);

    auto it = g_fdc_backward_plan_cache.find(key);

    if (it == g_fdc_backward_plan_cache.end()) {
        return false;
    }

    *plan = it->second;

    return true;
}

static inline void set_cached_fdc_backward_plan(
    const FDCBackwardCacheKey& key,
    FDCBackwardPlan plan
) {
    std::lock_guard<std::mutex> lock(g_fdc_backward_plan_cache_mutex);

    g_fdc_backward_plan_cache[key] = plan;
}

// ======================================================================================
// Shape helpers
// ======================================================================================

struct FDCShapeInfo {
    int B;
    int D;
    int L;
    int T;
    int N;
    int K;
    int sm;

    bool is_fp32;
    bool is_fp16;
    bool is_bf16;

    bool base_k3_dil1;
    bool base_d256_k3_dil1;

    bool shape_n6;
    bool shape_n6_d512;
    bool shape_manyn_k3_d256;

    bool shape_n16_k3_d256_gemm;
    bool shape_n32_k3_d256_gemm;
    bool shape_n64_k3_d256_gemm;
    bool shape_n128_k3_d256_gemm;

    bool full4096_offset0_n6;
};

static inline FDCShapeInfo make_fdc_shape_info(
    torch::Tensor h,
    torch::Tensor kc,
    int64_t off,
    int64_t dilation
) {
    FDCShapeInfo s;

    s.B = static_cast<int>(h.size(0));
    s.D = static_cast<int>(h.size(1));
    s.L = static_cast<int>(h.size(2));
    s.K = static_cast<int>(kc.size(1));
    s.N = static_cast<int>(kc.size(2));
    s.T = static_cast<int>(kc.size(3));
    s.sm = fdc_get_sm(h);

    s.is_fp32 = h.scalar_type() == at::ScalarType::Float;
    s.is_fp16 = h.scalar_type() == at::ScalarType::Half;
    s.is_bf16 = h.scalar_type() == at::ScalarType::BFloat16;

    s.base_k3_dil1 =
        s.B == 1 &&
        s.K == 3 &&
        static_cast<int>(dilation) == 1;

    s.base_d256_k3_dil1 =
        s.base_k3_dil1 &&
        s.D == 256;

    s.shape_n6 =
        s.base_d256_k3_dil1 &&
        s.N == 6;

    s.shape_n6_d512 =
        s.base_k3_dil1 &&
        s.D == 512 &&
        s.N == 6;

    s.shape_manyn_k3_d256 =
        s.base_d256_k3_dil1 &&
        s.N >= 16 &&
        s.N <= 128 &&
        s.T >= 2048;

    // 新增 GEMM backward 特化 shape。
    //
    // 注意:
    //   当前 GEMM backward 实现只支持 fp32。
    //   fp16/bf16 继续走现有 ManyNK3D256 或 Large 等方案。
    //
    // T 阈值设置为 >= 512，和 forward many-N 特化类似。
    // 如果后续 benchmark 发现小 T 不划算，可以提高到 2048。
    s.shape_n16_k3_d256_gemm =
        s.base_d256_k3_dil1 &&
        s.N == 16 &&
        s.is_fp32 &&
        s.T >= 512;

    s.shape_n32_k3_d256_gemm =
        s.base_d256_k3_dil1 &&
        s.N == 32 &&
        s.is_fp32 &&
        s.T >= 512;

    s.shape_n64_k3_d256_gemm =
        s.base_d256_k3_dil1 &&
        s.N == 64 &&
        s.is_fp32 &&
        s.T >= 512;

    s.shape_n128_k3_d256_gemm =
        s.base_d256_k3_dil1 &&
        s.N == 128 &&
        s.is_fp32 &&
        s.T >= 512;

    s.full4096_offset0_n6 =
        s.shape_n6 &&
        s.L == 4096 &&
        s.T == 4096 &&
        static_cast<int>(off) == 0;

    return s;
}

// ======================================================================================
// Validation
// ======================================================================================

static inline void check_fdc_common_inputs(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    TORCH_CHECK(h.is_cuda(), "h must be CUDA tensor.");
    TORCH_CHECK(kc.is_cuda(), "kc must be CUDA tensor.");
    TORCH_CHECK(mix.is_cuda(), "mix must be CUDA tensor.");

    TORCH_CHECK(h.is_contiguous(), "h must be contiguous.");
    TORCH_CHECK(kc.is_contiguous(), "kc must be contiguous.");
    TORCH_CHECK(mix.is_contiguous(), "mix must be contiguous.");

    TORCH_CHECK(h.dim() == 3, "h must be [B,D,L].");
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,K,N,T].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(h.scalar_type() == kc.scalar_type(), "h/kc dtype mismatch.");
    TORCH_CHECK(h.scalar_type() == mix.scalar_type(), "h/mix dtype mismatch.");

    TORCH_CHECK(kc.size(0) == h.size(0), "B mismatch.");
    TORCH_CHECK(mix.size(0) == h.size(1), "D mismatch.");
    TORCH_CHECK(mix.size(1) == kc.size(2), "N mismatch.");

    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(dilation > 0, "dilation must be > 0.");
    TORCH_CHECK(off + kc.size(3) <= h.size(2), "off + T must be <= L.");

    TORCH_CHECK(
        h.scalar_type() == at::ScalarType::Float ||
        h.scalar_type() == at::ScalarType::Half ||
        h.scalar_type() == at::ScalarType::BFloat16,
        "unsupported dtype"
    );
}

static inline void check_fdc_backward_inputs(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    check_fdc_common_inputs(
        h,
        kc,
        mix,
        off,
        dilation
    );

    TORCH_CHECK(go.is_cuda(), "go must be CUDA tensor.");
    TORCH_CHECK(go.is_contiguous(), "go must be contiguous.");
    TORCH_CHECK(go.dim() == 3, "go must be [B,D,T].");

    TORCH_CHECK(go.scalar_type() == h.scalar_type(), "go dtype mismatch.");
    TORCH_CHECK(go.size(0) == h.size(0), "go B mismatch.");
    TORCH_CHECK(go.size(1) == h.size(1), "go D mismatch.");
    TORCH_CHECK(go.size(2) == kc.size(3), "go T mismatch.");
}

// ======================================================================================
// Plan availability
// ======================================================================================

static inline bool fdc_plan_available(
    FDCBackwardPlan plan,
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDCShapeInfo s = make_fdc_shape_info(
        h,
        kc,
        off,
        dilation
    );

    switch (plan) {
        case FDCBackwardPlan::MidN6K3Warp:
            return s.shape_n6 || s.shape_n6_d512;

        case FDCBackwardPlan::ManyNK3D256:
            return s.shape_manyn_k3_d256 &&
                   (s.is_fp32 || s.is_fp16 || s.is_bf16);

        case FDCBackwardPlan::N16K3D256Gemm:
            return s.shape_n16_k3_d256_gemm &&
                   fdc_backward_n16_k3_d256_gemm_available_cuda(
                       go,
                       h,
                       kc,
                       mix,
                       off,
                       dilation
                   );

        case FDCBackwardPlan::N32K3D256Gemm:
            return s.shape_n32_k3_d256_gemm &&
                   fdc_backward_n32_k3_d256_gemm_available_cuda(
                       go,
                       h,
                       kc,
                       mix,
                       off,
                       dilation
                   );

        case FDCBackwardPlan::N64K3D256Gemm:
            return s.shape_n64_k3_d256_gemm &&
                   fdc_backward_n64_k3_d256_gemm_available_cuda(
                       go,
                       h,
                       kc,
                       mix,
                       off,
                       dilation
                   );

        case FDCBackwardPlan::N128K3D256Gemm:
            return s.shape_n128_k3_d256_gemm &&
                   fdc_backward_n128_k3_d256_gemm_available_cuda(
                       go,
                       h,
                       kc,
                       mix,
                       off,
                       dilation
                   );

        case FDCBackwardPlan::Large:
            return s.is_fp32 || s.is_fp16 || s.is_bf16;

        default:
            return false;
    }
}

// ======================================================================================
// Plan runner
// ======================================================================================

static std::vector<torch::Tensor> fdc_run_backward_plan(
    FDCBackwardPlan plan,
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDCShapeInfo s = make_fdc_shape_info(
        h,
        kc,
        off,
        dilation
    );

    switch (plan) {
        case FDCBackwardPlan::MidN6K3Warp:
            return fdc_backward_mid_n6k3_warp_cuda(
                go,
                h,
                kc,
                mix,
                off,
                s.full4096_offset0_n6
            );

        case FDCBackwardPlan::ManyNK3D256:
            return fdc_backward_manyn_k3_d256_cuda(
                go,
                h,
                kc,
                mix,
                off
            );

        case FDCBackwardPlan::N16K3D256Gemm:
            return fdc_backward_n16_k3_d256_gemm_cuda(
                go,
                h,
                kc,
                mix,
                off
            );

        case FDCBackwardPlan::N32K3D256Gemm:
            return fdc_backward_n32_k3_d256_gemm_cuda(
                go,
                h,
                kc,
                mix,
                off
            );

        case FDCBackwardPlan::N64K3D256Gemm:
            return fdc_backward_n64_k3_d256_gemm_cuda(
                go,
                h,
                kc,
                mix,
                off
            );

        case FDCBackwardPlan::N128K3D256Gemm:
            return fdc_backward_n128_k3_d256_gemm_cuda(
                go,
                h,
                kc,
                mix,
                off
            );

        case FDCBackwardPlan::Large:
            return fdc_backward_large_cuda(
                go,
                h,
                kc,
                mix,
                off,
                dilation
            );

        default:
            TORCH_CHECK(false, "unknown fused dynamic conv backward plan");
    }
}

// ======================================================================================
// Default heuristic
// ======================================================================================

static FDCBackwardPlan fdc_default_backward_plan(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDCShapeInfo s = make_fdc_shape_info(
        h,
        kc,
        off,
        dilation
    );

    // 注意:
    //   为了“不改变现有方案”，这里不默认选择新增 GEMM backward plan。
    //   新增 GEMM backward plan 只进入 warmup candidate list。
    //
    // 因此:
    //   - 未 warmup 时，行为与原逻辑保持一致。
    //   - warmup 后，candidate 实测谁快选谁，并写入 cache。

    if (s.shape_n6_d512) {
        return FDCBackwardPlan::MidN6K3Warp;
    }

    if (s.shape_manyn_k3_d256) {
        return FDCBackwardPlan::ManyNK3D256;
    }

    if (s.shape_n6 && s.is_bf16 && s.T == 8192) {
        return FDCBackwardPlan::MidN6K3Warp;
    }

    if (s.full4096_offset0_n6 && s.is_fp16) {
        return FDCBackwardPlan::MidN6K3Warp;
    }

    if (s.shape_n6) {
        return FDCBackwardPlan::MidN6K3Warp;
    }

    return FDCBackwardPlan::Large;

    (void)mix;
}

// ======================================================================================
// Candidate list
// ======================================================================================

static std::vector<FDCBackwardPlan> fdc_all_candidate_backward_plans(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    std::vector<FDCBackwardPlan> plans;

    FDCBackwardPlan all[] = {
        // 新增 GEMM backward 特化方案。
        //
        // 放在 ManyNK3D256 前面只是影响 warmup 测试顺序，
        // 不表示默认强制选择。
        //
        // warmup 会逐个计时:
        //   ms < best_ms
        // 谁快最终 cache 谁。
        FDCBackwardPlan::N16K3D256Gemm,
        FDCBackwardPlan::N32K3D256Gemm,
        FDCBackwardPlan::N64K3D256Gemm,
        FDCBackwardPlan::N128K3D256Gemm,

        // 原有方案保持不变。
        FDCBackwardPlan::ManyNK3D256,
        FDCBackwardPlan::MidN6K3Warp,
        FDCBackwardPlan::Large
    };

    for (FDCBackwardPlan p : all) {
        if (fdc_plan_available(
                p,
                go,
                h,
                kc,
                mix,
                off,
                dilation
            )) {
            plans.push_back(p);
        }
    }

    TORCH_CHECK(!plans.empty(), "no available fused dynamic conv backward plan");

    return plans;
}

// ======================================================================================
// Timing
// ======================================================================================

static float fdc_time_backward_plan_once_ms(
    FDCBackwardPlan plan,
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaEvent_t start;
    cudaEvent_t stop;

    TORCH_CHECK(cudaEventCreate(&start) == cudaSuccess, "cudaEventCreate start failed");
    TORCH_CHECK(cudaEventCreate(&stop) == cudaSuccess, "cudaEventCreate stop failed");
    TORCH_CHECK(cudaEventRecord(start, stream) == cudaSuccess, "cudaEventRecord start failed");

    auto outs = fdc_run_backward_plan(
        plan,
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );

    (void)outs;

    TORCH_CHECK(cudaEventRecord(stop, stream) == cudaSuccess, "cudaEventRecord stop failed");
    TORCH_CHECK(cudaEventSynchronize(stop) == cudaSuccess, "cudaEventSynchronize stop failed");

    float ms = 0.0f;

    TORCH_CHECK(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess, "cudaEventElapsedTime failed");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

static float fdc_time_backward_plan_median_ms(
    FDCBackwardPlan plan,
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation,
    int64_t repeat
) {
    std::vector<float> times;

    times.reserve(
        static_cast<size_t>(repeat)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int dry = 0; dry < 1; ++dry) {
        auto outs = fdc_run_backward_plan(
            plan,
            go,
            h,
            kc,
            mix,
            off,
            dilation
        );

        (void)outs;
    }

    TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "cudaStreamSynchronize failed before timing");

    for (int64_t i = 0; i < repeat; ++i) {
        times.push_back(
            fdc_time_backward_plan_once_ms(
                plan,
                go,
                h,
                kc,
                mix,
                off,
                dilation
            )
        );
    }

    std::sort(
        times.begin(),
        times.end()
    );

    return times[times.size() / 2];
}

// ======================================================================================
// Public CUDA entry points
// ======================================================================================

torch::Tensor fused_dynamic_conv_forward_chunk_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    check_fdc_common_inputs(
        h,
        kc,
        mix,
        off,
        dilation
    );

    return fused_dynamic_conv_forward_direct_cuda(
        h,
        kc,
        mix,
        off,
        dilation
    );
}

std::vector<torch::Tensor> fused_dynamic_conv_backward_chunk_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    check_fdc_backward_inputs(
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );

    auto key = make_fdc_backward_cache_key(
        h,
        kc,
        off,
        dilation
    );

    FDCBackwardPlan plan;

    if (get_cached_fdc_backward_plan(
            key,
            &plan
        )) {
        if (!fdc_plan_available(
                plan,
                go,
                h,
                kc,
                mix,
                off,
                dilation
            )) {
            plan = fdc_default_backward_plan(
                h,
                kc,
                mix,
                off,
                dilation
            );
        }

        FDC_DEBUG_PATH(fdc_backward_plan_name_impl(plan));

        return fdc_run_backward_plan(
            plan,
            go,
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    plan = fdc_default_backward_plan(
        h,
        kc,
        mix,
        off,
        dilation
    );

    FDC_DEBUG_PATH(fdc_backward_plan_name_impl(plan));

    return fdc_run_backward_plan(
        plan,
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );
}

int64_t fused_dynamic_conv_backward_chunk_warmup_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation,
    int64_t repeat
) {
    check_fdc_backward_inputs(
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );

    TORCH_CHECK(repeat >= 1, "repeat must be >= 1");

    auto key = make_fdc_backward_cache_key(
        h,
        kc,
        off,
        dilation
    );

    auto plans = fdc_all_candidate_backward_plans(
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );

    FDCBackwardPlan best_plan = plans[0];
    float best_ms = std::numeric_limits<float>::infinity();

    for (FDCBackwardPlan p : plans) {
        float ms = fdc_time_backward_plan_median_ms(
            p,
            go,
            h,
            kc,
            mix,
            off,
            dilation,
            repeat
        );

        if (fdc_debug_path_enabled()) {
            TORCH_WARN(
                "FDC warmup candidate ",
                fdc_backward_plan_name_impl(p),
                " median_ms=",
                ms
            );
        }

        if (ms < best_ms) {
            best_ms = ms;
            best_plan = p;
        }
    }

    set_cached_fdc_backward_plan(
        key,
        best_plan
    );

    if (fdc_debug_path_enabled()) {
        TORCH_WARN(
            "FDC warmup selected ",
            fdc_backward_plan_name_impl(best_plan),
            " median_ms=",
            best_ms
        );
    }

    return static_cast<int64_t>(best_plan);
}

int64_t fused_dynamic_conv_backward_chunk_cached_plan_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    int64_t off,
    int64_t dilation
) {
    auto key = make_fdc_backward_cache_key(
        h,
        kc,
        off,
        dilation
    );

    FDCBackwardPlan plan;

    if (!get_cached_fdc_backward_plan(
            key,
            &plan
        )) {
        return -1;
    }

    return static_cast<int64_t>(plan);
}

void fused_dynamic_conv_backward_chunk_clear_warmup_cache_cuda() {
    std::lock_guard<std::mutex> lock(g_fdc_backward_plan_cache_mutex);

    g_fdc_backward_plan_cache.clear();
}

std::string fused_dynamic_conv_backward_plan_name_cuda(
    int64_t plan_id
) {
    return std::string(
        fdc_backward_plan_name_impl(
            static_cast<FDCBackwardPlan>(plan_id)
        )
    );
}
