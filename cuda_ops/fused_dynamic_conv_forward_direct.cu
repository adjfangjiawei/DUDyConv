#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Forward plan enum
// ======================================================================================

enum class FDCForwardPlan : int {
    DirectGeneric2D = 0,
    DirectGenericSmallNPreload = 1,
    DirectN6K3D256 = 2,
    DirectN6K3D512 = 5,
    NewGemmNK3D256 = 6,
    N32K3D256 = 7,
    N16K3D256 = 8,
    N64K3D256 = 9,
    N128K3D256 = 10,

    // 新增方案：
    // forward_n128_k3_d256_batched_sgemm
    //
    // 不替换旧 N128K3D256。
    // 只是作为独立候选 plan 参与 warmup / cache / dispatch。
    N128K3D256BatchedSgemm = 11
};

static inline const char* fdc_forward_plan_name_impl(FDCForwardPlan p) {
    switch (p) {
        case FDCForwardPlan::DirectGeneric2D:
            return "forward_direct_generic_2d";

        case FDCForwardPlan::DirectGenericSmallNPreload:
            return "forward_direct_generic_smalln_preload";

        case FDCForwardPlan::DirectN6K3D256:
            return "forward_direct_n6k3_d256";

        case FDCForwardPlan::DirectN6K3D512:
            return "forward_direct_n6k3_d512";

        case FDCForwardPlan::NewGemmNK3D256:
            return "forward_new_gemm_nk3_d256";

        case FDCForwardPlan::N16K3D256:
            return "forward_n16_k3_d256";

        case FDCForwardPlan::N32K3D256:
            return "forward_n32_k3_d256";

        case FDCForwardPlan::N64K3D256:
            return "forward_n64_k3_d256";

        case FDCForwardPlan::N128K3D256:
            return "forward_n128_k3_d256";

        case FDCForwardPlan::N128K3D256BatchedSgemm:
            return "forward_n128_k3_d256_batched_sgemm";

        default:
            return "unknown";
    }
}

// ======================================================================================
// Cache key
// ======================================================================================

struct FDCForwardCacheKey {
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

    bool operator==(const FDCForwardCacheKey& o) const {
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

struct FDCForwardCacheKeyHash {
    std::size_t operator()(const FDCForwardCacheKey& k) const {
        std::size_t h = 1469598103934665603ull;

        auto mix_hash = [&](int v) {
            h ^= static_cast<std::size_t>(v);
            h *= 1099511628211ull;
        };

        mix_hash(k.device);
        mix_hash(k.sm);
        mix_hash(k.dtype);
        mix_hash(k.B);
        mix_hash(k.D);
        mix_hash(k.L);
        mix_hash(k.T);
        mix_hash(k.N);
        mix_hash(k.K);
        mix_hash(k.off);
        mix_hash(k.dilation);

        return h;
    }
};

static std::unordered_map<FDCForwardCacheKey, FDCForwardPlan, FDCForwardCacheKeyHash>
    g_fdc_forward_plan_cache;

static std::mutex g_fdc_forward_plan_cache_mutex;

static inline int fdc_forward_dtype_tag(at::ScalarType t) {
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

static inline int fdc_forward_get_sm(torch::Tensor x) {
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(
        &prop,
        x.get_device()
    );

    TORCH_CHECK(err == cudaSuccess, "cudaGetDeviceProperties failed");

    return prop.major * 10 + prop.minor;
}

static inline FDCForwardCacheKey make_fdc_forward_cache_key(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDCForwardCacheKey key;

    key.device = h.get_device();
    key.sm = fdc_forward_get_sm(h);
    key.dtype = fdc_forward_dtype_tag(h.scalar_type());
    key.B = static_cast<int>(h.size(0));
    key.D = static_cast<int>(h.size(1));
    key.L = static_cast<int>(h.size(2));
    key.K = static_cast<int>(kc.size(1));
    key.N = static_cast<int>(kc.size(2));
    key.T = static_cast<int>(kc.size(3));
    key.off = static_cast<int>(off);
    key.dilation = static_cast<int>(dilation);

    (void)mix;

    return key;
}

static inline bool get_cached_fdc_forward_plan(
    const FDCForwardCacheKey& key,
    FDCForwardPlan* plan
) {
    std::lock_guard<std::mutex> lock(g_fdc_forward_plan_cache_mutex);

    auto it = g_fdc_forward_plan_cache.find(key);

    if (it == g_fdc_forward_plan_cache.end()) {
        return false;
    }

    *plan = it->second;

    return true;
}

static inline void set_cached_fdc_forward_plan(
    const FDCForwardCacheKey& key,
    FDCForwardPlan plan
) {
    std::lock_guard<std::mutex> lock(g_fdc_forward_plan_cache_mutex);

    g_fdc_forward_plan_cache[key] = plan;
}

// ======================================================================================
// Shape helper
// ======================================================================================

struct FDCForwardShapeInfo {
    int B;
    int D;
    int L;
    int T;
    int N;
    int K;
    int off;
    int dilation;
    int sm;

    bool is_fp32;
    bool is_fp16;
    bool is_bf16;

    bool dil1;
    bool no_boundary;

    bool n6k3d256;
    bool n16k3d256;
    bool n32k3d256;
    bool n64k3d256;
    bool n128k3d256;
    bool n128k3d256_batched_sgemm;
    bool n6k3d512;
    bool smalln_preload;
};

static inline FDCForwardShapeInfo make_fdc_forward_shape_info(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDCForwardShapeInfo s;

    s.B = static_cast<int>(h.size(0));
    s.D = static_cast<int>(h.size(1));
    s.L = static_cast<int>(h.size(2));
    s.K = static_cast<int>(kc.size(1));
    s.N = static_cast<int>(kc.size(2));
    s.T = static_cast<int>(kc.size(3));
    s.off = static_cast<int>(off);
    s.dilation = static_cast<int>(dilation);
    s.sm = fdc_forward_get_sm(h);

    s.is_fp32 = h.scalar_type() == at::ScalarType::Float;
    s.is_fp16 = h.scalar_type() == at::ScalarType::Half;
    s.is_bf16 = h.scalar_type() == at::ScalarType::BFloat16;

    s.dil1 = s.dilation == 1;
    s.no_boundary = s.dil1 && s.off >= (s.K - 1);

    s.n6k3d256 =
        s.B == 1 &&
        s.D == 256 &&
        s.N == 6 &&
        s.K == 3 &&
        s.dil1;

    s.n16k3d256 =
        s.B == 1 &&
        s.D == 256 &&
        s.N == 16 &&
        s.K == 3 &&
        s.dil1;

    s.n32k3d256 =
        s.B == 1 &&
        s.D == 256 &&
        s.N == 32 &&
        s.K == 3 &&
        s.dil1 &&
        s.is_fp32 &&
        s.T >= 512;

    s.n64k3d256 =
        s.B == 1 &&
        s.D == 256 &&
        s.N == 64 &&
        s.K == 3 &&
        s.dil1 &&
        s.is_fp32 &&
        s.T >= 512;

    s.n128k3d256 =
        s.B == 1 &&
        s.D == 256 &&
        s.N == 128 &&
        s.K == 3 &&
        s.dil1 &&
        s.is_fp32 &&
        s.T >= 512;

    // 新增 batched SGEMM forward plan 的粗略 shape 标记。
    // 真正可用性仍由 fdc_forward_n128_k3_d256_batched_sgemm_available_cuda 判断。
    s.n128k3d256_batched_sgemm =
        s.B == 1 &&
        s.D == 256 &&
        s.N == 128 &&
        s.K == 3 &&
        s.dil1 &&
        s.is_fp32 &&
        s.T >= 512;

    s.n6k3d512 =
        s.B == 1 &&
        s.D == 512 &&
        s.N == 6 &&
        s.K == 3 &&
        s.dil1;

    s.smalln_preload =
        s.N <= 16 &&
        s.K <= 16;

    (void)mix;

    return s;
}

// ======================================================================================
// Forward validation
// ======================================================================================

static inline void check_fdc_forward_inputs(
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

// ======================================================================================
// Plan availability
// ======================================================================================

static inline bool fdc_forward_plan_available(
    FDCForwardPlan plan,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDCForwardShapeInfo s = make_fdc_forward_shape_info(
        h,
        kc,
        mix,
        off,
        dilation
    );

    switch (plan) {
        case FDCForwardPlan::DirectGeneric2D:
            return true;

        case FDCForwardPlan::DirectGenericSmallNPreload:
            return s.smalln_preload;

        case FDCForwardPlan::DirectN6K3D256:
            return s.n6k3d256;

        case FDCForwardPlan::DirectN6K3D512:
            return s.n6k3d512;

        case FDCForwardPlan::N16K3D256:
            return fdc_forward_n16_k3_d256_available_cuda(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::N32K3D256:
            return fdc_forward_n32_k3_d256_available_cuda(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::N64K3D256:
            return fdc_forward_n64_k3_d256_available_cuda(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::N128K3D256:
            return fdc_forward_n128_k3_d256_available_cuda(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::N128K3D256BatchedSgemm:
            return fdc_forward_n128_k3_d256_batched_sgemm_available_cuda(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::NewGemmNK3D256:
            return fdc_new_gemm_nk3_d256_available_cuda(
                h,
                kc,
                mix,
                off,
                dilation
            );

        default:
            return false;
    }
}

// ======================================================================================
// Plan runner
// ======================================================================================

static torch::Tensor fdc_run_forward_plan(
    FDCForwardPlan plan,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    switch (plan) {
        case FDCForwardPlan::DirectGeneric2D:
            return fdc_forward_direct_generic_2d_cuda(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::DirectGenericSmallNPreload:
            return fdc_forward_direct_generic_smalln_preload_cuda(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::DirectN6K3D256:
            return fdc_forward_direct_n6k3_d256_cuda(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::DirectN6K3D512:
            return fdc_forward_direct_n6k3_d512_cuda(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::N16K3D256:
            return fdc_forward_n16_k3_d256_cuda(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::N32K3D256:
            return fdc_forward_n32_k3_d256_cuda(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::N64K3D256:
            return fdc_forward_n64_k3_d256_cuda(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::N128K3D256:
            return fdc_forward_n128_k3_d256_cuda(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::N128K3D256BatchedSgemm:
            return fdc_forward_n128_k3_d256_batched_sgemm_cuda(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::NewGemmNK3D256:
            return fdc_new_forward_gemm_nk3_d256_cuda(
                h,
                kc,
                mix,
                off
            );

        default:
            TORCH_CHECK(false, "unknown fused dynamic conv forward plan");
    }
}

// ======================================================================================
// Default heuristic
// ======================================================================================

static FDCForwardPlan fdc_default_forward_plan(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDCForwardShapeInfo s = make_fdc_forward_shape_info(
        h,
        kc,
        mix,
        off,
        dilation
    );

    if (s.n16k3d256 && fdc_forward_n16_k3_d256_available_cuda(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return FDCForwardPlan::N16K3D256;
    }

    if (s.n32k3d256 && fdc_forward_n32_k3_d256_available_cuda(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return FDCForwardPlan::N32K3D256;
    }

    if (s.n64k3d256 && fdc_forward_n64_k3_d256_available_cuda(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return FDCForwardPlan::N64K3D256;
    }

    // 默认启发式中，把新增 batched SGEMM 放在旧 N128 前面。
    //
    // 注意：
    //   这不是删除或修改旧 N128 plan。
    //   只是当没有 warmup cache 时，N=128,K=3,D=256,fp32,dilation=1
    //   会优先尝试新增方案。
    //
    // 如果你希望“默认不改变旧方案选择，只在 warmup 中被选择”，
    // 可以删除这个 if，让它只出现在 candidate list。
    if (s.n128k3d256_batched_sgemm &&
        fdc_forward_n128_k3_d256_batched_sgemm_available_cuda(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return FDCForwardPlan::N128K3D256BatchedSgemm;
    }

    if (s.n128k3d256 && fdc_forward_n128_k3_d256_available_cuda(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return FDCForwardPlan::N128K3D256;
    }

    if (s.n6k3d256) {
        return FDCForwardPlan::DirectN6K3D256;
    }

    if (s.n6k3d512) {
        return FDCForwardPlan::DirectN6K3D512;
    }

    if (fdc_new_gemm_nk3_d256_available_cuda(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return FDCForwardPlan::NewGemmNK3D256;
    }

    if (s.smalln_preload) {
        return FDCForwardPlan::DirectGenericSmallNPreload;
    }

    return FDCForwardPlan::DirectGeneric2D;
}

// ======================================================================================
// Candidate list
// ======================================================================================

static std::vector<FDCForwardPlan> fdc_all_candidate_forward_plans(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    std::vector<FDCForwardPlan> plans;

    FDCForwardPlan all[] = {
        FDCForwardPlan::N16K3D256,
        FDCForwardPlan::N32K3D256,
        FDCForwardPlan::N64K3D256,

        // 新增 batched SGEMM 方案。
        // 和旧 N128K3D256 都进入候选，由 warmup 实测选择。
        FDCForwardPlan::N128K3D256BatchedSgemm,
        FDCForwardPlan::N128K3D256,

        FDCForwardPlan::DirectN6K3D256,
        FDCForwardPlan::DirectN6K3D512,
        FDCForwardPlan::DirectGenericSmallNPreload,
        FDCForwardPlan::NewGemmNK3D256,
        FDCForwardPlan::DirectGeneric2D
    };

    for (FDCForwardPlan p : all) {
        if (fdc_forward_plan_available(
                p,
                h,
                kc,
                mix,
                off,
                dilation
            )) {
            plans.push_back(p);
        }
    }

    TORCH_CHECK(!plans.empty(), "no available fused dynamic conv forward plan");

    return plans;
}

// ======================================================================================
// Timing
// ======================================================================================

static float fdc_time_forward_plan_once_ms(
    FDCForwardPlan plan,
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

    auto out = fdc_run_forward_plan(
        plan,
        h,
        kc,
        mix,
        off,
        dilation
    );

    (void)out;

    TORCH_CHECK(cudaEventRecord(stop, stream) == cudaSuccess, "cudaEventRecord stop failed");
    TORCH_CHECK(cudaEventSynchronize(stop) == cudaSuccess, "cudaEventSynchronize stop failed");

    float ms = 0.0f;

    TORCH_CHECK(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess, "cudaEventElapsedTime failed");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

static float fdc_time_forward_plan_median_ms(
    FDCForwardPlan plan,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation,
    int64_t repeat
) {
    std::vector<float> times;
    times.reserve(static_cast<size_t>(repeat));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int dry = 0; dry < 1; ++dry) {
        auto out = fdc_run_forward_plan(
            plan,
            h,
            kc,
            mix,
            off,
            dilation
        );

        (void)out;
    }

    TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "cudaStreamSynchronize failed before timing");

    for (int64_t i = 0; i < repeat; ++i) {
        times.push_back(
            fdc_time_forward_plan_once_ms(
                plan,
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
// Public forward entry
// ======================================================================================

torch::Tensor fused_dynamic_conv_forward_direct_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    check_fdc_forward_inputs(
        h,
        kc,
        mix,
        off,
        dilation
    );

    auto key = make_fdc_forward_cache_key(
        h,
        kc,
        mix,
        off,
        dilation
    );

    FDCForwardPlan plan;

    if (get_cached_fdc_forward_plan(
            key,
            &plan
        )) {
        if (!fdc_forward_plan_available(
                plan,
                h,
                kc,
                mix,
                off,
                dilation
            )) {
            plan = fdc_default_forward_plan(
                h,
                kc,
                mix,
                off,
                dilation
            );
        }

        FDC_DEBUG_PATH(fdc_forward_plan_name_impl(plan));

        return fdc_run_forward_plan(
            plan,
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    plan = fdc_default_forward_plan(
        h,
        kc,
        mix,
        off,
        dilation
    );

    FDC_DEBUG_PATH(fdc_forward_plan_name_impl(plan));

    return fdc_run_forward_plan(
        plan,
        h,
        kc,
        mix,
        off,
        dilation
    );
}

int64_t fused_dynamic_conv_forward_direct_warmup_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation,
    int64_t repeat
) {
    check_fdc_forward_inputs(
        h,
        kc,
        mix,
        off,
        dilation
    );

    TORCH_CHECK(repeat >= 1, "repeat must be >= 1");

    auto key = make_fdc_forward_cache_key(
        h,
        kc,
        mix,
        off,
        dilation
    );

    auto plans = fdc_all_candidate_forward_plans(
        h,
        kc,
        mix,
        off,
        dilation
    );

    FDCForwardPlan best_plan = plans[0];
    float best_ms = std::numeric_limits<float>::infinity();

    for (FDCForwardPlan p : plans) {
        float ms = fdc_time_forward_plan_median_ms(
            p,
            h,
            kc,
            mix,
            off,
            dilation,
            repeat
        );

        if (fdc_debug_path_enabled()) {
            TORCH_WARN(
                "FDC forward warmup candidate ",
                fdc_forward_plan_name_impl(p),
                " median_ms=",
                ms
            );
        }

        if (ms < best_ms) {
            best_ms = ms;
            best_plan = p;
        }
    }

    set_cached_fdc_forward_plan(
        key,
        best_plan
    );

    if (fdc_debug_path_enabled()) {
        TORCH_WARN(
            "FDC forward warmup selected ",
            fdc_forward_plan_name_impl(best_plan),
            " median_ms=",
            best_ms
        );
    }

    return static_cast<int64_t>(best_plan);
}

int64_t fused_dynamic_conv_forward_direct_cached_plan_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    check_fdc_forward_inputs(
        h,
        kc,
        mix,
        off,
        dilation
    );

    auto key = make_fdc_forward_cache_key(
        h,
        kc,
        mix,
        off,
        dilation
    );

    FDCForwardPlan plan;

    if (!get_cached_fdc_forward_plan(
            key,
            &plan
        )) {
        return -1;
    }

    if (!fdc_forward_plan_available(
            plan,
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return -1;
    }

    return static_cast<int64_t>(plan);
}

void fused_dynamic_conv_forward_direct_clear_warmup_cache_cuda() {
    std::lock_guard<std::mutex> lock(g_fdc_forward_plan_cache_mutex);

    g_fdc_forward_plan_cache.clear();
}

std::string fused_dynamic_conv_forward_direct_plan_name_cuda(
    int64_t plan_id
) {
    return std::string(
        fdc_forward_plan_name_impl(
            static_cast<FDCForwardPlan>(plan_id)
        )
    );
}
