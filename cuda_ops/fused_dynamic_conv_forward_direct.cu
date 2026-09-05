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
    DirectN16K3D256 = 3,
    DirectN6K7D256 = 4,
    DirectN6K3D512 = 5,
    NewGemmNK3D256 = 6
};

static inline const char* fdc_forward_plan_name_impl(FDCForwardPlan p) {
    switch (p) {
        case FDCForwardPlan::DirectGeneric2D:
            return "forward_direct_generic_2d";

        case FDCForwardPlan::DirectGenericSmallNPreload:
            return "forward_direct_generic_smalln_preload";

        case FDCForwardPlan::DirectN6K3D256:
            return "forward_direct_n6k3_d256";

        case FDCForwardPlan::DirectN16K3D256:
            return "forward_direct_n16k3_d256";

        case FDCForwardPlan::DirectN6K7D256:
            return "forward_direct_n6k7_d256";

        case FDCForwardPlan::DirectN6K3D512:
            return "forward_direct_n6k3_d512";

        case FDCForwardPlan::NewGemmNK3D256:
            return "forward_new_gemm_nk3_d256";

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
    key.T = static_cast<int>(kc.size(1));
    key.N = static_cast<int>(kc.size(2));
    key.K = static_cast<int>(kc.size(3));
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
    bool n6k7d256;
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
    s.T = static_cast<int>(kc.size(1));
    s.N = static_cast<int>(kc.size(2));
    s.K = static_cast<int>(kc.size(3));
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

    s.n6k7d256 =
        s.B == 1 &&
        s.D == 256 &&
        s.N == 6 &&
        s.K == 7 &&
        s.dil1;

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
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,T,N,K].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(h.scalar_type() == kc.scalar_type(), "h/kc dtype mismatch.");
    TORCH_CHECK(h.scalar_type() == mix.scalar_type(), "h/mix dtype mismatch.");

    TORCH_CHECK(kc.size(0) == h.size(0), "B mismatch.");
    TORCH_CHECK(mix.size(0) == h.size(1), "D mismatch.");
    TORCH_CHECK(mix.size(1) == kc.size(2), "N mismatch.");

    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(dilation > 0, "dilation must be > 0.");
    TORCH_CHECK(off + kc.size(1) <= h.size(2), "off + T must be <= L.");

    TORCH_CHECK(
        h.scalar_type() == at::ScalarType::Float ||
        h.scalar_type() == at::ScalarType::Half ||
        h.scalar_type() == at::ScalarType::BFloat16,
        "unsupported dtype"
    );
}

// ======================================================================================
// Kernels: generic 2D
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_forward_direct_generic_2d_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int B,
    int D,
    int L,
    int T,
    int N,
    int K,
    int off,
    int dilation
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= B || d >= D || t >= T) {
        return;
    }

    float acc = 0.0f;

    int64_t hbase = ((int64_t)b * D + d) * L;
    int64_t obase = ((int64_t)b * D + d) * T;
    int64_t kbase = ((int64_t)b * T + t) * N * K;
    int64_t mbase = (int64_t)d * N;

    for (int kk = 0; kk < K; ++kk) {
        int s = off + t - kk * dilation;

        if (s < 0 || s >= L) {
            continue;
        }

        float w = 0.0f;

        for (int n = 0; n < N; ++n) {
            w += fdc_to_float_dev(kc[kbase + n * K + kk]) *
                 fdc_to_float_dev(mix[mbase + n]);
        }

        acc += w * fdc_to_float_dev(h[hbase + s]);
    }

    out[obase + t] = fdc_from_float_dev<scalar_t>(acc);
}

// ======================================================================================
// Kernels: generic small-N preload
// ======================================================================================

template <typename scalar_t, bool NO_BOUNDARY>
__global__ void fdc_forward_direct_generic_smalln_preload_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int B,
    int D,
    int L,
    int T,
    int N,
    int K,
    int off,
    int dilation
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(B) * D * T;

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(idx % T);
    int64_t q = idx / T;
    int d = static_cast<int>(q % D);
    int b = static_cast<int>(q / D);

    int64_t hbase = ((int64_t)b * D + d) * L;
    int64_t obase = ((int64_t)b * D + d) * T;
    int64_t kbase = ((int64_t)b * T + t) * N * K;
    int64_t mbase = (int64_t)d * N;

    float m[16];

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        m[i] = 0.0f;
    }

    for (int n = 0; n < N; ++n) {
        m[n] = fdc_to_float_dev(mix[mbase + n]);
    }

    float acc = 0.0f;

    for (int kk = 0; kk < K; ++kk) {
        int s = off + t - kk * dilation;

        if constexpr (!NO_BOUNDARY) {
            if (s < 0 || s >= L) {
                continue;
            }
        }

        float w = 0.0f;

        for (int n = 0; n < N; ++n) {
            w += fdc_to_float_dev(kc[kbase + n * K + kk]) * m[n];
        }

        acc += w * fdc_to_float_dev(h[hbase + s]);
    }

    out[obase + t] = fdc_from_float_dev<scalar_t>(acc);
}

// ======================================================================================
// Kernels: B=1, D=256, N=6, K=3, dilation=1
// ======================================================================================

template <typename scalar_t, bool NO_BOUNDARY>
__global__ void fdc_forward_direct_n6k3_d256_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    float m0 = fdc_to_float_dev(mix[d * 6 + 0]);
    float m1 = fdc_to_float_dev(mix[d * 6 + 1]);
    float m2 = fdc_to_float_dev(mix[d * 6 + 2]);
    float m3 = fdc_to_float_dev(mix[d * 6 + 3]);
    float m4 = fdc_to_float_dev(mix[d * 6 + 4]);
    float m5 = fdc_to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int s = off + t - kk;

        if constexpr (!NO_BOUNDARY) {
            if (s < 0 || s >= L) {
                continue;
            }
        }

        int64_t kb = (int64_t)t * 18 + kk;

        float w =
            fdc_to_float_dev(kc[kb + 0 * 3]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * 3]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * 3]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * 3]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * 3]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * 3]) * m5;

        acc += w * fdc_to_float_dev(h[(int64_t)d * L + s]);
    }

    out[(int64_t)d * T + t] = fdc_from_float_dev<scalar_t>(acc);
}

// ======================================================================================
// Kernels: B=1, D=512, N=6, K=3, dilation=1
// ======================================================================================

template <typename scalar_t, bool NO_BOUNDARY>
__global__ void fdc_forward_direct_n6k3_d512_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 512 * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    float m0 = fdc_to_float_dev(mix[d * 6 + 0]);
    float m1 = fdc_to_float_dev(mix[d * 6 + 1]);
    float m2 = fdc_to_float_dev(mix[d * 6 + 2]);
    float m3 = fdc_to_float_dev(mix[d * 6 + 3]);
    float m4 = fdc_to_float_dev(mix[d * 6 + 4]);
    float m5 = fdc_to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int s = off + t - kk;

        if constexpr (!NO_BOUNDARY) {
            if (s < 0 || s >= L) {
                continue;
            }
        }

        int64_t kb = (int64_t)t * 18 + kk;

        float w =
            fdc_to_float_dev(kc[kb + 0 * 3]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * 3]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * 3]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * 3]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * 3]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * 3]) * m5;

        acc += w * fdc_to_float_dev(h[(int64_t)d * L + s]);
    }

    out[(int64_t)d * T + t] = fdc_from_float_dev<scalar_t>(acc);
}

// ======================================================================================
// Kernels: B=1, D=256, N=16, K=3, dilation=1
// ======================================================================================

template <typename scalar_t, bool NO_BOUNDARY>
__global__ void fdc_forward_direct_n16k3_d256_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int64_t mb = (int64_t)d * 16;

    float m0 = fdc_to_float_dev(mix[mb + 0]);
    float m1 = fdc_to_float_dev(mix[mb + 1]);
    float m2 = fdc_to_float_dev(mix[mb + 2]);
    float m3 = fdc_to_float_dev(mix[mb + 3]);
    float m4 = fdc_to_float_dev(mix[mb + 4]);
    float m5 = fdc_to_float_dev(mix[mb + 5]);
    float m6 = fdc_to_float_dev(mix[mb + 6]);
    float m7 = fdc_to_float_dev(mix[mb + 7]);
    float m8 = fdc_to_float_dev(mix[mb + 8]);
    float m9 = fdc_to_float_dev(mix[mb + 9]);
    float m10 = fdc_to_float_dev(mix[mb + 10]);
    float m11 = fdc_to_float_dev(mix[mb + 11]);
    float m12 = fdc_to_float_dev(mix[mb + 12]);
    float m13 = fdc_to_float_dev(mix[mb + 13]);
    float m14 = fdc_to_float_dev(mix[mb + 14]);
    float m15 = fdc_to_float_dev(mix[mb + 15]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int s = off + t - kk;

        if constexpr (!NO_BOUNDARY) {
            if (s < 0 || s >= L) {
                continue;
            }
        }

        int64_t kb = (int64_t)t * 48 + kk;

        float w =
            fdc_to_float_dev(kc[kb + 0 * 3]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * 3]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * 3]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * 3]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * 3]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * 3]) * m5 +
            fdc_to_float_dev(kc[kb + 6 * 3]) * m6 +
            fdc_to_float_dev(kc[kb + 7 * 3]) * m7 +
            fdc_to_float_dev(kc[kb + 8 * 3]) * m8 +
            fdc_to_float_dev(kc[kb + 9 * 3]) * m9 +
            fdc_to_float_dev(kc[kb + 10 * 3]) * m10 +
            fdc_to_float_dev(kc[kb + 11 * 3]) * m11 +
            fdc_to_float_dev(kc[kb + 12 * 3]) * m12 +
            fdc_to_float_dev(kc[kb + 13 * 3]) * m13 +
            fdc_to_float_dev(kc[kb + 14 * 3]) * m14 +
            fdc_to_float_dev(kc[kb + 15 * 3]) * m15;

        acc += w * fdc_to_float_dev(h[(int64_t)d * L + s]);
    }

    out[(int64_t)d * T + t] = fdc_from_float_dev<scalar_t>(acc);
}

// ======================================================================================
// Kernels: B=1, D=256, N=6, K=7, dilation=1
// ======================================================================================

template <typename scalar_t, bool NO_BOUNDARY>
__global__ void fdc_forward_direct_n6k7_d256_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    float m0 = fdc_to_float_dev(mix[d * 6 + 0]);
    float m1 = fdc_to_float_dev(mix[d * 6 + 1]);
    float m2 = fdc_to_float_dev(mix[d * 6 + 2]);
    float m3 = fdc_to_float_dev(mix[d * 6 + 3]);
    float m4 = fdc_to_float_dev(mix[d * 6 + 4]);
    float m5 = fdc_to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 7; ++kk) {
        int s = off + t - kk;

        if constexpr (!NO_BOUNDARY) {
            if (s < 0 || s >= L) {
                continue;
            }
        }

        int64_t kb = (int64_t)t * 42 + kk;

        float w =
            fdc_to_float_dev(kc[kb + 0 * 7]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * 7]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * 7]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * 7]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * 7]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * 7]) * m5;

        acc += w * fdc_to_float_dev(h[(int64_t)d * L + s]);
    }

    out[(int64_t)d * T + t] = fdc_from_float_dev<scalar_t>(acc);
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

        case FDCForwardPlan::DirectN16K3D256:
            return s.n16k3d256;

        case FDCForwardPlan::DirectN6K7D256:
            return s.n6k7d256;

        case FDCForwardPlan::DirectN6K3D512:
            return s.n6k3d512;

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
// Plan runners
// ======================================================================================

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_generic_2d_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    int B = static_cast<int>(h.size(0));
    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));
    int K = static_cast<int>(kc.size(3));

    auto out = torch::empty(
        {B, D, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 block(16, 16);
    dim3 grid(
        (T + 15) / 16,
        (D + 15) / 16,
        B
    );

    fdc_forward_direct_generic_2d_kernel<scalar_t><<<
        grid,
        block,
        0,
        stream
    >>>(
        h.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        B,
        D,
        L,
        T,
        N,
        K,
        static_cast<int>(off),
        static_cast<int>(dilation)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_generic_smalln_preload_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    int B = static_cast<int>(h.size(0));
    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));
    int K = static_cast<int>(kc.size(3));

    auto out = torch::empty(
        {B, D, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t total = static_cast<int64_t>(B) * D * T;

    bool no_boundary =
        static_cast<int>(dilation) == 1 &&
        static_cast<int>(off) >= K - 1;

    if (no_boundary) {
        fdc_forward_direct_generic_smalln_preload_kernel<scalar_t, true><<<
            (total + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            D,
            L,
            T,
            N,
            K,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    } else {
        fdc_forward_direct_generic_smalln_preload_kernel<scalar_t, false><<<
            (total + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            D,
            L,
            T,
            N,
            K,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_n6k3_d256_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    auto out = torch::empty(
        {1, 256, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool no_boundary = static_cast<int>(off) >= 2;

    if (no_boundary) {
        fdc_forward_direct_n6k3_d256_kernel<scalar_t, true><<<
            (256 * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    } else {
        fdc_forward_direct_n6k3_d256_kernel<scalar_t, false><<<
            (256 * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_n6k3_d512_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    auto out = torch::empty(
        {1, 512, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool no_boundary = static_cast<int>(off) >= 2;

    if (no_boundary) {
        fdc_forward_direct_n6k3_d512_kernel<scalar_t, true><<<
            (512 * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    } else {
        fdc_forward_direct_n6k3_d512_kernel<scalar_t, false><<<
            (512 * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_n16k3_d256_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    auto out = torch::empty(
        {1, 256, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool no_boundary = static_cast<int>(off) >= 2;

    if (no_boundary) {
        fdc_forward_direct_n16k3_d256_kernel<scalar_t, true><<<
            (256 * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    } else {
        fdc_forward_direct_n16k3_d256_kernel<scalar_t, false><<<
            (256 * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_n6k7_d256_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    auto out = torch::empty(
        {1, 256, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool no_boundary = static_cast<int>(off) >= 6;

    if (no_boundary) {
        fdc_forward_direct_n6k7_d256_kernel<scalar_t, true><<<
            (256 * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    } else {
        fdc_forward_direct_n6k7_d256_kernel<scalar_t, false><<<
            (256 * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

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
            if (h.scalar_type() == at::ScalarType::Float) {
                return fdc_run_forward_direct_generic_2d_typed<float>(
                    h,
                    kc,
                    mix,
                    off,
                    dilation
                );
            }

            if (h.scalar_type() == at::ScalarType::Half) {
                return fdc_run_forward_direct_generic_2d_typed<c10::Half>(
                    h,
                    kc,
                    mix,
                    off,
                    dilation
                );
            }

            return fdc_run_forward_direct_generic_2d_typed<c10::BFloat16>(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::DirectGenericSmallNPreload:
            if (h.scalar_type() == at::ScalarType::Float) {
                return fdc_run_forward_direct_generic_smalln_preload_typed<float>(
                    h,
                    kc,
                    mix,
                    off,
                    dilation
                );
            }

            if (h.scalar_type() == at::ScalarType::Half) {
                return fdc_run_forward_direct_generic_smalln_preload_typed<c10::Half>(
                    h,
                    kc,
                    mix,
                    off,
                    dilation
                );
            }

            return fdc_run_forward_direct_generic_smalln_preload_typed<c10::BFloat16>(
                h,
                kc,
                mix,
                off,
                dilation
            );

        case FDCForwardPlan::DirectN6K3D256:
            if (h.scalar_type() == at::ScalarType::Float) {
                return fdc_run_forward_direct_n6k3_d256_typed<float>(
                    h,
                    kc,
                    mix,
                    off
                );
            }

            if (h.scalar_type() == at::ScalarType::Half) {
                return fdc_run_forward_direct_n6k3_d256_typed<c10::Half>(
                    h,
                    kc,
                    mix,
                    off
                );
            }

            return fdc_run_forward_direct_n6k3_d256_typed<c10::BFloat16>(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::DirectN6K3D512:
            if (h.scalar_type() == at::ScalarType::Float) {
                return fdc_run_forward_direct_n6k3_d512_typed<float>(
                    h,
                    kc,
                    mix,
                    off
                );
            }

            if (h.scalar_type() == at::ScalarType::Half) {
                return fdc_run_forward_direct_n6k3_d512_typed<c10::Half>(
                    h,
                    kc,
                    mix,
                    off
                );
            }

            return fdc_run_forward_direct_n6k3_d512_typed<c10::BFloat16>(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::DirectN16K3D256:
            if (h.scalar_type() == at::ScalarType::Float) {
                return fdc_run_forward_direct_n16k3_d256_typed<float>(
                    h,
                    kc,
                    mix,
                    off
                );
            }

            if (h.scalar_type() == at::ScalarType::Half) {
                return fdc_run_forward_direct_n16k3_d256_typed<c10::Half>(
                    h,
                    kc,
                    mix,
                    off
                );
            }

            return fdc_run_forward_direct_n16k3_d256_typed<c10::BFloat16>(
                h,
                kc,
                mix,
                off
            );

        case FDCForwardPlan::DirectN6K7D256:
            if (h.scalar_type() == at::ScalarType::Float) {
                return fdc_run_forward_direct_n6k7_d256_typed<float>(
                    h,
                    kc,
                    mix,
                    off
                );
            }

            if (h.scalar_type() == at::ScalarType::Half) {
                return fdc_run_forward_direct_n6k7_d256_typed<c10::Half>(
                    h,
                    kc,
                    mix,
                    off
                );
            }

            return fdc_run_forward_direct_n6k7_d256_typed<c10::BFloat16>(
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

    if (s.n6k3d256) {
        return FDCForwardPlan::DirectN6K3D256;
    }

    if (s.n16k3d256) {
        return FDCForwardPlan::DirectN16K3D256;
    }

    if (s.n6k7d256) {
        return FDCForwardPlan::DirectN6K7D256;
    }

    if (s.n6k3d512) {
        return FDCForwardPlan::DirectN6K3D512;
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
        FDCForwardPlan::DirectN6K3D256,
        FDCForwardPlan::DirectN16K3D256,
        FDCForwardPlan::DirectN6K7D256,
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

    TORCH_CHECK(
        cudaEventCreate(&start) == cudaSuccess,
        "cudaEventCreate start failed"
    );

    TORCH_CHECK(
        cudaEventCreate(&stop) == cudaSuccess,
        "cudaEventCreate stop failed"
    );

    TORCH_CHECK(
        cudaEventRecord(start, stream) == cudaSuccess,
        "cudaEventRecord start failed"
    );

    auto out = fdc_run_forward_plan(
        plan,
        h,
        kc,
        mix,
        off,
        dilation
    );

    (void)out;

    TORCH_CHECK(
        cudaEventRecord(stop, stream) == cudaSuccess,
        "cudaEventRecord stop failed"
    );

    TORCH_CHECK(
        cudaEventSynchronize(stop) == cudaSuccess,
        "cudaEventSynchronize stop failed"
    );

    float ms = 0.0f;

    TORCH_CHECK(
        cudaEventElapsedTime(&ms, start, stop) == cudaSuccess,
        "cudaEventElapsedTime failed"
    );

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

    TORCH_CHECK(
        cudaStreamSynchronize(stream) == cudaSuccess,
        "cudaStreamSynchronize failed before timing"
    );

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
