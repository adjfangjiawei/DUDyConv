#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "fused_dynamic_conv_common.cuh"
#include "fused_source_conv_mixed_plans.h"

// ======================================================================================
// Plan F4:
//   No-boundary fast path for fused source conv + GELU + source mix forward.
//
// Provided public entries:
//
//   1. fused_source_conv_mixed_forward_direct_generic_noboundary_cuda
//        arbitrary D, C, S
//
//   2. fused_source_conv_mixed_forward_direct_c8s3_noboundary_cuda
//        arbitrary D, C=8, S=3
//
//   3. fused_source_conv_mixed_forward_direct_c32s3_noboundary_cuda
//        arbitrary D, C=32, S=3
//
//   4. fused_source_conv_mixed_forward_d256_c16s3_noboundary_cuda
//        D=256, C=16, S=3
//
// Assumption:
//   off >= dilation * (S - 1)
//
// Also checked:
//   off >= 0
//   T > 0
//   off + T <= L
//
// Therefore for all t in [0,T):
//   src_j = off + t - dilation * (S - 1 - j)
// is always in [0,L).
//
// Notes:
//   - No source-channel dropout.
//   - No post-mix dropout.
//   - No residual.
//   - No backward.
//   - Accumulation is float.
//   - Output dtype equals input dtype.
// ======================================================================================

namespace {

constexpr int FSCM_D256 = 256;
constexpr int FSCM_C8 = 8;
constexpr int FSCM_C16 = 16;
constexpr int FSCM_C32 = 32;
constexpr int FSCM_S3 = 3;

// ======================================================================================
// dtype helpers
// ======================================================================================

template <typename scalar_t>
__device__ __forceinline__ float fscm_nb_to_float(
    scalar_t x
) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float fscm_nb_to_float<c10::Half>(
    c10::Half x
) {
    return __half2float(
        static_cast<__half>(x)
    );
}

template <>
__device__ __forceinline__ float fscm_nb_to_float<c10::BFloat16>(
    c10::BFloat16 x
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __bfloat162float(
        static_cast<__nv_bfloat16>(x)
    );
#else
    return static_cast<float>(x);
#endif
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t fscm_nb_from_float(
    float x
) {
    return static_cast<scalar_t>(x);
}

template <>
__device__ __forceinline__ c10::Half fscm_nb_from_float<c10::Half>(
    float x
) {
    return c10::Half(
        __float2half_rn(x)
    );
}

template <>
__device__ __forceinline__ c10::BFloat16 fscm_nb_from_float<c10::BFloat16>(
    float x
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return c10::BFloat16(
        __float2bfloat16_rn(x)
    );
#else
    return c10::BFloat16(x);
#endif
}

// ======================================================================================
// exact GELU
// ======================================================================================

__device__ __forceinline__ float fscm_nb_gelu_exact(
    float x
) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;

    return 0.5f * x * (
        1.0f + erff(
            x * inv_sqrt2
        )
    );
}

// ======================================================================================
// Generic no-boundary kernel
// arbitrary D,C,S
// ======================================================================================

template <typename scalar_t, bool HAS_BIAS>
__global__ void fscm_forward_direct_generic_noboundary_kernel(
    const scalar_t* __restrict__ x_norm,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ mix_weight,
    const scalar_t* __restrict__ mix_bias,
    scalar_t* __restrict__ out,
    int B,
    int D,
    int L,
    int T,
    int C,
    int S,
    int off,
    int dilation
) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t total =
        static_cast<int64_t>(B) *
        static_cast<int64_t>(D) *
        static_cast<int64_t>(T);

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(
        idx % T
    );

    int64_t q = idx / T;

    int d = static_cast<int>(
        q % D
    );

    int b = static_cast<int>(
        q / D
    );

    int64_t x_base =
        (
            static_cast<int64_t>(b) * D + d
        ) * L;

    int64_t weight_d_base =
        static_cast<int64_t>(d) * C * S;

    int64_t mix_d_base =
        static_cast<int64_t>(d) * C;

    float mixed = fscm_nb_to_float(
        mix_bias[d]
    );

    for (int c = 0; c < C; ++c) {
        float z = 0.0f;

        if constexpr (HAS_BIAS) {
            z = fscm_nb_to_float(
                bias[static_cast<int64_t>(d) * C + c]
            );
        }

        int64_t weight_c_base =
            weight_d_base +
            static_cast<int64_t>(c) * S;

        for (int j = 0; j < S; ++j) {
            int src =
                off +
                t -
                dilation * (S - 1 - j);

            float x_val = fscm_nb_to_float(
                x_norm[x_base + src]
            );

            float w_val = fscm_nb_to_float(
                weight[weight_c_base + j]
            );

            z += x_val * w_val;
        }

        float a = fscm_nb_gelu_exact(
            z
        );

        float m = fscm_nb_to_float(
            mix_weight[mix_d_base + c]
        );

        mixed += a * m;
    }

    int64_t out_idx =
        (
            static_cast<int64_t>(b) * D + d
        ) * T + t;

    out[out_idx] =
        fscm_nb_from_float<scalar_t>(
            mixed
        );
}

// ======================================================================================
// S=3 unrolled no-boundary kernel for arbitrary D and compile-time C
// Used by C8S3 and C32S3.
// ======================================================================================

template <typename scalar_t, int C, bool HAS_BIAS>
__global__ void fscm_forward_direct_cs3_noboundary_kernel(
    const scalar_t* __restrict__ x_norm,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ mix_weight,
    const scalar_t* __restrict__ mix_bias,
    scalar_t* __restrict__ out,
    int B,
    int D,
    int L,
    int T,
    int off,
    int dilation
) {
    constexpr int S = FSCM_S3;

    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t total =
        static_cast<int64_t>(B) *
        static_cast<int64_t>(D) *
        static_cast<int64_t>(T);

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(
        idx % T
    );

    int64_t q = idx / T;

    int d = static_cast<int>(
        q % D
    );

    int b = static_cast<int>(
        q / D
    );

    int src0 = off + t - 2 * dilation;
    int src1 = off + t - dilation;
    int src2 = off + t;

    int64_t x_base =
        (
            static_cast<int64_t>(b) * D + d
        ) * L;

    float x0 = fscm_nb_to_float(
        x_norm[x_base + src0]
    );

    float x1 = fscm_nb_to_float(
        x_norm[x_base + src1]
    );

    float x2 = fscm_nb_to_float(
        x_norm[x_base + src2]
    );

    int64_t weight_d_base =
        static_cast<int64_t>(d) * C * S;

    int64_t mix_d_base =
        static_cast<int64_t>(d) * C;

    int64_t bias_d_base =
        static_cast<int64_t>(d) * C;

    float mixed = fscm_nb_to_float(
        mix_bias[d]
    );

#pragma unroll
    for (int c = 0; c < C; ++c) {
        int64_t w_base =
            weight_d_base +
            static_cast<int64_t>(c) * S;

        int64_t cm_idx =
            static_cast<int64_t>(c);

        float z = 0.0f;

        if constexpr (HAS_BIAS) {
            z = fscm_nb_to_float(
                bias[bias_d_base + cm_idx]
            );
        }

        z += x0 * fscm_nb_to_float(
            weight[w_base + 0]
        );

        z += x1 * fscm_nb_to_float(
            weight[w_base + 1]
        );

        z += x2 * fscm_nb_to_float(
            weight[w_base + 2]
        );

        float a = fscm_nb_gelu_exact(
            z
        );

        float m = fscm_nb_to_float(
            mix_weight[mix_d_base + cm_idx]
        );

        mixed += a * m;
    }

    int64_t out_idx =
        (
            static_cast<int64_t>(b) * D + d
        ) * T + t;

    out[out_idx] =
        fscm_nb_from_float<scalar_t>(
            mixed
        );
}

// ======================================================================================
// D=256,C=16,S=3 no-boundary kernel
// ======================================================================================

template <typename scalar_t, bool HAS_BIAS>
__global__ void fscm_forward_d256_c16s3_noboundary_kernel(
    const scalar_t* __restrict__ x_norm,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ mix_weight,
    const scalar_t* __restrict__ mix_bias,
    scalar_t* __restrict__ out,
    int B,
    int L,
    int T,
    int off,
    int dilation
) {
    constexpr int D = FSCM_D256;
    constexpr int C = FSCM_C16;
    constexpr int S = FSCM_S3;

    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t total =
        static_cast<int64_t>(B) *
        static_cast<int64_t>(D) *
        static_cast<int64_t>(T);

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(
        idx % T
    );

    int64_t q = idx / T;

    int d = static_cast<int>(
        q & 255
    );

    int b = static_cast<int>(
        q >> 8
    );

    int src0 = off + t - 2 * dilation;
    int src1 = off + t - dilation;
    int src2 = off + t;

    int64_t x_base =
        (
            static_cast<int64_t>(b) * D + d
        ) * L;

    float x0 = fscm_nb_to_float(
        x_norm[x_base + src0]
    );

    float x1 = fscm_nb_to_float(
        x_norm[x_base + src1]
    );

    float x2 = fscm_nb_to_float(
        x_norm[x_base + src2]
    );

    int64_t weight_d_base =
        static_cast<int64_t>(d) * C * S;

    int64_t mix_d_base =
        static_cast<int64_t>(d) * C;

    int64_t bias_d_base =
        static_cast<int64_t>(d) * C;

    float mixed = fscm_nb_to_float(
        mix_bias[d]
    );

#pragma unroll
    for (int c = 0; c < C; ++c) {
        int64_t w_base =
            weight_d_base +
            static_cast<int64_t>(c) * S;

        float z = 0.0f;

        if constexpr (HAS_BIAS) {
            z = fscm_nb_to_float(
                bias[bias_d_base + c]
            );
        }

        z += x0 * fscm_nb_to_float(
            weight[w_base + 0]
        );

        z += x1 * fscm_nb_to_float(
            weight[w_base + 1]
        );

        z += x2 * fscm_nb_to_float(
            weight[w_base + 2]
        );

        float a = fscm_nb_gelu_exact(
            z
        );

        float m = fscm_nb_to_float(
            mix_weight[mix_d_base + c]
        );

        mixed += a * m;
    }

    int64_t out_idx =
        (
            static_cast<int64_t>(b) * D + d
        ) * T + t;

    out[out_idx] =
        fscm_nb_from_float<scalar_t>(
            mixed
        );
}

// ======================================================================================
// Validation helpers
// ======================================================================================

static void fscm_nb_check_common(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    const char* plan_name
) {
    TORCH_CHECK(
        x_norm.defined(),
        plan_name,
        ": x_norm must be defined."
    );

    TORCH_CHECK(
        weight.defined(),
        plan_name,
        ": weight must be defined."
    );

    TORCH_CHECK(
        mix_weight.defined(),
        plan_name,
        ": mix_weight must be defined."
    );

    TORCH_CHECK(
        mix_bias.defined(),
        plan_name,
        ": mix_bias must be defined."
    );

    TORCH_CHECK(
        x_norm.is_cuda(),
        plan_name,
        ": x_norm must be CUDA tensor."
    );

    TORCH_CHECK(
        weight.is_cuda(),
        plan_name,
        ": weight must be CUDA tensor."
    );

    TORCH_CHECK(
        mix_weight.is_cuda(),
        plan_name,
        ": mix_weight must be CUDA tensor."
    );

    TORCH_CHECK(
        mix_bias.is_cuda(),
        plan_name,
        ": mix_bias must be CUDA tensor."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.is_cuda(),
            plan_name,
            ": bias must be CUDA tensor when defined."
        );
    }

    TORCH_CHECK(
        x_norm.is_contiguous(),
        plan_name,
        ": x_norm must be contiguous."
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        plan_name,
        ": weight must be contiguous."
    );

    TORCH_CHECK(
        mix_weight.is_contiguous(),
        plan_name,
        ": mix_weight must be contiguous."
    );

    TORCH_CHECK(
        mix_bias.is_contiguous(),
        plan_name,
        ": mix_bias must be contiguous."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.is_contiguous(),
            plan_name,
            ": bias must be contiguous when defined."
        );
    }

    TORCH_CHECK(
        x_norm.dim() == 3,
        plan_name,
        ": x_norm must be [B,D,L]."
    );

    TORCH_CHECK(
        weight.dim() == 3,
        plan_name,
        ": weight must be [D,C,S]."
    );

    TORCH_CHECK(
        mix_weight.dim() == 2,
        plan_name,
        ": mix_weight must be [D,C]."
    );

    TORCH_CHECK(
        mix_bias.dim() == 1,
        plan_name,
        ": mix_bias must be [D]."
    );

    int64_t B = x_norm.size(0);
    int64_t D = x_norm.size(1);
    int64_t L = x_norm.size(2);
    int64_t Dw = weight.size(0);
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    TORCH_CHECK(
        B > 0,
        plan_name,
        ": B must be positive."
    );

    TORCH_CHECK(
        D > 0,
        plan_name,
        ": D must be positive."
    );

    TORCH_CHECK(
        L > 0,
        plan_name,
        ": L must be positive."
    );

    TORCH_CHECK(
        C > 0,
        plan_name,
        ": C must be positive."
    );

    TORCH_CHECK(
        S > 0,
        plan_name,
        ": S must be positive."
    );

    TORCH_CHECK(
        Dw == D,
        plan_name,
        ": weight D mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(0) == D,
        plan_name,
        ": mix_weight D mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(1) == C,
        plan_name,
        ": mix_weight C mismatch."
    );

    TORCH_CHECK(
        mix_bias.size(0) == D,
        plan_name,
        ": mix_bias D mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.dim() == 2,
            plan_name,
            ": bias must be [D,C] when defined."
        );

        TORCH_CHECK(
            bias.size(0) == D,
            plan_name,
            ": bias D mismatch."
        );

        TORCH_CHECK(
            bias.size(1) == C,
            plan_name,
            ": bias C mismatch."
        );
    }

    TORCH_CHECK(
        off >= 0,
        plan_name,
        ": off must be >= 0."
    );

    TORCH_CHECK(
        T > 0,
        plan_name,
        ": T must be positive."
    );

    TORCH_CHECK(
        off + T <= L,
        plan_name,
        ": off + T must be <= L."
    );

    TORCH_CHECK(
        dilation > 0,
        plan_name,
        ": dilation must be positive."
    );

    TORCH_CHECK(
        off >= dilation * (S - 1),
        plan_name,
        ": no-boundary plan requires off >= dilation * (S - 1)."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == weight.scalar_type(),
        plan_name,
        ": x_norm and weight dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == mix_weight.scalar_type(),
        plan_name,
        ": x_norm and mix_weight dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == mix_bias.scalar_type(),
        plan_name,
        ": x_norm and mix_bias dtype mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            x_norm.scalar_type() == bias.scalar_type(),
            plan_name,
            ": x_norm and bias dtype mismatch."
        );
    }

    TORCH_CHECK(
        x_norm.scalar_type() == at::ScalarType::Float ||
        x_norm.scalar_type() == at::ScalarType::Half ||
        x_norm.scalar_type() == at::ScalarType::BFloat16,
        plan_name,
        ": unsupported dtype."
    );
}

static void fscm_nb_check_c_s(
    torch::Tensor weight,
    int64_t expected_C,
    int64_t expected_S,
    const char* plan_name
) {
    TORCH_CHECK(
        weight.size(1) == expected_C,
        plan_name,
        ": expected C=",
        expected_C,
        ", got C=",
        weight.size(1),
        "."
    );

    TORCH_CHECK(
        weight.size(2) == expected_S,
        plan_name,
        ": expected S=",
        expected_S,
        ", got S=",
        weight.size(2),
        "."
    );
}

static void fscm_nb_check_d256_c16s3(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    torch::Tensor bias,
    const char* plan_name
) {
    TORCH_CHECK(
        x_norm.size(1) == FSCM_D256,
        plan_name,
        ": requires D == 256."
    );

    TORCH_CHECK(
        weight.size(0) == FSCM_D256,
        plan_name,
        ": weight D mismatch."
    );

    TORCH_CHECK(
        weight.size(1) == FSCM_C16,
        plan_name,
        ": requires C == 16."
    );

    TORCH_CHECK(
        weight.size(2) == FSCM_S3,
        plan_name,
        ": requires S == 3."
    );

    TORCH_CHECK(
        mix_weight.size(0) == FSCM_D256,
        plan_name,
        ": mix_weight D mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(1) == FSCM_C16,
        plan_name,
        ": mix_weight C mismatch."
    );

    TORCH_CHECK(
        mix_bias.size(0) == FSCM_D256,
        plan_name,
        ": mix_bias D mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.size(0) == FSCM_D256,
            plan_name,
            ": bias D mismatch."
        );

        TORCH_CHECK(
            bias.size(1) == FSCM_C16,
            plan_name,
            ": bias C mismatch."
        );
    }
}

// ======================================================================================
// Launch helpers
// ======================================================================================

template <typename scalar_t>
static torch::Tensor fscm_forward_generic_noboundary_typed(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T_arg,
    int64_t dilation
) {
    int B = static_cast<int>(
        x_norm.size(0)
    );

    int D = static_cast<int>(
        x_norm.size(1)
    );

    int L = static_cast<int>(
        x_norm.size(2)
    );

    int C = static_cast<int>(
        weight.size(1)
    );

    int S = static_cast<int>(
        weight.size(2)
    );

    int T = static_cast<int>(
        T_arg
    );

    auto out = torch::empty(
        {
            B,
            D,
            T
        },
        x_norm.options()
    );

    int64_t total =
        static_cast<int64_t>(B) *
        static_cast<int64_t>(D) *
        static_cast<int64_t>(T);

    int threads = 256;

    int blocks = static_cast<int>(
        (total + threads - 1) / threads
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    bool has_bias =
        bias.defined() &&
        bias.numel() > 0;

    if (has_bias) {
        fscm_forward_direct_generic_noboundary_kernel<scalar_t, true><<<
            blocks,
            threads,
            0,
            stream
        >>>(
            x_norm.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            mix_weight.data_ptr<scalar_t>(),
            mix_bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            D,
            L,
            T,
            C,
            S,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    } else {
        fscm_forward_direct_generic_noboundary_kernel<scalar_t, false><<<
            blocks,
            threads,
            0,
            stream
        >>>(
            x_norm.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            nullptr,
            mix_weight.data_ptr<scalar_t>(),
            mix_bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            D,
            L,
            T,
            C,
            S,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

template <typename scalar_t, int C>
static torch::Tensor fscm_forward_cs3_noboundary_typed(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T_arg,
    int64_t dilation
) {
    int B = static_cast<int>(
        x_norm.size(0)
    );

    int D = static_cast<int>(
        x_norm.size(1)
    );

    int L = static_cast<int>(
        x_norm.size(2)
    );

    int T = static_cast<int>(
        T_arg
    );

    auto out = torch::empty(
        {
            B,
            D,
            T
        },
        x_norm.options()
    );

    int64_t total =
        static_cast<int64_t>(B) *
        static_cast<int64_t>(D) *
        static_cast<int64_t>(T);

    int threads = 256;

    int blocks = static_cast<int>(
        (total + threads - 1) / threads
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    bool has_bias =
        bias.defined() &&
        bias.numel() > 0;

    if (has_bias) {
        fscm_forward_direct_cs3_noboundary_kernel<scalar_t, C, true><<<
            blocks,
            threads,
            0,
            stream
        >>>(
            x_norm.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            mix_weight.data_ptr<scalar_t>(),
            mix_bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            D,
            L,
            T,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    } else {
        fscm_forward_direct_cs3_noboundary_kernel<scalar_t, C, false><<<
            blocks,
            threads,
            0,
            stream
        >>>(
            x_norm.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            nullptr,
            mix_weight.data_ptr<scalar_t>(),
            mix_bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            D,
            L,
            T,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

template <typename scalar_t>
static torch::Tensor fscm_forward_d256_c16s3_noboundary_typed(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T_arg,
    int64_t dilation
) {
    int B = static_cast<int>(
        x_norm.size(0)
    );

    int L = static_cast<int>(
        x_norm.size(2)
    );

    int T = static_cast<int>(
        T_arg
    );

    auto out = torch::empty(
        {
            B,
            FSCM_D256,
            T
        },
        x_norm.options()
    );

    int64_t total =
        static_cast<int64_t>(B) *
        static_cast<int64_t>(FSCM_D256) *
        static_cast<int64_t>(T);

    int threads = 256;

    int blocks = static_cast<int>(
        (total + threads - 1) / threads
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    bool has_bias =
        bias.defined() &&
        bias.numel() > 0;

    if (has_bias) {
        fscm_forward_d256_c16s3_noboundary_kernel<scalar_t, true><<<
            blocks,
            threads,
            0,
            stream
        >>>(
            x_norm.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            mix_weight.data_ptr<scalar_t>(),
            mix_bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            L,
            T,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    } else {
        fscm_forward_d256_c16s3_noboundary_kernel<scalar_t, false><<<
            blocks,
            threads,
            0,
            stream
        >>>(
            x_norm.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            nullptr,
            mix_weight.data_ptr<scalar_t>(),
            mix_bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            L,
            T,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

} // namespace

// ======================================================================================
// Public entry: generic no-boundary
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_direct_generic_noboundary_cuda(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FSCM path: forward_direct_generic_noboundary");

    const char* plan_name =
        "fused_source_conv_mixed_forward_direct_generic_noboundary_cuda";

    fscm_nb_check_common(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        plan_name
    );

    if (x_norm.scalar_type() == at::ScalarType::Float) {
        return fscm_forward_generic_noboundary_typed<float>(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );
    }

    if (x_norm.scalar_type() == at::ScalarType::Half) {
        return fscm_forward_generic_noboundary_typed<c10::Half>(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );
    }

    return fscm_forward_generic_noboundary_typed<c10::BFloat16>(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation
    );
}

// ======================================================================================
// Public entry: C8S3 no-boundary
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_direct_c8s3_noboundary_cuda(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FSCM path: forward_direct_c8s3_noboundary");

    const char* plan_name =
        "fused_source_conv_mixed_forward_direct_c8s3_noboundary_cuda";

    fscm_nb_check_common(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        plan_name
    );

    fscm_nb_check_c_s(
        weight,
        FSCM_C8,
        FSCM_S3,
        plan_name
    );

    if (x_norm.scalar_type() == at::ScalarType::Float) {
        return fscm_forward_cs3_noboundary_typed<float, FSCM_C8>(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );
    }

    if (x_norm.scalar_type() == at::ScalarType::Half) {
        return fscm_forward_cs3_noboundary_typed<c10::Half, FSCM_C8>(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );
    }

    return fscm_forward_cs3_noboundary_typed<c10::BFloat16, FSCM_C8>(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation
    );
}

// ======================================================================================
// Public entry: C32S3 no-boundary
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_direct_c32s3_noboundary_cuda(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FSCM path: forward_direct_c32s3_noboundary");

    const char* plan_name =
        "fused_source_conv_mixed_forward_direct_c32s3_noboundary_cuda";

    fscm_nb_check_common(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        plan_name
    );

    fscm_nb_check_c_s(
        weight,
        FSCM_C32,
        FSCM_S3,
        plan_name
    );

    if (x_norm.scalar_type() == at::ScalarType::Float) {
        return fscm_forward_cs3_noboundary_typed<float, FSCM_C32>(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );
    }

    if (x_norm.scalar_type() == at::ScalarType::Half) {
        return fscm_forward_cs3_noboundary_typed<c10::Half, FSCM_C32>(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );
    }

    return fscm_forward_cs3_noboundary_typed<c10::BFloat16, FSCM_C32>(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation
    );
}

// ======================================================================================
// Public entry: D256 C16S3 no-boundary
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_d256_c16s3_noboundary_cuda(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FSCM path: forward_d256_c16s3_noboundary");

    const char* plan_name =
        "fused_source_conv_mixed_forward_d256_c16s3_noboundary_cuda";

    fscm_nb_check_common(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        plan_name
    );

    fscm_nb_check_d256_c16s3(
        x_norm,
        weight,
        mix_weight,
        mix_bias,
        bias,
        plan_name
    );

    if (x_norm.scalar_type() == at::ScalarType::Float) {
        return fscm_forward_d256_c16s3_noboundary_typed<float>(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );
    }

    if (x_norm.scalar_type() == at::ScalarType::Half) {
        return fscm_forward_d256_c16s3_noboundary_typed<c10::Half>(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );
    }

    return fscm_forward_d256_c16s3_noboundary_typed<c10::BFloat16>(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation
    );
}
