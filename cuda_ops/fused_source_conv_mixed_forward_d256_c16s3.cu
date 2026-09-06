#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "fused_dynamic_conv_common.cuh"
#include "fused_source_conv_mixed_plans.h"

// ======================================================================================
// Plan F3:
//   D=256, C=16, S=3, T-large direct unrolled source conv + GELU + source mix forward
//
// Target:
//   D == 256
//   C == 16
//   S == 3
//
// Input:
//   x_norm:
//     [B,256,L], contiguous
//
//   weight:
//     [256,16,3], contiguous
//
//   bias:
//     [256,16], contiguous, optional / empty
//
//   mix_weight:
//     [256,16], contiguous
//
//   mix_bias:
//     [256], contiguous
//
// Output:
//   out:
//     [B,256,T], contiguous
//
// Math:
//   src0 = off + t - 2 * dilation
//   src1 = off + t - dilation
//   src2 = off + t
//
//   z[b,d,c,t] = bias[d,c]
//                + x_norm[b,d,src0] * weight[d,c,0]
//                + x_norm[b,d,src1] * weight[d,c,1]
//                + x_norm[b,d,src2] * weight[d,c,2]
//
//   out[b,d,t] = mix_bias[d]
//                + sum_c GELU(z[b,d,c,t]) * mix_weight[d,c]
//
// Notes:
//   - No dropout.
//   - No residual.
//   - No backward.
//   - Accumulation is float.
//   - Output dtype equals input dtype.
//   - This plan specializes D/C/S and avoids runtime D/C/S arithmetic.
// ======================================================================================

namespace {

constexpr int FSCM_D256 = 256;
constexpr int FSCM_C16 = 16;
constexpr int FSCM_S3 = 3;

// ======================================================================================
// dtype helpers
// ======================================================================================

template <typename scalar_t>
__device__ __forceinline__ float fscm_d256_to_float(
    scalar_t x
) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float fscm_d256_to_float<c10::Half>(
    c10::Half x
) {
    return __half2float(
        static_cast<__half>(x)
    );
}

template <>
__device__ __forceinline__ float fscm_d256_to_float<c10::BFloat16>(
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
__device__ __forceinline__ scalar_t fscm_d256_from_float(
    float x
) {
    return static_cast<scalar_t>(x);
}

template <>
__device__ __forceinline__ c10::Half fscm_d256_from_float<c10::Half>(
    float x
) {
    return c10::Half(
        __float2half_rn(x)
    );
}

template <>
__device__ __forceinline__ c10::BFloat16 fscm_d256_from_float<c10::BFloat16>(
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

__device__ __forceinline__ float fscm_d256_gelu_exact(
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
// one C channel contribution
// ======================================================================================

template <typename scalar_t, bool HAS_BIAS>
__device__ __forceinline__ float fscm_d256_c16_one_c(
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ mix_weight,
    float x0,
    float x1,
    float x2,
    int d,
    int c
) {
    constexpr int C = FSCM_C16;
    constexpr int S = FSCM_S3;

    int64_t w_base =
        (
            static_cast<int64_t>(d) * C + c
        ) * S;

    int64_t m_idx =
        static_cast<int64_t>(d) * C + c;

    float z = 0.0f;

    if constexpr (HAS_BIAS) {
        z = fscm_d256_to_float(
            bias[m_idx]
        );
    }

    z += x0 * fscm_d256_to_float(
        weight[w_base + 0]
    );

    z += x1 * fscm_d256_to_float(
        weight[w_base + 1]
    );

    z += x2 * fscm_d256_to_float(
        weight[w_base + 2]
    );

    float a = fscm_d256_gelu_exact(
        z
    );

    float m = fscm_d256_to_float(
        mix_weight[m_idx]
    );

    return a * m;
}

// ======================================================================================
// D=256,C=16,S=3 kernel
//
// Mapping:
//   total = B * 256 * T
//   idx -> b,d,t
//
// Since D=256:
//   q = idx / T
//   d = q & 255 if B dimension is outer and D exactly 256.
//   b = q >> 8
//
// We still keep simple arithmetic for clarity and safety.
// ======================================================================================

template <typename scalar_t, bool HAS_BIAS, bool NO_BOUNDARY>
__global__ void fscm_forward_d256_c16s3_kernel(
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

    float x0 = 0.0f;
    float x1 = 0.0f;
    float x2 = 0.0f;

    if constexpr (NO_BOUNDARY) {
        x0 = fscm_d256_to_float(
            x_norm[x_base + src0]
        );

        x1 = fscm_d256_to_float(
            x_norm[x_base + src1]
        );

        x2 = fscm_d256_to_float(
            x_norm[x_base + src2]
        );
    } else {
        if (src0 >= 0 && src0 < L) {
            x0 = fscm_d256_to_float(
                x_norm[x_base + src0]
            );
        }

        if (src1 >= 0 && src1 < L) {
            x1 = fscm_d256_to_float(
                x_norm[x_base + src1]
            );
        }

        if (src2 >= 0 && src2 < L) {
            x2 = fscm_d256_to_float(
                x_norm[x_base + src2]
            );
        }
    }

    float mixed = fscm_d256_to_float(
        mix_bias[d]
    );

#pragma unroll
    for (int c = 0; c < FSCM_C16; ++c) {
        mixed += fscm_d256_c16_one_c<scalar_t, HAS_BIAS>(
            weight,
            bias,
            mix_weight,
            x0,
            x1,
            x2,
            d,
            c
        );
    }

    int64_t out_idx =
        (
            static_cast<int64_t>(b) * D + d
        ) * T + t;

    out[out_idx] =
        fscm_d256_from_float<scalar_t>(
            mixed
        );
}

// ======================================================================================
// Validation
// ======================================================================================

static void fscm_check_d256_c16s3_inputs(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    const char* plan_name =
        "fused_source_conv_mixed_forward_d256_c16s3_cuda";

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
        ": x_norm must be [B,256,L]."
    );

    TORCH_CHECK(
        weight.dim() == 3,
        plan_name,
        ": weight must be [256,16,3]."
    );

    TORCH_CHECK(
        mix_weight.dim() == 2,
        plan_name,
        ": mix_weight must be [256,16]."
    );

    TORCH_CHECK(
        mix_bias.dim() == 1,
        plan_name,
        ": mix_bias must be [256]."
    );

    int64_t B = x_norm.size(0);
    int64_t D = x_norm.size(1);
    int64_t L = x_norm.size(2);

    TORCH_CHECK(
        B > 0,
        plan_name,
        ": B must be positive."
    );

    TORCH_CHECK(
        D == FSCM_D256,
        plan_name,
        ": requires D == 256."
    );

    TORCH_CHECK(
        L > 0,
        plan_name,
        ": L must be positive."
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
            bias.dim() == 2,
            plan_name,
            ": bias must be [256,16] when defined."
        );

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

// ======================================================================================
// Typed launcher
// ======================================================================================

template <typename scalar_t>
static torch::Tensor fscm_forward_d256_c16s3_typed(
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

    bool no_boundary =
        static_cast<int>(off) >=
        static_cast<int>(dilation) * 2;

    if (has_bias) {
        if (no_boundary) {
            fscm_forward_d256_c16s3_kernel<scalar_t, true, true><<<
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
            fscm_forward_d256_c16s3_kernel<scalar_t, true, false><<<
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
        }
    } else {
        if (no_boundary) {
            fscm_forward_d256_c16s3_kernel<scalar_t, false, true><<<
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
        } else {
            fscm_forward_d256_c16s3_kernel<scalar_t, false, false><<<
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
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

} // namespace

// ======================================================================================
// Public entry
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_d256_c16s3_cuda(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FSCM path: forward_d256_c16s3");

    fscm_check_d256_c16s3_inputs(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation
    );

    if (x_norm.scalar_type() == at::ScalarType::Float) {
        return fscm_forward_d256_c16s3_typed<float>(
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
        return fscm_forward_d256_c16s3_typed<c10::Half>(
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

    return fscm_forward_d256_c16s3_typed<c10::BFloat16>(
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
