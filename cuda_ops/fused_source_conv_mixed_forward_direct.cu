#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>

#include "fused_dynamic_conv_common.cuh"
#include "fused_source_conv_mixed_plans.h"

// ======================================================================================
// Generic direct fused source conv + GELU + source mix forward
//
// Input:
//   x_norm:
//     [B,D,L], contiguous
//
//   weight:
//     [D,C,S], contiguous
//
//   bias:
//     [D,C], contiguous, optional
//
//   mix_weight:
//     [D,C], contiguous
//
//   mix_bias:
//     [D], contiguous
//
// Output:
//   out:
//     [B,D,T], contiguous
//
// Math:
//   z[b,d,c,t] = bias[d,c] +
//                sum_j x_norm[b,d, off+t - dilation*(S-1-j)] * weight[d,c,j]
//
//   out[b,d,t] = mix_bias[d] +
//                sum_c GELU(z[b,d,c,t]) * mix_weight[d,c]
//
// Notes:
//   - No source-channel dropout.
//   - No post-mix dropout.
//   - No residual add.
//   - Accumulation is float.
//   - Output dtype equals input dtype.
// ======================================================================================

namespace {

template <typename scalar_t>
__device__ __forceinline__ float fscm_to_float(scalar_t x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float fscm_to_float<c10::Half>(c10::Half x) {
    return __half2float(static_cast<__half>(x));
}

template <>
__device__ __forceinline__ float fscm_to_float<c10::BFloat16>(c10::BFloat16 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __bfloat162float(static_cast<__nv_bfloat16>(x));
#else
    return static_cast<float>(x);
#endif
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t fscm_from_float(float x) {
    return static_cast<scalar_t>(x);
}

template <>
__device__ __forceinline__ c10::Half fscm_from_float<c10::Half>(float x) {
    return c10::Half(__float2half_rn(x));
}

template <>
__device__ __forceinline__ c10::BFloat16 fscm_from_float<c10::BFloat16>(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return c10::BFloat16(__float2bfloat16_rn(x));
#else
    return c10::BFloat16(x);
#endif
}

__device__ __forceinline__ float fscm_gelu_exact(float x) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;

    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

template <typename scalar_t, bool HAS_BIAS, bool NO_BOUNDARY>
__global__ void fscm_forward_direct_generic_kernel(
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

    int t = static_cast<int>(idx % T);

    int64_t q = idx / T;

    int d = static_cast<int>(q % D);
    int b = static_cast<int>(q / D);

    int64_t x_base =
        (
            static_cast<int64_t>(b) * D + d
        ) * L;

    int64_t out_idx =
        (
            static_cast<int64_t>(b) * D + d
        ) * T + t;

    int64_t weight_d_base =
        static_cast<int64_t>(d) * C * S;

    int64_t mix_d_base =
        static_cast<int64_t>(d) * C;

    float mixed = fscm_to_float(
        mix_bias[d]
    );

    for (int c = 0; c < C; ++c) {
        float z = 0.0f;

        if constexpr (HAS_BIAS) {
            z = fscm_to_float(
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

            if constexpr (NO_BOUNDARY) {
                float x_val = fscm_to_float(
                    x_norm[x_base + src]
                );

                float w_val = fscm_to_float(
                    weight[weight_c_base + j]
                );

                z += x_val * w_val;
            } else {
                if (src >= 0 && src < L) {
                    float x_val = fscm_to_float(
                        x_norm[x_base + src]
                    );

                    float w_val = fscm_to_float(
                        weight[weight_c_base + j]
                    );

                    z += x_val * w_val;
                }
            }
        }

        float a = fscm_gelu_exact(
            z
        );

        float m = fscm_to_float(
            mix_weight[mix_d_base + c]
        );

        mixed += a * m;
    }

    out[out_idx] = fscm_from_float<scalar_t>(
        mixed
    );
}

template <typename scalar_t>
static torch::Tensor fscm_forward_direct_generic_typed(
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

    TORCH_CHECK(
        T > 0,
        "fused_source_conv_mixed_forward_direct_generic_cuda got non-positive T."
    );

    TORCH_CHECK(
        off >= 0,
        "fused_source_conv_mixed_forward_direct_generic_cuda: off must be >= 0."
    );

    TORCH_CHECK(
        off + T <= L,
        "fused_source_conv_mixed_forward_direct_generic_cuda: off + T must be <= L."
    );

    auto out = torch::empty(
        {
            B,
            D,
            T
        },
        x_norm.options()
    );

    int threads = 256;

    int64_t total =
        static_cast<int64_t>(B) *
        static_cast<int64_t>(D) *
        static_cast<int64_t>(T);

    int blocks = static_cast<int>(
        (total + threads - 1) / threads
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    bool has_bias = bias.defined() && bias.numel() > 0;

    bool no_boundary =
        static_cast<int>(off) >=
        static_cast<int>(dilation) * (S - 1);

    if (has_bias) {
        if (no_boundary) {
            fscm_forward_direct_generic_kernel<scalar_t, true, true><<<
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
            fscm_forward_direct_generic_kernel<scalar_t, true, false><<<
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
        }
    } else {
        if (no_boundary) {
            fscm_forward_direct_generic_kernel<scalar_t, false, true><<<
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
        } else {
            fscm_forward_direct_generic_kernel<scalar_t, false, false><<<
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
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

} // namespace

// ======================================================================================
// Public forward direct generic entry
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_direct_generic_cuda(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FSCM path: forward_direct_generic");

    TORCH_CHECK(
        x_norm.defined(),
        "x_norm must be defined."
    );

    TORCH_CHECK(
        weight.defined(),
        "weight must be defined."
    );

    TORCH_CHECK(
        mix_weight.defined(),
        "mix_weight must be defined."
    );

    TORCH_CHECK(
        mix_bias.defined(),
        "mix_bias must be defined."
    );

    TORCH_CHECK(
        x_norm.is_cuda(),
        "x_norm must be CUDA tensor."
    );

    TORCH_CHECK(
        weight.is_cuda(),
        "weight must be CUDA tensor."
    );

    TORCH_CHECK(
        mix_weight.is_cuda(),
        "mix_weight must be CUDA tensor."
    );

    TORCH_CHECK(
        mix_bias.is_cuda(),
        "mix_bias must be CUDA tensor."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.is_cuda(),
            "bias must be CUDA tensor when defined."
        );
    }

    TORCH_CHECK(
        x_norm.is_contiguous(),
        "x_norm must be contiguous."
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous."
    );

    TORCH_CHECK(
        mix_weight.is_contiguous(),
        "mix_weight must be contiguous."
    );

    TORCH_CHECK(
        mix_bias.is_contiguous(),
        "mix_bias must be contiguous."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.is_contiguous(),
            "bias must be contiguous when defined."
        );
    }

    TORCH_CHECK(
        x_norm.dim() == 3,
        "x_norm must be [B,D,L]."
    );

    TORCH_CHECK(
        weight.dim() == 3,
        "weight must be [D,C,S]."
    );

    TORCH_CHECK(
        mix_weight.dim() == 2,
        "mix_weight must be [D,C]."
    );

    TORCH_CHECK(
        mix_bias.dim() == 1,
        "mix_bias must be [D]."
    );

    int64_t B = x_norm.size(0);
    int64_t D = x_norm.size(1);
    int64_t L = x_norm.size(2);

    int64_t Dw = weight.size(0);
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    TORCH_CHECK(
        B > 0,
        "B must be positive."
    );

    TORCH_CHECK(
        D > 0,
        "D must be positive."
    );

    TORCH_CHECK(
        L > 0,
        "L must be positive."
    );

    TORCH_CHECK(
        C > 0,
        "C must be positive."
    );

    TORCH_CHECK(
        S > 0,
        "S must be positive."
    );

    TORCH_CHECK(
        Dw == D,
        "weight D mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(0) == D,
        "mix_weight D mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(1) == C,
        "mix_weight C mismatch."
    );

    TORCH_CHECK(
        mix_bias.size(0) == D,
        "mix_bias D mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.dim() == 2,
            "bias must be [D,C] when defined."
        );

        TORCH_CHECK(
            bias.size(0) == D,
            "bias D mismatch."
        );

        TORCH_CHECK(
            bias.size(1) == C,
            "bias C mismatch."
        );
    }

    TORCH_CHECK(
        off >= 0,
        "off must be >= 0."
    );

    TORCH_CHECK(
        T > 0,
        "T must be positive."
    );

    TORCH_CHECK(
        off + T <= L,
        "off + T must be <= L."
    );

    TORCH_CHECK(
        dilation > 0,
        "dilation must be positive."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == weight.scalar_type(),
        "x_norm and weight dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == mix_weight.scalar_type(),
        "x_norm and mix_weight dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == mix_bias.scalar_type(),
        "x_norm and mix_bias dtype mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            x_norm.scalar_type() == bias.scalar_type(),
            "x_norm and bias dtype mismatch."
        );
    }

    TORCH_CHECK(
        x_norm.scalar_type() == at::ScalarType::Float ||
        x_norm.scalar_type() == at::ScalarType::Half ||
        x_norm.scalar_type() == at::ScalarType::BFloat16,
        "unsupported dtype."
    );

    if (x_norm.scalar_type() == at::ScalarType::Float) {
        return fscm_forward_direct_generic_typed<float>(
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
        return fscm_forward_direct_generic_typed<c10::Half>(
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

    return fscm_forward_direct_generic_typed<c10::BFloat16>(
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
