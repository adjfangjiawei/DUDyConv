#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"
#include "fused_source_conv_mixed_plans.h"

// ======================================================================================
// Plan B0:
//   Simple fused backward atomic for source conv + GELU + source mix.
//
// Forward:
//
//   z[b,d,c,t] = bias[d,c]
//                + sum_j x_norm[b,d,src_j] * weight[d,c,j]
//
//   a[b,d,c,t] = GELU(z[b,d,c,t])
//
//   out[b,d,t] = mix_bias[d]
//                + sum_c a[b,d,c,t] * mix_weight[d,c]
//
// Backward input:
//   grad_out:
//     [B,D,T]
//
// Backward outputs:
//   grad_x_norm:
//     [B,D,L]
//
//   grad_weight:
//     [D,C,S]
//
//   grad_bias:
//     [D,C] or empty when no bias
//
//   grad_mix_weight:
//     [D,C]
//
//   grad_mix_bias:
//     [D]
//
// Strategy:
//
//   One thread handles one output position (b,d,t).
//   It recomputes all z[b,d,c,t], GELU(z), GELU'(z), and atomically accumulates:
//
//     grad_mix_bias[d]       += gy
//     grad_mix_weight[d,c]   += gy * GELU(z)
//     grad_bias[d,c]         += gy * mix_weight[d,c] * GELU'(z)
//     grad_weight[d,c,j]     += gz * x_norm[b,d,src_j]
//     grad_x_norm[b,d,src_j] += gz * weight[d,c,j]
//
// Notes:
//   - No dropout.
//   - No residual.
//   - No post-mix dropout.
//   - No source nonlinear residual.
//   - Accumulation arithmetic is float.
//   - Gradient tensors are allocated with same dtype as input/weight.
// ======================================================================================

namespace {

// ======================================================================================
// dtype helpers
// ======================================================================================

template <typename scalar_t>
__device__ __forceinline__ float fscm_bwd_to_float(
    scalar_t x
) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float fscm_bwd_to_float<c10::Half>(
    c10::Half x
) {
    return __half2float(
        static_cast<__half>(x)
    );
}

template <>
__device__ __forceinline__ float fscm_bwd_to_float<c10::BFloat16>(
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

// ======================================================================================
// atomic add helpers
// ======================================================================================

template <typename scalar_t>
__device__ __forceinline__ void fscm_atomic_add(
    scalar_t* ptr,
    float val
) {
    atomicAdd(
        ptr,
        static_cast<scalar_t>(val)
    );
}

template <>
__device__ __forceinline__ void fscm_atomic_add<float>(
    float* ptr,
    float val
) {
    atomicAdd(
        ptr,
        val
    );
}

template <>
__device__ __forceinline__ void fscm_atomic_add<c10::Half>(
    c10::Half* ptr,
    float val
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    atomicAdd(
        reinterpret_cast<__half*>(ptr),
        __float2half_rn(val)
    );
#else
    // Half atomicAdd requires sm_70+.
    // This fallback intentionally does nothing on unsupported arch.
    // Extension build should target sm_70+ for fp16 backward.
#endif
}

template <>
__device__ __forceinline__ void fscm_atomic_add<c10::BFloat16>(
    c10::BFloat16* ptr,
    float val
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    atomicAdd(
        reinterpret_cast<__nv_bfloat16*>(ptr),
        __float2bfloat16_rn(val)
    );
#else
    // BF16 atomicAdd requires sm_80+.
    // This fallback intentionally does nothing on unsupported arch.
    // Extension build should target sm_80+ for bf16 backward.
#endif
}

// ======================================================================================
// GELU exact and derivative
// ======================================================================================

__device__ __forceinline__ float fscm_bwd_gelu_exact(
    float x
) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;

    return 0.5f * x * (
        1.0f + erff(
            x * inv_sqrt2
        )
    );
}

__device__ __forceinline__ float fscm_bwd_gelu_exact_grad(
    float x
) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    constexpr float inv_sqrt2pi = 0.39894228040143267794f;

    float cdf =
        0.5f * (
            1.0f + erff(
                x * inv_sqrt2
            )
        );

    float pdf =
        inv_sqrt2pi * expf(
            -0.5f * x * x
        );

    return cdf + x * pdf;
}

// ======================================================================================
// Generic atomic backward kernel
// ======================================================================================

template <typename scalar_t, bool HAS_BIAS, bool NO_BOUNDARY>
__global__ void fscm_backward_atomic_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x_norm,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ mix_weight,
    const scalar_t* __restrict__ mix_bias,
    scalar_t* __restrict__ grad_x_norm,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    scalar_t* __restrict__ grad_mix_weight,
    scalar_t* __restrict__ grad_mix_bias,
    int B,
    int D,
    int L,
    int T,
    int C,
    int S,
    int off,
    int dilation
) {
    (void)mix_bias;

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

    int64_t go_idx =
        (
            static_cast<int64_t>(b) * D + d
        ) * T + t;

    float gy =
        fscm_bwd_to_float(
            grad_out[go_idx]
        );

    // grad_mix_bias[d] += gy
    fscm_atomic_add<scalar_t>(
        grad_mix_bias + d,
        gy
    );

    int64_t weight_d_base =
        static_cast<int64_t>(d) * C * S;

    int64_t cm_d_base =
        static_cast<int64_t>(d) * C;

    for (int c = 0; c < C; ++c) {
        int64_t weight_c_base =
            weight_d_base +
            static_cast<int64_t>(c) * S;

        int64_t cm_idx =
            cm_d_base + c;

        float z = 0.0f;

        if constexpr (HAS_BIAS) {
            z = fscm_bwd_to_float(
                bias[cm_idx]
            );
        }

        // recompute z
        for (int j = 0; j < S; ++j) {
            int src =
                off +
                t -
                dilation * (S - 1 - j);

            if constexpr (NO_BOUNDARY) {
                float x_val =
                    fscm_bwd_to_float(
                        x_norm[x_base + src]
                    );

                float w_val =
                    fscm_bwd_to_float(
                        weight[weight_c_base + j]
                    );

                z += x_val * w_val;
            } else {
                if (src >= 0 && src < L) {
                    float x_val =
                        fscm_bwd_to_float(
                            x_norm[x_base + src]
                        );

                    float w_val =
                        fscm_bwd_to_float(
                            weight[weight_c_base + j]
                        );

                    z += x_val * w_val;
                }
            }
        }

        float a =
            fscm_bwd_gelu_exact(
                z
            );

        float gelu_g =
            fscm_bwd_gelu_exact_grad(
                z
            );

        float mix_val =
            fscm_bwd_to_float(
                mix_weight[cm_idx]
            );

        // grad_mix_weight[d,c] += gy * a
        fscm_atomic_add<scalar_t>(
            grad_mix_weight + cm_idx,
            gy * a
        );

        float gz =
            gy * mix_val * gelu_g;

        // grad_bias[d,c] += gz
        if constexpr (HAS_BIAS) {
            fscm_atomic_add<scalar_t>(
                grad_bias + cm_idx,
                gz
            );
        }

        // grad_weight[d,c,j] and grad_x_norm[b,d,src]
        for (int j = 0; j < S; ++j) {
            int src =
                off +
                t -
                dilation * (S - 1 - j);

            if constexpr (NO_BOUNDARY) {
                float x_val =
                    fscm_bwd_to_float(
                        x_norm[x_base + src]
                    );

                float w_val =
                    fscm_bwd_to_float(
                        weight[weight_c_base + j]
                    );

                fscm_atomic_add<scalar_t>(
                    grad_weight + weight_c_base + j,
                    gz * x_val
                );

                fscm_atomic_add<scalar_t>(
                    grad_x_norm + x_base + src,
                    gz * w_val
                );
            } else {
                if (src >= 0 && src < L) {
                    float x_val =
                        fscm_bwd_to_float(
                            x_norm[x_base + src]
                        );

                    float w_val =
                        fscm_bwd_to_float(
                            weight[weight_c_base + j]
                        );

                    fscm_atomic_add<scalar_t>(
                        grad_weight + weight_c_base + j,
                        gz * x_val
                    );

                    fscm_atomic_add<scalar_t>(
                        grad_x_norm + x_base + src,
                        gz * w_val
                    );
                }
            }
        }
    }
}

// ======================================================================================
// Validation
// ======================================================================================

static void fscm_backward_atomic_check_inputs(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    const char* fn_name =
        "fused_source_conv_mixed_backward_atomic_cuda";

    TORCH_CHECK(
        grad_out.defined(),
        fn_name,
        ": grad_out must be defined."
    );

    TORCH_CHECK(
        x_norm.defined(),
        fn_name,
        ": x_norm must be defined."
    );

    TORCH_CHECK(
        weight.defined(),
        fn_name,
        ": weight must be defined."
    );

    TORCH_CHECK(
        mix_weight.defined(),
        fn_name,
        ": mix_weight must be defined."
    );

    TORCH_CHECK(
        mix_bias.defined(),
        fn_name,
        ": mix_bias must be defined."
    );

    TORCH_CHECK(
        grad_out.is_cuda(),
        fn_name,
        ": grad_out must be CUDA tensor."
    );

    TORCH_CHECK(
        x_norm.is_cuda(),
        fn_name,
        ": x_norm must be CUDA tensor."
    );

    TORCH_CHECK(
        weight.is_cuda(),
        fn_name,
        ": weight must be CUDA tensor."
    );

    TORCH_CHECK(
        mix_weight.is_cuda(),
        fn_name,
        ": mix_weight must be CUDA tensor."
    );

    TORCH_CHECK(
        mix_bias.is_cuda(),
        fn_name,
        ": mix_bias must be CUDA tensor."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.is_cuda(),
            fn_name,
            ": bias must be CUDA tensor when defined."
        );
    }

    TORCH_CHECK(
        grad_out.is_contiguous(),
        fn_name,
        ": grad_out must be contiguous."
    );

    TORCH_CHECK(
        x_norm.is_contiguous(),
        fn_name,
        ": x_norm must be contiguous."
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        fn_name,
        ": weight must be contiguous."
    );

    TORCH_CHECK(
        mix_weight.is_contiguous(),
        fn_name,
        ": mix_weight must be contiguous."
    );

    TORCH_CHECK(
        mix_bias.is_contiguous(),
        fn_name,
        ": mix_bias must be contiguous."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.is_contiguous(),
            fn_name,
            ": bias must be contiguous when defined."
        );
    }

    TORCH_CHECK(
        grad_out.dim() == 3,
        fn_name,
        ": grad_out must be [B,D,T]."
    );

    TORCH_CHECK(
        x_norm.dim() == 3,
        fn_name,
        ": x_norm must be [B,D,L]."
    );

    TORCH_CHECK(
        weight.dim() == 3,
        fn_name,
        ": weight must be [D,C,S]."
    );

    TORCH_CHECK(
        mix_weight.dim() == 2,
        fn_name,
        ": mix_weight must be [D,C]."
    );

    TORCH_CHECK(
        mix_bias.dim() == 1,
        fn_name,
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
        fn_name,
        ": B must be positive."
    );

    TORCH_CHECK(
        D > 0,
        fn_name,
        ": D must be positive."
    );

    TORCH_CHECK(
        L > 0,
        fn_name,
        ": L must be positive."
    );

    TORCH_CHECK(
        C > 0,
        fn_name,
        ": C must be positive."
    );

    TORCH_CHECK(
        S > 0,
        fn_name,
        ": S must be positive."
    );

    TORCH_CHECK(
        Dw == D,
        fn_name,
        ": weight D mismatch."
    );

    TORCH_CHECK(
        grad_out.size(0) == B,
        fn_name,
        ": grad_out B mismatch."
    );

    TORCH_CHECK(
        grad_out.size(1) == D,
        fn_name,
        ": grad_out D mismatch."
    );

    TORCH_CHECK(
        grad_out.size(2) == T,
        fn_name,
        ": grad_out T mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(0) == D,
        fn_name,
        ": mix_weight D mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(1) == C,
        fn_name,
        ": mix_weight C mismatch."
    );

    TORCH_CHECK(
        mix_bias.size(0) == D,
        fn_name,
        ": mix_bias D mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.dim() == 2,
            fn_name,
            ": bias must be [D,C] when defined."
        );

        TORCH_CHECK(
            bias.size(0) == D,
            fn_name,
            ": bias D mismatch."
        );

        TORCH_CHECK(
            bias.size(1) == C,
            fn_name,
            ": bias C mismatch."
        );
    }

    TORCH_CHECK(
        off >= 0,
        fn_name,
        ": off must be >= 0."
    );

    TORCH_CHECK(
        T > 0,
        fn_name,
        ": T must be positive."
    );

    TORCH_CHECK(
        off + T <= L,
        fn_name,
        ": off + T must be <= L."
    );

    TORCH_CHECK(
        dilation > 0,
        fn_name,
        ": dilation must be positive."
    );

    TORCH_CHECK(
        grad_out.scalar_type() == x_norm.scalar_type(),
        fn_name,
        ": grad_out and x_norm dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == weight.scalar_type(),
        fn_name,
        ": x_norm and weight dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == mix_weight.scalar_type(),
        fn_name,
        ": x_norm and mix_weight dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == mix_bias.scalar_type(),
        fn_name,
        ": x_norm and mix_bias dtype mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            x_norm.scalar_type() == bias.scalar_type(),
            fn_name,
            ": x_norm and bias dtype mismatch."
        );
    }

    TORCH_CHECK(
        x_norm.scalar_type() == at::ScalarType::Float ||
        x_norm.scalar_type() == at::ScalarType::Half ||
        x_norm.scalar_type() == at::ScalarType::BFloat16,
        fn_name,
        ": unsupported dtype."
    );
}

// ======================================================================================
// Typed launcher
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> fscm_backward_atomic_typed(
    torch::Tensor grad_out,
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

    auto grad_x_norm =
        torch::zeros_like(
            x_norm
        );

    auto grad_weight =
        torch::zeros_like(
            weight
        );

    bool has_bias =
        bias.defined() &&
        bias.numel() > 0;

    torch::Tensor grad_bias;

    if (has_bias) {
        grad_bias =
            torch::zeros_like(
                bias
            );
    } else {
        grad_bias =
            torch::empty(
                {0},
                x_norm.options()
            );
    }

    auto grad_mix_weight =
        torch::zeros_like(
            mix_weight
        );

    auto grad_mix_bias =
        torch::zeros_like(
            mix_bias
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

    bool no_boundary =
        off >= dilation * (S - 1);

    if (has_bias) {
        if (no_boundary) {
            fscm_backward_atomic_kernel<scalar_t, true, true><<<
                blocks,
                threads,
                0,
                stream
            >>>(
                grad_out.data_ptr<scalar_t>(),
                x_norm.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                mix_weight.data_ptr<scalar_t>(),
                mix_bias.data_ptr<scalar_t>(),
                grad_x_norm.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                grad_bias.data_ptr<scalar_t>(),
                grad_mix_weight.data_ptr<scalar_t>(),
                grad_mix_bias.data_ptr<scalar_t>(),
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
            fscm_backward_atomic_kernel<scalar_t, true, false><<<
                blocks,
                threads,
                0,
                stream
            >>>(
                grad_out.data_ptr<scalar_t>(),
                x_norm.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                mix_weight.data_ptr<scalar_t>(),
                mix_bias.data_ptr<scalar_t>(),
                grad_x_norm.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                grad_bias.data_ptr<scalar_t>(),
                grad_mix_weight.data_ptr<scalar_t>(),
                grad_mix_bias.data_ptr<scalar_t>(),
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
            fscm_backward_atomic_kernel<scalar_t, false, true><<<
                blocks,
                threads,
                0,
                stream
            >>>(
                grad_out.data_ptr<scalar_t>(),
                x_norm.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                nullptr,
                mix_weight.data_ptr<scalar_t>(),
                mix_bias.data_ptr<scalar_t>(),
                grad_x_norm.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                nullptr,
                grad_mix_weight.data_ptr<scalar_t>(),
                grad_mix_bias.data_ptr<scalar_t>(),
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
            fscm_backward_atomic_kernel<scalar_t, false, false><<<
                blocks,
                threads,
                0,
                stream
            >>>(
                grad_out.data_ptr<scalar_t>(),
                x_norm.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                nullptr,
                mix_weight.data_ptr<scalar_t>(),
                mix_bias.data_ptr<scalar_t>(),
                grad_x_norm.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                nullptr,
                grad_mix_weight.data_ptr<scalar_t>(),
                grad_mix_bias.data_ptr<scalar_t>(),
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

    return {
        grad_x_norm,
        grad_weight,
        grad_bias,
        grad_mix_weight,
        grad_mix_bias
    };
}

} // namespace

// ======================================================================================
// Public entry
// ======================================================================================

std::vector<torch::Tensor> fused_source_conv_mixed_backward_atomic_cuda(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FSCM path: backward_atomic_generic");

    fscm_backward_atomic_check_inputs(
        grad_out,
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
        return fscm_backward_atomic_typed<float>(
            grad_out,
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
        return fscm_backward_atomic_typed<c10::Half>(
            grad_out,
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

    return fscm_backward_atomic_typed<c10::BFloat16>(
        grad_out,
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
