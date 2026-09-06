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
// Plan B1:
//   Per-D block reduction backward.
//
// Forward semantics:
//
//   z[b,d,c,t] = bias[d,c]
//                + sum_j x_norm[b,d,src_j] * weight[d,c,j]
//
//   a[b,d,c,t] = GELU(z[b,d,c,t])
//
//   out[b,d,t] = mix_bias[d]
//                + sum_c a[b,d,c,t] * mix_weight[d,c]
//
// Backward:
//
//   grad_out:
//     [B,D,T]
//
// Return:
//   grad_x_norm:
//     [B,D,L]
//
//   grad_weight:
//     [D,C,S]
//
//   grad_bias:
//     [D,C] or empty
//
//   grad_mix_weight:
//     [D,C]
//
//   grad_mix_bias:
//     [D]
//
// Main design:
//
//   Kernel 1:
//     grid = [D, B, tiles_per_b]
//     one block processes one fixed d, one fixed b, and one T tile.
//
//     It computes per-block partial parameter gradients:
//       partial_grad_weight     [num_partials, D, C, S]
//       partial_grad_bias       [num_partials, D, C]
//       partial_grad_mix_weight [num_partials, D, C]
//       partial_grad_mix_bias   [num_partials, D]
//
//     grad_x_norm is directly atomic-added.
//
//   Kernel 2:
//     Reduce partial dimension into final param gradients.
//
// Notes:
//   - Parameter partial buffers are float for stable accumulation.
//   - Final gradients are cast to original dtype.
//   - grad_x_norm is allocated in original dtype and atomic-added.
//   - fp16 atomicAdd requires sm_70+.
//   - bf16 atomicAdd requires sm_80+.
// ======================================================================================

namespace {

constexpr int FSCM_B1_MAX_C = 64;
constexpr int FSCM_B1_MAX_S = 8;

// ======================================================================================
// dtype helpers
// ======================================================================================

template <typename scalar_t>
__device__ __forceinline__ float fscm_b1_to_float(
    scalar_t x
) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float fscm_b1_to_float<c10::Half>(
    c10::Half x
) {
    return __half2float(
        static_cast<__half>(x)
    );
}

template <>
__device__ __forceinline__ float fscm_b1_to_float<c10::BFloat16>(
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
__device__ __forceinline__ scalar_t fscm_b1_from_float(
    float x
) {
    return static_cast<scalar_t>(x);
}

template <>
__device__ __forceinline__ c10::Half fscm_b1_from_float<c10::Half>(
    float x
) {
    return c10::Half(
        __float2half_rn(x)
    );
}

template <>
__device__ __forceinline__ c10::BFloat16 fscm_b1_from_float<c10::BFloat16>(
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
// atomic add helper for grad_x_norm
// ======================================================================================

template <typename scalar_t>
__device__ __forceinline__ void fscm_b1_atomic_add(
    scalar_t* ptr,
    float val
) {
    atomicAdd(
        ptr,
        static_cast<scalar_t>(val)
    );
}

template <>
__device__ __forceinline__ void fscm_b1_atomic_add<float>(
    float* ptr,
    float val
) {
    atomicAdd(
        ptr,
        val
    );
}

template <>
__device__ __forceinline__ void fscm_b1_atomic_add<c10::Half>(
    c10::Half* ptr,
    float val
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    atomicAdd(
        reinterpret_cast<__half*>(ptr),
        __float2half_rn(val)
    );
#endif
}

template <>
__device__ __forceinline__ void fscm_b1_atomic_add<c10::BFloat16>(
    c10::BFloat16* ptr,
    float val
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    atomicAdd(
        reinterpret_cast<__nv_bfloat16*>(ptr),
        __float2bfloat16_rn(val)
    );
#endif
}

// ======================================================================================
// GELU exact and derivative
// ======================================================================================

__device__ __forceinline__ float fscm_b1_gelu_exact(
    float x
) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;

    return 0.5f * x * (
        1.0f + erff(
            x * inv_sqrt2
        )
    );
}

__device__ __forceinline__ float fscm_b1_gelu_exact_grad(
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
// Block reduce sum
// ======================================================================================

__device__ __forceinline__ float fscm_b1_block_reduce_sum(
    float val
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    smem[tid] = val;

    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }

        __syncthreads();
    }

    return smem[0];
}

// ======================================================================================
// Kernel 1:
//   compute partial parameter grads and grad_x_norm
//
// Grid:
//   blockIdx.x = d
//   blockIdx.y = b
//   blockIdx.z = tile_id
//
// Partial index:
//   partial_id = b * tiles_per_b + tile_id
//
// partial buffers are float:
//
//   partial_grad_weight:
//     [num_partials, D, C, S]
//
//   partial_grad_bias:
//     [num_partials, D, C]
//
//   partial_grad_mix_weight:
//     [num_partials, D, C]
//
//   partial_grad_mix_bias:
//     [num_partials, D]
//
// For each block:
//   - every thread iterates t = tile_start + threadIdx.x, stride blockDim.x
//   - for each reduced scalar, block reduces across threads and writes one value.
// ======================================================================================

template <typename scalar_t, bool HAS_BIAS, bool NO_BOUNDARY>
__global__ void fscm_b1_partial_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x_norm,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ mix_weight,
    float* __restrict__ partial_grad_weight,
    float* __restrict__ partial_grad_bias,
    float* __restrict__ partial_grad_mix_weight,
    float* __restrict__ partial_grad_mix_bias,
    scalar_t* __restrict__ grad_x_norm,
    int B,
    int D,
    int L,
    int T,
    int C,
    int S,
    int off,
    int dilation,
    int tile_t,
    int tiles_per_b,
    int num_partials
) {
    int d = static_cast<int>(
        blockIdx.x
    );

    int b = static_cast<int>(
        blockIdx.y
    );

    int tile_id = static_cast<int>(
        blockIdx.z
    );

    int partial_id =
        b * tiles_per_b + tile_id;

    int t_begin =
        tile_id * tile_t;

    int t_end =
        min(
            t_begin + tile_t,
            T
        );

    int tid =
        threadIdx.x;

    int64_t x_base =
        (
            static_cast<int64_t>(b) * D + d
        ) * L;

    int64_t go_base =
        (
            static_cast<int64_t>(b) * D + d
        ) * T;

    int64_t weight_d_base =
        static_cast<int64_t>(d) * C * S;

    int64_t cm_d_base =
        static_cast<int64_t>(d) * C;

    // ----------------------------------------------------------------------
    // partial grad_mix_bias[d]
    // ----------------------------------------------------------------------

    float local_gmb = 0.0f;

    for (int t = t_begin + tid; t < t_end; t += blockDim.x) {
        local_gmb += fscm_b1_to_float(
            grad_out[go_base + t]
        );
    }

    float block_gmb =
        fscm_b1_block_reduce_sum(
            local_gmb
        );

    if (tid == 0) {
        int64_t idx =
            (
                static_cast<int64_t>(partial_id) * D + d
            );

        partial_grad_mix_bias[idx] =
            block_gmb;
    }

    __syncthreads();

    // ----------------------------------------------------------------------
    // partial grad per c:
    //   grad_mix_weight[d,c]
    //   grad_bias[d,c]
    //   grad_weight[d,c,j]
    //   grad_x_norm atomicAdd during recomputation
    // ----------------------------------------------------------------------

    for (int c = 0; c < C; ++c) {
        int64_t weight_c_base =
            weight_d_base +
            static_cast<int64_t>(c) * S;

        int64_t cm_idx =
            cm_d_base + c;

        float local_gmw = 0.0f;
        float local_gb = 0.0f;

        float local_gw[FSCM_B1_MAX_S];

#pragma unroll
        for (int jj = 0; jj < FSCM_B1_MAX_S; ++jj) {
            local_gw[jj] = 0.0f;
        }

        for (int t = t_begin + tid; t < t_end; t += blockDim.x) {
            float gy =
                fscm_b1_to_float(
                    grad_out[go_base + t]
                );

            float z = 0.0f;

            if constexpr (HAS_BIAS) {
                z = fscm_b1_to_float(
                    bias[cm_idx]
                );
            }

            for (int j = 0; j < S; ++j) {
                int src =
                    off +
                    t -
                    dilation * (S - 1 - j);

                if constexpr (NO_BOUNDARY) {
                    float x_val =
                        fscm_b1_to_float(
                            x_norm[x_base + src]
                        );

                    float w_val =
                        fscm_b1_to_float(
                            weight[weight_c_base + j]
                        );

                    z += x_val * w_val;
                } else {
                    if (src >= 0 && src < L) {
                        float x_val =
                            fscm_b1_to_float(
                                x_norm[x_base + src]
                            );

                        float w_val =
                            fscm_b1_to_float(
                                weight[weight_c_base + j]
                            );

                        z += x_val * w_val;
                    }
                }
            }

            float a =
                fscm_b1_gelu_exact(
                    z
                );

            float gelu_g =
                fscm_b1_gelu_exact_grad(
                    z
                );

            float mix_val =
                fscm_b1_to_float(
                    mix_weight[cm_idx]
                );

            float gz =
                gy * mix_val * gelu_g;

            local_gmw += gy * a;
            local_gb += gz;

            for (int j = 0; j < S; ++j) {
                int src =
                    off +
                    t -
                    dilation * (S - 1 - j);

                if constexpr (NO_BOUNDARY) {
                    float x_val =
                        fscm_b1_to_float(
                            x_norm[x_base + src]
                        );

                    float w_val =
                        fscm_b1_to_float(
                            weight[weight_c_base + j]
                        );

                    if (j < FSCM_B1_MAX_S) {
                        local_gw[j] += gz * x_val;
                    }

                    fscm_b1_atomic_add<scalar_t>(
                        grad_x_norm + x_base + src,
                        gz * w_val
                    );
                } else {
                    if (src >= 0 && src < L) {
                        float x_val =
                            fscm_b1_to_float(
                                x_norm[x_base + src]
                            );

                        float w_val =
                            fscm_b1_to_float(
                                weight[weight_c_base + j]
                            );

                        if (j < FSCM_B1_MAX_S) {
                            local_gw[j] += gz * x_val;
                        }

                        fscm_b1_atomic_add<scalar_t>(
                            grad_x_norm + x_base + src,
                            gz * w_val
                        );
                    }
                }
            }
        }

        float block_gmw =
            fscm_b1_block_reduce_sum(
                local_gmw
            );

        if (tid == 0) {
            int64_t idx =
                (
                    static_cast<int64_t>(partial_id) * D + d
                ) * C + c;

            partial_grad_mix_weight[idx] =
                block_gmw;
        }

        __syncthreads();

        float block_gb =
            fscm_b1_block_reduce_sum(
                local_gb
            );

        if constexpr (HAS_BIAS) {
            if (tid == 0) {
                int64_t idx =
                    (
                        static_cast<int64_t>(partial_id) * D + d
                    ) * C + c;

                partial_grad_bias[idx] =
                    block_gb;
            }
        }

        __syncthreads();

        for (int j = 0; j < S; ++j) {
            float block_gw =
                fscm_b1_block_reduce_sum(
                    local_gw[j]
                );

            if (tid == 0) {
                int64_t idx =
                    (
                        (
                            static_cast<int64_t>(partial_id) * D + d
                        ) * C + c
                    ) * S + j;

                partial_grad_weight[idx] =
                    block_gw;
            }

            __syncthreads();
        }
    }

    (void)num_partials;
}

// ======================================================================================
// Kernel 2:
//   finalize partial parameter gradients.
//
// one thread per final parameter element.
// ======================================================================================

template <typename scalar_t, bool HAS_BIAS>
__global__ void fscm_b1_finalize_kernel(
    const float* __restrict__ partial_grad_weight,
    const float* __restrict__ partial_grad_bias,
    const float* __restrict__ partial_grad_mix_weight,
    const float* __restrict__ partial_grad_mix_bias,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    scalar_t* __restrict__ grad_mix_weight,
    scalar_t* __restrict__ grad_mix_bias,
    int D,
    int C,
    int S,
    int num_partials
) {
    int64_t n_gw =
        static_cast<int64_t>(D) *
        static_cast<int64_t>(C) *
        static_cast<int64_t>(S);

    int64_t n_gm =
        static_cast<int64_t>(D) *
        static_cast<int64_t>(C);

    int64_t n_gmb =
        static_cast<int64_t>(D);

    int64_t total =
        n_gw + n_gm + n_gmb;

    if constexpr (HAS_BIAS) {
        total += n_gm;
    }

    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    if (idx < n_gw) {
        int64_t local = idx;

        int j = static_cast<int>(
            local % S
        );

        local /= S;

        int c = static_cast<int>(
            local % C
        );

        int d = static_cast<int>(
            local / C
        );

        float acc = 0.0f;

        for (int p = 0; p < num_partials; ++p) {
            int64_t pidx =
                (
                    (
                        static_cast<int64_t>(p) * D + d
                    ) * C + c
                ) * S + j;

            acc += partial_grad_weight[pidx];
        }

        grad_weight[idx] =
            fscm_b1_from_float<scalar_t>(
                acc
            );

        return;
    }

    idx -= n_gw;

    if (idx < n_gm) {
        int64_t local = idx;

        int c = static_cast<int>(
            local % C
        );

        int d = static_cast<int>(
            local / C
        );

        float acc = 0.0f;

        for (int p = 0; p < num_partials; ++p) {
            int64_t pidx =
                (
                    static_cast<int64_t>(p) * D + d
                ) * C + c;

            acc += partial_grad_mix_weight[pidx];
        }

        grad_mix_weight[idx] =
            fscm_b1_from_float<scalar_t>(
                acc
            );

        return;
    }

    idx -= n_gm;

    if constexpr (HAS_BIAS) {
        if (idx < n_gm) {
            int64_t local = idx;

            int c = static_cast<int>(
                local % C
            );

            int d = static_cast<int>(
                local / C
            );

            float acc = 0.0f;

            for (int p = 0; p < num_partials; ++p) {
                int64_t pidx =
                    (
                        static_cast<int64_t>(p) * D + d
                    ) * C + c;

                acc += partial_grad_bias[pidx];
            }

            grad_bias[idx] =
                fscm_b1_from_float<scalar_t>(
                    acc
                );

            return;
        }

        idx -= n_gm;
    }

    if (idx < n_gmb) {
        int d = static_cast<int>(
            idx
        );

        float acc = 0.0f;

        for (int p = 0; p < num_partials; ++p) {
            int64_t pidx =
                static_cast<int64_t>(p) * D + d;

            acc += partial_grad_mix_bias[pidx];
        }

        grad_mix_bias[d] =
            fscm_b1_from_float<scalar_t>(
                acc
            );
    }
}

// ======================================================================================
// Validation
// ======================================================================================

static void fscm_b1_check_inputs(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t tile_t
) {
    const char* fn_name =
        "fused_source_conv_mixed_backward_per_d_reduce_cuda";

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
        C <= FSCM_B1_MAX_C,
        fn_name,
        ": C too large for B1 implementation. max=",
        FSCM_B1_MAX_C,
        ", got ",
        C,
        "."
    );

    TORCH_CHECK(
        S <= FSCM_B1_MAX_S,
        fn_name,
        ": S too large for B1 implementation. max=",
        FSCM_B1_MAX_S,
        ", got ",
        S,
        "."
    );

    TORCH_CHECK(
        weight.size(0) == D,
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
        tile_t > 0,
        fn_name,
        ": tile_t must be positive."
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
static std::vector<torch::Tensor> fscm_b1_backward_typed(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T_arg,
    int64_t dilation,
    int64_t tile_t_arg
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

    int tile_t = static_cast<int>(
        tile_t_arg
    );

    int tiles_per_b =
        static_cast<int>(
            (T + tile_t - 1) / tile_t
        );

    int num_partials =
        B * tiles_per_b;

    auto grad_x_norm =
        torch::zeros_like(
            x_norm
        );

    auto grad_weight =
        torch::empty_like(
            weight
        );

    bool has_bias =
        bias.defined() &&
        bias.numel() > 0;

    torch::Tensor grad_bias;

    if (has_bias) {
        grad_bias =
            torch::empty_like(
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
        torch::empty_like(
            mix_weight
        );

    auto grad_mix_bias =
        torch::empty_like(
            mix_bias
        );

    auto float_opts =
        x_norm.options().dtype(
            torch::kFloat32
        );

    auto partial_grad_weight =
        torch::empty(
            {
                num_partials,
                D,
                C,
                S
            },
            float_opts
        );

    torch::Tensor partial_grad_bias;

    if (has_bias) {
        partial_grad_bias =
            torch::empty(
                {
                    num_partials,
                    D,
                    C
                },
                float_opts
            );
    } else {
        partial_grad_bias =
            torch::empty(
                {0},
                float_opts
            );
    }

    auto partial_grad_mix_weight =
        torch::empty(
            {
                num_partials,
                D,
                C
            },
            float_opts
        );

    auto partial_grad_mix_bias =
        torch::empty(
            {
                num_partials,
                D
            },
            float_opts
        );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    dim3 grid_partial(
        static_cast<unsigned int>(D),
        static_cast<unsigned int>(B),
        static_cast<unsigned int>(tiles_per_b)
    );

    int threads_partial = 256;

    size_t shared_bytes =
        static_cast<size_t>(threads_partial) *
        sizeof(float);

    bool no_boundary =
        off >= dilation * (S - 1);

    if (has_bias) {
        if (no_boundary) {
            fscm_b1_partial_kernel<scalar_t, true, true><<<
                grid_partial,
                threads_partial,
                shared_bytes,
                stream
            >>>(
                grad_out.data_ptr<scalar_t>(),
                x_norm.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                mix_weight.data_ptr<scalar_t>(),
                partial_grad_weight.data_ptr<float>(),
                partial_grad_bias.data_ptr<float>(),
                partial_grad_mix_weight.data_ptr<float>(),
                partial_grad_mix_bias.data_ptr<float>(),
                grad_x_norm.data_ptr<scalar_t>(),
                B,
                D,
                L,
                T,
                C,
                S,
                static_cast<int>(off),
                static_cast<int>(dilation),
                tile_t,
                tiles_per_b,
                num_partials
            );
        } else {
            fscm_b1_partial_kernel<scalar_t, true, false><<<
                grid_partial,
                threads_partial,
                shared_bytes,
                stream
            >>>(
                grad_out.data_ptr<scalar_t>(),
                x_norm.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                mix_weight.data_ptr<scalar_t>(),
                partial_grad_weight.data_ptr<float>(),
                partial_grad_bias.data_ptr<float>(),
                partial_grad_mix_weight.data_ptr<float>(),
                partial_grad_mix_bias.data_ptr<float>(),
                grad_x_norm.data_ptr<scalar_t>(),
                B,
                D,
                L,
                T,
                C,
                S,
                static_cast<int>(off),
                static_cast<int>(dilation),
                tile_t,
                tiles_per_b,
                num_partials
            );
        }
    } else {
        if (no_boundary) {
            fscm_b1_partial_kernel<scalar_t, false, true><<<
                grid_partial,
                threads_partial,
                shared_bytes,
                stream
            >>>(
                grad_out.data_ptr<scalar_t>(),
                x_norm.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                nullptr,
                mix_weight.data_ptr<scalar_t>(),
                partial_grad_weight.data_ptr<float>(),
                nullptr,
                partial_grad_mix_weight.data_ptr<float>(),
                partial_grad_mix_bias.data_ptr<float>(),
                grad_x_norm.data_ptr<scalar_t>(),
                B,
                D,
                L,
                T,
                C,
                S,
                static_cast<int>(off),
                static_cast<int>(dilation),
                tile_t,
                tiles_per_b,
                num_partials
            );
        } else {
            fscm_b1_partial_kernel<scalar_t, false, false><<<
                grid_partial,
                threads_partial,
                shared_bytes,
                stream
            >>>(
                grad_out.data_ptr<scalar_t>(),
                x_norm.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                nullptr,
                mix_weight.data_ptr<scalar_t>(),
                partial_grad_weight.data_ptr<float>(),
                nullptr,
                partial_grad_mix_weight.data_ptr<float>(),
                partial_grad_mix_bias.data_ptr<float>(),
                grad_x_norm.data_ptr<scalar_t>(),
                B,
                D,
                L,
                T,
                C,
                S,
                static_cast<int>(off),
                static_cast<int>(dilation),
                tile_t,
                tiles_per_b,
                num_partials
            );
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t n_gw =
        static_cast<int64_t>(D) *
        static_cast<int64_t>(C) *
        static_cast<int64_t>(S);

    int64_t n_gm =
        static_cast<int64_t>(D) *
        static_cast<int64_t>(C);

    int64_t n_gmb =
        static_cast<int64_t>(D);

    int64_t total_finalize =
        n_gw + n_gm + n_gmb;

    if (has_bias) {
        total_finalize += n_gm;
    }

    int threads_finalize = 256;

    int blocks_finalize =
        static_cast<int>(
            (total_finalize + threads_finalize - 1) /
            threads_finalize
        );

    if (has_bias) {
        fscm_b1_finalize_kernel<scalar_t, true><<<
            blocks_finalize,
            threads_finalize,
            0,
            stream
        >>>(
            partial_grad_weight.data_ptr<float>(),
            partial_grad_bias.data_ptr<float>(),
            partial_grad_mix_weight.data_ptr<float>(),
            partial_grad_mix_bias.data_ptr<float>(),
            grad_weight.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            grad_mix_weight.data_ptr<scalar_t>(),
            grad_mix_bias.data_ptr<scalar_t>(),
            D,
            C,
            S,
            num_partials
        );
    } else {
        fscm_b1_finalize_kernel<scalar_t, false><<<
            blocks_finalize,
            threads_finalize,
            0,
            stream
        >>>(
            partial_grad_weight.data_ptr<float>(),
            nullptr,
            partial_grad_mix_weight.data_ptr<float>(),
            partial_grad_mix_bias.data_ptr<float>(),
            grad_weight.data_ptr<scalar_t>(),
            nullptr,
            grad_mix_weight.data_ptr<scalar_t>(),
            grad_mix_bias.data_ptr<scalar_t>(),
            D,
            C,
            S,
            num_partials
        );
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

std::vector<torch::Tensor> fused_source_conv_mixed_backward_per_d_reduce_cuda(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t tile_t
) {
    FDC_DEBUG_PATH("FSCM path: backward_per_d_reduce");

    fscm_b1_check_inputs(
        grad_out,
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        tile_t
    );

    if (x_norm.scalar_type() == at::ScalarType::Float) {
        return fscm_b1_backward_typed<float>(
            grad_out,
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation,
            tile_t
        );
    }

    if (x_norm.scalar_type() == at::ScalarType::Half) {
        return fscm_b1_backward_typed<c10::Half>(
            grad_out,
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation,
            tile_t
        );
    }

    return fscm_b1_backward_typed<c10::BFloat16>(
        grad_out,
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        tile_t
    );
}
