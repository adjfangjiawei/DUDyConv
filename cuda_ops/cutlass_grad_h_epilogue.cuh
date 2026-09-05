#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>


template <typename scalar_t>
struct ScalarLoad;


template <>
struct ScalarLoad<float> {
    __device__ __forceinline__ static float to_float(float x) {
        return x;
    }
};


template <>
struct ScalarLoad<c10::Half> {
    __device__ __forceinline__ static float to_float(c10::Half x) {
        return __half2float(static_cast<__half>(x));
    }
};


template <>
struct ScalarLoad<c10::BFloat16> {
    __device__ __forceinline__ static float to_float(c10::BFloat16 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        return __bfloat162float(static_cast<__nv_bfloat16>(x));
#else
        return static_cast<float>(x);
#endif
    }
};


template <typename scalar_t>
struct ScalarStore;


template <>
struct ScalarStore<float> {
    __device__ __forceinline__ static float from_float(float x) {
        return x;
    }
};


template <>
struct ScalarStore<c10::Half> {
    __device__ __forceinline__ static c10::Half from_float(float x) {
        return c10::Half(__float2half_rn(x));
    }
};


template <>
struct ScalarStore<c10::BFloat16> {
    __device__ __forceinline__ static c10::BFloat16 from_float(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        return c10::BFloat16(__float2bfloat16_rn(x));
#else
        return c10::BFloat16(x);
#endif
    }
};


/*
    GEMM:
        kernel_btn[b]: [T, N]
        mix_dn:        [D, N]

    Want:
        weight[b,t,d] = sum_n kernel[b,t,n] * mix[d,n]

    Then:
        grad_h[b,d,src] += grad_out[b,d,t] * weight[b,t,d]

    We launch tiled kernel directly, not materializing weight.

    This is a CUTLASS-like fused epilogue kernel.
    注意：
        这里为了可控性，没有强行用 cutlass::gemm::device::Gemm 的 epilogue visitor，
        而是写成 tensor-core tile kernel 的外壳位置。
        真正 CUTLASS 3 EVT 版本需要接入 CuTe callback，代码量会更大。
*/


template <
    typename scalar_t,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K
>
__global__ void grad_h_fused_weight_epilogue_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ kernel_chunk,
    const scalar_t* __restrict__ kernel_mix,
    float* __restrict__ grad_h_float,
    int B,
    int D,
    int L,
    int T,
    int N,
    int K,
    int kk,
    int t_offset,
    int dilation
) {
    /*
        Block maps:
            M dimension = T
            N dimension = D

        Each block computes tile:
            t in [block_t, block_t + BLOCK_M)
            d in [block_d, block_d + BLOCK_N)

        Reduction over:
            n in [0, N)

        This is the same math as:
            weight[t,d] = dot(kernel[t,:], mix[d,:])
            grad_h[d,src] += grad_out[d,t] * weight[t,d]

        For your typical N=6, this kernel is faster than launching GEMM + extra elementwise kernel,
        because it avoids writing/reading weight_btd.
    */

    const int tile_t = blockIdx.x * BLOCK_M;
    const int tile_d = blockIdx.y * BLOCK_N;
    const int b = blockIdx.z;

    const int tid = threadIdx.x;

    constexpr int NUM_ELEMS = BLOCK_M * BLOCK_N;

    for (int linear = tid; linear < NUM_ELEMS; linear += blockDim.x) {
        const int local_d = linear % BLOCK_N;
        const int local_t = linear / BLOCK_N;

        const int t = tile_t + local_t;
        const int d = tile_d + local_d;

        if (b >= B || t >= T || d >= D) {
            continue;
        }

        const int src_t = t_offset + t - kk * dilation;

        if (src_t < 0 || src_t >= L) {
            continue;
        }

        float acc = 0.0f;

        for (int n = 0; n < N; ++n) {
            const int64_t kernel_idx =
                (
                    static_cast<int64_t>(b) * T + t
                ) * N * K + n * K + kk;

            const int64_t mix_idx =
                static_cast<int64_t>(d) * N + n;

            const float kval = ScalarLoad<scalar_t>::to_float(
                kernel_chunk[kernel_idx]
            );

            const float mval = ScalarLoad<scalar_t>::to_float(
                kernel_mix[mix_idx]
            );

            acc += kval * mval;
        }

        const int64_t go_idx =
            (
                static_cast<int64_t>(b) * D + d
            ) * T + t;

        const float go = ScalarLoad<scalar_t>::to_float(
            grad_out[go_idx]
        );

        const int64_t h_idx =
            (
                static_cast<int64_t>(b) * D + d
            ) * L + src_t;

        atomicAdd(
            grad_h_float + h_idx,
            go * acc
        );
    }
}
