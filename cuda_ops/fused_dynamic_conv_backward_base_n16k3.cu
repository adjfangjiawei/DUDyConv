#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <type_traits>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Local helpers
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_n16_cast_float_to_scalar_kernel(
    const float* __restrict__ src,
    scalar_t* __restrict__ dst,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        dst[i] = fdc_from_float_dev<scalar_t>(src[i]);
    }
}

// ======================================================================================
// grad_h generic typed N16K3, B=1, D=256, dilation=1
//
// gh[d,s] = sum_kk go[d,t] * dot(kc[t,:,kk], mix[d,:])
// t = s - off + kk
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_n16k3_d256_grad_h_typed_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gh,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * L;

    if (idx >= total) {
        return;
    }

    int s = idx % L;
    int d = idx / L;

    int64_t mb = static_cast<int64_t>(d) * 16;

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

    int tbase = s - off;

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = tbase + kk;

        if (t >= 0 && t < T) {
            int64_t kb = static_cast<int64_t>(t) * 48 + kk;

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

            acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) * w;
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc;
}

// ======================================================================================
// grad_h float-specialized N16K3, B=1, D=256, dilation=1
// ======================================================================================

__global__ void fdc_n16k3_d256_grad_h_float_kernel(
    const float* __restrict__ go,
    const float* __restrict__ kc,
    const float* __restrict__ mix,
    float* __restrict__ gh,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * L;

    if (idx >= total) {
        return;
    }

    int s = idx % L;
    int d = idx / L;

    int64_t mb = static_cast<int64_t>(d) * 16;

    float m0 = mix[mb + 0];
    float m1 = mix[mb + 1];
    float m2 = mix[mb + 2];
    float m3 = mix[mb + 3];
    float m4 = mix[mb + 4];
    float m5 = mix[mb + 5];
    float m6 = mix[mb + 6];
    float m7 = mix[mb + 7];
    float m8 = mix[mb + 8];
    float m9 = mix[mb + 9];
    float m10 = mix[mb + 10];
    float m11 = mix[mb + 11];
    float m12 = mix[mb + 12];
    float m13 = mix[mb + 13];
    float m14 = mix[mb + 14];
    float m15 = mix[mb + 15];

    int tbase = s - off;

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = tbase + kk;

        if (t >= 0 && t < T) {
            int64_t kb = static_cast<int64_t>(t) * 48 + kk;

            float w =
                kc[kb + 0 * 3] * m0 +
                kc[kb + 1 * 3] * m1 +
                kc[kb + 2 * 3] * m2 +
                kc[kb + 3 * 3] * m3 +
                kc[kb + 4 * 3] * m4 +
                kc[kb + 5 * 3] * m5 +
                kc[kb + 6 * 3] * m6 +
                kc[kb + 7 * 3] * m7 +
                kc[kb + 8 * 3] * m8 +
                kc[kb + 9 * 3] * m9 +
                kc[kb + 10 * 3] * m10 +
                kc[kb + 11 * 3] * m11 +
                kc[kb + 12 * 3] * m12 +
                kc[kb + 13 * 3] * m13 +
                kc[kb + 14 * 3] * m14 +
                kc[kb + 15 * 3] * m15;

            acc += go[static_cast<int64_t>(d) * T + t] * w;
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc;
}

// ======================================================================================
// grad_kernel typed, warp reduction over D=256
//
// Each warp handles:
//   one t
//   one kk
//   one n_tile of 4 n values
//
// gk[t,n,kk] = sum_d go[d,t] * h[d,off+t-kk] * mix[d,n]
// ======================================================================================

template <typename scalar_t, int T_TILE, int N_TILE>
__global__ void fdc_n16k3_d256_grad_kernel_typed_warp_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gk,
    int L,
    int T,
    int off
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    int t = blockIdx.x * T_TILE + warp;
    int kk = blockIdx.y;
    int ntile = blockIdx.z;
    int nbase = ntile * N_TILE;

    if (warp >= T_TILE || t >= T) {
        return;
    }

    int s = off + t - kk;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    if (s >= 0 && s < L) {
        for (int d = lane; d < 256; d += 32) {
            float base =
                fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) *
                fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);

            int64_t mb = static_cast<int64_t>(d) * 16;

            acc0 += base * fdc_to_float_dev(mix[mb + nbase + 0]);
            acc1 += base * fdc_to_float_dev(mix[mb + nbase + 1]);
            acc2 += base * fdc_to_float_dev(mix[mb + nbase + 2]);
            acc3 += base * fdc_to_float_dev(mix[mb + nbase + 3]);
        }
    }

    acc0 = fdc_warp_sum_float(acc0);
    acc1 = fdc_warp_sum_float(acc1);
    acc2 = fdc_warp_sum_float(acc2);
    acc3 = fdc_warp_sum_float(acc3);

    if (lane == 0) {
        int64_t ob = static_cast<int64_t>(t) * 48 + kk;

        gk[ob + (nbase + 0) * 3] = acc0;
        gk[ob + (nbase + 1) * 3] = acc1;
        gk[ob + (nbase + 2) * 3] = acc2;
        gk[ob + (nbase + 3) * 3] = acc3;
    }
}

// ======================================================================================
// grad_kernel float-specialized, warp reduction over D=256
// ======================================================================================

template <int T_TILE, int N_TILE>
__global__ void fdc_n16k3_d256_grad_kernel_float_warp_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    const float* __restrict__ mix,
    float* __restrict__ gk,
    int L,
    int T,
    int off
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    int t = blockIdx.x * T_TILE + warp;
    int kk = blockIdx.y;
    int ntile = blockIdx.z;
    int nbase = ntile * N_TILE;

    if (warp >= T_TILE || t >= T) {
        return;
    }

    int s = off + t - kk;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    if (s >= 0 && s < L) {
        for (int d = lane; d < 256; d += 32) {
            float base =
                go[static_cast<int64_t>(d) * T + t] *
                h[static_cast<int64_t>(d) * L + s];

            int64_t mb = static_cast<int64_t>(d) * 16;

            acc0 += base * mix[mb + nbase + 0];
            acc1 += base * mix[mb + nbase + 1];
            acc2 += base * mix[mb + nbase + 2];
            acc3 += base * mix[mb + nbase + 3];
        }
    }

    acc0 = fdc_warp_sum_float(acc0);
    acc1 = fdc_warp_sum_float(acc1);
    acc2 = fdc_warp_sum_float(acc2);
    acc3 = fdc_warp_sum_float(acc3);

    if (lane == 0) {
        int64_t ob = static_cast<int64_t>(t) * 48 + kk;

        gk[ob + (nbase + 0) * 3] = acc0;
        gk[ob + (nbase + 1) * 3] = acc1;
        gk[ob + (nbase + 2) * 3] = acc2;
        gk[ob + (nbase + 3) * 3] = acc3;
    }
}

// ======================================================================================
// grad_mix typed partial
//
// Each block handles:
//   d
//   n_tile of 4 n values
//   one T tile
//
// gm[d,n] = sum_t sum_kk go[d,t] * h[d,off+t-kk] * kc[t,n,kk]
// ======================================================================================

template <typename scalar_t, int TILE_T, int N_TILE>
__global__ void fdc_n16k3_d256_grad_mix_typed_partial_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    float* __restrict__ partial,
    int L,
    int T,
    int off
) {
    int d = blockIdx.x;
    int ntile = blockIdx.y;
    int tile = blockIdx.z;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int nbase = ntile * N_TILE;

    int start = tile * TILE_T;
    int end = start + TILE_T;

    if (end > T) {
        end = T;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int t = start + tid; t < end; t += blockDim.x) {
        float g = fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]);

#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            int s = off + t - kk;

            if (s >= 0 && s < L) {
                float base =
                    g *
                    fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);

                int64_t kb = static_cast<int64_t>(t) * 48 + kk;

                acc0 += base * fdc_to_float_dev(kc[kb + (nbase + 0) * 3]);
                acc1 += base * fdc_to_float_dev(kc[kb + (nbase + 1) * 3]);
                acc2 += base * fdc_to_float_dev(kc[kb + (nbase + 2) * 3]);
                acc3 += base * fdc_to_float_dev(kc[kb + (nbase + 3) * 3]);
            }
        }
    }

    acc0 = fdc_warp_sum_float(acc0);
    acc1 = fdc_warp_sum_float(acc1);
    acc2 = fdc_warp_sum_float(acc2);
    acc3 = fdc_warp_sum_float(acc3);

    __shared__ float sh[8 * N_TILE];

    if (lane == 0) {
        sh[warp * N_TILE + 0] = acc0;
        sh[warp * N_TILE + 1] = acc1;
        sh[warp * N_TILE + 2] = acc2;
        sh[warp * N_TILE + 3] = acc3;
    }

    __syncthreads();

    if (warp == 0) {
        float v0 = lane < 8 ? sh[lane * N_TILE + 0] : 0.0f;
        float v1 = lane < 8 ? sh[lane * N_TILE + 1] : 0.0f;
        float v2 = lane < 8 ? sh[lane * N_TILE + 2] : 0.0f;
        float v3 = lane < 8 ? sh[lane * N_TILE + 3] : 0.0f;

        v0 = fdc_warp_sum_float(v0);
        v1 = fdc_warp_sum_float(v1);
        v2 = fdc_warp_sum_float(v2);
        v3 = fdc_warp_sum_float(v3);

        if (lane == 0) {
            int64_t ob =
                (static_cast<int64_t>(tile) * 256 + d) * 16 + nbase;

            partial[ob + 0] = v0;
            partial[ob + 1] = v1;
            partial[ob + 2] = v2;
            partial[ob + 3] = v3;
        }
    }
}

// ======================================================================================
// grad_mix float-specialized partial
// ======================================================================================

template <int TILE_T, int N_TILE>
__global__ void fdc_n16k3_d256_grad_mix_float_partial_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    const float* __restrict__ kc,
    float* __restrict__ partial,
    int L,
    int T,
    int off
) {
    int d = blockIdx.x;
    int ntile = blockIdx.y;
    int tile = blockIdx.z;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int nbase = ntile * N_TILE;

    int start = tile * TILE_T;
    int end = start + TILE_T;

    if (end > T) {
        end = T;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int t = start + tid; t < end; t += blockDim.x) {
        float g = go[static_cast<int64_t>(d) * T + t];

#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            int s = off + t - kk;

            if (s >= 0 && s < L) {
                float base =
                    g *
                    h[static_cast<int64_t>(d) * L + s];

                int64_t kb = static_cast<int64_t>(t) * 48 + kk;

                acc0 += base * kc[kb + (nbase + 0) * 3];
                acc1 += base * kc[kb + (nbase + 1) * 3];
                acc2 += base * kc[kb + (nbase + 2) * 3];
                acc3 += base * kc[kb + (nbase + 3) * 3];
            }
        }
    }

    acc0 = fdc_warp_sum_float(acc0);
    acc1 = fdc_warp_sum_float(acc1);
    acc2 = fdc_warp_sum_float(acc2);
    acc3 = fdc_warp_sum_float(acc3);

    __shared__ float sh[8 * N_TILE];

    if (lane == 0) {
        sh[warp * N_TILE + 0] = acc0;
        sh[warp * N_TILE + 1] = acc1;
        sh[warp * N_TILE + 2] = acc2;
        sh[warp * N_TILE + 3] = acc3;
    }

    __syncthreads();

    if (warp == 0) {
        float v0 = lane < 8 ? sh[lane * N_TILE + 0] : 0.0f;
        float v1 = lane < 8 ? sh[lane * N_TILE + 1] : 0.0f;
        float v2 = lane < 8 ? sh[lane * N_TILE + 2] : 0.0f;
        float v3 = lane < 8 ? sh[lane * N_TILE + 3] : 0.0f;

        v0 = fdc_warp_sum_float(v0);
        v1 = fdc_warp_sum_float(v1);
        v2 = fdc_warp_sum_float(v2);
        v3 = fdc_warp_sum_float(v3);

        if (lane == 0) {
            int64_t ob =
                (static_cast<int64_t>(tile) * 256 + d) * 16 + nbase;

            partial[ob + 0] = v0;
            partial[ob + 1] = v1;
            partial[ob + 2] = v2;
            partial[ob + 3] = v3;
        }
    }
}

// ======================================================================================
// grad_mix finalize
// ======================================================================================

template <int N>
__global__ void fdc_n16k3_d256_grad_mix_finalize_kernel(
    const float* __restrict__ partial,
    float* __restrict__ gm,
    int tiles
) {
    int d = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;

    int lane = tid & 31;
    int warp = tid >> 5;

    float acc = 0.0f;

    for (int tile = tid; tile < tiles; tile += blockDim.x) {
        acc += partial[(static_cast<int64_t>(tile) * 256 + d) * N + n];
    }

    acc = fdc_warp_sum_float(acc);

    __shared__ float sh[8];

    if (lane == 0) {
        sh[warp] = acc;
    }

    __syncthreads();

    if (warp == 0) {
        float v = lane < 8 ? sh[lane] : 0.0f;

        v = fdc_warp_sum_float(v);

        if (lane == 0) {
            gm[d * N + n] = v;
        }
    }
}

// ======================================================================================
// Fused cast three outputs for fp16/bf16 path
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_n16_cast3_float_to_scalar_kernel(
    const float* __restrict__ src_gh,
    const float* __restrict__ src_gk,
    const float* __restrict__ src_gm,
    scalar_t* __restrict__ dst_gh,
    scalar_t* __restrict__ dst_gk,
    scalar_t* __restrict__ dst_gm,
    int64_t n_gh,
    int64_t n_gk,
    int64_t n_gm
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n_gh) {
        dst_gh[i] = fdc_from_float_dev<scalar_t>(src_gh[i]);
    }

    if (i < n_gk) {
        dst_gk[i] = fdc_from_float_dev<scalar_t>(src_gk[i]);
    }

    if (i < n_gm) {
        dst_gm[i] = fdc_from_float_dev<scalar_t>(src_gm[i]);
    }
}

// ======================================================================================
// fp32 specialized wrapper
// ======================================================================================

static std::vector<torch::Tensor> fdc_backward_base_n16k3_float(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_base_n16k3_float_d256_v3");

    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    TORCH_CHECK(D == 256, "fdc_backward_base_n16k3_float requires D == 256.");

    auto fopts = h.options().dtype(torch::kFloat32);

    auto gh = torch::empty(
        h.sizes(),
        fopts
    );

    auto gk = torch::empty(
        kc.sizes(),
        fopts
    );

    auto gm = torch::empty(
        mix.sizes(),
        fopts
    );

    constexpr int T_TILE = 8;
    constexpr int N_TILE = 4;
    constexpr int MIX_TILE_T = 2048;

    int tiles = (T + MIX_TILE_T - 1) / MIX_TILE_T;

    auto partial = torch::empty(
        {
            tiles,
            256,
            16,
        },
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fdc_n16k3_d256_grad_h_float_kernel<<<
        (static_cast<int64_t>(256) * L + 255) / 256,
        256,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        kc.data_ptr<float>(),
        mix.data_ptr<float>(),
        gh.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );


    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_gk(
        (T + T_TILE - 1) / T_TILE,
        3,
        4
    );

    fdc_n16k3_d256_grad_kernel_float_warp_kernel<T_TILE, N_TILE><<<
        grid_gk,
        256,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        h.data_ptr<float>(),
        mix.data_ptr<float>(),
        gk.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_partial(
        256,
        4,
        tiles
    );

    fdc_n16k3_d256_grad_mix_float_partial_kernel<MIX_TILE_T, N_TILE><<<
        grid_partial,
        256,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        h.data_ptr<float>(),
        kc.data_ptr<float>(),
        partial.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_final(
        256,
        16,
        1
    );

    fdc_n16k3_d256_grad_mix_finalize_kernel<16><<<
        grid_final,
        256,
        0,
        stream
    >>>(
        partial.data_ptr<float>(),
        gm.data_ptr<float>(),
        tiles
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        gh,
        gk,
        gm
    };
}

// ======================================================================================
// typed wrapper for half/bfloat16
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> fdc_backward_base_n16k3_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_base_n16k3_typed_d256_v3");

    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    TORCH_CHECK(D == 256, "fdc_backward_base_n16k3_typed requires D == 256.");

    auto fopts = h.options().dtype(torch::kFloat32);

    auto ghf = torch::empty(
        h.sizes(),
        fopts
    );

    auto gkf = torch::empty(
        kc.sizes(),
        fopts
    );

    auto gmf = torch::empty(
        mix.sizes(),
        fopts
    );

    constexpr int T_TILE = 8;
    constexpr int N_TILE = 4;
    constexpr int MIX_TILE_T = 2048;

    int tiles = (T + MIX_TILE_T - 1) / MIX_TILE_T;

    auto partial = torch::empty(
        {
            tiles,
            256,
            16,
        },
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fdc_n16k3_d256_grad_h_typed_kernel<scalar_t><<<
        (static_cast<int64_t>(256) * L + 255) / 256,
        256,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        ghf.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_gk(
        (T + T_TILE - 1) / T_TILE,
        3,
        4
    );

    fdc_n16k3_d256_grad_kernel_typed_warp_kernel<scalar_t, T_TILE, N_TILE><<<
        grid_gk,
        256,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        gkf.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_partial(
        256,
        4,
        tiles
    );

    fdc_n16k3_d256_grad_mix_typed_partial_kernel<scalar_t, MIX_TILE_T, N_TILE><<<
        grid_partial,
        256,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        partial.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_final(
        256,
        16,
        1
    );

    fdc_n16k3_d256_grad_mix_finalize_kernel<16><<<
        grid_final,
        256,
        0,
        stream
    >>>(
        partial.data_ptr<float>(),
        gmf.data_ptr<float>(),
        tiles
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto gh = torch::empty_like(h);
    auto gk = torch::empty_like(kc);
    auto gm = torch::empty_like(mix);

    int64_t n_gh = gh.numel();
    int64_t n_gk = gk.numel();
    int64_t n_gm = gm.numel();

    int64_t max_n = n_gh;

    if (n_gk > max_n) {
        max_n = n_gk;
    }

    if (n_gm > max_n) {
        max_n = n_gm;
    }

    fdc_n16_cast3_float_to_scalar_kernel<scalar_t><<<
        (max_n + 255) / 256,
        256,
        0,
        stream
    >>>(
        ghf.data_ptr<float>(),
        gkf.data_ptr<float>(),
        gmf.data_ptr<float>(),
        gh.data_ptr<scalar_t>(),
        gk.data_ptr<scalar_t>(),
        gm.data_ptr<scalar_t>(),
        n_gh,
        n_gk,
        n_gm
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        gh,
        gk,
        gm
    };
}

// ======================================================================================
// Public base N16K3 plan
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_base_n16k3_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    TORCH_CHECK(h.dim() == 3, "h must be [B,D,L].");
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,T,N,K].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");
    TORCH_CHECK(go.dim() == 3, "go must be [B,D,T].");

    TORCH_CHECK(h.size(0) == 1, "fdc_backward_base_n16k3 requires B == 1.");
    TORCH_CHECK(h.size(1) == 256, "fdc_backward_base_n16k3 requires D == 256.");
    TORCH_CHECK(kc.size(2) == 16, "fdc_backward_base_n16k3 requires N == 16.");
    TORCH_CHECK(kc.size(3) == 3, "fdc_backward_base_n16k3 requires K == 3.");

    TORCH_CHECK(go.size(0) == 1, "go B mismatch.");
    TORCH_CHECK(go.size(1) == 256, "go D mismatch.");
    TORCH_CHECK(go.size(2) == kc.size(1), "go T mismatch.");

    TORCH_CHECK(mix.size(0) == 256, "mix D mismatch.");
    TORCH_CHECK(mix.size(1) == 16, "mix N mismatch.");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_backward_base_n16k3_float(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_backward_base_n16k3_typed<c10::Half>(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    return fdc_backward_base_n16k3_typed<c10::BFloat16>(
        go,
        h,
        kc,
        mix,
        off
    );
}
