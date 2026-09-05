#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <type_traits>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Local cast helper
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_mid_cast_float_to_scalar_kernel(
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
// grad_h N6K3 shared by mid path, runtime D
//
// kc layout: [1,3,6,T]
// kc[kk,n,t] = (kk * 6 + n) * T + t
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_mid_n6k3_grad_h_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gh,
    int D,
    int L,
    int T,
    int off
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    if (d >= D || s >= L) {
        return;
    }

    int t0 = s - off;

    int64_t mb = static_cast<int64_t>(d) * 6;

    float m0 = fdc_to_float_dev(mix[mb + 0]);
    float m1 = fdc_to_float_dev(mix[mb + 1]);
    float m2 = fdc_to_float_dev(mix[mb + 2]);
    float m3 = fdc_to_float_dev(mix[mb + 3]);
    float m4 = fdc_to_float_dev(mix[mb + 4]);
    float m5 = fdc_to_float_dev(mix[mb + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = t0 + kk;

        if (t >= 0 && t < T) {
            int64_t kb = static_cast<int64_t>(kk) * 6 * T + t;

            float w =
                fdc_to_float_dev(kc[kb + 0 * T]) * m0 +
                fdc_to_float_dev(kc[kb + 1 * T]) * m1 +
                fdc_to_float_dev(kc[kb + 2 * T]) * m2 +
                fdc_to_float_dev(kc[kb + 3 * T]) * m3 +
                fdc_to_float_dev(kc[kb + 4 * T]) * m4 +
                fdc_to_float_dev(kc[kb + 5 * T]) * m5;

            acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) * w;
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc;
}

// ======================================================================================
// full4096 offset0 specialized grad_h, D=256 only
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_mid_full4096_n6k3_grad_h_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * 4096;

    if (idx >= total) {
        return;
    }

    int s = idx & 4095;
    int d = idx >> 12;

    float m0 = fdc_to_float_dev(mix[d * 6 + 0]);
    float m1 = fdc_to_float_dev(mix[d * 6 + 1]);
    float m2 = fdc_to_float_dev(mix[d * 6 + 2]);
    float m3 = fdc_to_float_dev(mix[d * 6 + 3]);
    float m4 = fdc_to_float_dev(mix[d * 6 + 4]);
    float m5 = fdc_to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

    {
        int t = s;
        int64_t kb = static_cast<int64_t>(0) * 6 * 4096 + t;

        float w =
            fdc_to_float_dev(kc[kb + 0 * 4096]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * 4096]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * 4096]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * 4096]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * 4096]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * 4096]) * m5;

        acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * 4096 + t]) * w;
    }

    if (s + 1 < 4096) {
        int t = s + 1;
        int64_t kb = static_cast<int64_t>(1) * 6 * 4096 + t;

        float w =
            fdc_to_float_dev(kc[kb + 0 * 4096]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * 4096]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * 4096]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * 4096]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * 4096]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * 4096]) * m5;

        acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * 4096 + t]) * w;
    }

    if (s + 2 < 4096) {
        int t = s + 2;
        int64_t kb = static_cast<int64_t>(2) * 6 * 4096 + t;

        float w =
            fdc_to_float_dev(kc[kb + 0 * 4096]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * 4096]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * 4096]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * 4096]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * 4096]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * 4096]) * m5;

        acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * 4096 + t]) * w;
    }

    gh[static_cast<int64_t>(d) * 4096 + s] = acc;
}

// ======================================================================================
// grad_kernel mid warp, runtime D
//
// output gk layout: [1,3,6,T]
// ======================================================================================

template <typename scalar_t, int T_TILE>
__global__ void fdc_mid_n6k3_grad_kernel_warp_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gk,
    int D,
    int L,
    int T,
    int off
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    int t = blockIdx.x * T_TILE + warp;
    int kk = blockIdx.y;

    if (warp >= T_TILE || t >= T) {
        return;
    }

    int s = off + t - kk;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    if (s >= 0 && s < L) {
        for (int d = lane; d < D; d += 32) {
            float base =
                fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) *
                fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);

            int64_t mb = static_cast<int64_t>(d) * 6;

            acc0 += base * fdc_to_float_dev(mix[mb + 0]);
            acc1 += base * fdc_to_float_dev(mix[mb + 1]);
            acc2 += base * fdc_to_float_dev(mix[mb + 2]);
            acc3 += base * fdc_to_float_dev(mix[mb + 3]);
            acc4 += base * fdc_to_float_dev(mix[mb + 4]);
            acc5 += base * fdc_to_float_dev(mix[mb + 5]);
        }
    }

    acc0 = fdc_warp_sum_float(acc0);
    acc1 = fdc_warp_sum_float(acc1);
    acc2 = fdc_warp_sum_float(acc2);
    acc3 = fdc_warp_sum_float(acc3);
    acc4 = fdc_warp_sum_float(acc4);
    acc5 = fdc_warp_sum_float(acc5);

    if (lane == 0) {
        int64_t base = static_cast<int64_t>(kk) * 6 * T + t;

        gk[base + 0 * T] = acc0;
        gk[base + 1 * T] = acc1;
        gk[base + 2 * T] = acc2;
        gk[base + 3 * T] = acc3;
        gk[base + 4 * T] = acc4;
        gk[base + 5 * T] = acc5;
    }
}

// ======================================================================================
// grad_mix mid all n, runtime D
//
// kc layout: [1,3,6,T]
// ======================================================================================

template <typename scalar_t, int TILE_T>
__global__ void fdc_mid_n6k3_grad_mix_partial_alln_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    float* __restrict__ partial,
    int D,
    int L,
    int T,
    int off
) {
    int d = blockIdx.x;
    int tile = blockIdx.y;
    int tid = threadIdx.x;

    if (d >= D) {
        return;
    }

    int start = tile * TILE_T;
    int end = start + TILE_T;

    if (end > T) {
        end = T;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    for (int t = start + tid; t < end; t += blockDim.x) {
        float g = fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]);

#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            int s = off + t - kk;

            if (s >= 0 && s < L) {
                float base =
                    g *
                    fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);

                int64_t kb = static_cast<int64_t>(kk) * 6 * T + t;

                acc0 += base * fdc_to_float_dev(kc[kb + 0 * T]);
                acc1 += base * fdc_to_float_dev(kc[kb + 1 * T]);
                acc2 += base * fdc_to_float_dev(kc[kb + 2 * T]);
                acc3 += base * fdc_to_float_dev(kc[kb + 3 * T]);
                acc4 += base * fdc_to_float_dev(kc[kb + 4 * T]);
                acc5 += base * fdc_to_float_dev(kc[kb + 5 * T]);
            }
        }
    }

    int lane = tid & 31;
    int warp = tid >> 5;

    acc0 = fdc_warp_sum_float(acc0);
    acc1 = fdc_warp_sum_float(acc1);
    acc2 = fdc_warp_sum_float(acc2);
    acc3 = fdc_warp_sum_float(acc3);
    acc4 = fdc_warp_sum_float(acc4);
    acc5 = fdc_warp_sum_float(acc5);

    __shared__ float sh[8 * 6];

    if (lane == 0) {
        sh[warp * 6 + 0] = acc0;
        sh[warp * 6 + 1] = acc1;
        sh[warp * 6 + 2] = acc2;
        sh[warp * 6 + 3] = acc3;
        sh[warp * 6 + 4] = acc4;
        sh[warp * 6 + 5] = acc5;
    }

    __syncthreads();

    if (warp == 0) {
        float v0 = lane < 8 ? sh[lane * 6 + 0] : 0.0f;
        float v1 = lane < 8 ? sh[lane * 6 + 1] : 0.0f;
        float v2 = lane < 8 ? sh[lane * 6 + 2] : 0.0f;
        float v3 = lane < 8 ? sh[lane * 6 + 3] : 0.0f;
        float v4 = lane < 8 ? sh[lane * 6 + 4] : 0.0f;
        float v5 = lane < 8 ? sh[lane * 6 + 5] : 0.0f;

        v0 = fdc_warp_sum_float(v0);
        v1 = fdc_warp_sum_float(v1);
        v2 = fdc_warp_sum_float(v2);
        v3 = fdc_warp_sum_float(v3);
        v4 = fdc_warp_sum_float(v4);
        v5 = fdc_warp_sum_float(v5);

        if (lane == 0) {
            int64_t ob = (static_cast<int64_t>(tile) * D + d) * 6;

            partial[ob + 0] = v0;
            partial[ob + 1] = v1;
            partial[ob + 2] = v2;
            partial[ob + 3] = v3;
            partial[ob + 4] = v4;
            partial[ob + 5] = v5;
        }
    }
}

__global__ void fdc_mid_n6k3_grad_mix_finalize_alln_kernel(
    const float* __restrict__ partial,
    float* __restrict__ gm,
    int D,
    int tiles
) {
    int d = blockIdx.x;
    int tid = threadIdx.x;

    if (d >= D) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    for (int tile = tid; tile < tiles; tile += blockDim.x) {
        int64_t ib = (static_cast<int64_t>(tile) * D + d) * 6;

        acc0 += partial[ib + 0];
        acc1 += partial[ib + 1];
        acc2 += partial[ib + 2];
        acc3 += partial[ib + 3];
        acc4 += partial[ib + 4];
        acc5 += partial[ib + 5];
    }

    int lane = tid & 31;
    int warp = tid >> 5;

    acc0 = fdc_warp_sum_float(acc0);
    acc1 = fdc_warp_sum_float(acc1);
    acc2 = fdc_warp_sum_float(acc2);
    acc3 = fdc_warp_sum_float(acc3);
    acc4 = fdc_warp_sum_float(acc4);
    acc5 = fdc_warp_sum_float(acc5);

    __shared__ float sh[8 * 6];

    if (lane == 0) {
        sh[warp * 6 + 0] = acc0;
        sh[warp * 6 + 1] = acc1;
        sh[warp * 6 + 2] = acc2;
        sh[warp * 6 + 3] = acc3;
        sh[warp * 6 + 4] = acc4;
        sh[warp * 6 + 5] = acc5;
    }

    __syncthreads();

    if (warp == 0) {
        float v0 = lane < 8 ? sh[lane * 6 + 0] : 0.0f;
        float v1 = lane < 8 ? sh[lane * 6 + 1] : 0.0f;
        float v2 = lane < 8 ? sh[lane * 6 + 2] : 0.0f;
        float v3 = lane < 8 ? sh[lane * 6 + 3] : 0.0f;
        float v4 = lane < 8 ? sh[lane * 6 + 4] : 0.0f;
        float v5 = lane < 8 ? sh[lane * 6 + 5] : 0.0f;

        v0 = fdc_warp_sum_float(v0);
        v1 = fdc_warp_sum_float(v1);
        v2 = fdc_warp_sum_float(v2);
        v3 = fdc_warp_sum_float(v3);
        v4 = fdc_warp_sum_float(v4);
        v5 = fdc_warp_sum_float(v5);

        if (lane == 0) {
            gm[d * 6 + 0] = v0;
            gm[d * 6 + 1] = v1;
            gm[d * 6 + 2] = v2;
            gm[d * 6 + 3] = v3;
            gm[d * 6 + 4] = v4;
            gm[d * 6 + 5] = v5;
        }
    }
}

// ======================================================================================
// Typed wrapper
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> fdc_backward_mid_n6k3_warp_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    bool full4096_offset0
) {
    FDC_DEBUG_PATH("FDC path: backward_mid_n6k3_warp_d256_d512_bknt");

    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(3));

    TORCH_CHECK(
        D == 256 || D == 512,
        "fdc_backward_mid_n6k3_warp_typed supports D == 256 or D == 512."
    );

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
    constexpr int MIX_TILE_T = 2048;

    int tiles = (T + MIX_TILE_T - 1) / MIX_TILE_T;

    auto partial = torch::empty(
        {
            tiles,
            D,
            6,
        },
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (full4096_offset0 && D == 256) {
        fdc_mid_full4096_n6k3_grad_h_kernel<scalar_t><<<
            (256 * 4096 + 255) / 256,
            256,
            0,
            stream
        >>>(
            go.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            ghf.data_ptr<float>()
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        dim3 block_h(
            16,
            16
        );

        dim3 grid_h(
            (L + 15) / 16,
            (D + 15) / 16,
            1
        );

        fdc_mid_n6k3_grad_h_kernel<scalar_t><<<
            grid_h,
            block_h,
            0,
            stream
        >>>(
            go.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            ghf.data_ptr<float>(),
            D,
            L,
            T,
            static_cast<int>(off)
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    dim3 grid_gk(
        (T + T_TILE - 1) / T_TILE,
        3,
        1
    );

    fdc_mid_n6k3_grad_kernel_warp_kernel<scalar_t, T_TILE><<<
        grid_gk,
        256,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        gkf.data_ptr<float>(),
        D,
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_partial(
        D,
        tiles,
        1
    );

    fdc_mid_n6k3_grad_mix_partial_alln_kernel<scalar_t, MIX_TILE_T><<<
        grid_partial,
        256,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        partial.data_ptr<float>(),
        D,
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_mid_n6k3_grad_mix_finalize_alln_kernel<<<
        D,
        256,
        0,
        stream
    >>>(
        partial.data_ptr<float>(),
        gmf.data_ptr<float>(),
        D,
        tiles
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if constexpr (std::is_same<scalar_t, float>::value) {
        return {
            ghf,
            gkf,
            gmf
        };
    } else {
        auto gh = torch::empty_like(h);
        auto gk = torch::empty_like(kc);
        auto gm = torch::empty_like(mix);

        int threads = 256;

        fdc_mid_cast_float_to_scalar_kernel<scalar_t><<<
            (gh.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            ghf.data_ptr<float>(),
            gh.data_ptr<scalar_t>(),
            gh.numel()
        );

        fdc_mid_cast_float_to_scalar_kernel<scalar_t><<<
            (gk.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gkf.data_ptr<float>(),
            gk.data_ptr<scalar_t>(),
            gk.numel()
        );

        fdc_mid_cast_float_to_scalar_kernel<scalar_t><<<
            (gm.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gmf.data_ptr<float>(),
            gm.data_ptr<scalar_t>(),
            gm.numel()
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return {
            gh,
            gk,
            gm
        };
    }
}

// ======================================================================================
// Public mid N6K3 plan
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_mid_n6k3_warp_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    bool full4096_offset0
) {
    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_backward_mid_n6k3_warp_typed<float>(
            go,
            h,
            kc,
            mix,
            off,
            full4096_offset0
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_backward_mid_n6k3_warp_typed<c10::Half>(
            go,
            h,
            kc,
            mix,
            off,
            full4096_offset0
        );
    }

    return fdc_backward_mid_n6k3_warp_typed<c10::BFloat16>(
        go,
        h,
        kc,
        mix,
        off,
        full4096_offset0
    );
}
