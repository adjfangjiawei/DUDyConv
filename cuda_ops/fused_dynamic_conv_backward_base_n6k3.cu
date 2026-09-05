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
__global__ void fdc_base_n6_cast_float_to_scalar_kernel(
    const float* __restrict__ src,
    scalar_t* __restrict__ dst,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        dst[i] = fdc_from_float_dev<scalar_t>(src[i]);
    }
}

template <typename scalar_t>
__global__ void fdc_base_n6_make_mix_transpose_float_kernel(
    const scalar_t* __restrict__ mix,
    float* __restrict__ mixT,
    int D,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * N;

    if (idx >= total) {
        return;
    }

    int n = idx % N;
    int d = idx / N;

    mixT[n * D + d] = fdc_to_float_dev(mix[d * N + n]);
}

template <typename scalar_t>
__global__ void fdc_base_n6_make_base_ktd_float_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    float* __restrict__ base,
    int D,
    int L,
    int T,
    int K,
    int off
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(K) * T * D;

    if (idx >= total) {
        return;
    }

    int d = idx % D;
    int64_t q = idx / D;
    int t = q % T;
    int kk = q / T;

    int s = off + t - kk;

    float v = 0.0f;

    if (s >= 0 && s < L) {
        v =
            fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) *
            fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);
    }

    base[(static_cast<int64_t>(kk) * T + t) * D + d] = v;
}

// ======================================================================================
// grad_h N6K3
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_base_n6_grad_h_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gh,
    int L,
    int T,
    int off
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    if (d >= 256 || s >= L) {
        return;
    }

    int t0 = s - off;

    float m0 = fdc_to_float_dev(mix[d * 6 + 0]);
    float m1 = fdc_to_float_dev(mix[d * 6 + 1]);
    float m2 = fdc_to_float_dev(mix[d * 6 + 2]);
    float m3 = fdc_to_float_dev(mix[d * 6 + 3]);
    float m4 = fdc_to_float_dev(mix[d * 6 + 4]);
    float m5 = fdc_to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = t0 + kk;

        if (t >= 0 && t < T) {
            int64_t kb = static_cast<int64_t>(t) * 18 + kk;

            float w =
                fdc_to_float_dev(kc[kb + 0 * 3]) * m0 +
                fdc_to_float_dev(kc[kb + 1 * 3]) * m1 +
                fdc_to_float_dev(kc[kb + 2 * 3]) * m2 +
                fdc_to_float_dev(kc[kb + 3 * 3]) * m3 +
                fdc_to_float_dev(kc[kb + 4 * 3]) * m4 +
                fdc_to_float_dev(kc[kb + 5 * 3]) * m5;

            acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) * w;
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc;
}

template <typename scalar_t>
__global__ void fdc_base_n6_full4096_grad_h_kernel(
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
        int64_t kb = static_cast<int64_t>(t) * 18;

        float w =
            fdc_to_float_dev(kc[kb + 0]) * m0 +
            fdc_to_float_dev(kc[kb + 3]) * m1 +
            fdc_to_float_dev(kc[kb + 6]) * m2 +
            fdc_to_float_dev(kc[kb + 9]) * m3 +
            fdc_to_float_dev(kc[kb + 12]) * m4 +
            fdc_to_float_dev(kc[kb + 15]) * m5;

        acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * 4096 + t]) * w;
    }

    if (s + 1 < 4096) {
        int t = s + 1;
        int64_t kb = static_cast<int64_t>(t) * 18;

        float w =
            fdc_to_float_dev(kc[kb + 1]) * m0 +
            fdc_to_float_dev(kc[kb + 4]) * m1 +
            fdc_to_float_dev(kc[kb + 7]) * m2 +
            fdc_to_float_dev(kc[kb + 10]) * m3 +
            fdc_to_float_dev(kc[kb + 13]) * m4 +
            fdc_to_float_dev(kc[kb + 16]) * m5;

        acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * 4096 + t]) * w;
    }

    if (s + 2 < 4096) {
        int t = s + 2;
        int64_t kb = static_cast<int64_t>(t) * 18;

        float w =
            fdc_to_float_dev(kc[kb + 2]) * m0 +
            fdc_to_float_dev(kc[kb + 5]) * m1 +
            fdc_to_float_dev(kc[kb + 8]) * m2 +
            fdc_to_float_dev(kc[kb + 11]) * m3 +
            fdc_to_float_dev(kc[kb + 14]) * m4 +
            fdc_to_float_dev(kc[kb + 17]) * m5;

        acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * 4096 + t]) * w;
    }

    gh[static_cast<int64_t>(d) * 4096 + s] = acc;
}

// ======================================================================================
// grad_kernel base N6K3
// ======================================================================================

template <int T_TILE>
__global__ void fdc_base_n6_grad_kernel_warp_kernel(
    const float* __restrict__ base,
    const float* __restrict__ mixT,
    float* __restrict__ gk,
    int D,
    int T
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    int t = blockIdx.x * T_TILE + warp;
    int kk = blockIdx.y;

    if (warp >= T_TILE || t >= T) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    int64_t bb = (static_cast<int64_t>(kk) * T + t) * D;

    for (int d = lane; d < D; d += 32) {
        float b = base[bb + d];

        acc0 += b * mixT[0 * D + d];
        acc1 += b * mixT[1 * D + d];
        acc2 += b * mixT[2 * D + d];
        acc3 += b * mixT[3 * D + d];
        acc4 += b * mixT[4 * D + d];
        acc5 += b * mixT[5 * D + d];
    }

    acc0 = fdc_warp_sum_float(acc0);
    acc1 = fdc_warp_sum_float(acc1);
    acc2 = fdc_warp_sum_float(acc2);
    acc3 = fdc_warp_sum_float(acc3);
    acc4 = fdc_warp_sum_float(acc4);
    acc5 = fdc_warp_sum_float(acc5);

    if (lane == 0) {
        int64_t ob = static_cast<int64_t>(t) * 18;

        gk[ob + 0 * 3 + kk] = acc0;
        gk[ob + 1 * 3 + kk] = acc1;
        gk[ob + 2 * 3 + kk] = acc2;
        gk[ob + 3 * 3 + kk] = acc3;
        gk[ob + 4 * 3 + kk] = acc4;
        gk[ob + 5 * 3 + kk] = acc5;
    }
}

// ======================================================================================
// grad_mix base N6K3
// ======================================================================================

template <int TILE_T>
__global__ void fdc_base_n6_grad_mix_partial_kernel(
    const float* __restrict__ base,
    const void* __restrict__ kc_void,
    float* __restrict__ partial,
    int T,
    int dtype_tag
) {
    int d = blockIdx.x;
    int tile = blockIdx.y;
    int tid = threadIdx.x;

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
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            float b = base[(static_cast<int64_t>(kk) * T + t) * 256 + d];
            int64_t kb = static_cast<int64_t>(t) * 18 + kk;

            if (dtype_tag == 0) {
                const float* kc = reinterpret_cast<const float*>(kc_void);

                acc0 += b * kc[kb + 0 * 3];
                acc1 += b * kc[kb + 1 * 3];
                acc2 += b * kc[kb + 2 * 3];
                acc3 += b * kc[kb + 3 * 3];
                acc4 += b * kc[kb + 4 * 3];
                acc5 += b * kc[kb + 5 * 3];
            } else if (dtype_tag == 1) {
                const c10::Half* kc = reinterpret_cast<const c10::Half*>(kc_void);

                acc0 += b * fdc_to_float_dev(kc[kb + 0 * 3]);
                acc1 += b * fdc_to_float_dev(kc[kb + 1 * 3]);
                acc2 += b * fdc_to_float_dev(kc[kb + 2 * 3]);
                acc3 += b * fdc_to_float_dev(kc[kb + 3 * 3]);
                acc4 += b * fdc_to_float_dev(kc[kb + 4 * 3]);
                acc5 += b * fdc_to_float_dev(kc[kb + 5 * 3]);
            } else {
                const c10::BFloat16* kc = reinterpret_cast<const c10::BFloat16*>(kc_void);

                acc0 += b * fdc_to_float_dev(kc[kb + 0 * 3]);
                acc1 += b * fdc_to_float_dev(kc[kb + 1 * 3]);
                acc2 += b * fdc_to_float_dev(kc[kb + 2 * 3]);
                acc3 += b * fdc_to_float_dev(kc[kb + 3 * 3]);
                acc4 += b * fdc_to_float_dev(kc[kb + 4 * 3]);
                acc5 += b * fdc_to_float_dev(kc[kb + 5 * 3]);
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
            int64_t ob = (static_cast<int64_t>(tile) * 256 + d) * 6;

            partial[ob + 0] = v0;
            partial[ob + 1] = v1;
            partial[ob + 2] = v2;
            partial[ob + 3] = v3;
            partial[ob + 4] = v4;
            partial[ob + 5] = v5;
        }
    }
}

template <int N>
__global__ void fdc_base_n6_grad_mix_finalize_n_all_kernel(
    const float* __restrict__ partial,
    float* __restrict__ gm,
    int tiles
) {
    int d = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;

    float acc = 0.0f;

    for (int tile = tid; tile < tiles; tile += blockDim.x) {
        acc += partial[(static_cast<int64_t>(tile) * 256 + d) * N + n];
    }

    acc = fdc_warp_sum_float(acc);

    __shared__ float sh[8];

    int lane = tid & 31;
    int warp = tid >> 5;

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
// Typed wrapper
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> fdc_backward_base_n6k3_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    bool full4096_offset0
) {
    FDC_DEBUG_PATH("FDC path: backward_base_n6k3");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    auto fopts = h.options().dtype(torch::kFloat32);

    auto ghf = torch::empty(h.sizes(), fopts);
    auto gkf = torch::empty(kc.sizes(), fopts);
    auto gmf = torch::empty(mix.sizes(), fopts);
    auto base = torch::empty({3, T, 256}, fopts);
    auto mixT = torch::empty({6, 256}, fopts);

    constexpr int T_TILE = 8;
    constexpr int MIX_TILE_T = 1024;

    int tiles = (T + MIX_TILE_T - 1) / MIX_TILE_T;

    auto partial = torch::empty({tiles, 256, 6}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;

    fdc_base_n6_make_mix_transpose_float_kernel<scalar_t><<<
        (256 * 6 + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        mix.data_ptr<scalar_t>(),
        mixT.data_ptr<float>(),
        256,
        6
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_base_n6_make_base_ktd_float_kernel<scalar_t><<<
        (static_cast<int64_t>(3) * T * 256 + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        base.data_ptr<float>(),
        256,
        L,
        T,
        3,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (full4096_offset0) {
        fdc_base_n6_full4096_grad_h_kernel<scalar_t><<<
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
        dim3 block_h(16, 16);
        dim3 grid_h((L + 15) / 16, 16, 1);

        fdc_base_n6_grad_h_kernel<scalar_t><<<
            grid_h,
            block_h,
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
    }

    dim3 grid_gk(
        (T + T_TILE - 1) / T_TILE,
        3,
        1
    );

    fdc_base_n6_grad_kernel_warp_kernel<T_TILE><<<
        grid_gk,
        256,
        0,
        stream
    >>>(
        base.data_ptr<float>(),
        mixT.data_ptr<float>(),
        gkf.data_ptr<float>(),
        256,
        T
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int dtype_tag = 0;

    if (h.scalar_type() == at::ScalarType::Half) {
        dtype_tag = 1;
    }

    if (h.scalar_type() == at::ScalarType::BFloat16) {
        dtype_tag = 2;
    }

    dim3 grid_partial(
        256,
        tiles,
        1
    );

    fdc_base_n6_grad_mix_partial_kernel<MIX_TILE_T><<<
        grid_partial,
        256,
        0,
        stream
    >>>(
        base.data_ptr<float>(),
        static_cast<const void*>(kc.data_ptr<scalar_t>()),
        partial.data_ptr<float>(),
        T,
        dtype_tag
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_final(
        256,
        6,
        1
    );

    fdc_base_n6_grad_mix_finalize_n_all_kernel<6><<<
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

    if constexpr (std::is_same<scalar_t, float>::value) {
        return {ghf, gkf, gmf};
    } else {
        auto gh = torch::empty_like(h);
        auto gk = torch::empty_like(kc);
        auto gm = torch::empty_like(mix);

        fdc_base_n6_cast_float_to_scalar_kernel<scalar_t><<<
            (gh.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            ghf.data_ptr<float>(),
            gh.data_ptr<scalar_t>(),
            gh.numel()
        );

        fdc_base_n6_cast_float_to_scalar_kernel<scalar_t><<<
            (gk.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gkf.data_ptr<float>(),
            gk.data_ptr<scalar_t>(),
            gk.numel()
        );

        fdc_base_n6_cast_float_to_scalar_kernel<scalar_t><<<
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

        return {gh, gk, gm};
    }
}

// ======================================================================================
// Public base N6K3 plan
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_base_n6k3_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    bool full4096_offset0
) {
    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_backward_base_n6k3_typed<float>(
            go,
            h,
            kc,
            mix,
            off,
            full4096_offset0
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_backward_base_n6k3_typed<c10::Half>(
            go,
            h,
            kc,
            mix,
            off,
            full4096_offset0
        );
    }

    return fdc_backward_base_n6k3_typed<c10::BFloat16>(
        go,
        h,
        kc,
        mix,
        off,
        full4096_offset0
    );
}
