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
__global__ void fdc_small_cast_float_to_scalar_kernel(
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
// grad_h for small N6K3
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_small_n6k3_grad_h_kernel(
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
            int64_t kb = (int64_t)t * 18 + kk;

            float w =
                fdc_to_float_dev(kc[kb + 0 * 3]) * m0 +
                fdc_to_float_dev(kc[kb + 1 * 3]) * m1 +
                fdc_to_float_dev(kc[kb + 2 * 3]) * m2 +
                fdc_to_float_dev(kc[kb + 3 * 3]) * m3 +
                fdc_to_float_dev(kc[kb + 4 * 3]) * m4 +
                fdc_to_float_dev(kc[kb + 5 * 3]) * m5;

            acc += fdc_to_float_dev(go[(int64_t)d * T + t]) * w;
        }
    }

    gh[(int64_t)d * L + s] = acc;
}

// ======================================================================================
// grad_kernel for small N6K3
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_small_n6k3_grad_kernel_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gk,
    int L,
    int T,
    int off
) {
    int t = blockIdx.x;
    int kk = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ float sh[6 * 256];

    float base = 0.0f;

    int s = off + t - kk;

    if (t < T && s >= 0 && s < L) {
        base =
            fdc_to_float_dev(go[(int64_t)tid * T + t]) *
            fdc_to_float_dev(h[(int64_t)tid * L + s]);
    }

#pragma unroll
    for (int n = 0; n < 6; ++n) {
        sh[n * 256 + tid] =
            base * fdc_to_float_dev(mix[tid * 6 + n]);
    }

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int n = 0; n < 6; ++n) {
                sh[n * 256 + tid] += sh[n * 256 + tid + stride];
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int n = 0; n < 6; ++n) {
            gk[(int64_t)t * 18 + n * 3 + kk] = sh[n * 256];
        }
    }
}

// ======================================================================================
// grad_mix for small N6K3
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_small_n6k3_grad_mix_partial_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    float* __restrict__ partial,
    int L,
    int T,
    int off,
    int tiles
) {
    int d = blockIdx.x;
    int n = blockIdx.y;
    int tile = blockIdx.z;
    int tid = threadIdx.x;

    int start = tile * 512;
    int end = start + 512;

    if (end > T) {
        end = T;
    }

    float acc = 0.0f;

    for (int t = start + tid; t < end; t += 256) {
        float g = fdc_to_float_dev(go[(int64_t)d * T + t]);

#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            int s = off + t - kk;

            if (s >= 0 && s < L) {
                acc +=
                    g *
                    fdc_to_float_dev(h[(int64_t)d * L + s]) *
                    fdc_to_float_dev(kc[(int64_t)t * 18 + n * 3 + kk]);
            }
        }
    }

    __shared__ float sh[256];

    sh[tid] = acc;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[((int64_t)tile * 256 + d) * 6 + n] = sh[0];
    }
}

__global__ void fdc_small_n6k3_grad_mix_finalize_kernel(
    const float* __restrict__ partial,
    float* __restrict__ gm,
    int tiles
) {
    int d = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;

    float acc = 0.0f;

    for (int tile = tid; tile < tiles; tile += 256) {
        acc += partial[((int64_t)tile * 256 + d) * 6 + n];
    }

    __shared__ float sh[256];

    sh[tid] = acc;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        gm[d * 6 + n] = sh[0];
    }
}

// ======================================================================================
// Typed wrapper
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> fdc_backward_small_n6k3_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_small_n6k3");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

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

    int tiles = (T + 511) / 512;

    auto partial = torch::empty(
        {tiles, 256, 6},
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 block_h(16, 16);
    dim3 grid_h(
        (L + 15) / 16,
        16,
        1
    );

    fdc_small_n6k3_grad_h_kernel<scalar_t><<<
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

    dim3 grid_gk(
        T,
        3,
        1
    );

    fdc_small_n6k3_grad_kernel_kernel<scalar_t><<<
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
        6,
        tiles
    );

    fdc_small_n6k3_grad_mix_partial_kernel<scalar_t><<<
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
        static_cast<int>(off),
        tiles
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid_final(
        256,
        6,
        1
    );

    fdc_small_n6k3_grad_mix_finalize_kernel<<<
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

        fdc_small_cast_float_to_scalar_kernel<scalar_t><<<
            (gh.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            ghf.data_ptr<float>(),
            gh.data_ptr<scalar_t>(),
            gh.numel()
        );

        fdc_small_cast_float_to_scalar_kernel<scalar_t><<<
            (gk.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gkf.data_ptr<float>(),
            gk.data_ptr<scalar_t>(),
            gk.numel()
        );

        fdc_small_cast_float_to_scalar_kernel<scalar_t><<<
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
// Public small N6K3 plan
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_small_n6k3_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_backward_small_n6k3_typed<float>(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_backward_small_n6k3_typed<c10::Half>(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    return fdc_backward_small_n6k3_typed<c10::BFloat16>(
        go,
        h,
        kc,
        mix,
        off
    );
}
