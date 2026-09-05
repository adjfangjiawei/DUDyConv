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
__global__ void fdc_base_n16_cast_float_to_scalar_kernel(
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
__global__ void fdc_base_n16_make_mix_transpose_float_kernel(
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
__global__ void fdc_base_n16_make_base_ktd_float_kernel(
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
// grad_h N16K3
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_base_n16_grad_h_kernel(
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

    int tb = s - off;

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = tb + kk;

        if (t >= 0 && t < T) {
            int64_t kb = static_cast<int64_t>(t) * 48 + kk;
            int64_t mb = static_cast<int64_t>(d) * 16;

            float w = 0.0f;

#pragma unroll
            for (int n = 0; n < 16; ++n) {
                w +=
                    fdc_to_float_dev(kc[kb + n * 3]) *
                    fdc_to_float_dev(mix[mb + n]);
            }

            acc += fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) * w;
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc;
}

// ======================================================================================
// grad_kernel N16K3
// ======================================================================================

template <int T_TILE, int N_TILE>
__global__ void fdc_base_n16_grad_kernel_warp_kernel(
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
    int ntile = blockIdx.z;
    int nbase = ntile * N_TILE;

    if (warp >= T_TILE || t >= T) {
        return;
    }

    float acc[N_TILE];

#pragma unroll
    for (int i = 0; i < N_TILE; ++i) {
        acc[i] = 0.0f;
    }

    int64_t bb = (static_cast<int64_t>(kk) * T + t) * D;

    for (int d = lane; d < D; d += 32) {
        float b = base[bb + d];

#pragma unroll
        for (int i = 0; i < N_TILE; ++i) {
            int n = nbase + i;

            if (n < 16) {
                acc[i] += b * mixT[n * D + d];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < N_TILE; ++i) {
        acc[i] = fdc_warp_sum_float(acc[i]);
    }

    if (lane == 0) {
        int64_t ob = static_cast<int64_t>(t) * 48;

#pragma unroll
        for (int i = 0; i < N_TILE; ++i) {
            int n = nbase + i;

            if (n < 16) {
                gk[ob + n * 3 + kk] = acc[i];
            }
        }
    }
}

// ======================================================================================
// grad_mix N16K3
// ======================================================================================

template <int TILE_T, int N_TILE>
__global__ void fdc_base_n16_grad_mix_partial_kernel(
    const float* __restrict__ base,
    const void* __restrict__ kc_void,
    float* __restrict__ partial,
    int T,
    int dtype_tag
) {
    int d = blockIdx.x;
    int ntile = blockIdx.y;
    int tile = blockIdx.z;
    int tid = threadIdx.x;
    int nbase = ntile * N_TILE;

    int start = tile * TILE_T;
    int end = start + TILE_T;

    if (end > T) {
        end = T;
    }

    float acc[N_TILE];

#pragma unroll
    for (int i = 0; i < N_TILE; ++i) {
        acc[i] = 0.0f;
    }

    for (int t = start + tid; t < end; t += blockDim.x) {
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            float b = base[(static_cast<int64_t>(kk) * T + t) * 256 + d];
            int64_t kb = static_cast<int64_t>(t) * 48 + kk;

#pragma unroll
            for (int i = 0; i < N_TILE; ++i) {
                int n = nbase + i;

                if (n < 16) {
                    if (dtype_tag == 0) {
                        const float* kc =
                            reinterpret_cast<const float*>(kc_void);

                        acc[i] += b * kc[kb + n * 3];
                    } else if (dtype_tag == 1) {
                        const c10::Half* kc =
                            reinterpret_cast<const c10::Half*>(kc_void);

                        acc[i] += b * fdc_to_float_dev(kc[kb + n * 3]);
                    } else {
                        const c10::BFloat16* kc =
                            reinterpret_cast<const c10::BFloat16*>(kc_void);

                        acc[i] += b * fdc_to_float_dev(kc[kb + n * 3]);
                    }
                }
            }
        }
    }

    int lane = tid & 31;
    int warp = tid >> 5;

#pragma unroll
    for (int i = 0; i < N_TILE; ++i) {
        acc[i] = fdc_warp_sum_float(acc[i]);
    }

    __shared__ float sh[8 * N_TILE];

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N_TILE; ++i) {
            sh[warp * N_TILE + i] = acc[i];
        }
    }

    __syncthreads();

    if (warp == 0) {
        float v[N_TILE];

#pragma unroll
        for (int i = 0; i < N_TILE; ++i) {
            v[i] = lane < 8 ? sh[lane * N_TILE + i] : 0.0f;
            v[i] = fdc_warp_sum_float(v[i]);
        }

        if (lane == 0) {
            int64_t ob =
                ((static_cast<int64_t>(tile) * 256 + d) * 16 + nbase);

#pragma unroll
            for (int i = 0; i < N_TILE; ++i) {
                int n = nbase + i;

                if (n < 16) {
                    partial[ob + i] = v[i];
                }
            }
        }
    }
}

template <int N>
__global__ void fdc_base_n16_grad_mix_finalize_n_all_kernel(
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
static std::vector<torch::Tensor> fdc_backward_base_n16k3_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_base_n16k3");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    auto fopts = h.options().dtype(torch::kFloat32);

    auto ghf = torch::empty(h.sizes(), fopts);
    auto gkf = torch::empty(kc.sizes(), fopts);
    auto gmf = torch::empty(mix.sizes(), fopts);
    auto base = torch::empty({3, T, 256}, fopts);
    auto mixT = torch::empty({16, 256}, fopts);

    constexpr int T_TILE = 8;
    constexpr int N_TILE = 4;
    constexpr int MIX_TILE_T = 1024;

    int tiles = (T + MIX_TILE_T - 1) / MIX_TILE_T;

    auto partial = torch::empty({tiles, 256, 16}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;

    fdc_base_n16_make_mix_transpose_float_kernel<scalar_t><<<
        (256 * 16 + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        mix.data_ptr<scalar_t>(),
        mixT.data_ptr<float>(),
        256,
        16
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_base_n16_make_base_ktd_float_kernel<scalar_t><<<
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

    dim3 block_h(16, 16);
    dim3 grid_h(
        (L + 15) / 16,
        16,
        1
    );

    fdc_base_n16_grad_h_kernel<scalar_t><<<
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
        (T + T_TILE - 1) / T_TILE,
        3,
        4
    );

    fdc_base_n16_grad_kernel_warp_kernel<T_TILE, N_TILE><<<
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
        4,
        tiles
    );

    fdc_base_n16_grad_mix_partial_kernel<MIX_TILE_T, N_TILE><<<
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
        16,
        1
    );

    fdc_base_n16_grad_mix_finalize_n_all_kernel<16><<<
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

        fdc_base_n16_cast_float_to_scalar_kernel<scalar_t><<<
            (gh.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            ghf.data_ptr<float>(),
            gh.data_ptr<scalar_t>(),
            gh.numel()
        );

        fdc_base_n16_cast_float_to_scalar_kernel<scalar_t><<<
            (gk.numel() + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gkf.data_ptr<float>(),
            gk.data_ptr<scalar_t>(),
            gk.numel()
        );

        fdc_base_n16_cast_float_to_scalar_kernel<scalar_t><<<
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
// Public base N16K3 plan
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_base_n16k3_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_backward_base_n16k3_typed<float>(
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
