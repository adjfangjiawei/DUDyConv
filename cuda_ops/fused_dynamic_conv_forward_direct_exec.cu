#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Forward direct execution implementations
//
// kc layout is now:
//
//   [B,K,N,T]
//
// contiguous offset:
//
//   kc[b,kk,n,t] = ((b * K + kk) * N + n) * T + t
// ======================================================================================

// ======================================================================================
// Kernels: generic 2D
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_forward_direct_generic_2d_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int B,
    int D,
    int L,
    int T,
    int N,
    int K,
    int off,
    int dilation
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= B || d >= D || t >= T) {
        return;
    }

    float acc = 0.0f;

    int64_t hbase = ((int64_t)b * D + d) * L;
    int64_t obase = ((int64_t)b * D + d) * T;
    int64_t mbase = (int64_t)d * N;

    for (int kk = 0; kk < K; ++kk) {
        int s = off + t - kk * dilation;

        if (s < 0 || s >= L) {
            continue;
        }

        float w = 0.0f;

        int64_t kbase = ((int64_t)b * K + kk) * N * T + t;

        for (int n = 0; n < N; ++n) {
            w += fdc_to_float_dev(kc[kbase + (int64_t)n * T]) *
                 fdc_to_float_dev(mix[mbase + n]);
        }

        acc += w * fdc_to_float_dev(h[hbase + s]);
    }

    out[obase + t] = fdc_from_float_dev<scalar_t>(acc);
}

// ======================================================================================
// Kernels: generic small-N preload
// ======================================================================================

template <typename scalar_t, bool NO_BOUNDARY>
__global__ void fdc_forward_direct_generic_smalln_preload_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int B,
    int D,
    int L,
    int T,
    int N,
    int K,
    int off,
    int dilation
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(B) * D * T;

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(idx % T);
    int64_t q = idx / T;
    int d = static_cast<int>(q % D);
    int b = static_cast<int>(q / D);

    int64_t hbase = ((int64_t)b * D + d) * L;
    int64_t obase = ((int64_t)b * D + d) * T;
    int64_t mbase = (int64_t)d * N;

    float m[16];

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        m[i] = 0.0f;
    }

    for (int n = 0; n < N; ++n) {
        m[n] = fdc_to_float_dev(mix[mbase + n]);
    }

    float acc = 0.0f;

    for (int kk = 0; kk < K; ++kk) {
        int s = off + t - kk * dilation;

        if constexpr (!NO_BOUNDARY) {
            if (s < 0 || s >= L) {
                continue;
            }
        }

        float w = 0.0f;

        int64_t kbase = ((int64_t)b * K + kk) * N * T + t;

        for (int n = 0; n < N; ++n) {
            w += fdc_to_float_dev(kc[kbase + (int64_t)n * T]) * m[n];
        }

        acc += w * fdc_to_float_dev(h[hbase + s]);
    }

    out[obase + t] = fdc_from_float_dev<scalar_t>(acc);
}

// ======================================================================================
// Kernels: B=1, D=256, N=6, K=3, dilation=1
// kc: [1,3,6,T]
// ======================================================================================

template <typename scalar_t, bool NO_BOUNDARY>
__global__ void fdc_forward_direct_n6k3_d256_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    float m0 = fdc_to_float_dev(mix[d * 6 + 0]);
    float m1 = fdc_to_float_dev(mix[d * 6 + 1]);
    float m2 = fdc_to_float_dev(mix[d * 6 + 2]);
    float m3 = fdc_to_float_dev(mix[d * 6 + 3]);
    float m4 = fdc_to_float_dev(mix[d * 6 + 4]);
    float m5 = fdc_to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int s = off + t - kk;

        if constexpr (!NO_BOUNDARY) {
            if (s < 0 || s >= L) {
                continue;
            }
        }

        int64_t kb = (int64_t)kk * 6 * T + t;

        float w =
            fdc_to_float_dev(kc[kb + 0 * T]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * T]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * T]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * T]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * T]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * T]) * m5;

        acc += w * fdc_to_float_dev(h[(int64_t)d * L + s]);
    }

    out[(int64_t)d * T + t] = fdc_from_float_dev<scalar_t>(acc);
}

// ======================================================================================
// Kernels: B=1, D=512, N=6, K=3, dilation=1
// kc: [1,3,6,T]
// ======================================================================================

template <typename scalar_t, bool NO_BOUNDARY>
__global__ void fdc_forward_direct_n6k3_d512_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 512 * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    float m0 = fdc_to_float_dev(mix[d * 6 + 0]);
    float m1 = fdc_to_float_dev(mix[d * 6 + 1]);
    float m2 = fdc_to_float_dev(mix[d * 6 + 2]);
    float m3 = fdc_to_float_dev(mix[d * 6 + 3]);
    float m4 = fdc_to_float_dev(mix[d * 6 + 4]);
    float m5 = fdc_to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int s = off + t - kk;

        if constexpr (!NO_BOUNDARY) {
            if (s < 0 || s >= L) {
                continue;
            }
        }

        int64_t kb = (int64_t)kk * 6 * T + t;

        float w =
            fdc_to_float_dev(kc[kb + 0 * T]) * m0 +
            fdc_to_float_dev(kc[kb + 1 * T]) * m1 +
            fdc_to_float_dev(kc[kb + 2 * T]) * m2 +
            fdc_to_float_dev(kc[kb + 3 * T]) * m3 +
            fdc_to_float_dev(kc[kb + 4 * T]) * m4 +
            fdc_to_float_dev(kc[kb + 5 * T]) * m5;

        acc += w * fdc_to_float_dev(h[(int64_t)d * L + s]);
    }

    out[(int64_t)d * T + t] = fdc_from_float_dev<scalar_t>(acc);
}

// =================================================================================================
// Typed runners: generic 2D
// =================================================================================================

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_generic_2d_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    int B = static_cast<int>(h.size(0));
    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int K = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));
    int T = static_cast<int>(kc.size(3));

    auto out = torch::empty(
        {B, D, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 block(16, 16);
    dim3 grid(
        (T + 15) / 16,
        (D + 15) / 16,
        B
    );

    fdc_forward_direct_generic_2d_kernel<scalar_t><<<
        grid,
        block,
        0,
        stream
    >>>(
        h.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        B,
        D,
        L,
        T,
        N,
        K,
        static_cast<int>(off),
        static_cast<int>(dilation)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ======================================================================================
// Typed runners: generic small-N preload
// ======================================================================================

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_generic_smalln_preload_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    int B = static_cast<int>(h.size(0));
    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int K = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));
    int T = static_cast<int>(kc.size(3));

    auto out = torch::empty(
        {B, D, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t total = static_cast<int64_t>(B) * D * T;

    bool no_boundary =
        static_cast<int>(dilation) == 1 &&
        static_cast<int>(off) >= K - 1;

    if (no_boundary) {
        fdc_forward_direct_generic_smalln_preload_kernel<scalar_t, true><<<
            static_cast<unsigned int>((total + 255) / 256),
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            D,
            L,
            T,
            N,
            K,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    } else {
        fdc_forward_direct_generic_smalln_preload_kernel<scalar_t, false><<<
            static_cast<unsigned int>((total + 255) / 256),
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B,
            D,
            L,
            T,
            N,
            K,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ======================================================================================
// Typed runners: B=1, D=256, N=6, K=3, dilation=1
// ======================================================================================

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_n6k3_d256_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(3));

    auto out = torch::empty(
        {1, 256, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool no_boundary = static_cast<int>(off) >= 2;

    if (no_boundary) {
        fdc_forward_direct_n6k3_d256_kernel<scalar_t, true><<<
            static_cast<unsigned int>((256 * T + 255) / 256),
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    } else {
        fdc_forward_direct_n6k3_d256_kernel<scalar_t, false><<<
            static_cast<unsigned int>((256 * T + 255) / 256),
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ======================================================================================
// Typed runners: B=1, D=512, N=6, K=3, dilation=1
// ======================================================================================

template <typename scalar_t>
static torch::Tensor fdc_run_forward_direct_n6k3_d512_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(3));

    auto out = torch::empty(
        {1, 512, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    bool no_boundary = static_cast<int>(off) >= 2;

    if (no_boundary) {
        fdc_forward_direct_n6k3_d512_kernel<scalar_t, true><<<
            static_cast<unsigned int>((512 * T + 255) / 256),
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    } else {
        fdc_forward_direct_n6k3_d512_kernel<scalar_t, false><<<
            static_cast<unsigned int>((512 * T + 255) / 256),
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            L,
            T,
            static_cast<int>(off)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// =================================================================================================
// Public wrappers used by forward selector
// =================================================================================================

torch::Tensor fdc_forward_direct_generic_2d_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FDC path: forward_direct_generic_2d_exec");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_run_forward_direct_generic_2d_typed<float>(
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_run_forward_direct_generic_2d_typed<c10::Half>(
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    return fdc_run_forward_direct_generic_2d_typed<c10::BFloat16>(
        h,
        kc,
        mix,
        off,
        dilation
    );
}

torch::Tensor fdc_forward_direct_generic_smalln_preload_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FDC path: forward_direct_generic_smalln_preload_exec");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_run_forward_direct_generic_smalln_preload_typed<float>(
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_run_forward_direct_generic_smalln_preload_typed<c10::Half>(
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    return fdc_run_forward_direct_generic_smalln_preload_typed<c10::BFloat16>(
        h,
        kc,
        mix,
        off,
        dilation
    );
}

torch::Tensor fdc_forward_direct_n6k3_d256_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: forward_direct_n6k3_d256_exec");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_run_forward_direct_n6k3_d256_typed<float>(
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_run_forward_direct_n6k3_d256_typed<c10::Half>(
            h,
            kc,
            mix,
            off
        );
    }

    return fdc_run_forward_direct_n6k3_d256_typed<c10::BFloat16>(
        h,
        kc,
        mix,
        off
    );
}

torch::Tensor fdc_forward_direct_n6k3_d512_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: forward_direct_n6k3_d512_exec");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_run_forward_direct_n6k3_d512_typed<float>(
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_run_forward_direct_n6k3_d512_typed<c10::Half>(
            h,
            kc,
            mix,
            off
        );
    }

    return fdc_run_forward_direct_n6k3_d512_typed<c10::BFloat16>(
        h,
        kc,
        mix,
        off
    );
}
