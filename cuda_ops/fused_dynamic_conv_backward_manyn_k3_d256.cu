#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// many-N K3 D256 backward GEMM plan
//
// Target:
//   B=1
//   D=256
//   K=3
//   dilation=1
//   N in [16,128]
//   T >= 2048
//
// Main idea:
//
// Let:
//
//   kc_flat[q, n] where q = t * 3 + kk
//   x[d, q] = go[d,t] * h[d, off + t - kk]
//
// Then:
//
//   grad_kernel_flat[q, n] = sum_d x[d,q] * mix[d,n]
//                          = x^T @ mix
//
//   grad_mix[d, n] = sum_q x[d,q] * kc_flat[q,n]
//                  = x @ kc_flat
//
// For grad_h:
//
//   w[d, q] = sum_n mix[d,n] * kc_flat[q,n]
//           = mix @ kc_flat^T
//
//   gh[d,s] = sum_kk go[d,t] * w[d, t*3+kk]
//             where t = s - off + kk
//
// This makes the expensive many-N path use optimized GEMM instead of slow custom reductions.
// ======================================================================================

// ======================================================================================
// cast helpers
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_manyn_cast_float_to_scalar_kernel(
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
static torch::Tensor fdc_manyn_cast_float_tensor_to(
    torch::Tensor src,
    torch::Tensor like
) {
    auto dst = torch::empty_like(like);

    int64_t n = src.numel();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fdc_manyn_cast_float_to_scalar_kernel<scalar_t><<<
        (n + 255) / 256,
        256,
        0,
        stream
    >>>(
        src.data_ptr<float>(),
        dst.data_ptr<scalar_t>(),
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return dst;
}

// ======================================================================================
// build kc_flat:
//
// input:
//   kc: [1,T,N,3], contiguous
//
// output:
//   kc_flat: [T*3,N]
//   kc_flat[t*3+kk, n] = kc[0,t,n,kk]
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_manyn_build_kc_flat_kernel(
    const scalar_t* __restrict__ kc,
    float* __restrict__ kc_flat,
    int T,
    int N
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(T) * 3 * N;

    if (idx >= total) {
        return;
    }

    int n = static_cast<int>(idx % N);
    int q = static_cast<int>(idx / N);
    int kk = q % 3;
    int t = q / 3;

    kc_flat[static_cast<int64_t>(q) * N + n] =
        fdc_to_float_dev(kc[(static_cast<int64_t>(t) * N + n) * 3 + kk]);
}

__global__ void fdc_manyn_build_kc_flat_float_kernel(
    const float* __restrict__ kc,
    float* __restrict__ kc_flat,
    int T,
    int N
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(T) * 3 * N;

    if (idx >= total) {
        return;
    }

    int n = static_cast<int>(idx % N);
    int q = static_cast<int>(idx / N);
    int kk = q % 3;
    int t = q / 3;

    kc_flat[static_cast<int64_t>(q) * N + n] =
        kc[(static_cast<int64_t>(t) * N + n) * 3 + kk];
}

// ======================================================================================
// build x:
//
// x[d, q] = go[d,t] * h[d, off + t - kk]
// q = t*3 + kk
//
// shape:
//   x: [256, T*3]
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_manyn_build_x_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    float* __restrict__ x,
    int L,
    int T,
    int off
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(256) * T * 3;

    if (idx >= total) {
        return;
    }

    int q = static_cast<int>(idx % (T * 3));
    int d = static_cast<int>(idx / (T * 3));

    int kk = q % 3;
    int t = q / 3;
    int s = off + t - kk;

    float v = 0.0f;

    if (s >= 0 && s < L) {
        v =
            fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) *
            fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);
    }

    x[static_cast<int64_t>(d) * T * 3 + q] = v;
}

__global__ void fdc_manyn_build_x_float_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    float* __restrict__ x,
    int L,
    int T,
    int off
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(256) * T * 3;

    if (idx >= total) {
        return;
    }

    int q = static_cast<int>(idx % (T * 3));
    int d = static_cast<int>(idx / (T * 3));

    int kk = q % 3;
    int t = q / 3;
    int s = off + t - kk;

    float v = 0.0f;

    if (s >= 0 && s < L) {
        v =
            go[static_cast<int64_t>(d) * T + t] *
            h[static_cast<int64_t>(d) * L + s];
    }

    x[static_cast<int64_t>(d) * T * 3 + q] = v;
}

// ======================================================================================
// build grad_h from w:
//
// w[d, q] = mix @ kc_flat^T
// q = t*3 + kk
//
// gh[d,s] = sum_kk go[d,t] * w[d,t*3+kk]
// t = s - off + kk
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_manyn_build_gh_kernel(
    const scalar_t* __restrict__ go,
    const float* __restrict__ w,
    float* __restrict__ gh,
    int L,
    int T,
    int off
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(256) * L;

    if (idx >= total) {
        return;
    }

    int s = static_cast<int>(idx % L);
    int d = static_cast<int>(idx / L);

    int tbase = s - off;

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = tbase + kk;

        if (t >= 0 && t < T) {
            int q = t * 3 + kk;

            acc +=
                fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) *
                w[static_cast<int64_t>(d) * T * 3 + q];
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc;
}

__global__ void fdc_manyn_build_gh_float_kernel(
    const float* __restrict__ go,
    const float* __restrict__ w,
    float* __restrict__ gh,
    int L,
    int T,
    int off
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(256) * L;

    if (idx >= total) {
        return;
    }

    int s = static_cast<int>(idx % L);
    int d = static_cast<int>(idx / L);

    int tbase = s - off;

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = tbase + kk;

        if (t >= 0 && t < T) {
            int q = t * 3 + kk;

            acc +=
                go[static_cast<int64_t>(d) * T + t] *
                w[static_cast<int64_t>(d) * T * 3 + q];
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc;
}

// ======================================================================================
// scatter gk_flat:
//
// gk_flat: [T*3,N]
// gk:      [1,T,N,3]
//
// gk[0,t,n,kk] = gk_flat[t*3+kk,n]
// ======================================================================================

__global__ void fdc_manyn_scatter_gk_float_kernel(
    const float* __restrict__ gk_flat,
    float* __restrict__ gk,
    int T,
    int N
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(T) * N * 3;

    if (idx >= total) {
        return;
    }

    int kk = static_cast<int>(idx % 3);
    int n = static_cast<int>((idx / 3) % N);
    int t = static_cast<int>(idx / (N * 3));

    gk[(static_cast<int64_t>(t) * N + n) * 3 + kk] =
        gk_flat[static_cast<int64_t>(t * 3 + kk) * N + n];
}

// ======================================================================================
// fp32 GEMM implementation
// ======================================================================================

static std::vector<torch::Tensor> fdc_backward_manyn_k3_d256_float(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_manyn_k3_d256_gemm_float_v2");

    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));

    TORCH_CHECK(D == 256, "manyn_k3_d256_gemm requires D == 256.");

    auto fopts = h.options().dtype(torch::kFloat32);

    auto gh = torch::empty(
        h.sizes(),
        fopts
    );

    auto gk = torch::empty(
        kc.sizes(),
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto kc_flat = torch::empty(
        {
            T * 3,
            N,
        },
        fopts
    );

    auto x = torch::empty(
        {
            256,
            T * 3,
        },
        fopts
    );

    fdc_manyn_build_kc_flat_float_kernel<<<
        (static_cast<int64_t>(T) * 3 * N + 255) / 256,
        256,
        0,
        stream
    >>>(
        kc.data_ptr<float>(),
        kc_flat.data_ptr<float>(),
        T,
        N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_manyn_build_x_float_kernel<<<
        (static_cast<int64_t>(256) * T * 3 + 255) / 256,
        256,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        h.data_ptr<float>(),
        x.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    /*
     * w = mix @ kc_flat.T
     *
     * mix:     [256, N]
     * kc_flat: [T*3, N]
     * w:       [256, T*3]
     */
    auto w = at::matmul(
        mix,
        kc_flat.transpose(0, 1)
    );

    /*
     * gh from w
     */
    fdc_manyn_build_gh_float_kernel<<<
        (static_cast<int64_t>(256) * L + 255) / 256,
        256,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        w.data_ptr<float>(),
        gh.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    /*
     * gk_flat = x.T @ mix
     *
     * x.T: [T*3,256]
     * mix: [256,N]
     * gk_flat: [T*3,N]
     */
    auto gk_flat = at::matmul(
        x.transpose(0, 1),
        mix
    );

    fdc_manyn_scatter_gk_float_kernel<<<
        (static_cast<int64_t>(T) * N * 3 + 255) / 256,
        256,
        0,
        stream
    >>>(
        gk_flat.data_ptr<float>(),
        gk.data_ptr<float>(),
        T,
        N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    /*
     * gm = x @ kc_flat
     *
     * x:       [256,T*3]
     * kc_flat: [T*3,N]
     * gm:      [256,N]
     */
    auto gm = at::matmul(
        x,
        kc_flat
    );

    return {
        gh,
        gk,
        gm
    };
}

// ======================================================================================
// typed implementation
//
// For fp16/bf16, this plan computes in fp32 using casted temporary tensors.
// It is primarily introduced for fp32 N16/N64/N128 performance, but remains correct for
// half/bfloat16. Autotune can still choose large if this path is slower.
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> fdc_backward_manyn_k3_d256_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_manyn_k3_d256_gemm_typed_v2");

    auto gof = go.to(torch::kFloat32);
    auto hf = h.to(torch::kFloat32);
    auto kcf = kc.to(torch::kFloat32);
    auto mixf = mix.to(torch::kFloat32);

    auto outs_f = fdc_backward_manyn_k3_d256_float(
        gof,
        hf,
        kcf,
        mixf,
        off
    );

    auto gh = fdc_manyn_cast_float_tensor_to<scalar_t>(
        outs_f[0],
        h
    );

    auto gk = fdc_manyn_cast_float_tensor_to<scalar_t>(
        outs_f[1],
        kc
    );

    auto gm = fdc_manyn_cast_float_tensor_to<scalar_t>(
        outs_f[2],
        mix
    );

    return {
        gh,
        gk,
        gm
    };
}

// ======================================================================================
// public
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_manyn_k3_d256_cuda(
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

    TORCH_CHECK(h.size(0) == 1, "manyn_k3_d256 requires B == 1.");
    TORCH_CHECK(h.size(1) == 256, "manyn_k3_d256 requires D == 256.");
    TORCH_CHECK(kc.size(3) == 3, "manyn_k3_d256 requires K == 3.");
    TORCH_CHECK(kc.size(2) >= 16, "manyn_k3_d256 requires N >= 16.");
    TORCH_CHECK(kc.size(2) <= 128, "manyn_k3_d256 requires N <= 128.");

    TORCH_CHECK(go.size(0) == 1, "go B mismatch.");
    TORCH_CHECK(go.size(1) == 256, "go D mismatch.");
    TORCH_CHECK(go.size(2) == kc.size(1), "go T mismatch.");

    TORCH_CHECK(mix.size(0) == 256, "mix D mismatch.");
    TORCH_CHECK(mix.size(1) == kc.size(2), "mix N mismatch.");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_backward_manyn_k3_d256_float(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_backward_manyn_k3_d256_typed<c10::Half>(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    return fdc_backward_manyn_k3_d256_typed<c10::BFloat16>(
        go,
        h,
        kc,
        mix,
        off
    );
}
