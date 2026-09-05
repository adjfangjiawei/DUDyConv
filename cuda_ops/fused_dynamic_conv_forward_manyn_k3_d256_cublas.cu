#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Forward many-N K3 D256 cuBLAS sliced plan
//
// Computes:
//
//   out[d,t] = sum_kk h[d, off + t - kk] * sum_n mix[d,n] * kc[t,n,kk]
//
// K-slice formulation:
//
//   for kk in 0..2:
//      kc_nt[kk] = [N,T]
//      w_kk      = mix[D,N] @ kc_nt[N,T] -> [D,T]
//      out      += w_kk * shifted_h
//
// This avoids generic direct loop for many-N and avoids flatten/scatter.
//
// Notes:
//   - This file intentionally does NOT define fdc_check_cublas because your
//     fused_dynamic_conv_common.cuh already defines it.
//   - This plan is a candidate only; warmup will choose it only if faster.
// ======================================================================================

// ======================================================================================
// Kernels
// ======================================================================================

__global__ void fdc_forward_manyn_zero_float_kernel(
    float* __restrict__ out,
    int64_t total
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < total) {
        out[i] = 0.0f;
    }
}

__global__ void fdc_forward_manyn_build_kcnt_float_kernel(
    const float* __restrict__ kc,
    float* __restrict__ kc_nt,
    int T,
    int N,
    int kk
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(N) * T;

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(idx % T);
    int n = static_cast<int>(idx / T);

    kc_nt[static_cast<int64_t>(n) * T + t] =
        kc[(static_cast<int64_t>(t) * N + n) * 3 + kk];
}

template <typename scalar_t>
__global__ void fdc_forward_manyn_build_kcnt_typed_kernel(
    const scalar_t* __restrict__ kc,
    float* __restrict__ kc_nt,
    int T,
    int N,
    int kk
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(N) * T;

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(idx % T);
    int n = static_cast<int>(idx / T);

    kc_nt[static_cast<int64_t>(n) * T + t] =
        fdc_to_float_dev(
            kc[(static_cast<int64_t>(t) * N + n) * 3 + kk]
        );
}

__global__ void fdc_forward_manyn_accum_out_float_kernel(
    const float* __restrict__ h,
    const float* __restrict__ w,
    float* __restrict__ out,
    int L,
    int T,
    int off,
    int kk
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(256) * T;

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(idx % T);
    int d = static_cast<int>(idx / T);

    int s = off + t - kk;

    if (s >= 0 && s < L) {
        out[static_cast<int64_t>(d) * T + t] +=
            w[static_cast<int64_t>(d) * T + t] *
            h[static_cast<int64_t>(d) * L + s];
    }
}

template <typename scalar_t>
__global__ void fdc_forward_manyn_accum_out_typed_kernel(
    const scalar_t* __restrict__ h,
    const float* __restrict__ w,
    float* __restrict__ out,
    int L,
    int T,
    int off,
    int kk
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(256) * T;

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(idx % T);
    int d = static_cast<int>(idx / T);

    int s = off + t - kk;

    if (s >= 0 && s < L) {
        out[static_cast<int64_t>(d) * T + t] +=
            w[static_cast<int64_t>(d) * T + t] *
            fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);
    }
}

template <typename scalar_t>
__global__ void fdc_forward_manyn_cast_out_kernel(
    const float* __restrict__ src,
    scalar_t* __restrict__ dst,
    int64_t total
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < total) {
        dst[i] = fdc_from_float_dev<scalar_t>(src[i]);
    }
}

// ======================================================================================
// Row-major GEMM helper
//
// C[M,N] = A[M,K] @ B[K,N]
//
// cuBLAS is column-major. For row-major matrices:
//   C_row[M,N] = A_row[M,K] @ B_row[K,N]
//
// use:
//   cublasSgemm(
//       handle,
//       CUBLAS_OP_N,
//       CUBLAS_OP_N,
//       N,
//       M,
//       K,
//       B,
//       N,
//       A,
//       K,
//       C,
//       N
//   )
// ======================================================================================

static void fdc_forward_manyn_sgemm_rowmajor(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    fdc_check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            B,
            N,
            A,
            K,
            &beta,
            C,
            N
        ),
        "cublasSgemm forward_manyn rowmajor failed."
    );
}

// ======================================================================================
// fp32 implementation
// ======================================================================================

static torch::Tensor fdc_forward_manyn_k3_d256_cublas_float(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: forward_manyn_k3_d256_cublas_sliced_float");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));

    auto fopts = h.options().dtype(torch::kFloat32);

    auto out = torch::empty(
        {
            1,
            256,
            T,
        },
        fopts
    );

    auto out2d = out.view(
        {
            256,
            T,
        }
    );

    auto kc_nt = torch::empty(
        {
            N,
            T,
        },
        fopts
    );

    auto w = torch::empty(
        {
            256,
            T,
        },
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fdc_forward_manyn_zero_float_kernel<<<
        (static_cast<int64_t>(256) * T + 255) / 256,
        256,
        0,
        stream
    >>>(
        out2d.data_ptr<float>(),
        static_cast<int64_t>(256) * T
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_check_cublas(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream forward_manyn failed."
    );

    for (int kk = 0; kk < 3; ++kk) {
        fdc_forward_manyn_build_kcnt_float_kernel<<<
            (static_cast<int64_t>(N) * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            kc.data_ptr<float>(),
            kc_nt.data_ptr<float>(),
            T,
            N,
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // w[256,T] = mix[256,N] @ kc_nt[N,T]
        fdc_forward_manyn_sgemm_rowmajor(
            handle,
            mix.data_ptr<float>(),
            kc_nt.data_ptr<float>(),
            w.data_ptr<float>(),
            256,
            T,
            N
        );

        fdc_forward_manyn_accum_out_float_kernel<<<
            (static_cast<int64_t>(256) * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<float>(),
            w.data_ptr<float>(),
            out2d.data_ptr<float>(),
            L,
            T,
            static_cast<int>(off),
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}

// ======================================================================================
// fp16/bf16 implementation
//
// Computes in fp32 temporary tensors and casts output back.
// ======================================================================================

template <typename scalar_t>
static torch::Tensor fdc_forward_manyn_k3_d256_cublas_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: forward_manyn_k3_d256_cublas_sliced_typed");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));

    auto fopts = h.options().dtype(torch::kFloat32);

    auto mixf = mix.to(torch::kFloat32);

    auto outf = torch::empty(
        {
            1,
            256,
            T,
        },
        fopts
    );

    auto out2d = outf.view(
        {
            256,
            T,
        }
    );

    auto kc_nt = torch::empty(
        {
            N,
            T,
        },
        fopts
    );

    auto w = torch::empty(
        {
            256,
            T,
        },
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fdc_forward_manyn_zero_float_kernel<<<
        (static_cast<int64_t>(256) * T + 255) / 256,
        256,
        0,
        stream
    >>>(
        out2d.data_ptr<float>(),
        static_cast<int64_t>(256) * T
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_check_cublas(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream forward_manyn typed failed."
    );

    for (int kk = 0; kk < 3; ++kk) {
        fdc_forward_manyn_build_kcnt_typed_kernel<scalar_t><<<
            (static_cast<int64_t>(N) * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            kc.data_ptr<scalar_t>(),
            kc_nt.data_ptr<float>(),
            T,
            N,
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // w[256,T] = mixf[256,N] @ kc_nt[N,T]
        fdc_forward_manyn_sgemm_rowmajor(
            handle,
            mixf.data_ptr<float>(),
            kc_nt.data_ptr<float>(),
            w.data_ptr<float>(),
            256,
            T,
            N
        );

        fdc_forward_manyn_accum_out_typed_kernel<scalar_t><<<
            (static_cast<int64_t>(256) * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            h.data_ptr<scalar_t>(),
            w.data_ptr<float>(),
            out2d.data_ptr<float>(),
            L,
            T,
            static_cast<int>(off),
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto out = torch::empty(
        {
            1,
            256,
            T,
        },
        h.options()
    );

    fdc_forward_manyn_cast_out_kernel<scalar_t><<<
        (static_cast<int64_t>(256) * T + 255) / 256,
        256,
        0,
        stream
    >>>(
        outf.data_ptr<float>(),
        out.data_ptr<scalar_t>(),
        static_cast<int64_t>(256) * T
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ======================================================================================
// availability
// ======================================================================================

bool fdc_forward_manyn_k3_d256_cublas_available_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    if (!h.defined() || !kc.defined() || !mix.defined()) {
        return false;
    }

    if (!h.is_cuda() || !kc.is_cuda() || !mix.is_cuda()) {
        return false;
    }

    if (!h.is_contiguous() || !kc.is_contiguous() || !mix.is_contiguous()) {
        return false;
    }

    if (h.dim() != 3 || kc.dim() != 4 || mix.dim() != 2) {
        return false;
    }

    if (h.scalar_type() != kc.scalar_type()) {
        return false;
    }

    if (h.scalar_type() != mix.scalar_type()) {
        return false;
    }

    if (
        h.scalar_type() != at::ScalarType::Float &&
        h.scalar_type() != at::ScalarType::Half &&
        h.scalar_type() != at::ScalarType::BFloat16
    ) {
        return false;
    }

    if (h.size(0) != 1 || kc.size(0) != 1) {
        return false;
    }

    if (h.size(1) != 256 || mix.size(0) != 256) {
        return false;
    }

    if (kc.size(3) != 3) {
        return false;
    }

    if (mix.size(1) != kc.size(2)) {
        return false;
    }

    if (dilation != 1) {
        return false;
    }

    if (kc.size(2) < 8 || kc.size(2) > 256) {
        return false;
    }

    if (kc.size(1) < 1024) {
        return false;
    }

    if (off < 0) {
        return false;
    }

    if (off + kc.size(1) > h.size(2)) {
        return false;
    }

    return true;
}

// ======================================================================================
// public
// ======================================================================================

torch::Tensor fdc_forward_manyn_k3_d256_cublas_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    TORCH_CHECK(
        fdc_forward_manyn_k3_d256_cublas_available_cuda(
            h,
            kc,
            mix,
            off,
            dilation
        ),
        "forward_manyn_k3_d256_cublas is not available for this shape."
    );

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_forward_manyn_k3_d256_cublas_float(
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_forward_manyn_k3_d256_cublas_typed<c10::Half>(
            h,
            kc,
            mix,
            off
        );
    }

    TORCH_CHECK(
        h.scalar_type() == at::ScalarType::BFloat16,
        "forward_manyn_k3_d256_cublas supports float32, float16, bfloat16."
    );

    return fdc_forward_manyn_k3_d256_cublas_typed<c10::BFloat16>(
        h,
        kc,
        mix,
        off
    );
}
