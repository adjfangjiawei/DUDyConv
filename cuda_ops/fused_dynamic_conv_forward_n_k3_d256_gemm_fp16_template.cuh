#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"

// ======================================================================================
// Forward N{16,32,64,128} K3 D256 GEMM FP16 template
//
// Target:
//   B == 1
//   D == 256
//   K == 3
//   dilation == 1
//   dtype == float16
//   N == template parameter
//
// Input layout:
//   h:
//     [1,256,L], contiguous half
//
//   kc:
//     [1,3,N,T], contiguous half
//     kc[0,kk,n,t] = ((kk * N + n) * T + t)
//
//   mix:
//     [256,N], contiguous half
//
// Output:
//   out:
//     [1,256,T], contiguous half
//
// Math:
//   W_kk[D,T] = mix[D,N] @ kc_kk[N,T]
//
//   out[d,t] =
//       h[d, off+t]     * W_0[d,t] +
//       h[d, off+t - 1] * W_1[d,t] +
//       h[d, off+t - 2] * W_2[d,t]
//
// Implementation:
//   1. Use cublasGemmStridedBatchedEx to compute W[3,D,T].
//   2. Use one CUDA kernel to apply shifted h and reduce K=3.
//
// cuBLAS column-major trick for W:
//
//   Desired row-major:
//     W_kk[D,T] = mix[D,N] @ kc_kk[N,T]
//
//   Row-major [D,T] is viewed by cuBLAS as column-major [T,D].
//   Row-major kc_kk[N,T] viewed as column-major [T,N].
//   Row-major mix[D,N] viewed as column-major [N,D].
//
//   Therefore:
//     C_col[T,D] = A_col[T,N] @ B_col[N,D]
//
//   cublas:
//     opA = N
//     opB = N
//     m = T
//     n = D
//     k = N
//
//     A = kc, lda = T, strideA = N*T
//     B = mix, ldb = N, strideB = 0
//     C = W,  ldc = T, strideC = D*T
//
// ======================================================================================

namespace fdc_forward_n_k3_d256_gemm_fp16_detail {

constexpr int FDC_FWD_FP16_D = 256;
constexpr int FDC_FWD_FP16_K = 3;

static inline void fdc_fwd_fp16_check_cublas(
    cublasStatus_t status,
    const char* msg
) {
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        msg,
        " cublasStatus=",
        static_cast<int>(status)
    );
}

// ======================================================================================
// apply h shift and reduce K=3
//
// W:
//   [3,D,T], half
//
// h:
//   [D,L], half
//
// out:
//   [D,T], half
//
// out[d,t] = sum_kk h[d, off+t-kk] * W[kk,d,t]
//
// Accumulate in float, cast to half.
// ======================================================================================

template<int N>
__global__ void fdc_forward_n_k3_d256_gemm_fp16_apply_kernel(
    const half* __restrict__ h,
    const half* __restrict__ W,
    half* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_FP16_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    float acc = 0.0f;

    int src0 = off + t;
    int src1 = off + t - 1;
    int src2 = off + t - 2;

    if (src0 >= 0 && src0 < L) {
        half h_v = h[
            static_cast<int64_t>(d) * L + src0
        ];

        half w_v = W[
            static_cast<int64_t>(0) * FDC_FWD_FP16_D * T +
            static_cast<int64_t>(d) * T +
            t
        ];

        acc += __half2float(h_v) * __half2float(w_v);
    }

    if (src1 >= 0 && src1 < L) {
        half h_v = h[
            static_cast<int64_t>(d) * L + src1
        ];

        half w_v = W[
            static_cast<int64_t>(1) * FDC_FWD_FP16_D * T +
            static_cast<int64_t>(d) * T +
            t
        ];

        acc += __half2float(h_v) * __half2float(w_v);
    }

    if (src2 >= 0 && src2 < L) {
        half h_v = h[
            static_cast<int64_t>(d) * L + src2
        ];

        half w_v = W[
            static_cast<int64_t>(2) * FDC_FWD_FP16_D * T +
            static_cast<int64_t>(d) * T +
            t
        ];

        acc += __half2float(h_v) * __half2float(w_v);
    }

    out[
        static_cast<int64_t>(d) * T + t
    ] = __float2half(acc);

    (void)N;
}

// ======================================================================================
// availability
// ======================================================================================

template<int N>
static inline bool available(
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

    if (h.scalar_type() != at::ScalarType::Half ||
        kc.scalar_type() != at::ScalarType::Half ||
        mix.scalar_type() != at::ScalarType::Half) {
        return false;
    }

    if (h.size(0) != 1 || kc.size(0) != 1) {
        return false;
    }

    if (h.size(1) != FDC_FWD_FP16_D) {
        return false;
    }

    if (kc.size(1) != FDC_FWD_FP16_K) {
        return false;
    }

    if (kc.size(2) != N) {
        return false;
    }

    if (mix.size(0) != FDC_FWD_FP16_D || mix.size(1) != N) {
        return false;
    }

    if (dilation != 1) {
        return false;
    }

    if (off < 0) {
        return false;
    }

    if (off + kc.size(3) > h.size(2)) {
        return false;
    }

    // T 太小的时候，cuBLAS launch + 临时 W 可能不划算。
    // 先放宽到 1024，让 warmup 自动决定。
    if (kc.size(3) < 1024) {
        return false;
    }

    return true;
}

// ======================================================================================
// forward_cuda
// ======================================================================================

template<int N>
static inline torch::Tensor forward_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    const char* debug_name
) {
    FDC_DEBUG_PATH(debug_name);

    TORCH_CHECK(h.defined(), "h must be defined.");
    TORCH_CHECK(kc.defined(), "kc must be defined.");
    TORCH_CHECK(mix.defined(), "mix must be defined.");

    TORCH_CHECK(h.is_cuda(), "h must be CUDA tensor.");
    TORCH_CHECK(kc.is_cuda(), "kc must be CUDA tensor.");
    TORCH_CHECK(mix.is_cuda(), "mix must be CUDA tensor.");

    TORCH_CHECK(h.is_contiguous(), "h must be contiguous.");
    TORCH_CHECK(kc.is_contiguous(), "kc must be contiguous.");
    TORCH_CHECK(mix.is_contiguous(), "mix must be contiguous.");

    TORCH_CHECK(h.dim() == 3, "h must be [B,D,L].");
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,K,N,T].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(h.scalar_type() == at::ScalarType::Half, "h must be float16.");
    TORCH_CHECK(kc.scalar_type() == at::ScalarType::Half, "kc must be float16.");
    TORCH_CHECK(mix.scalar_type() == at::ScalarType::Half, "mix must be float16.");

    TORCH_CHECK(h.size(0) == 1, "requires B == 1.");
    TORCH_CHECK(kc.size(0) == 1, "requires B == 1.");

    TORCH_CHECK(h.size(1) == FDC_FWD_FP16_D, "requires D == 256.");
    TORCH_CHECK(kc.size(1) == FDC_FWD_FP16_K, "requires K == 3.");
    TORCH_CHECK(kc.size(2) == N, "N mismatch.");

    TORCH_CHECK(mix.size(0) == FDC_FWD_FP16_D, "mix D mismatch.");
    TORCH_CHECK(mix.size(1) == N, "mix N mismatch.");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(3));

    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + T <= L, "off + T must be <= L.");

    auto out = torch::empty(
        {
            1,
            FDC_FWD_FP16_D,
            T,
        },
        h.options()
    );

    auto W = torch::empty(
        {
            FDC_FWD_FP16_K,
            FDC_FWD_FP16_D,
            T,
        },
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_fwd_fp16_check_cublas(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream failed"
    );

    // 对 sm75 / GTX 1660 Ti，开启 Tensor Core math。
    // GemmEx 里也使用 CUBLAS_GEMM_DEFAULT_TENSOR_OP。
    fdc_fwd_fp16_check_cublas(
        cublasSetMathMode(
            handle,
            CUBLAS_TENSOR_OP_MATH
        ),
        "cublasSetMathMode failed"
    );

    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    // ==================================================================================
    // W_kk[D,T] = mix[D,N] @ kc_kk[N,T]
    //
    // Batched over kk=0,1,2.
    //
    // cuBLAS:
    //   C_col[T,D] = A_col[T,N] @ B_col[N,D]
    //
    //   A = kc_kk row-major [N,T], viewed col-major [T,N]
    //   B = mix   row-major [D,N], viewed col-major [N,D]
    //   C = W_kk  row-major [D,T], viewed col-major [T,D]
    //
    //   opA = N
    //   opB = N
    //   m = T
    //   n = D
    //   k = N
    //
    // ==================================================================================

    fdc_fwd_fp16_check_cublas(
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            T,
            FDC_FWD_FP16_D,
            N,
            &alpha,
            kc.data_ptr<at::Half>(),
            CUDA_R_16F,
            T,
            static_cast<long long>(N) * T,
            mix.data_ptr<at::Half>(),
            CUDA_R_16F,
            N,
            0,
            &beta0,
            W.data_ptr<at::Half>(),
            CUDA_R_16F,
            T,
            static_cast<long long>(FDC_FWD_FP16_D) * T,
            FDC_FWD_FP16_K,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "forward fp16 W cublasGemmStridedBatchedEx failed"
    );

    // ==================================================================================
    // out[d,t] = sum_kk h[d,off+t-kk] * W[kk,d,t]
    // ==================================================================================

    int total = FDC_FWD_FP16_D * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fdc_forward_n_k3_d256_gemm_fp16_apply_kernel<N><<<
        blocks,
        threads,
        0,
        stream
    >>>(
        reinterpret_cast<const half*>(h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

} // namespace fdc_forward_n_k3_d256_gemm_fp16_detail
