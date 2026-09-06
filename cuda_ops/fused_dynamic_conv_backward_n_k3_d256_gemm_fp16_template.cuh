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
// Backward N{16,32,64,128} K3 D256 GEMM FP16 template
//
// Target:
//   B == 1
//   D == 256
//   K == 3
//   dilation == 1
//   dtype == torch.float16
//   N == template parameter
//
// Input layout:
//   go:
//     [1,256,T], contiguous half
//
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
//   grad_h:
//     [1,256,L], contiguous half
//
//   grad_kc:
//     [1,3,N,T], contiguous half
//
//   grad_mix:
//     [256,N], contiguous half
//
// Forward math:
//   out[d,t] = sum_kk sum_n h[d,off+t-kk] * kc[kk,n,t] * mix[d,n]
//
// Backward optimized FP16 plan:
//
//   1. make G:
//        G_kk[d,t] = go[d,t] * h[d,off+t-kk]
//      G layout:
//        [3,D,T], half
//
//   2. grad_kernel:
//        grad_kc_kk[N,T] = mix^T[N,D] @ G_kk[D,T]
//      Implemented by one cublasGemmStridedBatchedEx.
//
//   3. grad_mix:
//        gm_tmp_kk[D,N] = G_kk[D,T] @ kc_kk^T[T,N]
//        grad_mix[D,N] = gm_tmp_0 + gm_tmp_1 + gm_tmp_2
//      Implemented by one cublasGemmStridedBatchedEx + one reduce kernel.
//
//   4. grad_h:
//        W_kk[D,T] = mix[D,N] @ kc_kk[N,T]
//      Implemented by one cublasGemmStridedBatchedEx.
//
//   5. grad_h gather:
//        For each d,src:
//          gh[d,src] = sum_kk valid(go[d,t] * W_kk[d,t])
//          where t = src - off + kk
//
//      This avoids half atomicAdd.
//
// cuBLAS settings:
//   input/output type:
//     CUDA_R_16F
//
//   compute:
//     CUBLAS_COMPUTE_32F
//
//   algorithm:
//     CUBLAS_GEMM_DEFAULT_TENSOR_OP
//
// Important compile/link note:
//   This .cuh is included by four .cu files.
//   All __global__ kernels are templated by N to avoid duplicate symbols.
// ======================================================================================

namespace fdc_backward_n_k3_d256_gemm_fp16_detail {

constexpr int FDC_BWD_FP16_D = 256;
constexpr int FDC_BWD_FP16_K = 3;

static inline void fdc_bwd_fp16_check_cublas(
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
// make G[3,D,T]
//
// G[kk,d,t] = go[d,t] * h[d, off+t-kk]
//
// Invalid boundary -> 0.
// Accumulate/multiply in float and cast to half.
// ======================================================================================

template<int N>
__global__ void fdc_bwd_n_k3_d256_gemm_fp16_make_g_kernel(
    const half* __restrict__ go,
    const half* __restrict__ h,
    half* __restrict__ G,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_BWD_FP16_K * FDC_BWD_FP16_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int tmp = idx / T;
    int d = tmp % FDC_BWD_FP16_D;
    int kk = tmp / FDC_BWD_FP16_D;

    int src = off + t - kk;

    float v = 0.0f;

    if (src >= 0 && src < L) {
        half go_v = go[
            static_cast<int64_t>(d) * T + t
        ];

        half h_v = h[
            static_cast<int64_t>(d) * L + src
        ];

        v = __half2float(go_v) * __half2float(h_v);
    }

    G[
        static_cast<int64_t>(kk) * FDC_BWD_FP16_D * T +
        static_cast<int64_t>(d) * T +
        t
    ] = __float2half(v);

    (void)N;
}

// ======================================================================================
// reduce gm_tmp[3,D,N] -> gm[D,N]
//
// gm_tmp and gm are half.
// Internally reduce in float and cast to half.
// ======================================================================================

template<int N>
__global__ void fdc_bwd_n_k3_d256_gemm_fp16_reduce_gm_kernel(
    const half* __restrict__ gm_tmp,
    half* __restrict__ gm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_BWD_FP16_D * N;

    if (idx >= total) {
        return;
    }

    int n = idx % N;
    int d = idx / N;

    int64_t base = static_cast<int64_t>(d) * N + n;

    half v0_h = gm_tmp[
        static_cast<int64_t>(0) * FDC_BWD_FP16_D * N + base
    ];

    half v1_h = gm_tmp[
        static_cast<int64_t>(1) * FDC_BWD_FP16_D * N + base
    ];

    half v2_h = gm_tmp[
        static_cast<int64_t>(2) * FDC_BWD_FP16_D * N + base
    ];

    float v =
        __half2float(v0_h) +
        __half2float(v1_h) +
        __half2float(v2_h);

    gm[base] = __float2half(v);
}

// ======================================================================================
// grad_h gather kernel, no atomic
//
// W:
//   [3,D,T], half
//
// go:
//   [D,T], half
//
// gh:
//   [D,L], half
//
// Only writes valid src range:
//   src_start = max(0, off - 2)
//   src_end   = min(L, off + T)
//
// For each src:
//   t0 = src - off + 0
//   t1 = src - off + 1
//   t2 = src - off + 2
//
//   gh[d,src] =
//       valid(t0) ? go[d,t0] * W[0,d,t0] : 0 +
//       valid(t1) ? go[d,t1] * W[1,d,t1] : 0 +
//       valid(t2) ? go[d,t2] * W[2,d,t2] : 0
//
// Accumulate in float and cast to half.
// ======================================================================================

template<int N>
__global__ void fdc_bwd_n_k3_d256_gemm_fp16_grad_h_gather_kernel(
    const half* __restrict__ go,
    const half* __restrict__ W,
    half* __restrict__ gh,
    int L,
    int T,
    int off,
    int src_start,
    int src_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_BWD_FP16_D * src_len;

    if (idx >= total) {
        return;
    }

    int local_src = idx % src_len;
    int d = idx / src_len;
    int src = src_start + local_src;

    if (src < 0 || src >= L) {
        return;
    }

    float acc = 0.0f;

    int t0 = src - off + 0;
    int t1 = src - off + 1;
    int t2 = src - off + 2;

    if (t0 >= 0 && t0 < T) {
        half go_v = go[
            static_cast<int64_t>(d) * T + t0
        ];

        half w_v = W[
            static_cast<int64_t>(0) * FDC_BWD_FP16_D * T +
            static_cast<int64_t>(d) * T +
            t0
        ];

        acc += __half2float(go_v) * __half2float(w_v);
    }

    if (t1 >= 0 && t1 < T) {
        half go_v = go[
            static_cast<int64_t>(d) * T + t1
        ];

        half w_v = W[
            static_cast<int64_t>(1) * FDC_BWD_FP16_D * T +
            static_cast<int64_t>(d) * T +
            t1
        ];

        acc += __half2float(go_v) * __half2float(w_v);
    }

    if (t2 >= 0 && t2 < T) {
        half go_v = go[
            static_cast<int64_t>(d) * T + t2
        ];

        half w_v = W[
            static_cast<int64_t>(2) * FDC_BWD_FP16_D * T +
            static_cast<int64_t>(d) * T +
            t2
        ];

        acc += __half2float(go_v) * __half2float(w_v);
    }

    gh[
        static_cast<int64_t>(d) * L + src
    ] = __float2half(acc);

    (void)N;
}

// ======================================================================================
// availability
// ======================================================================================

template<int N>
static inline bool available(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    if (!go.defined() || !h.defined() || !kc.defined() || !mix.defined()) {
        return false;
    }

    if (!go.is_cuda() || !h.is_cuda() || !kc.is_cuda() || !mix.is_cuda()) {
        return false;
    }

    if (!go.is_contiguous() || !h.is_contiguous() || !kc.is_contiguous() || !mix.is_contiguous()) {
        return false;
    }

    if (go.dim() != 3 || h.dim() != 3 || kc.dim() != 4 || mix.dim() != 2) {
        return false;
    }

    if (go.scalar_type() != at::ScalarType::Half ||
        h.scalar_type() != at::ScalarType::Half ||
        kc.scalar_type() != at::ScalarType::Half ||
        mix.scalar_type() != at::ScalarType::Half) {
        return false;
    }

    if (go.size(0) != 1 || h.size(0) != 1 || kc.size(0) != 1) {
        return false;
    }

    if (go.size(1) != FDC_BWD_FP16_D || h.size(1) != FDC_BWD_FP16_D) {
        return false;
    }

    if (kc.size(1) != FDC_BWD_FP16_K) {
        return false;
    }

    if (kc.size(2) != N) {
        return false;
    }

    if (mix.size(0) != FDC_BWD_FP16_D || mix.size(1) != N) {
        return false;
    }

    if (go.size(2) != kc.size(3)) {
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

    // T 太小的时候，cuBLAS launch + 临时 G/W/gm_tmp 不划算。
    // 先设置为 >=1024，让 warmup 在候选之间继续实测选择。
    if (kc.size(3) < 1024) {
        return false;
    }

    return true;
}

// ======================================================================================
// backward_cuda
// ======================================================================================

template<int N>
static inline std::vector<torch::Tensor> backward_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    const char* debug_name
) {
    FDC_DEBUG_PATH(debug_name);

    TORCH_CHECK(go.defined(), "go must be defined.");
    TORCH_CHECK(h.defined(), "h must be defined.");
    TORCH_CHECK(kc.defined(), "kc must be defined.");
    TORCH_CHECK(mix.defined(), "mix must be defined.");

    TORCH_CHECK(go.is_cuda(), "go must be CUDA tensor.");
    TORCH_CHECK(h.is_cuda(), "h must be CUDA tensor.");
    TORCH_CHECK(kc.is_cuda(), "kc must be CUDA tensor.");
    TORCH_CHECK(mix.is_cuda(), "mix must be CUDA tensor.");

    TORCH_CHECK(go.is_contiguous(), "go must be contiguous.");
    TORCH_CHECK(h.is_contiguous(), "h must be contiguous.");
    TORCH_CHECK(kc.is_contiguous(), "kc must be contiguous.");
    TORCH_CHECK(mix.is_contiguous(), "mix must be contiguous.");

    TORCH_CHECK(go.dim() == 3, "go must be [B,D,T].");
    TORCH_CHECK(h.dim() == 3, "h must be [B,D,L].");
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,K,N,T].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(go.scalar_type() == at::ScalarType::Half, "go must be float16.");
    TORCH_CHECK(h.scalar_type() == at::ScalarType::Half, "h must be float16.");
    TORCH_CHECK(kc.scalar_type() == at::ScalarType::Half, "kc must be float16.");
    TORCH_CHECK(mix.scalar_type() == at::ScalarType::Half, "mix must be float16.");

    TORCH_CHECK(go.size(0) == 1, "requires B == 1.");
    TORCH_CHECK(h.size(0) == 1, "requires B == 1.");
    TORCH_CHECK(kc.size(0) == 1, "requires B == 1.");

    TORCH_CHECK(go.size(1) == FDC_BWD_FP16_D, "requires D == 256.");
    TORCH_CHECK(h.size(1) == FDC_BWD_FP16_D, "requires D == 256.");

    TORCH_CHECK(kc.size(1) == FDC_BWD_FP16_K, "requires K == 3.");
    TORCH_CHECK(kc.size(2) == N, "N mismatch.");

    TORCH_CHECK(mix.size(0) == FDC_BWD_FP16_D, "mix D mismatch.");
    TORCH_CHECK(mix.size(1) == N, "mix N mismatch.");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(go.size(2));

    TORCH_CHECK(kc.size(3) == T, "kc T must equal grad_out T.");
    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + T <= L, "off + T must be <= L.");

    auto gh = torch::zeros_like(h);
    auto gk = torch::empty_like(kc);
    auto gm = torch::empty_like(mix);

    auto G = torch::empty(
        {
            FDC_BWD_FP16_K,
            FDC_BWD_FP16_D,
            T,
        },
        go.options()
    );

    auto gm_tmp = torch::empty(
        {
            FDC_BWD_FP16_K,
            FDC_BWD_FP16_D,
            N,
        },
        go.options()
    );

    auto W = torch::empty(
        {
            FDC_BWD_FP16_K,
            FDC_BWD_FP16_D,
            T,
        },
        go.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;

    // ==================================================================================
    // 1. make G[3,D,T]
    // ==================================================================================

    int total_G = FDC_BWD_FP16_K * FDC_BWD_FP16_D * T;
    int blocks_G = (total_G + threads - 1) / threads;

    fdc_bwd_n_k3_d256_gemm_fp16_make_g_kernel<N><<<
        blocks_G,
        threads,
        0,
        stream
    >>>(
        reinterpret_cast<const half*>(go.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(h.data_ptr<at::Half>()),
        reinterpret_cast<half*>(G.data_ptr<at::Half>()),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_bwd_fp16_check_cublas(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream failed"
    );

    fdc_bwd_fp16_check_cublas(
        cublasSetMathMode(
            handle,
            CUBLAS_TENSOR_OP_MATH
        ),
        "cublasSetMathMode failed"
    );

    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    // ==================================================================================
    // 2. grad_kernel batched GemmEx
    //
    // Desired:
    //   gk_kk[N,T] = mix^T[N,D] @ G_kk[D,T]
    //
    // Row-major to cuBLAS column-major trick:
    //   C_col[T,N] = A_col[T,D] @ B_col[D,N]
    //
    //   A = G_kk row-major [D,T], viewed col-major [T,D]
    //       opA=N, lda=T
    //
    //   B = mix row-major [D,N], viewed col-major [N,D]
    //       opB=T, gives [D,N], ldb=N
    //
    //   C = gk_kk row-major [N,T], viewed col-major [T,N]
    //       ldc=T
    //
    // cublas:
    //   m = T
    //   n = N
    //   k = D
    //   batchCount = 3
    // ==================================================================================

    fdc_bwd_fp16_check_cublas(
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            T,
            N,
            FDC_BWD_FP16_D,
            &alpha,
            G.data_ptr<at::Half>(),
            CUDA_R_16F,
            T,
            static_cast<long long>(FDC_BWD_FP16_D) * T,
            mix.data_ptr<at::Half>(),
            CUDA_R_16F,
            N,
            0,
            &beta0,
            gk.data_ptr<at::Half>(),
            CUDA_R_16F,
            T,
            static_cast<long long>(N) * T,
            FDC_BWD_FP16_K,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "backward fp16 grad_kernel cublasGemmStridedBatchedEx failed"
    );

    // ==================================================================================
    // 3. grad_mix batched GemmEx into gm_tmp[3,D,N]
    //
    // Desired:
    //   gm_tmp_kk[D,N] = G_kk[D,T] @ kc_kk^T[T,N]
    //
    // Row-major to cuBLAS column-major trick:
    //   C_col[N,D] = A_col[N,T] @ B_col[T,D]
    //
    //   A = kc_kk row-major [N,T], viewed col-major [T,N]
    //       opA=T, lda=T
    //
    //   B = G_kk row-major [D,T], viewed col-major [T,D]
    //       opB=N, ldb=T
    //
    //   C = gm_tmp_kk row-major [D,N], viewed col-major [N,D]
    //       ldc=N
    //
    // cublas:
    //   m = N
    //   n = D
    //   k = T
    // ==================================================================================

    fdc_bwd_fp16_check_cublas(
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N,
            FDC_BWD_FP16_D,
            T,
            &alpha,
            kc.data_ptr<at::Half>(),
            CUDA_R_16F,
            T,
            static_cast<long long>(N) * T,
            G.data_ptr<at::Half>(),
            CUDA_R_16F,
            T,
            static_cast<long long>(FDC_BWD_FP16_D) * T,
            &beta0,
            gm_tmp.data_ptr<at::Half>(),
            CUDA_R_16F,
            N,
            static_cast<long long>(FDC_BWD_FP16_D) * N,
            FDC_BWD_FP16_K,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "backward fp16 grad_mix cublasGemmStridedBatchedEx failed"
    );

    // ==================================================================================
    // 4. reduce gm_tmp[3,D,N] -> gm[D,N]
    // ==================================================================================

    int total_gm = FDC_BWD_FP16_D * N;
    int blocks_gm = (total_gm + threads - 1) / threads;

    fdc_bwd_n_k3_d256_gemm_fp16_reduce_gm_kernel<N><<<
        blocks_gm,
        threads,
        0,
        stream
    >>>(
        reinterpret_cast<const half*>(gm_tmp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(gm.data_ptr<at::Half>())
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ==================================================================================
    // 5. W batched GemmEx for grad_h
    //
    // Desired:
    //   W_kk[D,T] = mix[D,N] @ kc_kk[N,T]
    //
    // Row-major to cuBLAS column-major trick:
    //   C_col[T,D] = A_col[T,N] @ B_col[N,D]
    //
    //   A = kc_kk row-major [N,T], viewed col-major [T,N]
    //       opA=N, lda=T
    //
    //   B = mix row-major [D,N], viewed col-major [N,D]
    //       opB=N, ldb=N
    //
    //   C = W_kk row-major [D,T], viewed col-major [T,D]
    //       ldc=T
    //
    // cublas:
    //   m = T
    //   n = D
    //   k = N
    // ==================================================================================

    fdc_bwd_fp16_check_cublas(
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            T,
            FDC_BWD_FP16_D,
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
            static_cast<long long>(FDC_BWD_FP16_D) * T,
            FDC_BWD_FP16_K,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "backward fp16 grad_h W cublasGemmStridedBatchedEx failed"
    );

    // ==================================================================================
    // 6. grad_h gather, no atomic
    //
    // Valid src range:
    //   kk=0 contributes src=off+t
    //   kk=1 contributes src=off+t-1
    //   kk=2 contributes src=off+t-2
    //
    // Therefore:
    //   src in [off-2, off+T-1]
    // ==================================================================================

    int src_start = static_cast<int>(off) - 2;
    if (src_start < 0) {
        src_start = 0;
    }

    int src_end = static_cast<int>(off) + T;
    if (src_end > L) {
        src_end = L;
    }

    int src_len = src_end - src_start;

    if (src_len > 0) {
        int total_h = FDC_BWD_FP16_D * src_len;
        int blocks_h = (total_h + threads - 1) / threads;

        fdc_bwd_n_k3_d256_gemm_fp16_grad_h_gather_kernel<N><<<
            blocks_h,
            threads,
            0,
            stream
        >>>(
            reinterpret_cast<const half*>(go.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
            reinterpret_cast<half*>(gh.data_ptr<at::Half>()),
            L,
            T,
            static_cast<int>(off),
            src_start,
            src_len
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        gh,
        gk,
        gm,
    };
}

} // namespace fdc_backward_n_k3_d256_gemm_fp16_detail
