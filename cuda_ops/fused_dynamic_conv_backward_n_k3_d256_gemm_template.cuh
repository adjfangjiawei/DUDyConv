#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"

// ======================================================================================
// Backward N{16,32,64,128} K3 D256 GEMM template
//
// Target:
//   B == 1
//   D == 256
//   K == 3
//   dilation == 1
//   dtype == float32
//   N == template parameter
//
// Layout:
//   grad_out:
//     [1,256,T]
//
//   h:
//     [1,256,L]
//
//   kc:
//     [1,3,N,T]
//
//   mix:
//     [256,N]
//
// Outputs:
//   grad_h:
//     [1,256,L]
//
//   grad_kc:
//     [1,3,N,T]
//
//   grad_mix:
//     [256,N]
//
// Kernel layout:
//   kc[0,kk,n,t] = ((kk * N + n) * T + t)
//
// Semantics:
//   kk=0 current token
//   kk=1 previous token
//   kk=2 token before previous
//
// Forward:
//   out[d,t] = sum_kk sum_n h[d, off+t-kk] * kc[kk,n,t] * mix[d,n]
//
// Optimized backward plan:
//
//   1. make G:
//        G_kk[d,t] = grad_out[d,t] * h[d, off+t-kk]
//
//      G:
//        [3,D,T]
//
//   2. grad_kernel with one strided batched SGEMM:
//
//        grad_kc_kk[N,T] = mix^T[N,D] @ G_kk[D,T]
//
//      batchCount = 3
//
//   3. grad_mix with one strided batched SGEMM + reduce:
//
//        gm_tmp_kk[D,N] = G_kk[D,T] @ kc_kk^T[T,N]
//        grad_mix[D,N] = gm_tmp_0 + gm_tmp_1 + gm_tmp_2
//
//   4. grad_h GEMM-assisted:
//
//        W_kk[D,T] = mix[D,N] @ kc_kk[N,T]
//
//      Then scatter:
//
//        grad_h[d, off+t-kk] += grad_out[d,t] * W_kk[d,t]
//
// Important compile/link note:
//   This file is included by multiple .cu files.
//   Therefore __global__ kernels in this file are templated by N.
// ======================================================================================

namespace fdc_backward_n_k3_d256_gemm_detail {

constexpr int FDC_BWD_GEMM_D = 256;
constexpr int FDC_BWD_GEMM_K = 3;

static inline void fdc_bwd_gemm_check_cublas(
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
// make G
//
// G:
//   [3,D,T]
//
// G[kk,d,t] = go[d,t] * h[d, off+t-kk]
//
// boundary invalid -> 0
// ======================================================================================

template<int N>
__global__ void fdc_bwd_n_k3_d256_make_g_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    float* __restrict__ G,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_BWD_GEMM_K * FDC_BWD_GEMM_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int tmp = idx / T;
    int d = tmp % FDC_BWD_GEMM_D;
    int kk = tmp / FDC_BWD_GEMM_D;

    int src = off + t - kk;

    float v = 0.0f;

    if (src >= 0 && src < L) {
        float go_v = go[
            static_cast<int64_t>(d) * T + t
        ];

        float h_v = h[
            static_cast<int64_t>(d) * L + src
        ];

        v = go_v * h_v;
    }

    G[
        static_cast<int64_t>(kk) * FDC_BWD_GEMM_D * T +
        static_cast<int64_t>(d) * T +
        t
    ] = v;

    (void)N;
}

// ======================================================================================
// reduce grad_mix
//
// gm_tmp:
//   [3,D,N]
//
// gm:
//   [D,N]
//
// gm[d,n] = gm_tmp[0,d,n] + gm_tmp[1,d,n] + gm_tmp[2,d,n]
// ======================================================================================

template<int N>
__global__ void fdc_bwd_n_k3_d256_reduce_gm_kernel(
    const float* __restrict__ gm_tmp,
    float* __restrict__ gm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_BWD_GEMM_D * N;

    if (idx >= total) {
        return;
    }

    int n = idx % N;
    int d = idx / N;

    int64_t base = static_cast<int64_t>(d) * N + n;

    float v0 = gm_tmp[
        static_cast<int64_t>(0) * FDC_BWD_GEMM_D * N + base
    ];

    float v1 = gm_tmp[
        static_cast<int64_t>(1) * FDC_BWD_GEMM_D * N + base
    ];

    float v2 = gm_tmp[
        static_cast<int64_t>(2) * FDC_BWD_GEMM_D * N + base
    ];

    gm[base] = v0 + v1 + v2;
}

// ======================================================================================
// grad_h scatter kernel
//
// W:
//   [3,D,T]
//
// W_kk[d,t] = sum_n mix[d,n] * kc[kk,n,t]
//
// grad_h[d,src] += go[d,t] * W[kk,d,t]
//
// src = off + t - kk
//
// Since K=3, each h position can receive multiple contributions.
// We use atomicAdd for correctness.
// ======================================================================================

template<int N>
__global__ void fdc_bwd_n_k3_d256_grad_h_scatter_kernel(
    const float* __restrict__ go,
    const float* __restrict__ W,
    float* __restrict__ gh,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_BWD_GEMM_K * FDC_BWD_GEMM_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int tmp = idx / T;
    int d = tmp % FDC_BWD_GEMM_D;
    int kk = tmp / FDC_BWD_GEMM_D;

    int src = off + t - kk;

    if (src < 0 || src >= L) {
        return;
    }

    float go_v = go[
        static_cast<int64_t>(d) * T + t
    ];

    float w_v = W[
        static_cast<int64_t>(kk) * FDC_BWD_GEMM_D * T +
        static_cast<int64_t>(d) * T +
        t
    ];

    atomicAdd(
        gh + static_cast<int64_t>(d) * L + src,
        go_v * w_v
    );

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

    if (go.scalar_type() != at::ScalarType::Float ||
        h.scalar_type() != at::ScalarType::Float ||
        kc.scalar_type() != at::ScalarType::Float ||
        mix.scalar_type() != at::ScalarType::Float) {
        return false;
    }

    if (go.size(0) != 1 || h.size(0) != 1 || kc.size(0) != 1) {
        return false;
    }

    if (go.size(1) != FDC_BWD_GEMM_D || h.size(1) != FDC_BWD_GEMM_D) {
        return false;
    }

    if (kc.size(1) != FDC_BWD_GEMM_K) {
        return false;
    }

    if (kc.size(2) != N) {
        return false;
    }

    if (mix.size(0) != FDC_BWD_GEMM_D || mix.size(1) != N) {
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

    TORCH_CHECK(go.scalar_type() == at::ScalarType::Float, "go must be float32.");
    TORCH_CHECK(h.scalar_type() == at::ScalarType::Float, "h must be float32.");
    TORCH_CHECK(kc.scalar_type() == at::ScalarType::Float, "kc must be float32.");
    TORCH_CHECK(mix.scalar_type() == at::ScalarType::Float, "mix must be float32.");

    TORCH_CHECK(go.size(0) == 1, "requires B == 1.");
    TORCH_CHECK(h.size(0) == 1, "requires B == 1.");
    TORCH_CHECK(kc.size(0) == 1, "requires B == 1.");

    TORCH_CHECK(go.size(1) == FDC_BWD_GEMM_D, "requires D == 256.");
    TORCH_CHECK(h.size(1) == FDC_BWD_GEMM_D, "requires D == 256.");

    TORCH_CHECK(kc.size(1) == FDC_BWD_GEMM_K, "requires K == 3.");
    TORCH_CHECK(kc.size(2) == N, "N mismatch.");

    TORCH_CHECK(mix.size(0) == FDC_BWD_GEMM_D, "mix D mismatch.");
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
            FDC_BWD_GEMM_K,
            FDC_BWD_GEMM_D,
            T,
        },
        go.options()
    );

    auto gm_tmp = torch::empty(
        {
            FDC_BWD_GEMM_K,
            FDC_BWD_GEMM_D,
            N,
        },
        go.options()
    );

    auto W = torch::empty(
        {
            FDC_BWD_GEMM_K,
            FDC_BWD_GEMM_D,
            T,
        },
        go.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;

    // ==================================================================================
    // 1. make G[3,D,T]
    // ==================================================================================

    int total_G = FDC_BWD_GEMM_K * FDC_BWD_GEMM_D * T;
    int blocks_G = (total_G + threads - 1) / threads;

    fdc_bwd_n_k3_d256_make_g_kernel<N><<<
        blocks_G,
        threads,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        h.data_ptr<float>(),
        G.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_bwd_gemm_check_cublas(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream failed"
    );

    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    // ==================================================================================
    // 2. grad_kernel batched SGEMM
    //
    // For each kk:
    //   gk_kk[N,T] = mix^T[N,D] @ G_kk[D,T]
    //
    // Memory:
    //   G:
    //     [3,D,T], G_kk row-major [D,T]
    //
    //   mix:
    //     [D,N], row-major
    //
    //   gk:
    //     [1,3,N,T], gk_kk row-major [N,T]
    //
    // cuBLAS column-major trick:
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
    // cublasSgemmStridedBatched:
    //   m = T
    //   n = N
    //   k = D
    //   strideA = D*T
    //   strideB = 0
    //   strideC = N*T
    //   batchCount = 3
    // ==================================================================================

    fdc_bwd_gemm_check_cublas(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            T,
            N,
            FDC_BWD_GEMM_D,
            &alpha,
            G.data_ptr<float>(),
            T,
            static_cast<long long>(FDC_BWD_GEMM_D) * T,
            mix.data_ptr<float>(),
            N,
            0,
            &beta0,
            gk.data_ptr<float>(),
            T,
            static_cast<long long>(N) * T,
            FDC_BWD_GEMM_K
        ),
        "grad_kernel cublasSgemmStridedBatched failed"
    );

    // ==================================================================================
    // 3. grad_mix batched SGEMM into gm_tmp[3,D,N]
    //
    // For each kk:
    //   gm_tmp_kk[D,N] = G_kk[D,T] @ kc_kk^T[T,N]
    //
    // Memory:
    //   kc_kk:
    //     row-major [N,T], viewed col-major [T,N]
    //
    //   G_kk:
    //     row-major [D,T], viewed col-major [T,D]
    //
    //   gm_tmp_kk:
    //     row-major [D,N], viewed col-major [N,D]
    //
    // cuBLAS:
    //   C_col[N,D] = A_col[N,T] @ B_col[T,D]
    //
    //   A = kc_kk, opA=T, lda=T
    //   B = G_kk,  opB=N, lda=T
    //   C = gm_tmp_kk, ldc=N
    //
    //   m = N
    //   n = D
    //   k = T
    //
    //   strideA = N*T
    //   strideB = D*T
    //   strideC = D*N
    //   batchCount = 3
    // ==================================================================================

    fdc_bwd_gemm_check_cublas(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N,
            FDC_BWD_GEMM_D,
            T,
            &alpha,
            kc.data_ptr<float>(),
            T,
            static_cast<long long>(N) * T,
            G.data_ptr<float>(),
            T,
            static_cast<long long>(FDC_BWD_GEMM_D) * T,
            &beta0,
            gm_tmp.data_ptr<float>(),
            N,
            static_cast<long long>(FDC_BWD_GEMM_D) * N,
            FDC_BWD_GEMM_K
        ),
        "grad_mix cublasSgemmStridedBatched failed"
    );

    // ==================================================================================
    // 4. reduce gm_tmp[3,D,N] -> gm[D,N]
    // ==================================================================================

    int total_gm = FDC_BWD_GEMM_D * N;
    int blocks_gm = (total_gm + threads - 1) / threads;

    fdc_bwd_n_k3_d256_reduce_gm_kernel<N><<<
        blocks_gm,
        threads,
        0,
        stream
    >>>(
        gm_tmp.data_ptr<float>(),
        gm.data_ptr<float>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ==================================================================================
    // 5. grad_h GEMM-assisted
    //
    // For each kk:
    //   W_kk[D,T] = mix[D,N] @ kc_kk[N,T]
    //
    // Memory:
    //   kc_kk:
    //     row-major [N,T], viewed col-major [T,N]
    //
    //   mix:
    //     row-major [D,N], viewed col-major [N,D]
    //
    //   W_kk:
    //     row-major [D,T], viewed col-major [T,D]
    //
    // cuBLAS:
    //   C_col[T,D] = A_col[T,N] @ B_col[N,D]
    //
    //   A = kc_kk, opA=N, lda=T
    //   B = mix,   opB=N, lda=N
    //   C = W_kk,  ldc=T
    //
    //   m = T
    //   n = D
    //   k = N
    //
    //   strideA = N*T
    //   strideB = 0
    //   strideC = D*T
    //   batchCount = 3
    // ==================================================================================

    fdc_bwd_gemm_check_cublas(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            T,
            FDC_BWD_GEMM_D,
            N,
            &alpha,
            kc.data_ptr<float>(),
            T,
            static_cast<long long>(N) * T,
            mix.data_ptr<float>(),
            N,
            0,
            &beta0,
            W.data_ptr<float>(),
            T,
            static_cast<long long>(FDC_BWD_GEMM_D) * T,
            FDC_BWD_GEMM_K
        ),
        "grad_h W cublasSgemmStridedBatched failed"
    );

    // ==================================================================================
    // 6. scatter grad_h
    //
    // gh[d, off+t-kk] += go[d,t] * W[kk,d,t]
    // ==================================================================================

    int total_h = FDC_BWD_GEMM_K * FDC_BWD_GEMM_D * T;
    int blocks_h = (total_h + threads - 1) / threads;

    fdc_bwd_n_k3_d256_grad_h_scatter_kernel<N><<<
        blocks_h,
        threads,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        W.data_ptr<float>(),
        gh.data_ptr<float>(),
        L,
        T,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        gh,
        gk,
        gm,
    };
}

} // namespace fdc_backward_n_k3_d256_gemm_detail
