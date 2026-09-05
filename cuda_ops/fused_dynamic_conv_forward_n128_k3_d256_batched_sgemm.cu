#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// forward_n128_k3_d256_batched_sgemm
//
// 新增独立 forward plan，不修改现有 forward plan。
//
// Target:
//   B == 1
//   D == 256
//   N == 128
//   K == 3
//   dtype == float32
//   dilation == 1
//
// Layout:
//   h:
//     [1,256,L]
//
//   kc:
//     [1,3,128,T]
//
//   mix:
//     [256,128]
//
//   out:
//     [1,256,T]
//
// kc contiguous offset:
//
//   kc[0,kk,n,t] = ((kk * N + n) * T + t)
//
// Main optimization:
//
//   Old style:
//     for kk in 0..2:
//         tmp[kk] = mix @ kc[kk]
//
//   New style:
//     one cublasSgemmStridedBatched with batchCount=3.
//
// Logical row-major:
//
//   for kk:
//     tmp_kk[D,T] = mix[D,N] @ kc_kk[N,T]
//
// cuBLAS column-major trick:
//
//   C_col[T,D] = A_col[T,N] @ B_col[N,D]
//
// where:
//   A = kc_kk row-major [N,T], seen as column-major [T,N]
//   B = mix   row-major [D,N], seen as column-major [N,D]
//   C = tmp   row-major [D,T], seen as column-major [T,D]
//
// cublasSgemmStridedBatched parameters:
//
//   m = T
//   n = D
//   k = N
//
//   A: kc
//      lda = T
//      strideA = N * T
//
//   B: mix
//      ldb = N
//      strideB = 0
//
//   C: tmp
//      ldc = T
//      strideC = D * T
//
//   batchCount = 3
// ======================================================================================

namespace {

constexpr int FDC_FWD_N128_K3_D256_D = 256;
constexpr int FDC_FWD_N128_K3_D256_N = 128;
constexpr int FDC_FWD_N128_K3_D256_K = 3;

static inline void fdc_n128_k3_d256_batched_check_cublas(
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
// combine kernels
//
// tmp:
//   [3,256,T]
//
// out[d,t] =
//   tmp[0,d,t] * h[d,off+t]
// + tmp[1,d,t] * h[d,off+t-1]
// + tmp[2,d,t] * h[d,off+t-2]
//
// 新语义：
//   kk=0 当前
//   kk=1 上一位置
//   kk=2 上上位置
// ======================================================================================

__global__ void fdc_forward_n128_k3_d256_batched_combine_noboundary_kernel(
    const float* __restrict__ h,
    const float* __restrict__ tmp,
    float* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_N128_K3_D256_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int64_t ht = static_cast<int64_t>(d) * L + off + t;
    int64_t ot = static_cast<int64_t>(d) * T + t;

    int64_t tmp0 = static_cast<int64_t>(0) * FDC_FWD_N128_K3_D256_D * T + ot;
    int64_t tmp1 = static_cast<int64_t>(1) * FDC_FWD_N128_K3_D256_D * T + ot;
    int64_t tmp2 = static_cast<int64_t>(2) * FDC_FWD_N128_K3_D256_D * T + ot;

    out[ot] =
        tmp[tmp0] * h[ht] +
        tmp[tmp1] * h[ht - 1] +
        tmp[tmp2] * h[ht - 2];
}

__global__ void fdc_forward_n128_k3_d256_batched_combine_boundary_kernel(
    const float* __restrict__ h,
    const float* __restrict__ tmp,
    float* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_N128_K3_D256_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int s0 = off + t;
    int s1 = off + t - 1;
    int s2 = off + t - 2;

    int64_t ot = static_cast<int64_t>(d) * T + t;

    float acc = 0.0f;

    if (s0 >= 0 && s0 < L) {
        acc +=
            tmp[static_cast<int64_t>(0) * FDC_FWD_N128_K3_D256_D * T + ot] *
            h[static_cast<int64_t>(d) * L + s0];
    }

    if (s1 >= 0 && s1 < L) {
        acc +=
            tmp[static_cast<int64_t>(1) * FDC_FWD_N128_K3_D256_D * T + ot] *
            h[static_cast<int64_t>(d) * L + s1];
    }

    if (s2 >= 0 && s2 < L) {
        acc +=
            tmp[static_cast<int64_t>(2) * FDC_FWD_N128_K3_D256_D * T + ot] *
            h[static_cast<int64_t>(d) * L + s2];
    }

    out[ot] = acc;
}

} // namespace

// ======================================================================================
// availability
// ======================================================================================

bool fdc_forward_n128_k3_d256_batched_sgemm_available_cuda(
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

    if (h.scalar_type() != at::ScalarType::Float ||
        kc.scalar_type() != at::ScalarType::Float ||
        mix.scalar_type() != at::ScalarType::Float) {
        return false;
    }

    if (h.size(0) != 1 || kc.size(0) != 1) {
        return false;
    }

    if (h.size(1) != FDC_FWD_N128_K3_D256_D) {
        return false;
    }

    if (kc.size(1) != FDC_FWD_N128_K3_D256_K) {
        return false;
    }

    if (kc.size(2) != FDC_FWD_N128_K3_D256_N) {
        return false;
    }

    if (mix.size(0) != FDC_FWD_N128_K3_D256_D) {
        return false;
    }

    if (mix.size(1) != FDC_FWD_N128_K3_D256_N) {
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
// forward
// ======================================================================================

torch::Tensor fdc_forward_n128_k3_d256_batched_sgemm_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("forward_n128_k3_d256_batched_sgemm_bknt");

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

    TORCH_CHECK(h.scalar_type() == at::ScalarType::Float, "h must be float32.");
    TORCH_CHECK(kc.scalar_type() == at::ScalarType::Float, "kc must be float32.");
    TORCH_CHECK(mix.scalar_type() == at::ScalarType::Float, "mix must be float32.");

    TORCH_CHECK(h.size(0) == 1, "forward_n128_k3_d256_batched_sgemm requires B == 1.");
    TORCH_CHECK(kc.size(0) == 1, "kc B mismatch.");

    TORCH_CHECK(h.size(1) == FDC_FWD_N128_K3_D256_D, "requires D == 256.");
    TORCH_CHECK(kc.size(1) == FDC_FWD_N128_K3_D256_K, "requires K == 3.");
    TORCH_CHECK(kc.size(2) == FDC_FWD_N128_K3_D256_N, "requires N == 128.");

    TORCH_CHECK(mix.size(0) == FDC_FWD_N128_K3_D256_D, "mix D mismatch.");
    TORCH_CHECK(mix.size(1) == FDC_FWD_N128_K3_D256_N, "mix N mismatch.");

    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + kc.size(3) <= h.size(2), "off + T must be <= L.");

    int D = FDC_FWD_N128_K3_D256_D;
    int N = FDC_FWD_N128_K3_D256_N;
    int K = FDC_FWD_N128_K3_D256_K;

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(3));

    auto out = torch::empty(
        {
            1,
            D,
            T,
        },
        h.options()
    );

    auto tmp = torch::empty(
        {
            K,
            D,
            T,
        },
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_n128_k3_d256_batched_check_cublas(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream failed in fdc_forward_n128_k3_d256_batched_sgemm_cuda"
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float* A = kc.data_ptr<float>();
    const float* Bptr = mix.data_ptr<float>();
    float* C = tmp.data_ptr<float>();

    /*
     * Batched GEMM:
     *
     * For kk in 0..2:
     *
     *   tmp[kk, D, T] = mix[D,N] @ kc[kk,N,T]
     *
     * cuBLAS column-major view:
     *
     *   C_col[T,D] = A_col[T,N] @ B_col[N,D]
     *
     * Parameters:
     *
     *   m = T
     *   n = D
     *   k = N
     *
     *   A = kc[kk], row-major [N,T], column-major [T,N]
     *       lda = T
     *       strideA = N*T
     *
     *   B = mix, row-major [D,N], column-major [N,D]
     *       ldb = N
     *       strideB = 0
     *
     *   C = tmp[kk], row-major [D,T], column-major [T,D]
     *       ldc = T
     *       strideC = D*T
     */
    fdc_n128_k3_d256_batched_check_cublas(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            T,
            D,
            N,
            &alpha,
            A,
            T,
            static_cast<long long>(N) * T,
            Bptr,
            N,
            0,
            &beta,
            C,
            T,
            static_cast<long long>(D) * T,
            K
        ),
        "cublasSgemmStridedBatched failed in fdc_forward_n128_k3_d256_batched_sgemm_cuda"
    );

    int total = D * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (static_cast<int>(off) >= 2) {
        fdc_forward_n128_k3_d256_batched_combine_noboundary_kernel<<<
            blocks,
            threads,
            0,
            stream
        >>>(
            h.data_ptr<float>(),
            tmp.data_ptr<float>(),
            out.data_ptr<float>(),
            L,
            T,
            static_cast<int>(off)
        );
    } else {
        fdc_forward_n128_k3_d256_batched_combine_boundary_kernel<<<
            blocks,
            threads,
            0,
            stream
        >>>(
            h.data_ptr<float>(),
            tmp.data_ptr<float>(),
            out.data_ptr<float>(),
            L,
            T,
            static_cast<int>(off)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
