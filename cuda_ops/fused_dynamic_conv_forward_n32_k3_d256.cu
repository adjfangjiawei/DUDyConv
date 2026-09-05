#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Specialized forward implementation:
//
//   B = 1
//   D = 256
//   N = 32
//   K = 3
//   dtype = float32
//   dilation = 1
//
// Mathematical form:
//
//   out[d,t] =
//       h[d, off + t    ] * sum_n mix[d,n] * kc[t,n,0]
//     + h[d, off + t - 1] * sum_n mix[d,n] * kc[t,n,1]
//     + h[d, off + t - 2] * sum_n mix[d,n] * kc[t,n,2]
//
// This implementation is NOT a wrapper around an existing plan.
// It creates a real N32-specific GEMM-based candidate:
//
//   1. Pack kc into three contiguous row-major matrices:
//
//        kc_pack[k, n, t] = kc[0, t, n, k]
//
//      Shape per k:
//
//        kc_pack_k: [N, T] row-major = [32, T]
//
//   2. For k = 0,1,2, compute:
//
//        tmp_k = mix @ kc_pack_k
//
//      where:
//
//        mix  : [D, N] = [256, 32] row-major
//        kc_k : [N, T] = [32, T] row-major
//        tmp_k: [D, T] = [256, T] row-major
//
//      cuBLAS is column-major. A row-major [D,T] result can be interpreted as
//      a column-major [T,D] matrix. The call is:
//
//        C_col[T,D] = B_col[T,N] * A_col[N,D]
//
//      with:
//        B_col = kc_pack_k memory, interpreted as col-major [T,N]
//        A_col = mix memory,      interpreted as col-major [N,D]
//        C_col = tmp_k memory,    interpreted as col-major [T,D]
//
//      Therefore:
//
//        cublasSgemm(
//            CUBLAS_OP_N,
//            CUBLAS_OP_N,
//            T,
//            D,
//            N,
//            alpha,
//            kc_pack_k,
//            T,
//            mix,
//            N,
//            beta,
//            tmp_k,
//            T
//        )
//
//   3. Combine:
//
//        out[d,t] = tmp0[d,t] * h[d, off+t]
//                 + tmp1[d,t] * h[d, off+t-1]
//                 + tmp2[d,t] * h[d, off+t-2]
//
// ======================================================================================

namespace {

constexpr int FDC_FWD_N32_D = 256;
constexpr int FDC_FWD_N32_N = 32;
constexpr int FDC_FWD_N32_K = 3;

// ======================================================================================
// Small helpers
// ======================================================================================

static inline void fdc_check_cublas_status(
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

static inline bool fdc_forward_n32_shape_ok(
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

    if (h.size(1) != FDC_FWD_N32_D) {
        return false;
    }

    if (mix.size(0) != FDC_FWD_N32_D) {
        return false;
    }

    if (kc.size(2) != FDC_FWD_N32_N) {
        return false;
    }

    if (mix.size(1) != FDC_FWD_N32_N) {
        return false;
    }

    if (kc.size(3) != FDC_FWD_N32_K) {
        return false;
    }

    if (dilation != 1) {
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
// Kernel: pack kc [1,T,32,3] -> kc_pack [3,32,T]
//
// Input contiguous kc layout:
//
//   kc[0,t,n,k] offset = ((t * 32 + n) * 3 + k)
//
// Output contiguous kc_pack layout:
//
//   kc_pack[k,n,t] offset = (k * 32 + n) * T + t
//
// ======================================================================================

__global__ void fdc_forward_n32_pack_kc_kernel(
    const float* __restrict__ kc,
    float* __restrict__ kc_pack,
    int T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_N32_K * FDC_FWD_N32_N * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int q = idx / T;
    int n = q % FDC_FWD_N32_N;
    int k = q / FDC_FWD_N32_N;

    int64_t src = ((int64_t)t * FDC_FWD_N32_N + n) * FDC_FWD_N32_K + k;
    int64_t dst = ((int64_t)k * FDC_FWD_N32_N + n) * T + t;

    kc_pack[dst] = kc[src];
}

// ======================================================================================
// Kernel: combine no-boundary
//
// Valid when:
//
//   off >= 2
//
// because K=3, dilation=1, so:
//   s0 = off + t
//   s1 = off + t - 1
//   s2 = off + t - 2
//
// and check_fdc_forward_inputs already guarantees:
//
//   off + T <= L
//
// ======================================================================================

__global__ void fdc_forward_n32_combine_noboundary_kernel(
    const float* __restrict__ h,
    const float* __restrict__ tmp,
    float* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_N32_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int64_t ht = (int64_t)d * L + off + t;
    int64_t ot = (int64_t)d * T + t;

    int64_t base0 = (int64_t)0 * FDC_FWD_N32_D * T + ot;
    int64_t base1 = (int64_t)1 * FDC_FWD_N32_D * T + ot;
    int64_t base2 = (int64_t)2 * FDC_FWD_N32_D * T + ot;

    float v0 = tmp[base0] * h[ht];
    float v1 = tmp[base1] * h[ht - 1];
    float v2 = tmp[base2] * h[ht - 2];

    out[ot] = v0 + v1 + v2;
}

// ======================================================================================
// Kernel: combine boundary
//
// Handles off < 2 safely.
// ======================================================================================

__global__ void fdc_forward_n32_combine_boundary_kernel(
    const float* __restrict__ h,
    const float* __restrict__ tmp,
    float* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_N32_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int64_t ot = (int64_t)d * T + t;

    float acc = 0.0f;

    int s0 = off + t;
    int s1 = off + t - 1;
    int s2 = off + t - 2;

    if (s0 >= 0 && s0 < L) {
        int64_t tmp0 = (int64_t)0 * FDC_FWD_N32_D * T + ot;
        acc += tmp[tmp0] * h[(int64_t)d * L + s0];
    }

    if (s1 >= 0 && s1 < L) {
        int64_t tmp1 = (int64_t)1 * FDC_FWD_N32_D * T + ot;
        acc += tmp[tmp1] * h[(int64_t)d * L + s1];
    }

    if (s2 >= 0 && s2 < L) {
        int64_t tmp2 = (int64_t)2 * FDC_FWD_N32_D * T + ot;
        acc += tmp[tmp2] * h[(int64_t)d * L + s2];
    }

    out[ot] = acc;
}

} // namespace

// ======================================================================================
// Public availability
// ======================================================================================

bool fdc_forward_n32_k3_d256_available_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    if (!fdc_forward_n32_shape_ok(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return false;
    }

    // This implementation uses 3 SGEMMs. For very small T, direct kernels may win due to
    // cublas launch overhead. For the reported target case T=4096 this should participate.
    int64_t T = kc.size(1);

    if (T < 1024) {
        return false;
    }

    return true;
}

// ======================================================================================
// Public execution
// ======================================================================================

torch::Tensor fdc_forward_n32_k3_d256_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
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
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,T,N,K].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(h.scalar_type() == at::ScalarType::Float, "h must be float32.");
    TORCH_CHECK(kc.scalar_type() == at::ScalarType::Float, "kc must be float32.");
    TORCH_CHECK(mix.scalar_type() == at::ScalarType::Float, "mix must be float32.");

    TORCH_CHECK(h.size(0) == 1, "forward_n32_k3_d256 requires B == 1.");
    TORCH_CHECK(kc.size(0) == 1, "forward_n32_k3_d256 requires kc B == 1.");
    TORCH_CHECK(h.size(1) == FDC_FWD_N32_D, "forward_n32_k3_d256 requires D == 256.");
    TORCH_CHECK(mix.size(0) == FDC_FWD_N32_D, "forward_n32_k3_d256 requires mix D == 256.");
    TORCH_CHECK(kc.size(2) == FDC_FWD_N32_N, "forward_n32_k3_d256 requires N == 32.");
    TORCH_CHECK(mix.size(1) == FDC_FWD_N32_N, "forward_n32_k3_d256 requires mix N == 32.");
    TORCH_CHECK(kc.size(3) == FDC_FWD_N32_K, "forward_n32_k3_d256 requires K == 3.");
    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + kc.size(1) <= h.size(2), "off + T must be <= L.");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    auto out = torch::empty(
        {1, FDC_FWD_N32_D, T},
        h.options()
    );

    auto kc_pack = torch::empty(
        {FDC_FWD_N32_K, FDC_FWD_N32_N, T},
        h.options()
    );

    auto tmp = torch::empty(
        {FDC_FWD_N32_K, FDC_FWD_N32_D, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --------------------------------------------------------------------------
    // 1. Pack kc
    // --------------------------------------------------------------------------

    int pack_total = FDC_FWD_N32_K * FDC_FWD_N32_N * T;
    int pack_block = 256;
    int pack_grid = (pack_total + pack_block - 1) / pack_block;

    fdc_forward_n32_pack_kc_kernel<<<
        pack_grid,
        pack_block,
        0,
        stream
    >>>(
        kc.data_ptr<float>(),
        kc_pack.data_ptr<float>(),
        T
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // --------------------------------------------------------------------------
    // 2. Three SGEMMs:
    //
    //      tmp_k[D,T] = mix[D,N] @ kc_pack_k[N,T]
    //
    //    Implemented as column-major:
    //
    //      tmp_k_col[T,D] = kc_pack_k_col[T,N] @ mix_col[N,D]
    //
    // --------------------------------------------------------------------------

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    fdc_check_cublas_status(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream failed"
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float* mix_ptr = mix.data_ptr<float>();
    const float* kc_pack_ptr = kc_pack.data_ptr<float>();
    float* tmp_ptr = tmp.data_ptr<float>();

    // Dimensions for cublas column-major GEMM:
    //
    // C[T, D] = A[T, N] * B[N, D]
    //
    // A = kc_pack_k interpreted as column-major [T, N], lda = T
    // B = mix       interpreted as column-major [N, D], ldb = N
    // C = tmp_k     interpreted as column-major [T, D], ldc = T
    //
    // m = T
    // n = D=256
    // k = N=32

#pragma unroll
    for (int kk = 0; kk < FDC_FWD_N32_K; ++kk) {
        const float* A = kc_pack_ptr + (int64_t)kk * FDC_FWD_N32_N * T;
        const float* B = mix_ptr;
        float* C = tmp_ptr + (int64_t)kk * FDC_FWD_N32_D * T;

        fdc_check_cublas_status(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                T,
                FDC_FWD_N32_D,
                FDC_FWD_N32_N,
                &alpha,
                A,
                T,
                B,
                FDC_FWD_N32_N,
                &beta,
                C,
                T
            ),
            "cublasSgemm failed in forward_n32_k3_d256"
        );
    }

    // --------------------------------------------------------------------------
    // 3. Combine tmp and h
    // --------------------------------------------------------------------------

    int combine_total = FDC_FWD_N32_D * T;
    int combine_block = 256;
    int combine_grid = (combine_total + combine_block - 1) / combine_block;

    if (static_cast<int>(off) >= 2) {
        fdc_forward_n32_combine_noboundary_kernel<<<
            combine_grid,
            combine_block,
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
        fdc_forward_n32_combine_boundary_kernel<<<
            combine_grid,
            combine_block,
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

    FDC_DEBUG_PATH("forward_n32_k3_d256_sgemm3_pack_combine");

    return out;
}
