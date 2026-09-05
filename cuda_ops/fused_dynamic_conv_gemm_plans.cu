#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// New GEMM NK3 D256 plan
//
// Supported shape:
//
//   B = 1
//   D = 256
//   K = 3
//   dtype = float32
//   dilation = 1
//   N >= 8
//
// Layout:
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
//   out:
//     [1,256,T]
//
// Contiguous kc offset:
//
//   kc[0,kk,n,t] = (kk * N + n) * T + t
//
// ======================================================================================

namespace {

constexpr int FDC_NEW_D = 256;
constexpr int FDC_NEW_K = 3;

// ======================================================================================
// cuBLAS check
// ======================================================================================

static inline void fdc_new_check_cublas_status(
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
// shape check
// ======================================================================================

static inline bool fdc_new_shape_ok(
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

    if (h.size(1) != FDC_NEW_D) {
        return false;
    }

    if (kc.size(1) != FDC_NEW_K) {
        return false;
    }

    if (kc.size(2) < 8) {
        return false;
    }

    if (mix.size(0) != FDC_NEW_D) {
        return false;
    }

    if (mix.size(1) != kc.size(2)) {
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
// forward combine
//
// tmp:
//   [3,256,T]
//
// out[d,t] =
//   tmp[0,d,t] * h[d,off+t]
// + tmp[1,d,t] * h[d,off+t-1]
// + tmp[2,d,t] * h[d,off+t-2]
// ======================================================================================

__global__ void fdc_new_forward_combine_noboundary_kernel(
    const float* __restrict__ h,
    const float* __restrict__ tmp,
    float* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_NEW_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int64_t ht = static_cast<int64_t>(d) * L + off + t;
    int64_t ot = static_cast<int64_t>(d) * T + t;

    int64_t tmp0 = static_cast<int64_t>(0) * FDC_NEW_D * T + ot;
    int64_t tmp1 = static_cast<int64_t>(1) * FDC_NEW_D * T + ot;
    int64_t tmp2 = static_cast<int64_t>(2) * FDC_NEW_D * T + ot;

    out[ot] =
        tmp[tmp0] * h[ht] +
        tmp[tmp1] * h[ht - 1] +
        tmp[tmp2] * h[ht - 2];
}

__global__ void fdc_new_forward_combine_boundary_kernel(
    const float* __restrict__ h,
    const float* __restrict__ tmp,
    float* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_NEW_D * T;

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
        acc += tmp[static_cast<int64_t>(0) * FDC_NEW_D * T + ot] *
               h[static_cast<int64_t>(d) * L + s0];
    }

    if (s1 >= 0 && s1 < L) {
        acc += tmp[static_cast<int64_t>(1) * FDC_NEW_D * T + ot] *
               h[static_cast<int64_t>(d) * L + s1];
    }

    if (s2 >= 0 && s2 < L) {
        acc += tmp[static_cast<int64_t>(2) * FDC_NEW_D * T + ot] *
               h[static_cast<int64_t>(d) * L + s2];
    }

    out[ot] = acc;
}

// ======================================================================================
// backward helpers
// ======================================================================================

__global__ void fdc_new_zero_float_kernel(
    float* __restrict__ ptr,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        ptr[i] = 0.0f;
    }
}

// ======================================================================================
// build x for each kk:
//
// x[d,t] = go[d,t] * h[d,off+t-kk]
//
// x_dt:
//   [256,T]
//
// x_td:
//   [T,256]
// ======================================================================================

__global__ void fdc_new_backward_build_x_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    float* __restrict__ x_dt,
    float* __restrict__ x_td,
    int L,
    int T,
    int off,
    int kk
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(FDC_NEW_D) * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int s = off + t - kk;

    float v = 0.0f;

    if (s >= 0 && s < L) {
        v =
            go[static_cast<int64_t>(d) * T + t] *
            h[static_cast<int64_t>(d) * L + s];
    }

    x_dt[static_cast<int64_t>(d) * T + t] = v;
    x_td[static_cast<int64_t>(t) * FDC_NEW_D + d] = v;
}

// ======================================================================================
// build kc_tn / kc_nt for each kk:
//
// kc_tn:
//   [T,N]
//
// kc_nt:
//   [N,T]
//
// source:
//
//   kc[0,kk,n,t]
// ======================================================================================

__global__ void fdc_new_backward_build_kc_kernel(
    const float* __restrict__ kc,
    float* __restrict__ kc_tn,
    float* __restrict__ kc_nt,
    int T,
    int N,
    int kk
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(T) * N;

    if (idx >= total) {
        return;
    }

    int n = idx % N;
    int t = idx / N;

    float v = kc[(static_cast<int64_t>(kk) * N + n) * T + t];

    kc_tn[static_cast<int64_t>(t) * N + n] = v;
    kc_nt[static_cast<int64_t>(n) * T + t] = v;
}

// ======================================================================================
// write gk from gk_tn:
//
// gk_tn:
//   [T,N]
//
// gk:
//   [1,3,N,T]
// ======================================================================================

__global__ void fdc_new_backward_write_gk_kernel(
    const float* __restrict__ gk_tn,
    float* __restrict__ gk,
    int T,
    int N,
    int kk
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(T) * N;

    if (idx >= total) {
        return;
    }

    int n = idx % N;
    int t = idx / N;

    gk[(static_cast<int64_t>(kk) * N + n) * T + t] =
        gk_tn[static_cast<int64_t>(t) * N + n];
}

// ======================================================================================
// accumulate grad_h:
//
// w:
//   [256,T]
//
// gh[d,off+t-kk] += go[d,t] * w[d,t]
// ======================================================================================

__global__ void fdc_new_backward_accum_gh_kernel(
    const float* __restrict__ go,
    const float* __restrict__ w,
    float* __restrict__ gh,
    int L,
    int T,
    int off,
    int kk
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(FDC_NEW_D) * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int s = off + t - kk;

    if (s >= 0 && s < L) {
        gh[static_cast<int64_t>(d) * L + s] +=
            go[static_cast<int64_t>(d) * T + t] *
            w[static_cast<int64_t>(d) * T + t];
    }
}

// ======================================================================================
// row-major SGEMM helper
//
// Computes:
//
//   C[M,N] = A[M,K] @ B[K,N]
//
// for row-major contiguous tensors, using cuBLAS column-major trick.
// ======================================================================================

static inline void fdc_new_sgemm_rowmajor(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float beta_value
) {
    const float alpha = 1.0f;
    const float beta = beta_value;

    fdc_new_check_cublas_status(
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
        "fdc_new_sgemm_rowmajor failed"
    );
}

} // namespace

// ======================================================================================
// Availability
// ======================================================================================

bool fdc_new_gemm_nk3_d256_available_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    if (!fdc_new_shape_ok(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return false;
    }

    int64_t T = kc.size(3);
    int64_t N = kc.size(2);

    if (N < 8) {
        return false;
    }

    if (T < 512) {
        return false;
    }

    return true;
}

// ======================================================================================
// Forward
// ======================================================================================

torch::Tensor fdc_new_forward_gemm_nk3_d256_cuda(
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
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,K,N,T].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(h.scalar_type() == at::ScalarType::Float, "h must be float32.");
    TORCH_CHECK(kc.scalar_type() == at::ScalarType::Float, "kc must be float32.");
    TORCH_CHECK(mix.scalar_type() == at::ScalarType::Float, "mix must be float32.");

    TORCH_CHECK(h.size(0) == 1, "new_gemm_nk3_d256 requires B == 1.");
    TORCH_CHECK(kc.size(0) == 1, "new_gemm_nk3_d256 requires kc B == 1.");
    TORCH_CHECK(h.size(1) == FDC_NEW_D, "new_gemm_nk3_d256 requires D == 256.");
    TORCH_CHECK(kc.size(1) == FDC_NEW_K, "new_gemm_nk3_d256 requires K == 3.");
    TORCH_CHECK(mix.size(0) == FDC_NEW_D, "mix D mismatch.");
    TORCH_CHECK(mix.size(1) == kc.size(2), "mix N mismatch.");
    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + kc.size(3) <= h.size(2), "off + T must be <= L.");

    int L = static_cast<int>(h.size(2));
    int N = static_cast<int>(kc.size(2));
    int T = static_cast<int>(kc.size(3));

    auto out = torch::empty(
        {
            1,
            FDC_NEW_D,
            T,
        },
        h.options()
    );

    auto tmp = torch::empty(
        {
            FDC_NEW_K,
            FDC_NEW_D,
            T,
        },
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_new_check_cublas_status(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream failed in fdc_new_forward_gemm_nk3_d256_cuda"
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float* mix_ptr = mix.data_ptr<float>();
    const float* kc_ptr = kc.data_ptr<float>();
    float* tmp_ptr = tmp.data_ptr<float>();

#pragma unroll
    for (int kk = 0; kk < FDC_NEW_K; ++kk) {
        const float* A = kc_ptr + static_cast<int64_t>(kk) * N * T;
        const float* B = mix_ptr;
        float* C = tmp_ptr + static_cast<int64_t>(kk) * FDC_NEW_D * T;

        /*
         * Row-major logical:
         *
         *   C_row[256,T] = mix[256,N] @ kc_kk[N,T]
         *
         * cuBLAS column-major trick:
         *
         *   C_col[T,256] = kc_kk_col[T,N] @ mix_col[N,256]
         */
        fdc_new_check_cublas_status(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                T,
                FDC_NEW_D,
                N,
                &alpha,
                A,
                T,
                B,
                N,
                &beta,
                C,
                T
            ),
            "cublasSgemm failed in fdc_new_forward_gemm_nk3_d256_cuda"
        );
    }

    int total = FDC_NEW_D * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (static_cast<int>(off) >= 2) {
        fdc_new_forward_combine_noboundary_kernel<<<
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
        fdc_new_forward_combine_boundary_kernel<<<
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

    FDC_DEBUG_PATH("forward_new_gemm_nk3_d256_bknt");

    return out;
}

// ======================================================================================
// Backward
// ======================================================================================

std::vector<torch::Tensor> fdc_new_backward_gemm_nk3_d256_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_new_gemm_nk3_d256_bknt");

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

    TORCH_CHECK(h.size(0) == 1, "new backward requires B == 1.");
    TORCH_CHECK(go.size(0) == 1, "go B mismatch.");
    TORCH_CHECK(kc.size(0) == 1, "kc B mismatch.");

    TORCH_CHECK(h.size(1) == FDC_NEW_D, "new backward requires h D == 256.");
    TORCH_CHECK(go.size(1) == FDC_NEW_D, "new backward requires go D == 256.");
    TORCH_CHECK(mix.size(0) == FDC_NEW_D, "mix D mismatch.");

    TORCH_CHECK(kc.size(1) == FDC_NEW_K, "new backward requires K == 3.");
    TORCH_CHECK(mix.size(1) == kc.size(2), "mix N mismatch.");
    TORCH_CHECK(go.size(2) == kc.size(3), "go T mismatch.");

    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + kc.size(3) <= h.size(2), "off + T must be <= L.");

    int L = static_cast<int>(h.size(2));
    int N = static_cast<int>(kc.size(2));
    int T = static_cast<int>(kc.size(3));

    auto gh = torch::empty(
        h.sizes(),
        h.options()
    );

    auto gk = torch::empty(
        kc.sizes(),
        kc.options()
    );

    auto gm = torch::empty(
        mix.sizes(),
        mix.options()
    );

    auto x_dt = torch::empty(
        {
            FDC_NEW_D,
            T,
        },
        h.options()
    );

    auto x_td = torch::empty(
        {
            T,
            FDC_NEW_D,
        },
        h.options()
    );

    auto kc_tn = torch::empty(
        {
            T,
            N,
        },
        h.options()
    );

    auto kc_nt = torch::empty(
        {
            N,
            T,
        },
        h.options()
    );

    auto w = torch::empty(
        {
            FDC_NEW_D,
            T,
        },
        h.options()
    );

    auto gk_tn = torch::empty(
        {
            T,
            N,
        },
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_new_check_cublas_status(
        cublasSetStream(
            handle,
            stream
        ),
        "cublasSetStream failed in fdc_new_backward_gemm_nk3_d256_cuda"
    );

    int threads = 256;

    fdc_new_zero_float_kernel<<<
        (gh.numel() + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        gh.data_ptr<float>(),
        gh.numel()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_new_zero_float_kernel<<<
        (gm.numel() + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        gm.data_ptr<float>(),
        gm.numel()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int kk = 0; kk < FDC_NEW_K; ++kk) {
        fdc_new_backward_build_x_kernel<<<
            (static_cast<int64_t>(FDC_NEW_D) * T + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            go.data_ptr<float>(),
            h.data_ptr<float>(),
            x_dt.data_ptr<float>(),
            x_td.data_ptr<float>(),
            L,
            T,
            static_cast<int>(off),
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_new_backward_build_kc_kernel<<<
            (static_cast<int64_t>(T) * N + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            kc.data_ptr<float>(),
            kc_tn.data_ptr<float>(),
            kc_nt.data_ptr<float>(),
            T,
            N,
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        /*
         * w = mix @ kc_nt
         *
         * mix:
         *   [256,N]
         *
         * kc_nt:
         *   [N,T]
         *
         * w:
         *   [256,T]
         */
        fdc_new_sgemm_rowmajor(
            handle,
            mix.data_ptr<float>(),
            kc_nt.data_ptr<float>(),
            w.data_ptr<float>(),
            FDC_NEW_D,
            T,
            N,
            0.0f
        );

        fdc_new_backward_accum_gh_kernel<<<
            (static_cast<int64_t>(FDC_NEW_D) * T + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            go.data_ptr<float>(),
            w.data_ptr<float>(),
            gh.data_ptr<float>(),
            L,
            T,
            static_cast<int>(off),
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        /*
         * gk_tn = x_td @ mix
         *
         * x_td:
         *   [T,256]
         *
         * mix:
         *   [256,N]
         *
         * gk_tn:
         *   [T,N]
         */
        fdc_new_sgemm_rowmajor(
            handle,
            x_td.data_ptr<float>(),
            mix.data_ptr<float>(),
            gk_tn.data_ptr<float>(),
            T,
            N,
            FDC_NEW_D,
            0.0f
        );

        fdc_new_backward_write_gk_kernel<<<
            (static_cast<int64_t>(T) * N + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gk_tn.data_ptr<float>(),
            gk.data_ptr<float>(),
            T,
            N,
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        /*
         * gm += x_dt @ kc_tn
         *
         * x_dt:
         *   [256,T]
         *
         * kc_tn:
         *   [T,N]
         *
         * gm:
         *   [256,N]
         */
        fdc_new_sgemm_rowmajor(
            handle,
            x_dt.data_ptr<float>(),
            kc_tn.data_ptr<float>(),
            gm.data_ptr<float>(),
            FDC_NEW_D,
            N,
            T,
            kk == 0 ? 0.0f : 1.0f
        );
    }

    return {
        gh,
        gk,
        gm
    };
}
