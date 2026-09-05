#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <vector>
#include <type_traits>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Local helpers
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_new_cast_float_to_scalar_kernel(
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
__global__ void fdc_new_make_mix_t_float_kernel(
    const scalar_t* __restrict__ mix,
    float* __restrict__ mix_t,
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

    // mix:   [D, N]
    // mix_t: [N, D]
    mix_t[n * D + d] = fdc_to_float_dev(mix[d * N + n]);
}

template <typename scalar_t>
__global__ void fdc_new_make_kc_ktn_float_kernel(
    const scalar_t* __restrict__ kc,
    float* __restrict__ kc_ktn,
    int T,
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * T * N;

    if (idx >= total) {
        return;
    }

    int n = idx % N;
    int q = idx / N;
    int t = q % T;
    int kk = q / T;

    // kc:     [B=1, T, N, K]
    // kc_ktn: [K, T, N]
    kc_ktn[(static_cast<int64_t>(kk) * T + t) * N + n] =
        fdc_to_float_dev(kc[(static_cast<int64_t>(t) * N + n) * K + kk]);
}

template <typename scalar_t>
__global__ void fdc_new_make_base_ktd_float_kernel(
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

    // base: [K, T, D], row-major, each kk matrix is [T, D]
    base[(static_cast<int64_t>(kk) * T + t) * D + d] = v;
}

template <typename scalar_t>
__global__ void fdc_new_forward_from_weight_kernel(
    const scalar_t* __restrict__ h,
    const float* __restrict__ weight,
    scalar_t* __restrict__ out,
    int D,
    int L,
    int T,
    int K,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    float acc = 0.0f;

    for (int kk = 0; kk < K; ++kk) {
        int s = off + t - kk;

        if (s >= 0 && s < L) {
            acc +=
                weight[(static_cast<int64_t>(kk) * T + t) * D + d] *
                fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);
        }
    }

    out[static_cast<int64_t>(d) * T + t] =
        fdc_from_float_dev<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void fdc_new_grad_h_from_weight_kernel(
    const scalar_t* __restrict__ go,
    const float* __restrict__ weight,
    float* __restrict__ gh,
    int D,
    int L,
    int T,
    int K,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * L;

    if (idx >= total) {
        return;
    }

    int s = idx % L;
    int d = idx / L;

    int tb = s - off;

    float acc = 0.0f;

    for (int kk = 0; kk < K; ++kk) {
        int t = tb + kk;

        if (t >= 0 && t < T) {
            acc +=
                fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) *
                weight[(static_cast<int64_t>(kk) * T + t) * D + d];
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc;
}

template <typename scalar_t>
__global__ void fdc_new_scatter_gk_ktn_to_btnk_kernel(
    const float* __restrict__ src,
    scalar_t* __restrict__ dst,
    int T,
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * N * K;

    if (idx >= total) {
        return;
    }

    int kk = idx % K;
    int q = idx / K;
    int n = q % N;
    int t = q / N;

    float v = src[(static_cast<int64_t>(kk) * T + t) * N + n];

    dst[(static_cast<int64_t>(t) * N + n) * K + kk] =
        fdc_from_float_dev<scalar_t>(v);
}

template <typename scalar_t>
__global__ void fdc_new_scatter_gm_float_to_scalar_kernel(
    const float* __restrict__ src,
    scalar_t* __restrict__ dst,
    int D,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * N;

    if (idx >= total) {
        return;
    }

    dst[idx] = fdc_from_float_dev<scalar_t>(src[idx]);
}

// ======================================================================================
// Forward GEMM materialized plan
// ======================================================================================

template <typename scalar_t>
static torch::Tensor fdc_new_forward_gemm_nk3_d256_typed(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    int B = static_cast<int>(h.size(0));
    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));
    int K = static_cast<int>(kc.size(3));

    TORCH_CHECK(B == 1, "fdc_new_forward_gemm_nk3_d256 requires B == 1.");
    TORCH_CHECK(D == 256, "fdc_new_forward_gemm_nk3_d256 requires D == 256.");
    TORCH_CHECK(K == 3, "fdc_new_forward_gemm_nk3_d256 requires K == 3.");
    TORCH_CHECK(N == 6 || N == 16, "fdc_new_forward_gemm_nk3_d256 requires N == 6 or 16.");

    auto fopts = h.options().dtype(torch::kFloat32);

    auto mix_t = torch::empty({N, D}, fopts);
    auto kc_ktn = torch::empty({K, T, N}, fopts);
    auto weight = torch::empty({K, T, D}, fopts);
    auto out = torch::empty({B, D, T}, h.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_check_cublas(
        cublasSetStream(handle, stream),
        "fdc_new_forward_gemm_nk3_d256 cublasSetStream failed"
    );

    int threads = 256;

    fdc_new_make_mix_t_float_kernel<scalar_t><<<
        (D * N + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        mix.data_ptr<scalar_t>(),
        mix_t.data_ptr<float>(),
        D,
        N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_new_make_kc_ktn_float_kernel<scalar_t><<<
        (K * T * N + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        kc.data_ptr<scalar_t>(),
        kc_ktn.data_ptr<float>(),
        T,
        N,
        K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float alpha = 1.0f;
    float beta0 = 0.0f;

    // Compute weight[kk] = kc_kk[T,N] x mix_t[N,D] => [T,D].
    //
    // Row-major [T,D] is represented to cuBLAS as column-major [D,T].
    // C_col[D,T] = mix_t_col[D,N] x kc_col[N,T].
    //
    // mix_t memory [N,D] row-major equals column-major [D,N] with ld=D.
    // kc_ktn memory [T,N] row-major equals column-major [N,T] with ld=N.
    // weight memory [T,D] row-major equals column-major [D,T] with ld=D.
    fdc_check_cublas(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            D,
            T,
            N,
            &alpha,
            mix_t.data_ptr<float>(),
            D,
            0,
            kc_ktn.data_ptr<float>(),
            N,
            static_cast<long long>(T) * N,
            &beta0,
            weight.data_ptr<float>(),
            D,
            static_cast<long long>(T) * D,
            K
        ),
        "fdc_new_forward_gemm_nk3_d256 weight SGEMM failed"
    );

    fdc_new_forward_from_weight_kernel<scalar_t><<<
        (D * T + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        h.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        out.data_ptr<scalar_t>(),
        D,
        L,
        T,
        K,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor fdc_new_forward_gemm_nk3_d256_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    TORCH_CHECK(h.is_cuda(), "h must be CUDA tensor.");
    TORCH_CHECK(kc.is_cuda(), "kc must be CUDA tensor.");
    TORCH_CHECK(mix.is_cuda(), "mix must be CUDA tensor.");

    TORCH_CHECK(h.is_contiguous(), "h must be contiguous.");
    TORCH_CHECK(kc.is_contiguous(), "kc must be contiguous.");
    TORCH_CHECK(mix.is_contiguous(), "mix must be contiguous.");

    TORCH_CHECK(h.dim() == 3, "h must be [B,D,L].");
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,T,N,K].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(h.size(0) == 1, "B must be 1.");
    TORCH_CHECK(h.size(1) == 256, "D must be 256.");
    TORCH_CHECK(kc.size(0) == 1, "kc B must be 1.");
    TORCH_CHECK(kc.size(3) == 3, "K must be 3.");
    TORCH_CHECK(mix.size(0) == 256, "mix D must be 256.");
    TORCH_CHECK(kc.size(2) == mix.size(1), "N mismatch.");
    TORCH_CHECK(kc.size(2) == 6 || kc.size(2) == 16, "N must be 6 or 16.");

    TORCH_CHECK(h.scalar_type() == kc.scalar_type(), "h/kc dtype mismatch.");
    TORCH_CHECK(h.scalar_type() == mix.scalar_type(), "h/mix dtype mismatch.");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_new_forward_gemm_nk3_d256_typed<float>(
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_new_forward_gemm_nk3_d256_typed<c10::Half>(
            h,
            kc,
            mix,
            off
        );
    }

    TORCH_CHECK(
        h.scalar_type() == at::ScalarType::BFloat16,
        "unsupported dtype"
    );

    return fdc_new_forward_gemm_nk3_d256_typed<c10::BFloat16>(
        h,
        kc,
        mix,
        off
    );
}

// ======================================================================================
// Backward GEMM materialized plan
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> fdc_new_backward_gemm_nk3_d256_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_new_gemm_nk3_d256");

    int B = static_cast<int>(h.size(0));
    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));
    int K = static_cast<int>(kc.size(3));

    TORCH_CHECK(B == 1, "fdc_new_backward_gemm_nk3_d256 requires B == 1.");
    TORCH_CHECK(D == 256, "fdc_new_backward_gemm_nk3_d256 requires D == 256.");
    TORCH_CHECK(K == 3, "fdc_new_backward_gemm_nk3_d256 requires K == 3.");
    TORCH_CHECK(N == 6 || N == 16, "fdc_new_backward_gemm_nk3_d256 requires N == 6 or 16.");

    auto fopts = h.options().dtype(torch::kFloat32);

    auto mix_t = torch::empty({N, D}, fopts);
    auto kc_ktn = torch::empty({K, T, N}, fopts);
    auto base = torch::empty({K, T, D}, fopts);
    auto weight = torch::empty({K, T, D}, fopts);
    auto gk_ktn = torch::empty({K, T, N}, fopts);
    auto gmf = torch::zeros({D, N}, fopts);
    auto ghf = torch::empty({B, D, L}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_check_cublas(
        cublasSetStream(handle, stream),
        "fdc_new_backward_gemm_nk3_d256 cublasSetStream failed"
    );

    int threads = 256;

    fdc_new_make_mix_t_float_kernel<scalar_t><<<
        (D * N + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        mix.data_ptr<scalar_t>(),
        mix_t.data_ptr<float>(),
        D,
        N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_new_make_kc_ktn_float_kernel<scalar_t><<<
        (K * T * N + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        kc.data_ptr<scalar_t>(),
        kc_ktn.data_ptr<float>(),
        T,
        N,
        K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_new_make_base_ktd_float_kernel<scalar_t><<<
        (static_cast<int64_t>(K) * T * D + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        base.data_ptr<float>(),
        D,
        L,
        T,
        K,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    // ------------------------------------------------------------------
    // grad_kernel:
    //
    // gk_kk[T,N] = base_kk[T,D] x mix[D,N]
    //
    // Row-major gk[T,N] as column-major [N,T].
    // C_col[N,T] = mix_t_col[N,D] x base_col[D,T].
    // mix_t memory [N,D] row-major -> column-major [D,N]? 
    //
    // We need C [N,T].
    // A should be [N,D], B should be [D,T].
    //
    // mix_t is row-major [N,D], equivalent column-major [D,N].
    // To see it as [N,D] column-major, use transpose over [D,N].
    //
    // Easier:
    // gk^T[N,T] = mix^T[N,D] x base^T[D,T].
    //
    // mix_t memory stores mix^T row-major [N,D].
    // In cuBLAS column-major, that same memory is [D,N].
    // Therefore use CUBLAS_OP_T on mix_t with dimensions D x N to get N x D.
    //
    // base row-major [T,D] is column-major [D,T], no transpose.
    // ------------------------------------------------------------------
    fdc_check_cublas(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N,
            T,
            D,
            &alpha,
            mix_t.data_ptr<float>(),
            D,
            0,
            base.data_ptr<float>(),
            D,
            static_cast<long long>(T) * D,
            &beta0,
            gk_ktn.data_ptr<float>(),
            N,
            static_cast<long long>(T) * N,
            K
        ),
        "fdc_new_backward_gemm_nk3_d256 grad_kernel SGEMM failed"
    );

    // ------------------------------------------------------------------
    // weight for grad_h:
    //
    // weight_kk[T,D] = kc_kk[T,N] x mix_t[N,D]
    // same as forward materialization.
    // ------------------------------------------------------------------
    fdc_check_cublas(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            D,
            T,
            N,
            &alpha,
            mix_t.data_ptr<float>(),
            D,
            0,
            kc_ktn.data_ptr<float>(),
            N,
            static_cast<long long>(T) * N,
            &beta0,
            weight.data_ptr<float>(),
            D,
            static_cast<long long>(T) * D,
            K
        ),
        "fdc_new_backward_gemm_nk3_d256 weight SGEMM failed"
    );

    // ------------------------------------------------------------------
    // grad_mix:
    //
    // gm[D,N] += base_kk[T,D]^T x kc_kk[T,N]
    //
    // Row-major gm[D,N] equivalent column-major [N,D].
    // Compute gm_col[N,D] += kc_col[N,T] x base_col[T,D].
    //
    // kc_ktn row-major [T,N] equals column-major [N,T], no transpose.
    // base row-major [T,D] equals column-major [D,T].
    // Need base_col[T,D], so use CUBLAS_OP_T over [D,T].
    // ------------------------------------------------------------------
    for (int kk = 0; kk < K; ++kk) {
        fdc_check_cublas(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                N,
                D,
                T,
                &alpha,
                kc_ktn.data_ptr<float>() + static_cast<int64_t>(kk) * T * N,
                N,
                base.data_ptr<float>() + static_cast<int64_t>(kk) * T * D,
                D,
                &beta1,
                gmf.data_ptr<float>(),
                N
            ),
            "fdc_new_backward_gemm_nk3_d256 grad_mix SGEMM failed"
        );
    }

    fdc_new_grad_h_from_weight_kernel<scalar_t><<<
        (D * L + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        ghf.data_ptr<float>(),
        D,
        L,
        T,
        K,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto gh = torch::empty_like(h);
    auto gk = torch::empty_like(kc);
    auto gm = torch::empty_like(mix);

    fdc_new_cast_float_to_scalar_kernel<scalar_t><<<
        (gh.numel() + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        ghf.data_ptr<float>(),
        gh.data_ptr<scalar_t>(),
        gh.numel()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_new_scatter_gk_ktn_to_btnk_kernel<scalar_t><<<
        (T * N * K + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        gk_ktn.data_ptr<float>(),
        gk.data_ptr<scalar_t>(),
        T,
        N,
        K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_new_scatter_gm_float_to_scalar_kernel<scalar_t><<<
        (D * N + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        gmf.data_ptr<float>(),
        gm.data_ptr<scalar_t>(),
        D,
        N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        gh,
        gk,
        gm
    };
}

std::vector<torch::Tensor> fdc_new_backward_gemm_nk3_d256_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
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
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,T,N,K].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(h.size(0) == 1, "B must be 1.");
    TORCH_CHECK(h.size(1) == 256, "D must be 256.");
    TORCH_CHECK(kc.size(0) == 1, "kc B must be 1.");
    TORCH_CHECK(kc.size(3) == 3, "K must be 3.");
    TORCH_CHECK(mix.size(0) == 256, "mix D must be 256.");
    TORCH_CHECK(kc.size(2) == mix.size(1), "N mismatch.");
    TORCH_CHECK(kc.size(2) == 6 || kc.size(2) == 16, "N must be 6 or 16.");

    TORCH_CHECK(go.size(0) == h.size(0), "go B mismatch.");
    TORCH_CHECK(go.size(1) == h.size(1), "go D mismatch.");
    TORCH_CHECK(go.size(2) == kc.size(1), "go T mismatch.");

    TORCH_CHECK(h.scalar_type() == go.scalar_type(), "go dtype mismatch.");
    TORCH_CHECK(h.scalar_type() == kc.scalar_type(), "kc dtype mismatch.");
    TORCH_CHECK(h.scalar_type() == mix.scalar_type(), "mix dtype mismatch.");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_new_backward_gemm_nk3_d256_typed<float>(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_new_backward_gemm_nk3_d256_typed<c10::Half>(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    TORCH_CHECK(
        h.scalar_type() == at::ScalarType::BFloat16,
        "unsupported dtype"
    );

    return fdc_new_backward_gemm_nk3_d256_typed<c10::BFloat16>(
        go,
        h,
        kc,
        mix,
        off
    );
}

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
    if (!h.is_cuda()) {
        return false;
    }

    if (!kc.is_cuda()) {
        return false;
    }

    if (!mix.is_cuda()) {
        return false;
    }

    if (!h.is_contiguous()) {
        return false;
    }

    if (!kc.is_contiguous()) {
        return false;
    }

    if (!mix.is_contiguous()) {
        return false;
    }

    if (h.dim() != 3) {
        return false;
    }

    if (kc.dim() != 4) {
        return false;
    }

    if (mix.dim() != 2) {
        return false;
    }

    if (
        h.scalar_type() != at::ScalarType::Float &&
        h.scalar_type() != at::ScalarType::Half &&
        h.scalar_type() != at::ScalarType::BFloat16
    ) {
        return false;
    }

    if (kc.scalar_type() != h.scalar_type()) {
        return false;
    }

    if (mix.scalar_type() != h.scalar_type()) {
        return false;
    }

    if (h.size(0) != 1) {
        return false;
    }

    if (h.size(1) != 256) {
        return false;
    }

    if (kc.size(0) != 1) {
        return false;
    }

    if (kc.size(3) != 3) {
        return false;
    }

    if (mix.size(0) != 256) {
        return false;
    }

    if (kc.size(2) != mix.size(1)) {
        return false;
    }

    if (kc.size(2) != 6 && kc.size(2) != 16) {
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
