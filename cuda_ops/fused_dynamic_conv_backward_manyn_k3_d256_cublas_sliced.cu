#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Backward many-N K3 D256 raw cuBLAS sliced plan
//
// kc layout is now:
//
//   kc: [1,3,N,T]
//
// For each kk:
//
//   x_dt[d,t] = go[d,t] * h[d, off+t-kk]
//   x_td[t,d] = x_dt[d,t]
//
//   kc_tn[t,n] = kc[0,kk,n,t]
//   kc_nt[n,t] = kc[0,kk,n,t]
//
//   w[d,t]    = mix[d,n] @ kc_nt[n,t]
//   gh[d,s]  += go[d,t] * w[d,t], where s=off+t-kk
//
//   gk[0,kk,n,t] = x_td[t,d] @ mix[d,n]
//   gm[d,n]     += x_dt[d,t] @ kc_tn[t,n]
// ======================================================================================

static inline void fdc_check_cublas_sliced(cublasStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg, " cublas status=", static_cast<int>(status));
}

static void fdc_sgemm_rowmajor_sliced(
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

    fdc_check_cublas_sliced(
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
        "cublasSgemm rowmajor sliced failed."
    );
}

__global__ void fdc_manyn_sliced_zero_float_kernel(
    float* __restrict__ ptr,
    int64_t total
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < total) {
        ptr[i] = 0.0f;
    }
}

__global__ void fdc_manyn_sliced_build_x_float_kernel(
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
    int64_t total = static_cast<int64_t>(256) * T;

    if (idx >= total) {
        return;
    }

    int t = static_cast<int>(idx % T);
    int d = static_cast<int>(idx / T);

    int s = off + t - kk;

    float v = 0.0f;

    if (s >= 0 && s < L) {
        v =
            go[static_cast<int64_t>(d) * T + t] *
            h[static_cast<int64_t>(d) * L + s];
    }

    x_dt[static_cast<int64_t>(d) * T + t] = v;
    x_td[static_cast<int64_t>(t) * 256 + d] = v;
}

template <typename scalar_t>
__global__ void fdc_manyn_sliced_build_x_typed_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    float* __restrict__ x_dt,
    float* __restrict__ x_td,
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

    float v = 0.0f;

    if (s >= 0 && s < L) {
        v =
            fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) *
            fdc_to_float_dev(h[static_cast<int64_t>(d) * L + s]);
    }

    x_dt[static_cast<int64_t>(d) * T + t] = v;
    x_td[static_cast<int64_t>(t) * 256 + d] = v;
}

__global__ void fdc_manyn_sliced_build_kc_float_kernel(
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

    int n = static_cast<int>(idx % N);
    int t = static_cast<int>(idx / N);

    float v = kc[(static_cast<int64_t>(kk) * N + n) * T + t];

    kc_tn[static_cast<int64_t>(t) * N + n] = v;
    kc_nt[static_cast<int64_t>(n) * T + t] = v;
}

template <typename scalar_t>
__global__ void fdc_manyn_sliced_build_kc_typed_kernel(
    const scalar_t* __restrict__ kc,
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

    int n = static_cast<int>(idx % N);
    int t = static_cast<int>(idx / N);

    float v = fdc_to_float_dev(
        kc[(static_cast<int64_t>(kk) * N + n) * T + t]
    );

    kc_tn[static_cast<int64_t>(t) * N + n] = v;
    kc_nt[static_cast<int64_t>(n) * T + t] = v;
}

__global__ void fdc_manyn_sliced_accum_gh_float_kernel(
    const float* __restrict__ go,
    const float* __restrict__ w,
    float* __restrict__ gh,
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
        gh[static_cast<int64_t>(d) * L + s] +=
            go[static_cast<int64_t>(d) * T + t] *
            w[static_cast<int64_t>(d) * T + t];
    }
}

template <typename scalar_t>
__global__ void fdc_manyn_sliced_accum_gh_typed_kernel(
    const scalar_t* __restrict__ go,
    const float* __restrict__ w,
    float* __restrict__ gh,
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
        gh[static_cast<int64_t>(d) * L + s] +=
            fdc_to_float_dev(go[static_cast<int64_t>(d) * T + t]) *
            w[static_cast<int64_t>(d) * T + t];
    }
}

__global__ void fdc_manyn_sliced_write_gk_float_kernel(
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

    int n = static_cast<int>(idx % N);
    int t = static_cast<int>(idx / N);

    gk[(static_cast<int64_t>(kk) * N + n) * T + t] =
        gk_tn[static_cast<int64_t>(t) * N + n];
}

template <typename scalar_t>
__global__ void fdc_manyn_sliced_cast_float_to_scalar_kernel(
    const float* __restrict__ src,
    scalar_t* __restrict__ dst,
    int64_t total
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < total) {
        dst[i] = fdc_from_float_dev<scalar_t>(src[i]);
    }
}

static std::vector<torch::Tensor> fdc_backward_manyn_k3_d256_cublas_sliced_float(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_manyn_k3_d256_cublas_sliced_float_bknt");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(3));
    int N = static_cast<int>(kc.size(2));

    auto fopts = h.options().dtype(torch::kFloat32);

    auto gh = torch::empty(
        h.sizes(),
        fopts
    );

    auto gk = torch::empty(
        kc.sizes(),
        fopts
    );

    auto gm = torch::empty(
        mix.sizes(),
        fopts
    );

    auto x_dt = torch::empty(
        {
            256,
            T,
        },
        fopts
    );

    auto x_td = torch::empty(
        {
            T,
            256,
        },
        fopts
    );

    auto kc_tn = torch::empty(
        {
            T,
            N,
        },
        fopts
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

    auto gk_tn = torch::empty(
        {
            T,
            N,
        },
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fdc_manyn_sliced_zero_float_kernel<<<
        (static_cast<int64_t>(256) * L + 255) / 256,
        256,
        0,
        stream
    >>>(
        gh.data_ptr<float>(),
        static_cast<int64_t>(256) * L
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_manyn_sliced_zero_float_kernel<<<
        (static_cast<int64_t>(256) * N + 255) / 256,
        256,
        0,
        stream
    >>>(
        gm.data_ptr<float>(),
        static_cast<int64_t>(256) * N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_check_cublas_sliced(
        cublasSetStream(handle, stream),
        "cublasSetStream failed."
    );

    for (int kk = 0; kk < 3; ++kk) {
        fdc_manyn_sliced_build_x_float_kernel<<<
            (static_cast<int64_t>(256) * T + 255) / 256,
            256,
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

        fdc_manyn_sliced_build_kc_float_kernel<<<
            (static_cast<int64_t>(T) * N + 255) / 256,
            256,
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

        fdc_sgemm_rowmajor_sliced(
            handle,
            mix.data_ptr<float>(),
            kc_nt.data_ptr<float>(),
            w.data_ptr<float>(),
            256,
            T,
            N,
            0.0f
        );

        fdc_manyn_sliced_accum_gh_float_kernel<<<
            (static_cast<int64_t>(256) * T + 255) / 256,
            256,
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

        fdc_sgemm_rowmajor_sliced(
            handle,
            x_td.data_ptr<float>(),
            mix.data_ptr<float>(),
            gk_tn.data_ptr<float>(),
            T,
            N,
            256,
            0.0f
        );

        fdc_manyn_sliced_write_gk_float_kernel<<<
            (static_cast<int64_t>(T) * N + 255) / 256,
            256,
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

        fdc_sgemm_rowmajor_sliced(
            handle,
            x_dt.data_ptr<float>(),
            kc_tn.data_ptr<float>(),
            gm.data_ptr<float>(),
            256,
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

template <typename scalar_t>
static std::vector<torch::Tensor> fdc_backward_manyn_k3_d256_cublas_sliced_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_manyn_k3_d256_cublas_sliced_typed_bknt");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(3));
    int N = static_cast<int>(kc.size(2));

    auto fopts = h.options().dtype(torch::kFloat32);

    auto mixf = mix.to(torch::kFloat32);

    auto ghf = torch::empty(
        h.sizes(),
        fopts
    );

    auto gkf = torch::empty(
        kc.sizes(),
        fopts
    );

    auto gmf = torch::empty(
        mix.sizes(),
        fopts
    );

    auto x_dt = torch::empty(
        {
            256,
            T,
        },
        fopts
    );

    auto x_td = torch::empty(
        {
            T,
            256,
        },
        fopts
    );

    auto kc_tn = torch::empty(
        {
            T,
            N,
        },
        fopts
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

    auto gk_tn = torch::empty(
        {
            T,
            N,
        },
        fopts
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fdc_manyn_sliced_zero_float_kernel<<<
        (static_cast<int64_t>(256) * L + 255) / 256,
        256,
        0,
        stream
    >>>(
        ghf.data_ptr<float>(),
        static_cast<int64_t>(256) * L
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_manyn_sliced_zero_float_kernel<<<
        (static_cast<int64_t>(256) * N + 255) / 256,
        256,
        0,
        stream
    >>>(
        gmf.data_ptr<float>(),
        static_cast<int64_t>(256) * N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_check_cublas_sliced(
        cublasSetStream(handle, stream),
        "cublasSetStream failed."
    );

    for (int kk = 0; kk < 3; ++kk) {
        fdc_manyn_sliced_build_x_typed_kernel<scalar_t><<<
            (static_cast<int64_t>(256) * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            go.data_ptr<scalar_t>(),
            h.data_ptr<scalar_t>(),
            x_dt.data_ptr<float>(),
            x_td.data_ptr<float>(),
            L,
            T,
            static_cast<int>(off),
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_manyn_sliced_build_kc_typed_kernel<scalar_t><<<
            (static_cast<int64_t>(T) * N + 255) / 256,
            256,
            0,
            stream
        >>>(
            kc.data_ptr<scalar_t>(),
            kc_tn.data_ptr<float>(),
            kc_nt.data_ptr<float>(),
            T,
            N,
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_sgemm_rowmajor_sliced(
            handle,
            mixf.data_ptr<float>(),
            kc_nt.data_ptr<float>(),
            w.data_ptr<float>(),
            256,
            T,
            N,
            0.0f
        );

        fdc_manyn_sliced_accum_gh_typed_kernel<scalar_t><<<
            (static_cast<int64_t>(256) * T + 255) / 256,
            256,
            0,
            stream
        >>>(
            go.data_ptr<scalar_t>(),
            w.data_ptr<float>(),
            ghf.data_ptr<float>(),
            L,
            T,
            static_cast<int>(off),
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_sgemm_rowmajor_sliced(
            handle,
            x_td.data_ptr<float>(),
            mixf.data_ptr<float>(),
            gk_tn.data_ptr<float>(),
            T,
            N,
            256,
            0.0f
        );

        fdc_manyn_sliced_write_gk_float_kernel<<<
            (static_cast<int64_t>(T) * N + 255) / 256,
            256,
            0,
            stream
        >>>(
            gk_tn.data_ptr<float>(),
            gkf.data_ptr<float>(),
            T,
            N,
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_sgemm_rowmajor_sliced(
            handle,
            x_dt.data_ptr<float>(),
            kc_tn.data_ptr<float>(),
            gmf.data_ptr<float>(),
            256,
            N,
            T,
            kk == 0 ? 0.0f : 1.0f
        );
    }

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

    fdc_manyn_sliced_cast_float_to_scalar_kernel<scalar_t><<<
        (ghf.numel() + 255) / 256,
        256,
        0,
        stream
    >>>(
        ghf.data_ptr<float>(),
        gh.data_ptr<scalar_t>(),
        ghf.numel()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_manyn_sliced_cast_float_to_scalar_kernel<scalar_t><<<
        (gkf.numel() + 255) / 256,
        256,
        0,
        stream
    >>>(
        gkf.data_ptr<float>(),
        gk.data_ptr<scalar_t>(),
        gkf.numel()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    fdc_manyn_sliced_cast_float_to_scalar_kernel<scalar_t><<<
        (gmf.numel() + 255) / 256,
        256,
        0,
        stream
    >>>(
        gmf.data_ptr<float>(),
        gm.data_ptr<scalar_t>(),
        gmf.numel()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        gh,
        gk,
        gm
    };
}

bool fdc_backward_manyn_k3_d256_cublas_sliced_available_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    if (!go.is_cuda() || !h.is_cuda() || !kc.is_cuda() || !mix.is_cuda()) {
        return false;
    }

    if (go.dim() != 3 || h.dim() != 3 || kc.dim() != 4 || mix.dim() != 2) {
        return false;
    }

    if (!go.is_contiguous() || !h.is_contiguous() || !kc.is_contiguous() || !mix.is_contiguous()) {
        return false;
    }

    if (h.size(0) != 1 || go.size(0) != 1 || kc.size(0) != 1) {
        return false;
    }

    if (h.size(1) != 256 || go.size(1) != 256 || mix.size(0) != 256) {
        return false;
    }

    if (kc.size(1) != 3) {
        return false;
    }

    if (mix.size(1) != kc.size(2)) {
        return false;
    }

    if (go.size(2) != kc.size(3)) {
        return false;
    }

    if (dilation != 1) {
        return false;
    }

    if (kc.size(2) < 8 || kc.size(2) > 256) {
        return false;
    }

    if (kc.size(3) < 1024) {
        return false;
    }

    if (off < 0) {
        return false;
    }

    if (off + kc.size(3) > h.size(2)) {
        return false;
    }

    if (h.scalar_type() != go.scalar_type()) {
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

    return true;
}

std::vector<torch::Tensor> fdc_backward_manyn_k3_d256_cublas_sliced_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    TORCH_CHECK(h.dim() == 3, "h must be [B,D,L].");
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,K,N,T].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");
    TORCH_CHECK(go.dim() == 3, "go must be [B,D,T].");

    TORCH_CHECK(h.size(0) == 1, "sliced manyn requires B == 1.");
    TORCH_CHECK(h.size(1) == 256, "sliced manyn requires D == 256.");
    TORCH_CHECK(go.size(1) == 256, "sliced manyn requires go D == 256.");
    TORCH_CHECK(kc.size(1) == 3, "sliced manyn requires K == 3.");
    TORCH_CHECK(mix.size(0) == 256, "sliced manyn requires mix D == 256.");
    TORCH_CHECK(mix.size(1) == kc.size(2), "mix N mismatch.");
    TORCH_CHECK(go.size(2) == kc.size(3), "go T mismatch.");

    if (h.scalar_type() == at::ScalarType::Float) {
        return fdc_backward_manyn_k3_d256_cublas_sliced_float(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    if (h.scalar_type() == at::ScalarType::Half) {
        return fdc_backward_manyn_k3_d256_cublas_sliced_typed<c10::Half>(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    TORCH_CHECK(
        h.scalar_type() == at::ScalarType::BFloat16,
        "backward_manyn_k3_d256_cublas_sliced supports float32, float16, bfloat16."
    );

    return fdc_backward_manyn_k3_d256_cublas_sliced_typed<c10::BFloat16>(
        go,
        h,
        kc,
        mix,
        off
    );
}
