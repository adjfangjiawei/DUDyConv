#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

namespace {

constexpr int FDC_FWD_N256_D = 256;
constexpr int FDC_FWD_N256_N = 256;
constexpr int FDC_FWD_N256_K = 3;

static inline void fdc_forward_n256_check_cublas_status(
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

static inline bool fdc_forward_n256_shape_ok(
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

    if (h.size(1) != FDC_FWD_N256_D) {
        return false;
    }

    if (mix.size(0) != FDC_FWD_N256_D) {
        return false;
    }

    if (kc.size(2) != FDC_FWD_N256_N) {
        return false;
    }

    if (mix.size(1) != FDC_FWD_N256_N) {
        return false;
    }

    if (kc.size(3) != FDC_FWD_N256_K) {
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

__global__ void fdc_forward_n256_pack_kc_kernel(
    const float* __restrict__ kc,
    float* __restrict__ kc_pack,
    int T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_N256_K * FDC_FWD_N256_N * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int q = idx / T;
    int n = q % FDC_FWD_N256_N;
    int k = q / FDC_FWD_N256_N;

    int64_t src = ((int64_t)t * FDC_FWD_N256_N + n) * FDC_FWD_N256_K + k;
    int64_t dst = ((int64_t)k * FDC_FWD_N256_N + n) * T + t;

    kc_pack[dst] = kc[src];
}

__global__ void fdc_forward_n256_combine_noboundary_kernel(
    const float* __restrict__ h,
    const float* __restrict__ tmp,
    float* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_N256_D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int d = idx / T;

    int64_t ht = (int64_t)d * L + off + t;
    int64_t ot = (int64_t)d * T + t;

    int64_t tmp0 = (int64_t)0 * FDC_FWD_N256_D * T + ot;
    int64_t tmp1 = (int64_t)1 * FDC_FWD_N256_D * T + ot;
    int64_t tmp2 = (int64_t)2 * FDC_FWD_N256_D * T + ot;

    out[ot] =
        tmp[tmp0] * h[ht] +
        tmp[tmp1] * h[ht - 1] +
        tmp[tmp2] * h[ht - 2];
}

__global__ void fdc_forward_n256_combine_boundary_kernel(
    const float* __restrict__ h,
    const float* __restrict__ tmp,
    float* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = FDC_FWD_N256_D * T;

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
        int64_t tmp0 = (int64_t)0 * FDC_FWD_N256_D * T + ot;
        acc += tmp[tmp0] * h[(int64_t)d * L + s0];
    }

    if (s1 >= 0 && s1 < L) {
        int64_t tmp1 = (int64_t)1 * FDC_FWD_N256_D * T + ot;
        acc += tmp[tmp1] * h[(int64_t)d * L + s1];
    }

    if (s2 >= 0 && s2 < L) {
        int64_t tmp2 = (int64_t)2 * FDC_FWD_N256_D * T + ot;
        acc += tmp[tmp2] * h[(int64_t)d * L + s2];
    }

    out[ot] = acc;
}

} // namespace

bool fdc_forward_n256_k3_d256_available_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    if (!fdc_forward_n256_shape_ok(
            h,
            kc,
            mix,
            off,
            dilation
        )) {
        return false;
    }

    int64_t T = kc.size(1);

    if (T < 512) {
        return false;
    }

    return true;
}

torch::Tensor fdc_forward_n256_k3_d256_cuda(
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

    TORCH_CHECK(h.size(0) == 1, "forward_n256_k3_d256 requires B == 1.");
    TORCH_CHECK(kc.size(0) == 1, "forward_n256_k3_d256 requires kc B == 1.");
    TORCH_CHECK(h.size(1) == FDC_FWD_N256_D, "forward_n256_k3_d256 requires D == 256.");
    TORCH_CHECK(mix.size(0) == FDC_FWD_N256_D, "forward_n256_k3_d256 requires mix D == 256.");
    TORCH_CHECK(kc.size(2) == FDC_FWD_N256_N, "forward_n256_k3_d256 requires N == 256.");
    TORCH_CHECK(mix.size(1) == FDC_FWD_N256_N, "forward_n256_k3_d256 requires mix N == 256.");
    TORCH_CHECK(kc.size(3) == FDC_FWD_N256_K, "forward_n256_k3_d256 requires K == 3.");
    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + kc.size(1) <= h.size(2), "off + T must be <= L.");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));

    auto out = torch::empty(
        {1, FDC_FWD_N256_D, T},
        h.options()
    );

    auto kc_pack = torch::empty(
        {FDC_FWD_N256_K, FDC_FWD_N256_N, T},
        h.options()
    );

    auto tmp = torch::empty(
        {FDC_FWD_N256_K, FDC_FWD_N256_D, T},
        h.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int pack_total = FDC_FWD_N256_K * FDC_FWD_N256_N * T;
    int pack_block = 256;
    int pack_grid = (pack_total + pack_block - 1) / pack_block;

    fdc_forward_n256_pack_kc_kernel<<<
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

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_forward_n256_check_cublas_status(
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

#pragma unroll
    for (int kk = 0; kk < FDC_FWD_N256_K; ++kk) {
        const float* A = kc_pack_ptr + (int64_t)kk * FDC_FWD_N256_N * T;
        const float* B = mix_ptr;
        float* C = tmp_ptr + (int64_t)kk * FDC_FWD_N256_D * T;

        fdc_forward_n256_check_cublas_status(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                T,
                FDC_FWD_N256_D,
                FDC_FWD_N256_N,
                &alpha,
                A,
                T,
                B,
                FDC_FWD_N256_N,
                &beta,
                C,
                T
            ),
            "cublasSgemm failed in forward_n256_k3_d256"
        );
    }

    int combine_total = FDC_FWD_N256_D * T;
    int combine_block = 256;
    int combine_grid = (combine_total + combine_block - 1) / combine_block;

    if (static_cast<int>(off) >= 2) {
        fdc_forward_n256_combine_noboundary_kernel<<<
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
        fdc_forward_n256_combine_boundary_kernel<<<
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

    FDC_DEBUG_PATH("forward_n256_k3_d256_sgemm3_pack_combine");

    return out;
}
