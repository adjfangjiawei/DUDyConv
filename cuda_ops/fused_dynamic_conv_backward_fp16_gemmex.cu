#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Local helpers
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_fp16_gemmex_cast_float_to_scalar_kernel(
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
__global__ void fdc_fp16_gemmex_make_mix_half_kernel(
    const scalar_t* __restrict__ mix,
    __half* __restrict__ mixh,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total) {
        mixh[i] = __float2half_rn(fdc_to_float_dev(mix[i]));
    }
}

template <typename scalar_t>
__global__ void fdc_fp16_gemmex_make_base_half_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    __half* __restrict__ base,
    __half* __restrict__ kbtn,
    int B,
    int D,
    int L,
    int T,
    int N,
    int K,
    int kk,
    int off,
    int dilation
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t total_base = static_cast<int64_t>(B) * T * D;
    int64_t total_k = static_cast<int64_t>(B) * T * N;
    int64_t total = total_base > total_k ? total_base : total_k;

    if (idx >= total) {
        return;
    }

    if (idx < total_base) {
        int d = idx % D;
        int64_t q = idx / D;
        int t = q % T;
        int b = q / T;

        int s = off + t - kk * dilation;

        float v = 0.0f;

        if (s >= 0 && s < L) {
            v =
                fdc_to_float_dev(go[(static_cast<int64_t>(b) * D + d) * T + t]) *
                fdc_to_float_dev(h[(static_cast<int64_t>(b) * D + d) * L + s]);
        }

        base[(static_cast<int64_t>(b) * T + t) * D + d] =
            __float2half_rn(v);
    }

    if (idx < total_k) {
        int n = idx % N;
        int64_t q = idx / N;
        int t = q % T;
        int b = q / T;

        float v = fdc_to_float_dev(
            kc[((static_cast<int64_t>(b) * K + kk) * N + n) * T + t]
        );

        kbtn[(static_cast<int64_t>(b) * T + t) * N + n] =
            __float2half_rn(v);
    }
}

template <typename scalar_t>
__global__ void fdc_fp16_gemmex_copy_gk_float_to_scalar_kernel(
    const float* __restrict__ src,
    scalar_t* __restrict__ dst,
    int B,
    int T,
    int N,
    int K,
    int kk
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * N;

    if (i >= total) {
        return;
    }

    int n = i % N;
    int q = i / N;
    int t = q % T;
    int b = q / T;

    dst[((static_cast<int64_t>(b) * K + kk) * N + n) * T + t] =
        fdc_from_float_dev<scalar_t>(src[i]);
}

// ======================================================================================
// grad_h gather for fp16 GemmEx plan
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_fp16_gemmex_grad_h_gather_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gh,
    int B,
    int D,
    int L,
    int T,
    int N,
    int K,
    int off
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= B || d >= D || s >= L) {
        return;
    }

    int tb = s - off;

    float acc = 0.0f;

    int64_t mbase = static_cast<int64_t>(d) * N;
    int64_t gobase = (static_cast<int64_t>(b) * D + d) * T;

    for (int kk = 0; kk < K; ++kk) {
        int t = tb + kk;

        if (t < 0 || t >= T) {
            continue;
        }

        float w = 0.0f;

        for (int n = 0; n < N; ++n) {
            w +=
                fdc_to_float_dev(
                    kc[((static_cast<int64_t>(b) * K + kk) * N + n) * T + t]
                ) *
                fdc_to_float_dev(mix[mbase + n]);
        }

        acc += fdc_to_float_dev(go[gobase + t]) * w;
    }

    gh[(static_cast<int64_t>(b) * D + d) * L + s] = acc;
}

// ======================================================================================
// Public FP16 GemmEx v3 plan
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_fp16_gemmex_v3_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FDC path: backward_fp16_gemmex_v3_bknt");

    TORCH_CHECK(
        h.scalar_type() == at::ScalarType::Half,
        "fdc_backward_fp16_gemmex_v3_cuda requires fp16 input"
    );

    int B = static_cast<int>(h.size(0));
    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int K = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));
    int T = static_cast<int>(kc.size(3));

    auto fopts = h.options().dtype(torch::kFloat32);
    auto hopts = h.options().dtype(torch::kFloat16);

    auto ghf = torch::empty(h.sizes(), fopts);
    auto gk = torch::empty_like(kc);
    auto gmf = torch::zeros(mix.sizes(), fopts);

    auto mixh = torch::empty({D, N}, hopts);
    auto baseh = torch::empty({B, T, D}, hopts);
    auto kbtnh = torch::empty({B, T, N}, hopts);
    auto gkbtnf = torch::empty({B, T, N}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_check_cublas(
        cublasSetStream(handle, stream),
        "fdc_backward_fp16_gemmex_v3_cuda cublasSetStream failed"
    );

    int threads = 256;

    fdc_fp16_gemmex_make_mix_half_kernel<c10::Half><<<
        (D * N + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        mix.data_ptr<c10::Half>(),
        reinterpret_cast<__half*>(mixh.data_ptr<c10::Half>()),
        D * N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 block_h(16, 16);
    dim3 grid_h(
        (L + 15) / 16,
        (D + 15) / 16,
        B
    );

    fdc_fp16_gemmex_grad_h_gather_kernel<c10::Half><<<
        grid_h,
        block_h,
        0,
        stream
    >>>(
        go.data_ptr<c10::Half>(),
        kc.data_ptr<c10::Half>(),
        mix.data_ptr<c10::Half>(),
        ghf.data_ptr<float>(),
        B,
        D,
        L,
        T,
        N,
        K,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    for (int kk = 0; kk < K; ++kk) {
        int64_t total_base = static_cast<int64_t>(B) * T * D;
        int64_t total_k = static_cast<int64_t>(B) * T * N;
        int64_t total = total_base > total_k ? total_base : total_k;

        fdc_fp16_gemmex_make_base_half_kernel<c10::Half><<<
            (total + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            go.data_ptr<c10::Half>(),
            h.data_ptr<c10::Half>(),
            kc.data_ptr<c10::Half>(),
            reinterpret_cast<__half*>(baseh.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(kbtnh.data_ptr<c10::Half>()),
            B,
            D,
            L,
            T,
            N,
            K,
            kk,
            static_cast<int>(off),
            static_cast<int>(dilation)
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_check_cublas(
            cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                T,
                D,
                &alpha,
                mixh.data_ptr<c10::Half>(),
                CUDA_R_16F,
                N,
                0,
                baseh.data_ptr<c10::Half>(),
                CUDA_R_16F,
                D,
                static_cast<long long>(T) * D,
                &beta0,
                gkbtnf.data_ptr<float>(),
                CUDA_R_32F,
                N,
                static_cast<long long>(T) * N,
                B,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "fdc_backward_fp16_gemmex_v3_cuda grad_kernel GemmEx failed"
        );

        fdc_fp16_gemmex_copy_gk_float_to_scalar_kernel<c10::Half><<<
            (B * T * N + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gkbtnf.data_ptr<float>(),
            gk.data_ptr<c10::Half>(),
            B,
            T,
            N,
            K,
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_check_cublas(
            cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                N,
                D,
                T,
                &alpha,
                kbtnh.data_ptr<c10::Half>(),
                CUDA_R_16F,
                N,
                static_cast<long long>(T) * N,
                baseh.data_ptr<c10::Half>(),
                CUDA_R_16F,
                D,
                static_cast<long long>(T) * D,
                &beta1,
                gmf.data_ptr<float>(),
                CUDA_R_32F,
                N,
                0,
                B,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "fdc_backward_fp16_gemmex_v3_cuda grad_mix GemmEx failed"
        );
    }

    auto gh = torch::empty_like(h);
    auto gm = torch::empty_like(mix);

    fdc_fp16_gemmex_cast_float_to_scalar_kernel<c10::Half><<<
        (gh.numel() + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        ghf.data_ptr<float>(),
        gh.data_ptr<c10::Half>(),
        gh.numel()
    );

    fdc_fp16_gemmex_cast_float_to_scalar_kernel<c10::Half><<<
        (gm.numel() + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        gmf.data_ptr<float>(),
        gm.data_ptr<c10::Half>(),
        gm.numel()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        gh,
        gk,
        gm
    };
}
