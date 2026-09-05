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
// Local helpers
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_large_cast_float_to_scalar_kernel(
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
__global__ void fdc_large_cast_scalar_to_float_kernel(
    const scalar_t* __restrict__ src,
    float* __restrict__ dst,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        dst[i] = fdc_to_float_dev(src[i]);
    }
}

template <typename scalar_t>
__global__ void fdc_large_make_mix_float_kernel(
    const scalar_t* __restrict__ mix,
    float* __restrict__ mixf,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total) {
        mixf[i] = fdc_to_float_dev(mix[i]);
    }
}

template <typename scalar_t>
__global__ void fdc_large_make_base_kernel_float_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    float* __restrict__ base,
    float* __restrict__ kbtn,
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

        base[(static_cast<int64_t>(b) * T + t) * D + d] = v;
    }

    if (idx < total_k) {
        int n = idx % N;
        int64_t q = idx / N;
        int t = q % T;
        int b = q / T;

        kbtn[(static_cast<int64_t>(b) * T + t) * N + n] =
            fdc_to_float_dev(
                kc[((static_cast<int64_t>(b) * K + kk) * N + n) * T + t]
            );
    }
}

__global__ void fdc_large_copy_gk_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
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

    dst[((static_cast<int64_t>(b) * K + kk) * N + n) * T + t] = src[i];
}

// ======================================================================================
// grad_h generic gather
// ======================================================================================

template <typename scalar_t>
__global__ void fdc_large_grad_h_gather_kernel(
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
// grad_h atomic for non-gather case
// ======================================================================================

template <typename scalar_t, int BT, int BD>
__global__ void fdc_large_grad_h_atomic_kernel(
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
    int kk,
    int off,
    int dilation
) {
    int base_t = blockIdx.x * BT;
    int base_d = blockIdx.y * BD;
    int b = blockIdx.z;
    int tid = threadIdx.x;

    for (int x = tid; x < BT * BD; x += blockDim.x) {
        int ld = x % BD;
        int lt = x / BD;

        int t = base_t + lt;
        int d = base_d + ld;

        if (b >= B || t >= T || d >= D) {
            continue;
        }

        int s = off + t - kk * dilation;

        if (s < 0 || s >= L) {
            continue;
        }

        float w = 0.0f;

        int64_t mb = static_cast<int64_t>(d) * N;

        for (int n = 0; n < N; ++n) {
            w +=
                fdc_to_float_dev(
                    kc[((static_cast<int64_t>(b) * K + kk) * N + n) * T + t]
                ) *
                fdc_to_float_dev(mix[mb + n]);
        }

        atomicAdd(
            gh + (static_cast<int64_t>(b) * D + d) * L + s,
            fdc_to_float_dev(go[(static_cast<int64_t>(b) * D + d) * T + t]) * w
        );
    }
}

// ======================================================================================
// grad_h using materialized weight
// ======================================================================================

__global__ void fdc_large_grad_h_materialized_kernel(
    const float* __restrict__ gof,
    const float* __restrict__ weight,
    float* __restrict__ gh,
    int B,
    int D,
    int L,
    int T,
    int kk,
    int off,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * T;

    if (idx >= total) {
        return;
    }

    int t = idx % T;
    int q = idx / T;
    int d = q % D;
    int b = q / D;

    int s = off + t - kk * dilation;

    if (s < 0 || s >= L) {
        return;
    }

    atomicAdd(
        gh + (static_cast<int64_t>(b) * D + d) * L + s,
        gof[(static_cast<int64_t>(b) * D + d) * T + t] *
            weight[(static_cast<int64_t>(b) * T + t) * D + d]
    );
}

// ======================================================================================
// Public large plan
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_large_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FDC path: backward_large_bknt");

    int B = static_cast<int>(h.size(0));
    int D = static_cast<int>(h.size(1));
    int L = static_cast<int>(h.size(2));
    int K = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));
    int T = static_cast<int>(kc.size(3));

    cudaDeviceProp prop;
    cudaError_t prop_err = cudaGetDeviceProperties(&prop, h.get_device());
    TORCH_CHECK(prop_err == cudaSuccess, "cudaGetDeviceProperties failed");

    int sm = prop.major * 10 + prop.minor;

    bool is_bf16 = h.scalar_type() == at::ScalarType::BFloat16;
    bool materialize_h = is_bf16 && sm < 80 && N >= 16;
    bool gather_h = static_cast<int>(dilation) == 1 && !materialize_h;

    auto fopts = h.options().dtype(torch::kFloat32);

    auto ghf = torch::empty(h.sizes(), fopts);
    auto gkf = torch::empty(kc.sizes(), fopts);
    auto gmf = torch::zeros(mix.sizes(), fopts);

    auto mixf = torch::empty({D, N}, fopts);
    auto base = torch::empty({B, T, D}, fopts);
    auto kbtn = torch::empty({B, T, N}, fopts);
    auto gkbtn = torch::empty({B, T, N}, fopts);

    torch::Tensor weight;
    torch::Tensor gof;

    if (materialize_h) {
        weight = torch::empty({B, T, D}, fopts);
        gof = torch::empty(go.sizes(), fopts);
        ghf.zero_();
    } else if (!gather_h) {
        ghf.zero_();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    fdc_check_cublas(
        cublasSetStream(handle, stream),
        "fdc_backward_large_cuda cublasSetStream failed"
    );

    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        h.scalar_type(),
        "fdc_large_make_mix_float",
        [&] {
            fdc_large_make_mix_float_kernel<scalar_t><<<
                (D * N + threads - 1) / threads,
                threads,
                0,
                stream
            >>>(
                mix.data_ptr<scalar_t>(),
                mixf.data_ptr<float>(),
                D * N
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (materialize_h) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            h.scalar_type(),
            "fdc_large_go_float",
            [&] {
                fdc_large_cast_scalar_to_float_kernel<scalar_t><<<
                    (go.numel() + threads - 1) / threads,
                    threads,
                    0,
                    stream
                >>>(
                    go.data_ptr<scalar_t>(),
                    gof.data_ptr<float>(),
                    go.numel()
                );
            }
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (gather_h) {
        dim3 block(16, 16);
        dim3 grid(
            (L + 15) / 16,
            (D + 15) / 16,
            B
        );

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            h.scalar_type(),
            "fdc_large_grad_h_gather",
            [&] {
                fdc_large_grad_h_gather_kernel<scalar_t><<<
                    grid,
                    block,
                    0,
                    stream
                >>>(
                    go.data_ptr<scalar_t>(),
                    kc.data_ptr<scalar_t>(),
                    mix.data_ptr<scalar_t>(),
                    ghf.data_ptr<float>(),
                    B,
                    D,
                    L,
                    T,
                    N,
                    K,
                    static_cast<int>(off)
                );
            }
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    for (int kk = 0; kk < K; ++kk) {
        int64_t total_base = static_cast<int64_t>(B) * T * D;
        int64_t total_k = static_cast<int64_t>(B) * T * N;
        int64_t total = total_base > total_k ? total_base : total_k;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            h.scalar_type(),
            "fdc_large_make_base_kernel_float",
            [&] {
                fdc_large_make_base_kernel_float_kernel<scalar_t><<<
                    (total + threads - 1) / threads,
                    threads,
                    0,
                    stream
                >>>(
                    go.data_ptr<scalar_t>(),
                    h.data_ptr<scalar_t>(),
                    kc.data_ptr<scalar_t>(),
                    base.data_ptr<float>(),
                    kbtn.data_ptr<float>(),
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
            }
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_check_cublas(
            cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                T,
                D,
                &alpha,
                mixf.data_ptr<float>(),
                N,
                0,
                base.data_ptr<float>(),
                D,
                static_cast<long long>(T) * D,
                &beta0,
                gkbtn.data_ptr<float>(),
                N,
                static_cast<long long>(T) * N,
                B
            ),
            "fdc_backward_large_cuda sgemm grad_kernel failed"
        );

        fdc_large_copy_gk_kernel<<<
            (B * T * N + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gkbtn.data_ptr<float>(),
            gkf.data_ptr<float>(),
            B,
            T,
            N,
            K,
            kk
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        fdc_check_cublas(
            cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                N,
                D,
                T,
                &alpha,
                kbtn.data_ptr<float>(),
                N,
                static_cast<long long>(T) * N,
                base.data_ptr<float>(),
                D,
                static_cast<long long>(T) * D,
                &beta1,
                gmf.data_ptr<float>(),
                N,
                0,
                B
            ),
            "fdc_backward_large_cuda sgemm grad_mix failed"
        );

        if (materialize_h) {
            fdc_check_cublas(
                cublasSgemmStridedBatched(
                    handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    D,
                    T,
                    N,
                    &alpha,
                    mixf.data_ptr<float>(),
                    N,
                    0,
                    kbtn.data_ptr<float>(),
                    N,
                    static_cast<long long>(T) * N,
                    &beta0,
                    weight.data_ptr<float>(),
                    D,
                    static_cast<long long>(T) * D,
                    B
                ),
                "fdc_backward_large_cuda sgemm materialized weight failed"
            );

            fdc_large_grad_h_materialized_kernel<<<
                (B * D * T + threads - 1) / threads,
                threads,
                0,
                stream
            >>>(
                gof.data_ptr<float>(),
                weight.data_ptr<float>(),
                ghf.data_ptr<float>(),
                B,
                D,
                L,
                T,
                kk,
                static_cast<int>(off),
                static_cast<int>(dilation)
            );

            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (!gather_h) {
            dim3 block(256);
            dim3 grid(
                (T + 7) / 8,
                (D + 31) / 32,
                B
            );

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                h.scalar_type(),
                "fdc_large_grad_h_atomic",
                [&] {
                    fdc_large_grad_h_atomic_kernel<scalar_t, 8, 32><<<
                        grid,
                        block,
                        0,
                        stream
                    >>>(
                        go.data_ptr<scalar_t>(),
                        kc.data_ptr<scalar_t>(),
                        mix.data_ptr<scalar_t>(),
                        ghf.data_ptr<float>(),
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
                }
            );

            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    auto gh = torch::empty_like(h);
    auto gk = torch::empty_like(kc);
    auto gm = torch::empty_like(mix);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        h.scalar_type(),
        "fdc_large_cast_out",
        [&] {
            fdc_large_cast_float_to_scalar_kernel<scalar_t><<<
                (gh.numel() + threads - 1) / threads,
                threads,
                0,
                stream
            >>>(
                ghf.data_ptr<float>(),
                gh.data_ptr<scalar_t>(),
                gh.numel()
            );

            fdc_large_cast_float_to_scalar_kernel<scalar_t><<<
                (gk.numel() + threads - 1) / threads,
                threads,
                0,
                stream
            >>>(
                gkf.data_ptr<float>(),
                gk.data_ptr<scalar_t>(),
                gk.numel()
            );

            fdc_large_cast_float_to_scalar_kernel<scalar_t><<<
                (gm.numel() + threads - 1) / threads,
                threads,
                0,
                stream
            >>>(
                gmf.data_ptr<float>(),
                gm.data_ptr<scalar_t>(),
                gm.numel()
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        gh,
        gk,
        gm
    };
}
