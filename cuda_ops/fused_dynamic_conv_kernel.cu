#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>

template <typename T>
__device__ __forceinline__ float to_float_dev(T x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float to_float_dev<c10::Half>(c10::Half x) {
    return __half2float(static_cast<__half>(x));
}

template <>
__device__ __forceinline__ float to_float_dev<c10::BFloat16>(c10::BFloat16 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __bfloat162float(static_cast<__nv_bfloat16>(x));
#else
    return static_cast<float>(x);
#endif
}

template <typename T>
__device__ __forceinline__ T from_float_dev(float x) {
    return static_cast<T>(x);
}

template <>
__device__ __forceinline__ c10::Half from_float_dev<c10::Half>(float x) {
    return c10::Half(__float2half_rn(x));
}

template <>
__device__ __forceinline__ c10::BFloat16 from_float_dev<c10::BFloat16>(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return c10::BFloat16(__float2bfloat16_rn(x));
#else
    return c10::BFloat16(x);
#endif
}

static inline void check_cublas(cublasStatus_t s, const char* msg) {
    TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, msg);
}

template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int B,
    int D,
    int L,
    int T,
    int N,
    int K,
    int off,
    int dilation
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (b >= B || d >= D || t >= T) return;

    float acc = 0.0f;
    int gt = off + t;
    int64_t hbase = ((int64_t)b * D + d) * L;
    int64_t obase = ((int64_t)b * D + d) * T;
    int64_t kbase = ((int64_t)b * T + t) * N * K;
    int64_t mbase = (int64_t)d * N;

    for (int kk = 0; kk < K; ++kk) {
        int s = gt - kk * dilation;
        if (s < 0 || s >= L) continue;
        float w = 0.0f;
        for (int n = 0; n < N; ++n) {
            w += to_float_dev(kc[kbase + n * K + kk]) *
                 to_float_dev(mix[mbase + n]);
        }
        acc += w * to_float_dev(h[hbase + s]);
    }

    out[obase + t] = from_float_dev<scalar_t>(acc);
}

torch::Tensor fused_dynamic_conv_forward_chunk_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    int B = h.size(0);
    int D = h.size(1);
    int L = h.size(2);
    int T = kc.size(1);
    int N = kc.size(2);
    int K = kc.size(3);

    auto out = torch::empty({B, D, T}, h.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 block(16, 16);
    dim3 grid((T + 15) / 16, (D + 15) / 16, B);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        h.scalar_type(),
        "forward_kernel",
        [&] {
            forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
                h.data_ptr<scalar_t>(),
                kc.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                B,
                D,
                L,
                T,
                N,
                K,
                (int)off,
                (int)dilation
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

template <typename scalar_t>
__global__ void cast_float_to_scalar_kernel(
    const float* __restrict__ src,
    scalar_t* __restrict__ dst,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = from_float_dev<scalar_t>(src[i]);
}

template <typename scalar_t>
__global__ void cast_scalar_to_float_kernel(
    const scalar_t* __restrict__ src,
    float* __restrict__ dst,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = to_float_dev(src[i]);
}

template <typename scalar_t>
__global__ void make_mix_float_kernel(
    const scalar_t* __restrict__ mix,
    float* __restrict__ mixf,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) mixf[i] = to_float_dev(mix[i]);
}

template <typename scalar_t>
__global__ void make_base_kernel_float_kernel(
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
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_base = (int64_t)B * T * D;
    int64_t total_k = (int64_t)B * T * N;
    int64_t total = total_base > total_k ? total_base : total_k;
    if (idx >= total) return;

    if (idx < total_base) {
        int d = idx % D;
        int64_t q = idx / D;
        int t = q % T;
        int b = q / T;
        int s = off + t - kk * dilation;
        float v = 0.0f;
        if (s >= 0 && s < L) {
            v = to_float_dev(go[((int64_t)b * D + d) * T + t]) *
                to_float_dev(h[((int64_t)b * D + d) * L + s]);
        }
        base[((int64_t)b * T + t) * D + d] = v;
    }

    if (idx < total_k) {
        int n = idx % N;
        int64_t q = idx / N;
        int t = q % T;
        int b = q / T;
        kbtn[((int64_t)b * T + t) * N + n] =
            to_float_dev(kc[((int64_t)b * T + t) * N * K + n * K + kk]);
    }
}

__global__ void copy_gk_kernel(
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
    if (i >= total) return;
    int n = i % N;
    int q = i / N;
    int t = q % T;
    int b = q / T;
    dst[((int64_t)b * T + t) * N * K + n * K + kk] = src[i];
}

template <typename scalar_t>
__global__ void grad_h_gather_kernel(
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
    if (b >= B || d >= D || s >= L) return;

    int tb = s - off;
    float acc = 0.0f;
    int64_t mbase = (int64_t)d * N;
    int64_t gobase = ((int64_t)b * D + d) * T;

    for (int kk = 0; kk < K; ++kk) {
        int t = tb + kk;
        if (t < 0 || t >= T) continue;
        int64_t kbase = ((int64_t)b * T + t) * N * K;
        float w = 0.0f;
        for (int n = 0; n < N; ++n) {
            w += to_float_dev(kc[kbase + n * K + kk]) *
                 to_float_dev(mix[mbase + n]);
        }
        acc += to_float_dev(go[gobase + t]) * w;
    }

    gh[((int64_t)b * D + d) * L + s] = acc;
}

template <typename scalar_t, int BT, int BD>
__global__ void grad_h_atomic_kernel(
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
        if (b >= B || t >= T || d >= D) continue;
        int s = off + t - kk * dilation;
        if (s < 0 || s >= L) continue;
        float w = 0.0f;
        int64_t kb = ((int64_t)b * T + t) * N * K;
        int64_t mb = (int64_t)d * N;
        for (int n = 0; n < N; ++n) {
            w += to_float_dev(kc[kb + n * K + kk]) *
                 to_float_dev(mix[mb + n]);
        }
        float v = to_float_dev(go[((int64_t)b * D + d) * T + t]) * w;
        atomicAdd(gh + ((int64_t)b * D + d) * L + s, v);
    }
}

__global__ void grad_h_materialized_kernel(
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
    if (idx >= total) return;

    int t = idx % T;
    int q = idx / T;
    int d = q % D;
    int b = q / D;
    int s = off + t - kk * dilation;
    if (s < 0 || s >= L) return;

    atomicAdd(
        gh + ((int64_t)b * D + d) * L + s,
        gof[((int64_t)b * D + d) * T + t] *
            weight[((int64_t)b * T + t) * D + d]
    );
}

template <typename scalar_t>
__global__ void make_base_half_kernel(
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
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_base = (int64_t)B * T * D;
    int64_t total_k = (int64_t)B * T * N;
    int64_t total = total_base > total_k ? total_base : total_k;
    if (idx >= total) return;

    if (idx < total_base) {
        int d = idx % D;
        int64_t q = idx / D;
        int t = q % T;
        int b = q / T;
        int s = off + t - kk * dilation;
        float v = 0.0f;
        if (s >= 0 && s < L) {
            v = to_float_dev(go[((int64_t)b * D + d) * T + t]) *
                to_float_dev(h[((int64_t)b * D + d) * L + s]);
        }
        base[((int64_t)b * T + t) * D + d] = __float2half_rn(v);
    }

    if (idx < total_k) {
        int n = idx % N;
        int64_t q = idx / N;
        int t = q % T;
        int b = q / T;
        float v = to_float_dev(kc[((int64_t)b * T + t) * N * K + n * K + kk]);
        kbtn[((int64_t)b * T + t) * N + n] = __float2half_rn(v);
    }
}

template <typename scalar_t>
__global__ void make_mix_half_kernel(
    const scalar_t* __restrict__ mix,
    __half* __restrict__ mixh,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) mixh[i] = __float2half_rn(to_float_dev(mix[i]));
}

template <int T_TILE>
__global__ void fp32_n6k3d256_grad_kernel_v2(
    const float* __restrict__ go,
    const float* __restrict__ h,
    const float* __restrict__ mix,
    float* __restrict__ gk,
    int L,
    int T,
    int off
) {
    int tile = blockIdx.x;
    int tid = threadIdx.x;
    int t0 = tile * T_TILE;

    __shared__ float sh[T_TILE * 18 * 256];

    for (int tt = 0; tt < T_TILE; ++tt) {
        int t = t0 + tt;
        for (int kk = 0; kk < 3; ++kk) {
            int s = off + t - kk;
            for (int n = 0; n < 6; ++n) {
                float v = 0.0f;
                if (t < T && s >= 0 && s < L) {
                    int d = tid;
                    v = go[(int64_t)d * T + t] *
                        h[(int64_t)d * L + s] *
                        mix[(int64_t)d * 6 + n];
                }
                sh[((tt * 18 + kk * 6 + n) * 256) + tid] = v;
            }
        }
    }

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int i = 0; i < T_TILE * 18; ++i) {
                sh[i * 256 + tid] += sh[i * 256 + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int tt = 0; tt < T_TILE; ++tt) {
            int t = t0 + tt;
            if (t < T) {
                for (int kk = 0; kk < 3; ++kk) {
                    for (int n = 0; n < 6; ++n) {
                        gk[(int64_t)t * 18 + n * 3 + kk] =
                            sh[(tt * 18 + kk * 6 + n) * 256];
                    }
                }
            }
        }
    }
}

__global__ void fp32_n6k3d256_grad_mix_partial(
    const float* __restrict__ go,
    const float* __restrict__ h,
    const float* __restrict__ kc,
    float* __restrict__ partial,
    int L,
    int T,
    int off,
    int tiles
) {
    int d = blockIdx.x;
    int n = blockIdx.y;
    int tile = blockIdx.z;
    int tid = threadIdx.x;
    int start = tile * 512;
    int end = start + 512;
    if (end > T) end = T;

    float acc = 0.0f;
    for (int t = start + tid; t < end; t += 256) {
        float g = go[(int64_t)d * T + t];
        for (int kk = 0; kk < 3; ++kk) {
            int s = off + t - kk;
            if (s >= 0 && s < L) {
                acc += g *
                       h[(int64_t)d * L + s] *
                       kc[(int64_t)t * 18 + n * 3 + kk];
            }
        }
    }

    __shared__ float sh[256];
    sh[tid] = acc;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        partial[((int64_t)tile * 256 + d) * 6 + n] = sh[0];
    }
}

__global__ void fp32_n6k3d256_grad_mix_finalize(
    const float* __restrict__ partial,
    float* __restrict__ gm,
    int tiles
) {
    int d = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int tile = tid; tile < tiles; tile += 256) {
        acc += partial[((int64_t)tile * 256 + d) * 6 + n];
    }

    __shared__ float sh[256];
    sh[tid] = acc;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    if (tid == 0) gm[d * 6 + n] = sh[0];
}

static std::vector<torch::Tensor> backward_fp32_special_v2(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    int L = h.size(2);
    int T = kc.size(1);
    auto opts = h.options().dtype(torch::kFloat32);

    auto ghf = torch::empty(h.sizes(), opts);
    auto gkf = torch::empty(kc.sizes(), opts);
    auto gmf = torch::empty(mix.sizes(), opts);
    int tiles = (T + 511) / 512;
    auto partial = torch::empty({tiles, 256, 6}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 bh(16, 16);
    dim3 gh((L + 15) / 16, 16, 1);
    grad_h_gather_kernel<float><<<gh, bh, 0, stream>>>(
        go.data_ptr<float>(),
        kc.data_ptr<float>(),
        mix.data_ptr<float>(),
        ghf.data_ptr<float>(),
        1,
        256,
        L,
        T,
        6,
        3,
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    constexpr int T_TILE = 2;
    fp32_n6k3d256_grad_kernel_v2<T_TILE><<<
        (T + T_TILE - 1) / T_TILE,
        256,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        h.data_ptr<float>(),
        mix.data_ptr<float>(),
        gkf.data_ptr<float>(),
        L,
        T,
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 pg(256, 6, tiles);
    fp32_n6k3d256_grad_mix_partial<<<pg, 256, 0, stream>>>(
        go.data_ptr<float>(),
        h.data_ptr<float>(),
        kc.data_ptr<float>(),
        partial.data_ptr<float>(),
        L,
        T,
        (int)off,
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 fg(256, 6);
    fp32_n6k3d256_grad_mix_finalize<<<fg, 256, 0, stream>>>(
        partial.data_ptr<float>(),
        gmf.data_ptr<float>(),
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {ghf, gkf, gmf};
}

static std::vector<torch::Tensor> backward_large(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    int B = h.size(0);
    int D = h.size(1);
    int L = h.size(2);
    int T = kc.size(1);
    int N = kc.size(2);
    int K = kc.size(3);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, h.get_device());
    int sm = prop.major * 10 + prop.minor;
    bool is_bf16 = h.scalar_type() == at::ScalarType::BFloat16;
    bool materialize_h = is_bf16 && sm < 80 && N >= 16;
    bool gather_h = ((int)dilation == 1) && !materialize_h;

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
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream failed");

    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        h.scalar_type(),
        "make_mix_float",
        [&] {
            make_mix_float_kernel<scalar_t><<<
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
            "go_float",
            [&] {
                cast_scalar_to_float_kernel<scalar_t><<<
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
        dim3 grid((L + 15) / 16, (D + 15) / 16, B);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            h.scalar_type(),
            "grad_h_gather",
            [&] {
                grad_h_gather_kernel<scalar_t><<<grid, block, 0, stream>>>(
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
                    (int)off
                );
            }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    for (int kk = 0; kk < K; ++kk) {
        int64_t total_base = (int64_t)B * T * D;
        int64_t total_k = (int64_t)B * T * N;
        int64_t total = total_base > total_k ? total_base : total_k;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            h.scalar_type(),
            "make_base_kernel_float",
            [&] {
                make_base_kernel_float_kernel<scalar_t><<<
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
                    (int)off,
                    (int)dilation
                );
            }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        check_cublas(
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
                (long long)T * D,
                &beta0,
                gkbtn.data_ptr<float>(),
                N,
                (long long)T * N,
                B
            ),
            "sgemm grad_kernel failed"
        );

        copy_gk_kernel<<<
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

        check_cublas(
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
                (long long)T * N,
                base.data_ptr<float>(),
                D,
                (long long)T * D,
                &beta1,
                gmf.data_ptr<float>(),
                N,
                0,
                B
            ),
            "sgemm grad_mix failed"
        );

        if (materialize_h) {
            check_cublas(
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
                    (long long)T * N,
                    &beta0,
                    weight.data_ptr<float>(),
                    D,
                    (long long)T * D,
                    B
                ),
                "sgemm materialized weight failed"
            );

            grad_h_materialized_kernel<<<
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
                (int)off,
                (int)dilation
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (!gather_h) {
            dim3 block(256);
            dim3 grid((T + 7) / 8, (D + 31) / 32, B);
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                h.scalar_type(),
                "grad_h_atomic",
                [&] {
                    grad_h_atomic_kernel<scalar_t, 8, 32><<<grid, block, 0, stream>>>(
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
                        (int)off,
                        (int)dilation
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
        "cast_out",
        [&] {
            cast_float_to_scalar_kernel<scalar_t><<<
                (gh.numel() + threads - 1) / threads,
                threads,
                0,
                stream
            >>>(
                ghf.data_ptr<float>(),
                gh.data_ptr<scalar_t>(),
                gh.numel()
            );
            cast_float_to_scalar_kernel<scalar_t><<<
                (gk.numel() + threads - 1) / threads,
                threads,
                0,
                stream
            >>>(
                gkf.data_ptr<float>(),
                gk.data_ptr<scalar_t>(),
                gk.numel()
            );
            cast_float_to_scalar_kernel<scalar_t><<<
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

    return {gh, gk, gm};
}

static std::vector<torch::Tensor> backward_fp16_gemmex(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    int B = h.size(0);
    int D = h.size(1);
    int L = h.size(2);
    int T = kc.size(1);
    int N = kc.size(2);
    int K = kc.size(3);

    auto fopts = h.options().dtype(torch::kFloat32);
    auto hopts = h.options().dtype(torch::kFloat16);

    auto ghf = torch::empty(h.sizes(), fopts);
    auto gkf = torch::empty(kc.sizes(), fopts);
    auto gmf = torch::zeros(mix.sizes(), fopts);
    auto mixh = torch::empty({D, N}, hopts);
    auto baseh = torch::empty({B, T, D}, hopts);
    auto kbtnh = torch::empty({B, T, N}, hopts);
    auto gkbtnf = torch::empty({B, T, N}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream failed");

    int threads = 256;

    make_mix_half_kernel<c10::Half><<<
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

    if ((int)dilation == 1) {
        dim3 block(16, 16);
        dim3 grid((L + 15) / 16, (D + 15) / 16, B);
        grad_h_gather_kernel<c10::Half><<<grid, block, 0, stream>>>(
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
            (int)off
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        ghf.zero_();
    }

    float alpha = 1.0f;
    float beta0 = 0.0f;
    float beta1 = 1.0f;

    for (int kk = 0; kk < K; ++kk) {
        int64_t total_base = (int64_t)B * T * D;
        int64_t total_k = (int64_t)B * T * N;
        int64_t total = total_base > total_k ? total_base : total_k;

        make_base_half_kernel<c10::Half><<<
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
            (int)off,
            (int)dilation
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        check_cublas(
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
                (long long)T * D,
                &beta0,
                gkbtnf.data_ptr<float>(),
                CUDA_R_32F,
                N,
                (long long)T * N,
                B,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "GemmEx grad_kernel failed"
        );

        copy_gk_kernel<<<
            (B * T * N + threads - 1) / threads,
            threads,
            0,
            stream
        >>>(
            gkbtnf.data_ptr<float>(),
            gkf.data_ptr<float>(),
            B,
            T,
            N,
            K,
            kk
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        check_cublas(
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
                (long long)T * N,
                baseh.data_ptr<c10::Half>(),
                CUDA_R_16F,
                D,
                (long long)T * D,
                &beta1,
                gmf.data_ptr<float>(),
                CUDA_R_32F,
                N,
                0,
                B,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "GemmEx grad_mix failed"
        );

        if ((int)dilation != 1) {
            dim3 block(256);
            dim3 grid((T + 7) / 8, (D + 31) / 32, B);
            grad_h_atomic_kernel<c10::Half, 8, 32><<<grid, block, 0, stream>>>(
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
                kk,
                (int)off,
                (int)dilation
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    auto gh = torch::empty_like(h);
    auto gk = torch::empty_like(kc);
    auto gm = torch::empty_like(mix);

    cast_float_to_scalar_kernel<c10::Half><<<
        (gh.numel() + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        ghf.data_ptr<float>(),
        gh.data_ptr<c10::Half>(),
        gh.numel()
    );

    cast_float_to_scalar_kernel<c10::Half><<<
        (gk.numel() + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        gkf.data_ptr<float>(),
        gk.data_ptr<c10::Half>(),
        gk.numel()
    );

    cast_float_to_scalar_kernel<c10::Half><<<
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

    return {gh, gk, gm};
}

std::vector<torch::Tensor> fused_dynamic_conv_backward_chunk_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    int B = h.size(0);
    int D = h.size(1);
    int T = kc.size(1);
    int N = kc.size(2);
    int K = kc.size(3);

    bool is_fp32 = h.scalar_type() == at::ScalarType::Float;
    bool is_fp16 = h.scalar_type() == at::ScalarType::Half;
    bool is_bf16 = h.scalar_type() == at::ScalarType::BFloat16;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, h.get_device());
    int sm = prop.major * 10 + prop.minor;

    bool use_fp32_special_v2 =
        is_fp32 &&
        B == 1 &&
        D == 256 &&
        N == 6 &&
        K == 3 &&
        (int)dilation == 1 &&
        T <= 8192;

    bool use_fp16_gemmex =
        is_fp16 &&
        B == 1 &&
        D == 256 &&
        N == 6 &&
        K == 3 &&
        T >= 4096 &&
        sm >= 75;

    bool use_bf16_large =
        is_bf16;

    if (use_fp32_special_v2) {
        return backward_fp32_special_v2(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    if (use_fp16_gemmex) {
        return backward_fp16_gemmex(
            go,
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    if (use_bf16_large) {
        return backward_large(
            go,
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    return backward_large(
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );
}
