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
#include <cstdlib>

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

static inline bool fdc_debug_path_enabled() {
    const char* v = std::getenv("FDC_DEBUG_PATH");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

#define FDC_DEBUG_PATH(msg) \
    do { \
        if (fdc_debug_path_enabled()) { \
            TORCH_WARN(msg); \
        } \
    } while (0)

__device__ __forceinline__ float warp_sum_float(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// ======================================================================================
// Forward
// ======================================================================================

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
    int64_t hbase = ((int64_t)b * D + d) * L;
    int64_t obase = ((int64_t)b * D + d) * T;
    int64_t kbase = ((int64_t)b * T + t) * N * K;
    int64_t mbase = (int64_t)d * N;

    for (int kk = 0; kk < K; ++kk) {
        int s = off + t - kk * dilation;
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

template <typename scalar_t>
__global__ void forward_n6k3_d256_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ out,
    int L,
    int T,
    int off
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * T;
    if (idx >= total) return;

    int t = idx % T;
    int d = idx / T;

    float m0 = to_float_dev(mix[d * 6 + 0]);
    float m1 = to_float_dev(mix[d * 6 + 1]);
    float m2 = to_float_dev(mix[d * 6 + 2]);
    float m3 = to_float_dev(mix[d * 6 + 3]);
    float m4 = to_float_dev(mix[d * 6 + 4]);
    float m5 = to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int s = off + t - kk;
        if (s >= 0 && s < L) {
            int64_t kb = (int64_t)t * 18 + kk;
            float w =
                to_float_dev(kc[kb + 0 * 3]) * m0 +
                to_float_dev(kc[kb + 1 * 3]) * m1 +
                to_float_dev(kc[kb + 2 * 3]) * m2 +
                to_float_dev(kc[kb + 3 * 3]) * m3 +
                to_float_dev(kc[kb + 4 * 3]) * m4 +
                to_float_dev(kc[kb + 5 * 3]) * m5;
            acc += w * to_float_dev(h[(int64_t)d * L + s]);
        }
    }

    out[(int64_t)d * T + t] = from_float_dev<scalar_t>(acc);
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

    if (B == 1 && D == 256 && N == 6 && K == 3 && (int)dilation == 1) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            h.scalar_type(),
            "forward_n6k3_d256_kernel",
            [&] {
                forward_n6k3_d256_kernel<scalar_t><<<
                    (256 * T + 255) / 256,
                    256,
                    0,
                    stream
                >>>(
                    h.data_ptr<scalar_t>(),
                    kc.data_ptr<scalar_t>(),
                    mix.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    L,
                    T,
                    (int)off
                );
            }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

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

// ======================================================================================
// Common kernels
// ======================================================================================

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
__global__ void make_mix_transpose_float_kernel(
    const scalar_t* __restrict__ mix,
    float* __restrict__ mixT,
    int D,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * N;
    if (idx >= total) return;

    int n = idx % N;
    int d = idx / N;
    mixT[n * D + d] = to_float_dev(mix[d * N + n]);
}

template <typename scalar_t>
__global__ void make_base_ktd_float_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    float* __restrict__ base,
    int D,
    int L,
    int T,
    int K,
    int off
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)K * T * D;
    if (idx >= total) return;

    int d = idx % D;
    int64_t q = idx / D;
    int t = q % T;
    int kk = q / T;

    int s = off + t - kk;
    float v = 0.0f;

    if (s >= 0 && s < L) {
        v = to_float_dev(go[(int64_t)d * T + t]) *
            to_float_dev(h[(int64_t)d * L + s]);
    }

    base[((int64_t)kk * T + t) * D + d] = v;
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
__global__ void copy_gk_float_to_scalar_kernel(
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
    if (i >= total) return;

    int n = i % N;
    int q = i / N;
    int t = q % T;
    int b = q / T;

    dst[((int64_t)b * T + t) * N * K + n * K + kk] =
        from_float_dev<scalar_t>(src[i]);
}

// ======================================================================================
// grad_h
// ======================================================================================

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

        atomicAdd(
            gh + ((int64_t)b * D + d) * L + s,
            to_float_dev(go[((int64_t)b * D + d) * T + t]) * w
        );
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
__global__ void small_n6k3_grad_h_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gh,
    int L,
    int T,
    int off
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (d >= 256 || s >= L) return;

    int t0 = s - off;

    float m0 = to_float_dev(mix[d * 6 + 0]);
    float m1 = to_float_dev(mix[d * 6 + 1]);
    float m2 = to_float_dev(mix[d * 6 + 2]);
    float m3 = to_float_dev(mix[d * 6 + 3]);
    float m4 = to_float_dev(mix[d * 6 + 4]);
    float m5 = to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = t0 + kk;
        if (t >= 0 && t < T) {
            int64_t kb = (int64_t)t * 18 + kk;
            float w =
                to_float_dev(kc[kb + 0 * 3]) * m0 +
                to_float_dev(kc[kb + 1 * 3]) * m1 +
                to_float_dev(kc[kb + 2 * 3]) * m2 +
                to_float_dev(kc[kb + 3 * 3]) * m3 +
                to_float_dev(kc[kb + 4 * 3]) * m4 +
                to_float_dev(kc[kb + 5 * 3]) * m5;

            acc += to_float_dev(go[(int64_t)d * T + t]) * w;
        }
    }

    gh[(int64_t)d * L + s] = acc;
}

template <typename scalar_t>
__global__ void full4096_n6k3_grad_h_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 256 * 4096;
    if (idx >= total) return;

    int s = idx & 4095;
    int d = idx >> 12;

    float m0 = to_float_dev(mix[d * 6 + 0]);
    float m1 = to_float_dev(mix[d * 6 + 1]);
    float m2 = to_float_dev(mix[d * 6 + 2]);
    float m3 = to_float_dev(mix[d * 6 + 3]);
    float m4 = to_float_dev(mix[d * 6 + 4]);
    float m5 = to_float_dev(mix[d * 6 + 5]);

    float acc = 0.0f;

    {
        int t = s;
        int64_t kb = (int64_t)t * 18;
        float w =
            to_float_dev(kc[kb + 0]) * m0 +
            to_float_dev(kc[kb + 3]) * m1 +
            to_float_dev(kc[kb + 6]) * m2 +
            to_float_dev(kc[kb + 9]) * m3 +
            to_float_dev(kc[kb + 12]) * m4 +
            to_float_dev(kc[kb + 15]) * m5;
        acc += to_float_dev(go[(int64_t)d * 4096 + t]) * w;
    }

    if (s + 1 < 4096) {
        int t = s + 1;
        int64_t kb = (int64_t)t * 18;
        float w =
            to_float_dev(kc[kb + 1]) * m0 +
            to_float_dev(kc[kb + 4]) * m1 +
            to_float_dev(kc[kb + 7]) * m2 +
            to_float_dev(kc[kb + 10]) * m3 +
            to_float_dev(kc[kb + 13]) * m4 +
            to_float_dev(kc[kb + 16]) * m5;
        acc += to_float_dev(go[(int64_t)d * 4096 + t]) * w;
    }

    if (s + 2 < 4096) {
        int t = s + 2;
        int64_t kb = (int64_t)t * 18;
        float w =
            to_float_dev(kc[kb + 2]) * m0 +
            to_float_dev(kc[kb + 5]) * m1 +
            to_float_dev(kc[kb + 8]) * m2 +
            to_float_dev(kc[kb + 11]) * m3 +
            to_float_dev(kc[kb + 14]) * m4 +
            to_float_dev(kc[kb + 17]) * m5;
        acc += to_float_dev(go[(int64_t)d * 4096 + t]) * w;
    }

    gh[(int64_t)d * 4096 + s] = acc;
}

// ======================================================================================
// Old small N6 kernels
// ======================================================================================

template <typename scalar_t>
__global__ void small_n6k3_grad_kernel_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gk,
    int L,
    int T,
    int off
) {
    int t = blockIdx.x;
    int kk = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ float sh[6 * 256];

    float base = 0.0f;
    int s = off + t - kk;

    if (t < T && s >= 0 && s < L) {
        base = to_float_dev(go[(int64_t)tid * T + t]) *
               to_float_dev(h[(int64_t)tid * L + s]);
    }

#pragma unroll
    for (int n = 0; n < 6; ++n) {
        sh[n * 256 + tid] = base * to_float_dev(mix[tid * 6 + n]);
    }

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int n = 0; n < 6; ++n) {
                sh[n * 256 + tid] += sh[n * 256 + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int n = 0; n < 6; ++n) {
            gk[(int64_t)t * 18 + n * 3 + kk] = sh[n * 256];
        }
    }
}

template <typename scalar_t>
__global__ void small_n6k3_grad_mix_partial_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
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
        float g = to_float_dev(go[(int64_t)d * T + t]);

#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            int s = off + t - kk;
            if (s >= 0 && s < L) {
                acc += g *
                       to_float_dev(h[(int64_t)d * L + s]) *
                       to_float_dev(kc[(int64_t)t * 18 + n * 3 + kk]);
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

__global__ void small_n6k3_grad_mix_finalize_kernel(
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

    if (tid == 0) {
        gm[d * 6 + n] = sh[0];
    }
}

// ======================================================================================
// Mid N6 warp path
// ======================================================================================

template <typename scalar_t, int T_TILE>
__global__ void mid_n6k3_grad_kernel_warp_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gk,
    int L,
    int T,
    int off
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int t = blockIdx.x * T_TILE + warp;
    int kk = blockIdx.y;

    if (warp >= T_TILE || t >= T) return;

    int s = off + t - kk;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    if (s >= 0 && s < L) {
        for (int d = lane; d < 256; d += 32) {
            float base =
                to_float_dev(go[(int64_t)d * T + t]) *
                to_float_dev(h[(int64_t)d * L + s]);

            int mb = d * 6;
            acc0 += base * to_float_dev(mix[mb + 0]);
            acc1 += base * to_float_dev(mix[mb + 1]);
            acc2 += base * to_float_dev(mix[mb + 2]);
            acc3 += base * to_float_dev(mix[mb + 3]);
            acc4 += base * to_float_dev(mix[mb + 4]);
            acc5 += base * to_float_dev(mix[mb + 5]);
        }
    }

    acc0 = warp_sum_float(acc0);
    acc1 = warp_sum_float(acc1);
    acc2 = warp_sum_float(acc2);
    acc3 = warp_sum_float(acc3);
    acc4 = warp_sum_float(acc4);
    acc5 = warp_sum_float(acc5);

    if (lane == 0) {
        int64_t base = (int64_t)t * 18;
        gk[base + 0 * 3 + kk] = acc0;
        gk[base + 1 * 3 + kk] = acc1;
        gk[base + 2 * 3 + kk] = acc2;
        gk[base + 3 * 3 + kk] = acc3;
        gk[base + 4 * 3 + kk] = acc4;
        gk[base + 5 * 3 + kk] = acc5;
    }
}

template <typename scalar_t, int TILE_T>
__global__ void mid_n6k3_grad_mix_partial_alln_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ kc,
    float* __restrict__ partial,
    int L,
    int T,
    int off,
    int tiles
) {
    int d = blockIdx.x;
    int tile = blockIdx.y;
    int tid = threadIdx.x;

    int start = tile * TILE_T;
    int end = start + TILE_T;
    if (end > T) end = T;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    for (int t = start + tid; t < end; t += blockDim.x) {
        float g = to_float_dev(go[(int64_t)d * T + t]);

#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            int s = off + t - kk;
            if (s >= 0 && s < L) {
                float base = g * to_float_dev(h[(int64_t)d * L + s]);
                int64_t kb = (int64_t)t * 18 + kk;

                acc0 += base * to_float_dev(kc[kb + 0 * 3]);
                acc1 += base * to_float_dev(kc[kb + 1 * 3]);
                acc2 += base * to_float_dev(kc[kb + 2 * 3]);
                acc3 += base * to_float_dev(kc[kb + 3 * 3]);
                acc4 += base * to_float_dev(kc[kb + 4 * 3]);
                acc5 += base * to_float_dev(kc[kb + 5 * 3]);
            }
        }
    }

    int lane = tid & 31;
    int warp = tid >> 5;

    acc0 = warp_sum_float(acc0);
    acc1 = warp_sum_float(acc1);
    acc2 = warp_sum_float(acc2);
    acc3 = warp_sum_float(acc3);
    acc4 = warp_sum_float(acc4);
    acc5 = warp_sum_float(acc5);

    __shared__ float sh[8 * 6];

    if (lane == 0) {
        sh[warp * 6 + 0] = acc0;
        sh[warp * 6 + 1] = acc1;
        sh[warp * 6 + 2] = acc2;
        sh[warp * 6 + 3] = acc3;
        sh[warp * 6 + 4] = acc4;
        sh[warp * 6 + 5] = acc5;
    }

    __syncthreads();

    if (warp == 0) {
        float v0 = lane < 8 ? sh[lane * 6 + 0] : 0.0f;
        float v1 = lane < 8 ? sh[lane * 6 + 1] : 0.0f;
        float v2 = lane < 8 ? sh[lane * 6 + 2] : 0.0f;
        float v3 = lane < 8 ? sh[lane * 6 + 3] : 0.0f;
        float v4 = lane < 8 ? sh[lane * 6 + 4] : 0.0f;
        float v5 = lane < 8 ? sh[lane * 6 + 5] : 0.0f;

        v0 = warp_sum_float(v0);
        v1 = warp_sum_float(v1);
        v2 = warp_sum_float(v2);
        v3 = warp_sum_float(v3);
        v4 = warp_sum_float(v4);
        v5 = warp_sum_float(v5);

        if (lane == 0) {
            int64_t ob = ((int64_t)tile * 256 + d) * 6;
            partial[ob + 0] = v0;
            partial[ob + 1] = v1;
            partial[ob + 2] = v2;
            partial[ob + 3] = v3;
            partial[ob + 4] = v4;
            partial[ob + 5] = v5;
        }
    }
}

__global__ void mid_n6k3_grad_mix_finalize_alln_kernel(
    const float* __restrict__ partial,
    float* __restrict__ gm,
    int tiles
) {
    int d = blockIdx.x;
    int tid = threadIdx.x;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    for (int tile = tid; tile < tiles; tile += blockDim.x) {
        int64_t ib = ((int64_t)tile * 256 + d) * 6;
        acc0 += partial[ib + 0];
        acc1 += partial[ib + 1];
        acc2 += partial[ib + 2];
        acc3 += partial[ib + 3];
        acc4 += partial[ib + 4];
        acc5 += partial[ib + 5];
    }

    int lane = tid & 31;
    int warp = tid >> 5;

    acc0 = warp_sum_float(acc0);
    acc1 = warp_sum_float(acc1);
    acc2 = warp_sum_float(acc2);
    acc3 = warp_sum_float(acc3);
    acc4 = warp_sum_float(acc4);
    acc5 = warp_sum_float(acc5);

    __shared__ float sh[8 * 6];

    if (lane == 0) {
        sh[warp * 6 + 0] = acc0;
        sh[warp * 6 + 1] = acc1;
        sh[warp * 6 + 2] = acc2;
        sh[warp * 6 + 3] = acc3;
        sh[warp * 6 + 4] = acc4;
        sh[warp * 6 + 5] = acc5;
    }

    __syncthreads();

    if (warp == 0) {
        float v0 = lane < 8 ? sh[lane * 6 + 0] : 0.0f;
        float v1 = lane < 8 ? sh[lane * 6 + 1] : 0.0f;
        float v2 = lane < 8 ? sh[lane * 6 + 2] : 0.0f;
        float v3 = lane < 8 ? sh[lane * 6 + 3] : 0.0f;
        float v4 = lane < 8 ? sh[lane * 6 + 4] : 0.0f;
        float v5 = lane < 8 ? sh[lane * 6 + 5] : 0.0f;

        v0 = warp_sum_float(v0);
        v1 = warp_sum_float(v1);
        v2 = warp_sum_float(v2);
        v3 = warp_sum_float(v3);
        v4 = warp_sum_float(v4);
        v5 = warp_sum_float(v5);

        if (lane == 0) {
            gm[d * 6 + 0] = v0;
            gm[d * 6 + 1] = v1;
            gm[d * 6 + 2] = v2;
            gm[d * 6 + 3] = v3;
            gm[d * 6 + 4] = v4;
            gm[d * 6 + 5] = v5;
        }
    }
}

// ======================================================================================
// Base-coalesced N6/N16 grad_kernel
// ======================================================================================

template <int T_TILE>
__global__ void base_n6k3_grad_kernel_warp_kernel(
    const float* __restrict__ base,
    const float* __restrict__ mixT,
    float* __restrict__ gk,
    int D,
    int T
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int t = blockIdx.x * T_TILE + warp;
    int kk = blockIdx.y;

    if (warp >= T_TILE || t >= T) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    int64_t bb = ((int64_t)kk * T + t) * D;

    for (int d = lane; d < D; d += 32) {
        float b = base[bb + d];

        acc0 += b * mixT[0 * D + d];
        acc1 += b * mixT[1 * D + d];
        acc2 += b * mixT[2 * D + d];
        acc3 += b * mixT[3 * D + d];
        acc4 += b * mixT[4 * D + d];
        acc5 += b * mixT[5 * D + d];
    }

    acc0 = warp_sum_float(acc0);
    acc1 = warp_sum_float(acc1);
    acc2 = warp_sum_float(acc2);
    acc3 = warp_sum_float(acc3);
    acc4 = warp_sum_float(acc4);
    acc5 = warp_sum_float(acc5);

    if (lane == 0) {
        int64_t ob = (int64_t)t * 18;
        gk[ob + 0 * 3 + kk] = acc0;
        gk[ob + 1 * 3 + kk] = acc1;
        gk[ob + 2 * 3 + kk] = acc2;
        gk[ob + 3 * 3 + kk] = acc3;
        gk[ob + 4 * 3 + kk] = acc4;
        gk[ob + 5 * 3 + kk] = acc5;
    }
}

template <int T_TILE, int N_TILE>
__global__ void base_n16k3_grad_kernel_warp_kernel(
    const float* __restrict__ base,
    const float* __restrict__ mixT,
    float* __restrict__ gk,
    int D,
    int T
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int t = blockIdx.x * T_TILE + warp;
    int kk = blockIdx.y;
    int ntile = blockIdx.z;
    int nbase = ntile * N_TILE;

    if (warp >= T_TILE || t >= T) return;

    float acc[N_TILE];

#pragma unroll
    for (int i = 0; i < N_TILE; ++i) acc[i] = 0.0f;

    int64_t bb = ((int64_t)kk * T + t) * D;

    for (int d = lane; d < D; d += 32) {
        float b = base[bb + d];

#pragma unroll
        for (int i = 0; i < N_TILE; ++i) {
            int n = nbase + i;
            if (n < 16) acc[i] += b * mixT[n * D + d];
        }
    }

#pragma unroll
    for (int i = 0; i < N_TILE; ++i) {
        acc[i] = warp_sum_float(acc[i]);
    }

    if (lane == 0) {
        int64_t ob = (int64_t)t * 48;

#pragma unroll
        for (int i = 0; i < N_TILE; ++i) {
            int n = nbase + i;
            if (n < 16) gk[ob + n * 3 + kk] = acc[i];
        }
    }
}

// ======================================================================================
// Base-coalesced grad_mix N6/N16
// ======================================================================================

template <int TILE_T>
__global__ void base_n6k3_grad_mix_partial_kernel(
    const float* __restrict__ base,
    const void* __restrict__ kc_void,
    float* __restrict__ partial,
    int T,
    int dtype_tag
) {
    int d = blockIdx.x;
    int tile = blockIdx.y;
    int tid = threadIdx.x;

    int start = tile * TILE_T;
    int end = start + TILE_T;
    if (end > T) end = T;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;

    for (int t = start + tid; t < end; t += blockDim.x) {
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            float b = base[((int64_t)kk * T + t) * 256 + d];
            int64_t kb = (int64_t)t * 18 + kk;

            if (dtype_tag == 0) {
                const float* kc = reinterpret_cast<const float*>(kc_void);
                acc0 += b * kc[kb + 0 * 3];
                acc1 += b * kc[kb + 1 * 3];
                acc2 += b * kc[kb + 2 * 3];
                acc3 += b * kc[kb + 3 * 3];
                acc4 += b * kc[kb + 4 * 3];
                acc5 += b * kc[kb + 5 * 3];
            } else if (dtype_tag == 1) {
                const c10::Half* kc = reinterpret_cast<const c10::Half*>(kc_void);
                acc0 += b * to_float_dev(kc[kb + 0 * 3]);
                acc1 += b * to_float_dev(kc[kb + 1 * 3]);
                acc2 += b * to_float_dev(kc[kb + 2 * 3]);
                acc3 += b * to_float_dev(kc[kb + 3 * 3]);
                acc4 += b * to_float_dev(kc[kb + 4 * 3]);
                acc5 += b * to_float_dev(kc[kb + 5 * 3]);
            } else {
                const c10::BFloat16* kc = reinterpret_cast<const c10::BFloat16*>(kc_void);
                acc0 += b * to_float_dev(kc[kb + 0 * 3]);
                acc1 += b * to_float_dev(kc[kb + 1 * 3]);
                acc2 += b * to_float_dev(kc[kb + 2 * 3]);
                acc3 += b * to_float_dev(kc[kb + 3 * 3]);
                acc4 += b * to_float_dev(kc[kb + 4 * 3]);
                acc5 += b * to_float_dev(kc[kb + 5 * 3]);
            }
        }
    }

    int lane = tid & 31;
    int warp = tid >> 5;

    acc0 = warp_sum_float(acc0);
    acc1 = warp_sum_float(acc1);
    acc2 = warp_sum_float(acc2);
    acc3 = warp_sum_float(acc3);
    acc4 = warp_sum_float(acc4);
    acc5 = warp_sum_float(acc5);

    __shared__ float sh[8 * 6];

    if (lane == 0) {
        sh[warp * 6 + 0] = acc0;
        sh[warp * 6 + 1] = acc1;
        sh[warp * 6 + 2] = acc2;
        sh[warp * 6 + 3] = acc3;
        sh[warp * 6 + 4] = acc4;
        sh[warp * 6 + 5] = acc5;
    }

    __syncthreads();

    if (warp == 0) {
        float v0 = lane < 8 ? sh[lane * 6 + 0] : 0.0f;
        float v1 = lane < 8 ? sh[lane * 6 + 1] : 0.0f;
        float v2 = lane < 8 ? sh[lane * 6 + 2] : 0.0f;
        float v3 = lane < 8 ? sh[lane * 6 + 3] : 0.0f;
        float v4 = lane < 8 ? sh[lane * 6 + 4] : 0.0f;
        float v5 = lane < 8 ? sh[lane * 6 + 5] : 0.0f;

        v0 = warp_sum_float(v0);
        v1 = warp_sum_float(v1);
        v2 = warp_sum_float(v2);
        v3 = warp_sum_float(v3);
        v4 = warp_sum_float(v4);
        v5 = warp_sum_float(v5);

        if (lane == 0) {
            int64_t ob = ((int64_t)tile * 256 + d) * 6;
            partial[ob + 0] = v0;
            partial[ob + 1] = v1;
            partial[ob + 2] = v2;
            partial[ob + 3] = v3;
            partial[ob + 4] = v4;
            partial[ob + 5] = v5;
        }
    }
}

template <int TILE_T, int N_TILE>
__global__ void base_n16k3_grad_mix_partial_kernel(
    const float* __restrict__ base,
    const void* __restrict__ kc_void,
    float* __restrict__ partial,
    int T,
    int dtype_tag
) {
    int d = blockIdx.x;
    int ntile = blockIdx.y;
    int tile = blockIdx.z;
    int tid = threadIdx.x;
    int nbase = ntile * N_TILE;

    int start = tile * TILE_T;
    int end = start + TILE_T;
    if (end > T) end = T;

    float acc[N_TILE];

#pragma unroll
    for (int i = 0; i < N_TILE; ++i) acc[i] = 0.0f;

    for (int t = start + tid; t < end; t += blockDim.x) {
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            float b = base[((int64_t)kk * T + t) * 256 + d];
            int64_t kb = (int64_t)t * 48 + kk;

#pragma unroll
            for (int i = 0; i < N_TILE; ++i) {
                int n = nbase + i;
                if (n < 16) {
                    if (dtype_tag == 0) {
                        const float* kc = reinterpret_cast<const float*>(kc_void);
                        acc[i] += b * kc[kb + n * 3];
                    } else if (dtype_tag == 1) {
                        const c10::Half* kc = reinterpret_cast<const c10::Half*>(kc_void);
                        acc[i] += b * to_float_dev(kc[kb + n * 3]);
                    } else {
                        const c10::BFloat16* kc = reinterpret_cast<const c10::BFloat16*>(kc_void);
                        acc[i] += b * to_float_dev(kc[kb + n * 3]);
                    }
                }
            }
        }
    }

    int lane = tid & 31;
    int warp = tid >> 5;

#pragma unroll
    for (int i = 0; i < N_TILE; ++i) {
        acc[i] = warp_sum_float(acc[i]);
    }

    __shared__ float sh[8 * N_TILE];

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N_TILE; ++i) {
            sh[warp * N_TILE + i] = acc[i];
        }
    }

    __syncthreads();

    if (warp == 0) {
        float v[N_TILE];

#pragma unroll
        for (int i = 0; i < N_TILE; ++i) {
            v[i] = lane < 8 ? sh[lane * N_TILE + i] : 0.0f;
            v[i] = warp_sum_float(v[i]);
        }

        if (lane == 0) {
            int64_t ob = (((int64_t)tile * 256 + d) * 16 + nbase);

#pragma unroll
            for (int i = 0; i < N_TILE; ++i) {
                int n = nbase + i;
                if (n < 16) partial[ob + i] = v[i];
            }
        }
    }
}

template <int N>
__global__ void grad_mix_finalize_n_all_kernel(
    const float* __restrict__ partial,
    float* __restrict__ gm,
    int tiles
) {
    int d = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;

    float acc = 0.0f;

    for (int tile = tid; tile < tiles; tile += blockDim.x) {
        acc += partial[((int64_t)tile * 256 + d) * N + n];
    }

    acc = warp_sum_float(acc);

    __shared__ float sh[8];

    int lane = tid & 31;
    int warp = tid >> 5;

    if (lane == 0) sh[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        float v = lane < 8 ? sh[lane] : 0.0f;
        v = warp_sum_float(v);
        if (lane == 0) gm[d * N + n] = v;
    }
}

// ======================================================================================
// N16 grad_h
// ======================================================================================

template <typename scalar_t>
__global__ void mid_n16k3_grad_h_kernel(
    const scalar_t* __restrict__ go,
    const scalar_t* __restrict__ kc,
    const scalar_t* __restrict__ mix,
    float* __restrict__ gh,
    int L,
    int T,
    int off
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (d >= 256 || s >= L) return;

    int tb = s - off;
    float acc = 0.0f;

#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        int t = tb + kk;
        if (t >= 0 && t < T) {
            int64_t kb = (int64_t)t * 48 + kk;
            int64_t mb = (int64_t)d * 16;

            float w = 0.0f;

#pragma unroll
            for (int n = 0; n < 16; ++n) {
                w += to_float_dev(kc[kb + n * 3]) *
                     to_float_dev(mix[mb + n]);
            }

            acc += to_float_dev(go[(int64_t)d * T + t]) * w;
        }
    }

    gh[(int64_t)d * L + s] = acc;
}

// ======================================================================================
// Small wrappers
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> backward_small_n6k3_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_small_n6k3");

    int L = h.size(2);
    int T = kc.size(1);

    auto fopts = h.options().dtype(torch::kFloat32);
    auto ghf = torch::empty(h.sizes(), fopts);
    auto gkf = torch::empty(kc.sizes(), fopts);
    auto gmf = torch::empty(mix.sizes(), fopts);

    int tiles = (T + 511) / 512;
    auto partial = torch::empty({tiles, 256, 6}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 bh(16, 16);
    dim3 gh((L + 15) / 16, 16, 1);

    small_n6k3_grad_h_kernel<scalar_t><<<gh, bh, 0, stream>>>(
        go.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        ghf.data_ptr<float>(),
        L,
        T,
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 gkg(T, 3, 1);
    small_n6k3_grad_kernel_kernel<scalar_t><<<gkg, 256, 0, stream>>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        gkf.data_ptr<float>(),
        L,
        T,
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 pg(256, 6, tiles);
    small_n6k3_grad_mix_partial_kernel<scalar_t><<<pg, 256, 0, stream>>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        partial.data_ptr<float>(),
        L,
        T,
        (int)off,
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 fg(256, 6);
    small_n6k3_grad_mix_finalize_kernel<<<fg, 256, 0, stream>>>(
        partial.data_ptr<float>(),
        gmf.data_ptr<float>(),
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if constexpr (std::is_same<scalar_t, float>::value) {
        return {ghf, gkf, gmf};
    } else {
        auto gh = torch::empty_like(h);
        auto gk = torch::empty_like(kc);
        auto gm = torch::empty_like(mix);

        int threads = 256;

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

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {gh, gk, gm};
    }
}

static std::vector<torch::Tensor> backward_small_n6k3(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    if (h.scalar_type() == at::ScalarType::Float) {
        return backward_small_n6k3_typed<float>(go, h, kc, mix, off);
    }
    if (h.scalar_type() == at::ScalarType::Half) {
        return backward_small_n6k3_typed<c10::Half>(go, h, kc, mix, off);
    }
    return backward_small_n6k3_typed<c10::BFloat16>(go, h, kc, mix, off);
}

// ======================================================================================
// Mid N6 warp wrapper
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> backward_mid_n6k3_warp_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    bool full4096_offset0
) {
    FDC_DEBUG_PATH("FDC path: backward_mid_n6k3_warp");

    int L = h.size(2);
    int T = kc.size(1);

    auto fopts = h.options().dtype(torch::kFloat32);
    auto ghf = torch::empty(h.sizes(), fopts);
    auto gkf = torch::empty(kc.sizes(), fopts);
    auto gmf = torch::empty(mix.sizes(), fopts);

    constexpr int T_TILE = 8;
    constexpr int MIX_TILE_T = 2048;

    int tiles = (T + MIX_TILE_T - 1) / MIX_TILE_T;
    auto partial = torch::empty({tiles, 256, 6}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (full4096_offset0) {
        full4096_n6k3_grad_h_kernel<scalar_t><<<
            (256 * 4096 + 255) / 256,
            256,
            0,
            stream
        >>>(
            go.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            ghf.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        dim3 bh(16, 16);
        dim3 gh((L + 15) / 16, 16, 1);

        small_n6k3_grad_h_kernel<scalar_t><<<gh, bh, 0, stream>>>(
            go.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            ghf.data_ptr<float>(),
            L,
            T,
            (int)off
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    dim3 gkg((T + T_TILE - 1) / T_TILE, 3, 1);
    mid_n6k3_grad_kernel_warp_kernel<scalar_t, T_TILE><<<
        gkg,
        256,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        gkf.data_ptr<float>(),
        L,
        T,
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 pg(256, tiles, 1);
    mid_n6k3_grad_mix_partial_alln_kernel<scalar_t, MIX_TILE_T><<<
        pg,
        256,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        partial.data_ptr<float>(),
        L,
        T,
        (int)off,
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    mid_n6k3_grad_mix_finalize_alln_kernel<<<
        256,
        256,
        0,
        stream
    >>>(
        partial.data_ptr<float>(),
        gmf.data_ptr<float>(),
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if constexpr (std::is_same<scalar_t, float>::value) {
        return {ghf, gkf, gmf};
    } else {
        auto gh = torch::empty_like(h);
        auto gk = torch::empty_like(kc);
        auto gm = torch::empty_like(mix);

        int threads = 256;

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

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {gh, gk, gm};
    }
}

static std::vector<torch::Tensor> backward_mid_n6k3_warp(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    bool full4096_offset0
) {
    if (h.scalar_type() == at::ScalarType::Float) {
        return backward_mid_n6k3_warp_typed<float>(
            go,
            h,
            kc,
            mix,
            off,
            full4096_offset0
        );
    }
    if (h.scalar_type() == at::ScalarType::Half) {
        return backward_mid_n6k3_warp_typed<c10::Half>(
            go,
            h,
            kc,
            mix,
            off,
            full4096_offset0
        );
    }
    return backward_mid_n6k3_warp_typed<c10::BFloat16>(
        go,
        h,
        kc,
        mix,
        off,
        full4096_offset0
    );
}

// ======================================================================================
// Base coalesced wrappers
// ======================================================================================

template <typename scalar_t>
static std::vector<torch::Tensor> backward_base_n6k3_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    bool full4096_offset0
) {
    FDC_DEBUG_PATH("FDC path: backward_base_n6k3");

    int L = h.size(2);
    int T = kc.size(1);

    auto fopts = h.options().dtype(torch::kFloat32);
    auto ghf = torch::empty(h.sizes(), fopts);
    auto gkf = torch::empty(kc.sizes(), fopts);
    auto gmf = torch::empty(mix.sizes(), fopts);
    auto base = torch::empty({3, T, 256}, fopts);
    auto mixT = torch::empty({6, 256}, fopts);

    constexpr int T_TILE = 8;
    constexpr int MIX_TILE_T = 1024;

    int tiles = (T + MIX_TILE_T - 1) / MIX_TILE_T;
    auto partial = torch::empty({tiles, 256, 6}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;

    make_mix_transpose_float_kernel<scalar_t><<<
        (256 * 6 + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        mix.data_ptr<scalar_t>(),
        mixT.data_ptr<float>(),
        256,
        6
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    make_base_ktd_float_kernel<scalar_t><<<
        ((int64_t)3 * T * 256 + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        base.data_ptr<float>(),
        256,
        L,
        T,
        3,
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (full4096_offset0) {
        full4096_n6k3_grad_h_kernel<scalar_t><<<
            (256 * 4096 + 255) / 256,
            256,
            0,
            stream
        >>>(
            go.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            ghf.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        dim3 bh(16, 16);
        dim3 gh((L + 15) / 16, 16, 1);
        small_n6k3_grad_h_kernel<scalar_t><<<gh, bh, 0, stream>>>(
            go.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            mix.data_ptr<scalar_t>(),
            ghf.data_ptr<float>(),
            L,
            T,
            (int)off
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    dim3 gkg((T + T_TILE - 1) / T_TILE, 3, 1);
    base_n6k3_grad_kernel_warp_kernel<T_TILE><<<
        gkg,
        256,
        0,
        stream
    >>>(
        base.data_ptr<float>(),
        mixT.data_ptr<float>(),
        gkf.data_ptr<float>(),
        256,
        T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int dtype_tag = 0;
    if (h.scalar_type() == at::ScalarType::Half) dtype_tag = 1;
    if (h.scalar_type() == at::ScalarType::BFloat16) dtype_tag = 2;

    dim3 pg(256, tiles, 1);
    base_n6k3_grad_mix_partial_kernel<MIX_TILE_T><<<
        pg,
        256,
        0,
        stream
    >>>(
        base.data_ptr<float>(),
        static_cast<const void*>(kc.data_ptr<scalar_t>()),
        partial.data_ptr<float>(),
        T,
        dtype_tag
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 fg(256, 6, 1);
    grad_mix_finalize_n_all_kernel<6><<<
        fg,
        256,
        0,
        stream
    >>>(
        partial.data_ptr<float>(),
        gmf.data_ptr<float>(),
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if constexpr (std::is_same<scalar_t, float>::value) {
        return {ghf, gkf, gmf};
    } else {
        auto gh = torch::empty_like(h);
        auto gk = torch::empty_like(kc);
        auto gm = torch::empty_like(mix);

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

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {gh, gk, gm};
    }
}

template <typename scalar_t>
static std::vector<torch::Tensor> backward_base_n16k3_typed(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    FDC_DEBUG_PATH("FDC path: backward_base_n16k3");

    int L = h.size(2);
    int T = kc.size(1);

    auto fopts = h.options().dtype(torch::kFloat32);
    auto ghf = torch::empty(h.sizes(), fopts);
    auto gkf = torch::empty(kc.sizes(), fopts);
    auto gmf = torch::empty(mix.sizes(), fopts);
    auto base = torch::empty({3, T, 256}, fopts);
    auto mixT = torch::empty({16, 256}, fopts);

    constexpr int T_TILE = 8;
    constexpr int N_TILE = 4;
    constexpr int MIX_TILE_T = 1024;

    int tiles = (T + MIX_TILE_T - 1) / MIX_TILE_T;
    auto partial = torch::empty({tiles, 256, 16}, fopts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;

    make_mix_transpose_float_kernel<scalar_t><<<
        (256 * 16 + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        mix.data_ptr<scalar_t>(),
        mixT.data_ptr<float>(),
        256,
        16
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    make_base_ktd_float_kernel<scalar_t><<<
        ((int64_t)3 * T * 256 + threads - 1) / threads,
        threads,
        0,
        stream
    >>>(
        go.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        base.data_ptr<float>(),
        256,
        L,
        T,
        3,
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 bh(16, 16);
    dim3 gh((L + 15) / 16, 16, 1);
    mid_n16k3_grad_h_kernel<scalar_t><<<gh, bh, 0, stream>>>(
        go.data_ptr<scalar_t>(),
        kc.data_ptr<scalar_t>(),
        mix.data_ptr<scalar_t>(),
        ghf.data_ptr<float>(),
        L,
        T,
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 gkg((T + T_TILE - 1) / T_TILE, 3, 4);
    base_n16k3_grad_kernel_warp_kernel<T_TILE, N_TILE><<<
        gkg,
        256,
        0,
        stream
    >>>(
        base.data_ptr<float>(),
        mixT.data_ptr<float>(),
        gkf.data_ptr<float>(),
        256,
        T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int dtype_tag = 0;
    if (h.scalar_type() == at::ScalarType::Half) dtype_tag = 1;
    if (h.scalar_type() == at::ScalarType::BFloat16) dtype_tag = 2;

    dim3 pg(256, 4, tiles);
    base_n16k3_grad_mix_partial_kernel<MIX_TILE_T, N_TILE><<<
        pg,
        256,
        0,
        stream
    >>>(
        base.data_ptr<float>(),
        static_cast<const void*>(kc.data_ptr<scalar_t>()),
        partial.data_ptr<float>(),
        T,
        dtype_tag
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 fg(256, 16, 1);
    grad_mix_finalize_n_all_kernel<16><<<
        fg,
        256,
        0,
        stream
    >>>(
        partial.data_ptr<float>(),
        gmf.data_ptr<float>(),
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if constexpr (std::is_same<scalar_t, float>::value) {
        return {ghf, gkf, gmf};
    } else {
        auto gh = torch::empty_like(h);
        auto gk = torch::empty_like(kc);
        auto gm = torch::empty_like(mix);

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

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {gh, gk, gm};
    }
}

static std::vector<torch::Tensor> backward_base_n6k3(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    bool full4096_offset0
) {
    if (h.scalar_type() == at::ScalarType::Float) {
        return backward_base_n6k3_typed<float>(go, h, kc, mix, off, full4096_offset0);
    }
    if (h.scalar_type() == at::ScalarType::Half) {
        return backward_base_n6k3_typed<c10::Half>(go, h, kc, mix, off, full4096_offset0);
    }
    return backward_base_n6k3_typed<c10::BFloat16>(go, h, kc, mix, off, full4096_offset0);
}

static std::vector<torch::Tensor> backward_base_n16k3(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    if (h.scalar_type() == at::ScalarType::Float) {
        return backward_base_n16k3_typed<float>(go, h, kc, mix, off);
    }
    if (h.scalar_type() == at::ScalarType::Half) {
        return backward_base_n16k3_typed<c10::Half>(go, h, kc, mix, off);
    }
    return backward_base_n16k3_typed<c10::BFloat16>(go, h, kc, mix, off);
}

// ======================================================================================
// Large path
// ======================================================================================

static std::vector<torch::Tensor> backward_large(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FDC path: backward_large");

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

    check_cublas(
        cublasSetStream(handle, stream),
        "cublasSetStream failed"
    );

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
                    grad_h_atomic_kernel<scalar_t, 8, 32><<<
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

// ======================================================================================
// fp16 GemmEx v3
// ======================================================================================

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

        float v = to_float_dev(
            kc[((int64_t)b * T + t) * N * K + n * K + kk]
        );

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
    if (i < total) {
        mixh[i] = __float2half_rn(to_float_dev(mix[i]));
    }
}

static std::vector<torch::Tensor> backward_fp16_gemmex_v3(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    FDC_DEBUG_PATH("FDC path: backward_fp16_gemmex_v3");

    int B = h.size(0);
    int D = h.size(1);
    int L = h.size(2);
    int T = kc.size(1);
    int N = kc.size(2);
    int K = kc.size(3);

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

    check_cublas(
        cublasSetStream(handle, stream),
        "cublasSetStream failed"
    );

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

    dim3 block_h(16, 16);
    dim3 grid_h((L + 15) / 16, (D + 15) / 16, B);

    grad_h_gather_kernel<c10::Half><<<
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
        (int)off
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

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
            "GemmEx v3 grad_kernel float output failed"
        );

        copy_gk_float_to_scalar_kernel<c10::Half><<<
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
            "GemmEx v3 grad_mix float output failed"
        );
    }

    auto gh = torch::empty_like(h);
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

// ======================================================================================
// Public dispatch
// ======================================================================================

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
    int L = h.size(2);
    int T = kc.size(1);
    int N = kc.size(2);
    int K = kc.size(3);

    bool is_fp32 = h.scalar_type() == at::ScalarType::Float;
    bool is_fp16 = h.scalar_type() == at::ScalarType::Half;
    bool is_bf16 = h.scalar_type() == at::ScalarType::BFloat16;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, h.get_device());
    int sm = prop.major * 10 + prop.minor;

    bool base_d256_k3 =
        B == 1 &&
        D == 256 &&
        K == 3 &&
        (int)dilation == 1;

    bool shape_n6 =
        base_d256_k3 &&
        N == 6;

    bool shape_n16 =
        base_d256_k3 &&
        N == 16;

    bool full4096_offset0_n6 =
        shape_n6 &&
        L == 4096 &&
        T == 4096 &&
        (int)off == 0;

    if (is_fp16 && shape_n6 && T == 8192 && sm >= 75) {
        return backward_fp16_gemmex_v3(
            go,
            h,
            kc,
            mix,
            off,
            dilation
        );
    }

    if (shape_n16 && T == 8192) {
        return backward_base_n16k3(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    if (shape_n6 && is_fp32 && (T == 4096 || T == 8192)) {
        return backward_base_n6k3(
            go,
            h,
            kc,
            mix,
            off,
            full4096_offset0_n6
        );
    }

    if (shape_n6 && is_bf16 && T == 8192) {
        return backward_mid_n6k3_warp(
            go,
            h,
            kc,
            mix,
            off,
            false
        );
    }

    if (full4096_offset0_n6 && is_fp16) {
        return backward_mid_n6k3_warp(
            go,
            h,
            kc,
            mix,
            off,
            true
        );
    }

    if (shape_n6 && T <= 1024) {
        return backward_small_n6k3(
            go,
            h,
            kc,
            mix,
            off
        );
    }

    if (shape_n6 && is_bf16 && T == 4096) {
        return backward_small_n6k3(
            go,
            h,
            kc,
            mix,
            off
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
