#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Optimized specialized backward plan:
//   N <= 32
//   K = 3
//   D = 256
//   B = 1
//   dilation = 1
//   dtype = float32
//
// Main optimizations:
//   1. grad_h: gather style, no global atomicAdd.
//   2. grad_kc: one block computes all N<=32 values for one (t,k).
//      Old: T*N*K blocks.
//      New: T*K blocks.
//   3. grad_mix: one block computes all N<=32 values for one d.
//      Old: D*N blocks.
//      New: D blocks.
// ======================================================================================

namespace {

constexpr int FDC_N32_D = 256;
constexpr int FDC_N32_MAX_N = 32;
constexpr int FDC_N32_K = 3;
constexpr int FDC_N32_THREADS = 256;

// shared memory layout for 32 reductions of 256 lanes:
// smem[n * 256 + tid]
constexpr int FDC_N32_REDUCE_SMEM_SIZE = FDC_N32_MAX_N * FDC_N32_THREADS;

static inline bool fdc_backward_n32_k3_d256_check_common(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    if (!go.defined() || !h.defined() || !kc.defined() || !mix.defined()) {
        return false;
    }

    if (!go.is_cuda() || !h.is_cuda() || !kc.is_cuda() || !mix.is_cuda()) {
        return false;
    }

    if (!go.is_contiguous() || !h.is_contiguous() || !kc.is_contiguous() || !mix.is_contiguous()) {
        return false;
    }

    if (go.dim() != 3 || h.dim() != 3 || kc.dim() != 4 || mix.dim() != 2) {
        return false;
    }

    if (go.scalar_type() != at::ScalarType::Float ||
        h.scalar_type() != at::ScalarType::Float ||
        kc.scalar_type() != at::ScalarType::Float ||
        mix.scalar_type() != at::ScalarType::Float) {
        return false;
    }

    if (go.size(0) != 1 || h.size(0) != 1 || kc.size(0) != 1) {
        return false;
    }

    if (go.size(1) != FDC_N32_D || h.size(1) != FDC_N32_D || mix.size(0) != FDC_N32_D) {
        return false;
    }

    if (kc.size(3) != FDC_N32_K) {
        return false;
    }

    if (go.size(2) != kc.size(1)) {
        return false;
    }

    int64_t N = kc.size(2);

    if (N <= 0 || N > FDC_N32_MAX_N) {
        return false;
    }

    if (mix.size(1) != N) {
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

    if (kc.size(1) < 512) {
        return false;
    }

    return true;
}

// ======================================================================================
// Zero kernel
// ======================================================================================

__global__ void fdc_n32_zero_kernel(
    float* __restrict__ ptr,
    int64_t total
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < total) {
        ptr[idx] = 0.0f;
    }
}

// ======================================================================================
// grad_h gather kernel, no atomic.
//
// One thread computes one gh[d,s].
//
// gh[d,s] = sum_k valid(t=s-off+k):
//              go[d,t] * sum_n kc[t,n,k] * mix[d,n]
// ======================================================================================

__global__ void fdc_backward_n32_grad_h_gather_kernel(
    const float* __restrict__ go,
    const float* __restrict__ kc,
    const float* __restrict__ mix,
    float* __restrict__ gh,
    int L,
    int T,
    int N,
    int off,
    int s_begin,
    int s_count
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(FDC_N32_D) * static_cast<int64_t>(s_count);

    if (idx >= total) {
        return;
    }

    int local_s = static_cast<int>(idx % s_count);
    int d = static_cast<int>(idx / s_count);
    int s = s_begin + local_s;

    const float* __restrict__ mix_ptr = mix + static_cast<int64_t>(d) * N;

    float acc_total = 0.0f;

    #pragma unroll
    for (int k = 0; k < FDC_N32_K; ++k) {
        int t = s - off + k;

        if (t >= 0 && t < T) {
            const float* __restrict__ kc_ptr =
                kc + (static_cast<int64_t>(t) * N) * FDC_N32_K + k;

            float sum_n = 0.0f;

            #pragma unroll
            for (int n = 0; n < FDC_N32_MAX_N; ++n) {
                if (n < N) {
                    sum_n += kc_ptr[static_cast<int64_t>(n) * FDC_N32_K] * mix_ptr[n];
                }
            }

            acc_total += go[static_cast<int64_t>(d) * T + t] * sum_n;
        }
    }

    gh[static_cast<int64_t>(d) * L + s] = acc_total;
}

// ======================================================================================
// Optimized grad_kc kernel.
//
// One block computes gkc[t, 0:N, k] for one pair (t,k).
//
// Old:
//   one block per (t,n,k)
//   blocks = T * N * K
//
// New:
//   one block per (t,k)
//   blocks = T * K
//
// For each n:
//   gkc[t,n,k] = sum_d go[d,t] * h[d, off+t-k] * mix[d,n]
//
// Shared memory:
//   smem[n][d] = base_d * mix[d,n]
//   base_d = go[d,t] * h[d,s]
// ======================================================================================

__global__ void fdc_backward_n32_grad_kc_alln_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    const float* __restrict__ mix,
    float* __restrict__ gkc,
    int L,
    int T,
    int N,
    int off
) {
    __shared__ float smem[FDC_N32_REDUCE_SMEM_SIZE];

    int flat = blockIdx.x;
    int k = flat % FDC_N32_K;
    int t = flat / FDC_N32_K;

    int tid = threadIdx.x;
    int d = tid;

    int s = off + t - k;

    float base = 0.0f;

    if (s >= 0 && s < L) {
        float go_val = go[static_cast<int64_t>(d) * T + t];
        float h_val = h[static_cast<int64_t>(d) * L + s];
        base = go_val * h_val;
    }

    // Compute all n reductions.
    #pragma unroll
    for (int n = 0; n < FDC_N32_MAX_N; ++n) {
        float v = 0.0f;

        if (n < N) {
            v = base * mix[static_cast<int64_t>(d) * N + n];
        }

        smem[n * FDC_N32_THREADS + tid] = v;
    }

    __syncthreads();

    // Reduce D dimension for every n.
    // 256 -> 128 -> 64
    if (tid < 128) {
        #pragma unroll
        for (int n = 0; n < FDC_N32_MAX_N; ++n) {
            if (n < N) {
                smem[n * FDC_N32_THREADS + tid] +=
                    smem[n * FDC_N32_THREADS + tid + 128];
            }
        }
    }

    __syncthreads();

    if (tid < 64) {
        #pragma unroll
        for (int n = 0; n < FDC_N32_MAX_N; ++n) {
            if (n < N) {
                smem[n * FDC_N32_THREADS + tid] +=
                    smem[n * FDC_N32_THREADS + tid + 64];
            }
        }
    }

    __syncthreads();

    // Warp reduction and write.
    if (tid < 32) {
        #pragma unroll
        for (int n = 0; n < FDC_N32_MAX_N; ++n) {
            if (n < N) {
                volatile float* vmem = smem + n * FDC_N32_THREADS;

                float v = vmem[tid];

                v += vmem[tid + 32];
                vmem[tid] = v;

                v += vmem[tid + 16];
                vmem[tid] = v;

                v += vmem[tid + 8];
                vmem[tid] = v;

                v += vmem[tid + 4];
                vmem[tid] = v;

                v += vmem[tid + 2];
                vmem[tid] = v;

                v += vmem[tid + 1];
                vmem[tid] = v;
            }
        }
    }

    if (tid == 0) {
        #pragma unroll
        for (int n = 0; n < FDC_N32_MAX_N; ++n) {
            if (n < N) {
                gkc[(static_cast<int64_t>(t) * N + n) * FDC_N32_K + k] =
                    smem[n * FDC_N32_THREADS];
            }
        }
    }
}

// ======================================================================================
// Optimized grad_mix kernel.
//
// One block computes gmix[d, 0:N] for one d.
//
// Old:
//   one block per (d,n)
//   blocks = D * N
//
// New:
//   one block per d
//   blocks = D
//
// For each n:
//   gmix[d,n] = sum_t,k go[d,t] * h[d, off+t-k] * kc[t,n,k]
//
// Each thread accumulates 32 n values in registers over strided t.
// Then block reduces 32 independent reductions.
// ======================================================================================

__global__ void fdc_backward_n32_grad_mix_alln_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    const float* __restrict__ kc,
    float* __restrict__ gmix,
    int L,
    int T,
    int N,
    int off
) {
    __shared__ float smem[FDC_N32_REDUCE_SMEM_SIZE];

    int d = blockIdx.x;
    int tid = threadIdx.x;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc8 = 0.0f;
    float acc9 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    float acc16 = 0.0f;
    float acc17 = 0.0f;
    float acc18 = 0.0f;
    float acc19 = 0.0f;
    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;
    float acc24 = 0.0f;
    float acc25 = 0.0f;
    float acc26 = 0.0f;
    float acc27 = 0.0f;
    float acc28 = 0.0f;
    float acc29 = 0.0f;
    float acc30 = 0.0f;
    float acc31 = 0.0f;

    const float* __restrict__ go_ptr = go + static_cast<int64_t>(d) * T;
    const float* __restrict__ h_ptr = h + static_cast<int64_t>(d) * L;

    for (int t = tid; t < T; t += FDC_N32_THREADS) {
        float go_val = go_ptr[t];

        int s0 = off + t;
        int s1 = off + t - 1;
        int s2 = off + t - 2;

        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;

        if (s0 >= 0 && s0 < L) {
            a0 = go_val * h_ptr[s0];
        }

        if (s1 >= 0 && s1 < L) {
            a1 = go_val * h_ptr[s1];
        }

        if (s2 >= 0 && s2 < L) {
            a2 = go_val * h_ptr[s2];
        }

        const float* __restrict__ kc_base =
            kc + static_cast<int64_t>(t) * N * FDC_N32_K;

        if (N > 0)  acc0  += a0 * kc_base[0  * 3 + 0] + a1 * kc_base[0  * 3 + 1] + a2 * kc_base[0  * 3 + 2];
        if (N > 1)  acc1  += a0 * kc_base[1  * 3 + 0] + a1 * kc_base[1  * 3 + 1] + a2 * kc_base[1  * 3 + 2];
        if (N > 2)  acc2  += a0 * kc_base[2  * 3 + 0] + a1 * kc_base[2  * 3 + 1] + a2 * kc_base[2  * 3 + 2];
        if (N > 3)  acc3  += a0 * kc_base[3  * 3 + 0] + a1 * kc_base[3  * 3 + 1] + a2 * kc_base[3  * 3 + 2];
        if (N > 4)  acc4  += a0 * kc_base[4  * 3 + 0] + a1 * kc_base[4  * 3 + 1] + a2 * kc_base[4  * 3 + 2];
        if (N > 5)  acc5  += a0 * kc_base[5  * 3 + 0] + a1 * kc_base[5  * 3 + 1] + a2 * kc_base[5  * 3 + 2];
        if (N > 6)  acc6  += a0 * kc_base[6  * 3 + 0] + a1 * kc_base[6  * 3 + 1] + a2 * kc_base[6  * 3 + 2];
        if (N > 7)  acc7  += a0 * kc_base[7  * 3 + 0] + a1 * kc_base[7  * 3 + 1] + a2 * kc_base[7  * 3 + 2];
        if (N > 8)  acc8  += a0 * kc_base[8  * 3 + 0] + a1 * kc_base[8  * 3 + 1] + a2 * kc_base[8  * 3 + 2];
        if (N > 9)  acc9  += a0 * kc_base[9  * 3 + 0] + a1 * kc_base[9  * 3 + 1] + a2 * kc_base[9  * 3 + 2];
        if (N > 10) acc10 += a0 * kc_base[10 * 3 + 0] + a1 * kc_base[10 * 3 + 1] + a2 * kc_base[10 * 3 + 2];
        if (N > 11) acc11 += a0 * kc_base[11 * 3 + 0] + a1 * kc_base[11 * 3 + 1] + a2 * kc_base[11 * 3 + 2];
        if (N > 12) acc12 += a0 * kc_base[12 * 3 + 0] + a1 * kc_base[12 * 3 + 1] + a2 * kc_base[12 * 3 + 2];
        if (N > 13) acc13 += a0 * kc_base[13 * 3 + 0] + a1 * kc_base[13 * 3 + 1] + a2 * kc_base[13 * 3 + 2];
        if (N > 14) acc14 += a0 * kc_base[14 * 3 + 0] + a1 * kc_base[14 * 3 + 1] + a2 * kc_base[14 * 3 + 2];
        if (N > 15) acc15 += a0 * kc_base[15 * 3 + 0] + a1 * kc_base[15 * 3 + 1] + a2 * kc_base[15 * 3 + 2];
        if (N > 16) acc16 += a0 * kc_base[16 * 3 + 0] + a1 * kc_base[16 * 3 + 1] + a2 * kc_base[16 * 3 + 2];
        if (N > 17) acc17 += a0 * kc_base[17 * 3 + 0] + a1 * kc_base[17 * 3 + 1] + a2 * kc_base[17 * 3 + 2];
        if (N > 18) acc18 += a0 * kc_base[18 * 3 + 0] + a1 * kc_base[18 * 3 + 1] + a2 * kc_base[18 * 3 + 2];
        if (N > 19) acc19 += a0 * kc_base[19 * 3 + 0] + a1 * kc_base[19 * 3 + 1] + a2 * kc_base[19 * 3 + 2];
        if (N > 20) acc20 += a0 * kc_base[20 * 3 + 0] + a1 * kc_base[20 * 3 + 1] + a2 * kc_base[20 * 3 + 2];
        if (N > 21) acc21 += a0 * kc_base[21 * 3 + 0] + a1 * kc_base[21 * 3 + 1] + a2 * kc_base[21 * 3 + 2];
        if (N > 22) acc22 += a0 * kc_base[22 * 3 + 0] + a1 * kc_base[22 * 3 + 1] + a2 * kc_base[22 * 3 + 2];
        if (N > 23) acc23 += a0 * kc_base[23 * 3 + 0] + a1 * kc_base[23 * 3 + 1] + a2 * kc_base[23 * 3 + 2];
        if (N > 24) acc24 += a0 * kc_base[24 * 3 + 0] + a1 * kc_base[24 * 3 + 1] + a2 * kc_base[24 * 3 + 2];
        if (N > 25) acc25 += a0 * kc_base[25 * 3 + 0] + a1 * kc_base[25 * 3 + 1] + a2 * kc_base[25 * 3 + 2];
        if (N > 26) acc26 += a0 * kc_base[26 * 3 + 0] + a1 * kc_base[26 * 3 + 1] + a2 * kc_base[26 * 3 + 2];
        if (N > 27) acc27 += a0 * kc_base[27 * 3 + 0] + a1 * kc_base[27 * 3 + 1] + a2 * kc_base[27 * 3 + 2];
        if (N > 28) acc28 += a0 * kc_base[28 * 3 + 0] + a1 * kc_base[28 * 3 + 1] + a2 * kc_base[28 * 3 + 2];
        if (N > 29) acc29 += a0 * kc_base[29 * 3 + 0] + a1 * kc_base[29 * 3 + 1] + a2 * kc_base[29 * 3 + 2];
        if (N > 30) acc30 += a0 * kc_base[30 * 3 + 0] + a1 * kc_base[30 * 3 + 1] + a2 * kc_base[30 * 3 + 2];
        if (N > 31) acc31 += a0 * kc_base[31 * 3 + 0] + a1 * kc_base[31 * 3 + 1] + a2 * kc_base[31 * 3 + 2];
    }

    if (N > 0)  smem[0  * 256 + tid] = acc0;
    if (N > 1)  smem[1  * 256 + tid] = acc1;
    if (N > 2)  smem[2  * 256 + tid] = acc2;
    if (N > 3)  smem[3  * 256 + tid] = acc3;
    if (N > 4)  smem[4  * 256 + tid] = acc4;
    if (N > 5)  smem[5  * 256 + tid] = acc5;
    if (N > 6)  smem[6  * 256 + tid] = acc6;
    if (N > 7)  smem[7  * 256 + tid] = acc7;
    if (N > 8)  smem[8  * 256 + tid] = acc8;
    if (N > 9)  smem[9  * 256 + tid] = acc9;
    if (N > 10) smem[10 * 256 + tid] = acc10;
    if (N > 11) smem[11 * 256 + tid] = acc11;
    if (N > 12) smem[12 * 256 + tid] = acc12;
    if (N > 13) smem[13 * 256 + tid] = acc13;
    if (N > 14) smem[14 * 256 + tid] = acc14;
    if (N > 15) smem[15 * 256 + tid] = acc15;
    if (N > 16) smem[16 * 256 + tid] = acc16;
    if (N > 17) smem[17 * 256 + tid] = acc17;
    if (N > 18) smem[18 * 256 + tid] = acc18;
    if (N > 19) smem[19 * 256 + tid] = acc19;
    if (N > 20) smem[20 * 256 + tid] = acc20;
    if (N > 21) smem[21 * 256 + tid] = acc21;
    if (N > 22) smem[22 * 256 + tid] = acc22;
    if (N > 23) smem[23 * 256 + tid] = acc23;
    if (N > 24) smem[24 * 256 + tid] = acc24;
    if (N > 25) smem[25 * 256 + tid] = acc25;
    if (N > 26) smem[26 * 256 + tid] = acc26;
    if (N > 27) smem[27 * 256 + tid] = acc27;
    if (N > 28) smem[28 * 256 + tid] = acc28;
    if (N > 29) smem[29 * 256 + tid] = acc29;
    if (N > 30) smem[30 * 256 + tid] = acc30;
    if (N > 31) smem[31 * 256 + tid] = acc31;

    __syncthreads();

    if (tid < 128) {
        #pragma unroll
        for (int n = 0; n < FDC_N32_MAX_N; ++n) {
            if (n < N) {
                smem[n * 256 + tid] += smem[n * 256 + tid + 128];
            }
        }
    }

    __syncthreads();

    if (tid < 64) {
        #pragma unroll
        for (int n = 0; n < FDC_N32_MAX_N; ++n) {
            if (n < N) {
                smem[n * 256 + tid] += smem[n * 256 + tid + 64];
            }
        }
    }

    __syncthreads();

    if (tid < 32) {
        #pragma unroll
        for (int n = 0; n < FDC_N32_MAX_N; ++n) {
            if (n < N) {
                volatile float* v = smem + n * 256;
                float x = v[tid];

                x += v[tid + 32];
                v[tid] = x;
                x += v[tid + 16];
                v[tid] = x;
                x += v[tid + 8];
                v[tid] = x;
                x += v[tid + 4];
                v[tid] = x;
                x += v[tid + 2];
                v[tid] = x;
                x += v[tid + 1];
                v[tid] = x;
            }
        }
    }

    if (tid == 0) {
        #pragma unroll
        for (int n = 0; n < FDC_N32_MAX_N; ++n) {
            if (n < N) {
                gmix[static_cast<int64_t>(d) * N + n] = smem[n * 256];
            }
        }
    }
}

} // namespace

bool fdc_backward_n32_k3_d256_available_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    return fdc_backward_n32_k3_d256_check_common(
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );
}

std::vector<torch::Tensor> fdc_backward_n32_k3_d256_cuda(
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

    TORCH_CHECK(go.scalar_type() == at::ScalarType::Float, "n32_k3_d256 supports fp32 only.");
    TORCH_CHECK(h.scalar_type() == at::ScalarType::Float, "n32_k3_d256 supports fp32 only.");
    TORCH_CHECK(kc.scalar_type() == at::ScalarType::Float, "n32_k3_d256 supports fp32 only.");
    TORCH_CHECK(mix.scalar_type() == at::ScalarType::Float, "n32_k3_d256 supports fp32 only.");

    TORCH_CHECK(go.dim() == 3, "go must be [B,D,T].");
    TORCH_CHECK(h.dim() == 3, "h must be [B,D,L].");
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,T,N,K].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(go.size(0) == 1, "go B must be 1.");
    TORCH_CHECK(h.size(0) == 1, "h B must be 1.");
    TORCH_CHECK(kc.size(0) == 1, "kc B must be 1.");

    TORCH_CHECK(go.size(1) == FDC_N32_D, "go D must be 256.");
    TORCH_CHECK(h.size(1) == FDC_N32_D, "h D must be 256.");
    TORCH_CHECK(mix.size(0) == FDC_N32_D, "mix D must be 256.");
    TORCH_CHECK(kc.size(3) == FDC_N32_K, "kc K must be 3.");
    TORCH_CHECK(go.size(2) == kc.size(1), "go T must equal kc T.");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));

    TORCH_CHECK(N > 0 && N <= FDC_N32_MAX_N, "n32_k3_d256 supports 1 <= N <= 32.");
    TORCH_CHECK(mix.size(1) == N, "mix N must equal kc N.");

    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + T <= L, "off + T must be <= L.");

    FDC_DEBUG_PATH("FDC path: backward_n32_k3_d256_alln_optimized");

    auto gh = torch::empty_like(h);
    auto gkc = torch::empty_like(kc);
    auto gmix = torch::empty_like(mix);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int threads = FDC_N32_THREADS;

    // zero gh because grad_h only writes touched range.
    int64_t gh_total = h.numel();

    fdc_n32_zero_kernel<<<
        static_cast<unsigned int>((gh_total + threads - 1) / threads),
        threads,
        0,
        stream
    >>>(
        gh.data_ptr<float>(),
        gh_total
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // grad_h gather
    int s_begin = off >= 2 ? static_cast<int>(off - 2) : 0;
    int s_end = static_cast<int>(off + T);

    if (s_end > L) {
        s_end = L;
    }

    int s_count = s_end - s_begin;

    if (s_count > 0) {
        int64_t grad_h_total =
            static_cast<int64_t>(FDC_N32_D) *
            static_cast<int64_t>(s_count);

        fdc_backward_n32_grad_h_gather_kernel<<<
            static_cast<unsigned int>((grad_h_total + threads - 1) / threads),
            threads,
            0,
            stream
        >>>(
            go.data_ptr<float>(),
            kc.data_ptr<float>(),
            mix.data_ptr<float>(),
            gh.data_ptr<float>(),
            L,
            T,
            N,
            static_cast<int>(off),
            s_begin,
            s_count
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // grad_kc: one block per (t,k), computes all n.
    int grad_kc_blocks = T * FDC_N32_K;

    fdc_backward_n32_grad_kc_alln_kernel<<<
        static_cast<unsigned int>(grad_kc_blocks),
        threads,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        h.data_ptr<float>(),
        mix.data_ptr<float>(),
        gkc.data_ptr<float>(),
        L,
        T,
        N,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // grad_mix: one block per d, computes all n.
    fdc_backward_n32_grad_mix_alln_kernel<<<
        static_cast<unsigned int>(FDC_N32_D),
        threads,
        0,
        stream
    >>>(
        go.data_ptr<float>(),
        h.data_ptr<float>(),
        kc.data_ptr<float>(),
        gmix.data_ptr<float>(),
        L,
        T,
        N,
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        gh,
        gkc,
        gmix
    };
}
