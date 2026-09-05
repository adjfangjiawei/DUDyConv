#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "fused_dynamic_conv_common.cuh"
#include "fused_dynamic_conv_plans.h"

// ======================================================================================
// Specialized backward plan:
//   N <= 16
//   K = 3
//   D = 256
//   B = 1
//   dilation = 1
//   dtype = float32
//
// Important:
//
//   This plan is named n16_k3_d256 because its kernel is optimized for N up to 16.
//   It is NOT restricted to exactly N == 16.
//
//   For example:
//     N=6  can enter this plan and let warmup compare it with n6/base/mid/large.
//     N=16 can enter this plan.
//     N=32 should NOT enter this plan, but can enter n32 plan if that plan supports it.
//
// Forward definition assumed:
//
//   out[b,d,t] = sum_{n=0}^{N-1} sum_{k=0}^{2}
//                    h[b,d,off + t - k] * kc[b,t,n,k] * mix[d,n]
//
// Backward:
//
//   grad_h[b,d,s] += go[b,d,t] * kc[b,t,n,k] * mix[d,n]
//      where s = off + t - k
//
//   grad_kc[b,t,n,k] = sum_d go[b,d,t] * h[b,d,off+t-k] * mix[d,n]
//
//   grad_mix[d,n] = sum_t,k go[b,d,t] * h[b,d,off+t-k] * kc[b,t,n,k]
//
// This implementation is a real specialized kernel.
// It does NOT fallback to large, many-N, cuBLAS, or any other plan.
// ======================================================================================

namespace {

constexpr int FDC_N16_D = 256;
constexpr int FDC_N16_MAX_N = 16;
constexpr int FDC_N16_K = 3;

// ======================================================================================
// Availability helper
// ======================================================================================

static inline bool fdc_backward_n16_k3_d256_check_common(
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

    if (h.size(0) != 1 || go.size(0) != 1 || kc.size(0) != 1) {
        return false;
    }

    if (h.size(1) != FDC_N16_D || go.size(1) != FDC_N16_D || mix.size(0) != FDC_N16_D) {
        return false;
    }

    if (kc.size(3) != FDC_N16_K) {
        return false;
    }

    if (go.size(2) != kc.size(1)) {
        return false;
    }

    int64_t N = kc.size(2);

    if (N <= 0 || N > FDC_N16_MAX_N) {
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

    // 这个阈值只限制过小 T 的 launch/reduction 开销。
    // 不按 N 精确限制，让 warmup 决定是否值得用。
    if (kc.size(1) < 512) {
        return false;
    }

    return true;
}

// ======================================================================================
// Kernel 1: zero tensor
// ======================================================================================

__global__ void fdc_n16_zero_kernel(
    float* __restrict__ ptr,
    int64_t total
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < total) {
        ptr[idx] = 0.0f;
    }
}

// ======================================================================================
// Kernel 2: grad_h
//
// Grid:
//   idx over D * T * K
//
// For each d,t,k:
//   s = off + t - k
//   mixed = sum_n kc[t,n,k] * mix[d,n]
//   gh[d,s] += go[d,t] * mixed
//
// Since K=3, multiple t/k can hit the same s, so atomicAdd is used.
// ======================================================================================

__global__ void fdc_backward_n16_grad_h_kernel(
    const float* __restrict__ go,
    const float* __restrict__ kc,
    const float* __restrict__ mix,
    float* __restrict__ gh,
    int L,
    int T,
    int N,
    int off
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(FDC_N16_D) * T * FDC_N16_K;

    if (idx >= total) {
        return;
    }

    int k = static_cast<int>(idx % FDC_N16_K);
    int t = static_cast<int>((idx / FDC_N16_K) % T);
    int d = static_cast<int>(idx / (static_cast<int64_t>(T) * FDC_N16_K));

    int s = off + t - k;

    if (s < 0 || s >= L) {
        return;
    }

    float sum_n = 0.0f;

    #pragma unroll
    for (int n = 0; n < FDC_N16_MAX_N; ++n) {
        if (n < N) {
            float kc_val = kc[(static_cast<int64_t>(t) * N + n) * FDC_N16_K + k];
            float mix_val = mix[static_cast<int64_t>(d) * N + n];
            sum_n += kc_val * mix_val;
        }
    }

    float go_val = go[static_cast<int64_t>(d) * T + t];
    float v = go_val * sum_n;

    atomicAdd(
        gh + static_cast<int64_t>(d) * L + s,
        v
    );
}

// ======================================================================================
// Kernel 3: grad_kc
//
// One block computes one element:
//
//   grad_kc[t,n,k] = sum_d go[d,t] * h[d,off+t-k] * mix[d,n]
//
// Grid:
//   blockIdx.x = flattened t,n,k
//
// Block:
//   256 threads over d
// ======================================================================================

__global__ void fdc_backward_n16_grad_kc_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    const float* __restrict__ mix,
    float* __restrict__ gkc,
    int L,
    int T,
    int N,
    int off
) {
    __shared__ float smem[FDC_N16_D];

    int flat = blockIdx.x;

    int k = flat % FDC_N16_K;
    int n = (flat / FDC_N16_K) % N;
    int t = flat / (FDC_N16_K * N);

    int d = threadIdx.x;
    int s = off + t - k;

    float acc = 0.0f;

    if (d < FDC_N16_D && s >= 0 && s < L) {
        float go_val = go[static_cast<int64_t>(d) * T + t];
        float h_val = h[static_cast<int64_t>(d) * L + s];
        float mix_val = mix[static_cast<int64_t>(d) * N + n];

        acc = go_val * h_val * mix_val;
    }

    smem[d] = acc;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (d < stride) {
            smem[d] += smem[d + stride];
        }

        __syncthreads();
    }

    if (d == 0) {
        gkc[(static_cast<int64_t>(t) * N + n) * FDC_N16_K + k] = smem[0];
    }
}

// ======================================================================================
// Kernel 4: grad_mix
//
// One block computes one element:
//
//   grad_mix[d,n] = sum_t,k go[d,t] * h[d,off+t-k] * kc[t,n,k]
//
// Grid:
//   blockIdx.x = d
//   blockIdx.y = n
//
// Block:
//   256 threads reduce over T*K
// ======================================================================================

__global__ void fdc_backward_n16_grad_mix_kernel(
    const float* __restrict__ go,
    const float* __restrict__ h,
    const float* __restrict__ kc,
    float* __restrict__ gmix,
    int L,
    int T,
    int N,
    int off
) {
    __shared__ float smem[256];

    int d = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;

    float acc = 0.0f;

    int total = T * FDC_N16_K;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int k = idx % FDC_N16_K;
        int t = idx / FDC_N16_K;
        int s = off + t - k;

        if (s >= 0 && s < L) {
            float go_val = go[static_cast<int64_t>(d) * T + t];
            float h_val = h[static_cast<int64_t>(d) * L + s];
            float kc_val = kc[(static_cast<int64_t>(t) * N + n) * FDC_N16_K + k];

            acc += go_val * h_val * kc_val;
        }
    }

    smem[tid] = acc;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        gmix[static_cast<int64_t>(d) * N + n] = smem[0];
    }
}

} // namespace

// ======================================================================================
// Public availability
// ======================================================================================

bool fdc_backward_n16_k3_d256_available_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    return fdc_backward_n16_k3_d256_check_common(
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );
}

// ======================================================================================
// Public runner
// ======================================================================================

std::vector<torch::Tensor> fdc_backward_n16_k3_d256_cuda(
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

    TORCH_CHECK(go.scalar_type() == at::ScalarType::Float, "n16_k3_d256 specialized backward currently supports fp32 only.");
    TORCH_CHECK(h.scalar_type() == at::ScalarType::Float, "n16_k3_d256 specialized backward currently supports fp32 only.");
    TORCH_CHECK(kc.scalar_type() == at::ScalarType::Float, "n16_k3_d256 specialized backward currently supports fp32 only.");
    TORCH_CHECK(mix.scalar_type() == at::ScalarType::Float, "n16_k3_d256 specialized backward currently supports fp32 only.");

    TORCH_CHECK(go.dim() == 3, "go must be [B,D,T].");
    TORCH_CHECK(h.dim() == 3, "h must be [B,D,L].");
    TORCH_CHECK(kc.dim() == 4, "kc must be [B,T,N,K].");
    TORCH_CHECK(mix.dim() == 2, "mix must be [D,N].");

    TORCH_CHECK(go.size(0) == 1, "go B must be 1.");
    TORCH_CHECK(h.size(0) == 1, "h B must be 1.");
    TORCH_CHECK(kc.size(0) == 1, "kc B must be 1.");

    TORCH_CHECK(go.size(1) == FDC_N16_D, "go D must be 256.");
    TORCH_CHECK(h.size(1) == FDC_N16_D, "h D must be 256.");
    TORCH_CHECK(mix.size(0) == FDC_N16_D, "mix D must be 256.");

    TORCH_CHECK(kc.size(3) == FDC_N16_K, "kc K must be 3.");
    TORCH_CHECK(go.size(2) == kc.size(1), "go T must equal kc T.");

    int L = static_cast<int>(h.size(2));
    int T = static_cast<int>(kc.size(1));
    int N = static_cast<int>(kc.size(2));

    TORCH_CHECK(N > 0 && N <= FDC_N16_MAX_N, "n16_k3_d256 supports 1 <= N <= 16.");
    TORCH_CHECK(mix.size(1) == N, "mix N must equal kc N.");

    TORCH_CHECK(off >= 0, "off must be >= 0.");
    TORCH_CHECK(off + T <= L, "off + T must be <= L.");

    FDC_DEBUG_PATH("FDC path: backward_n16_k3_d256");

    auto gh = torch::empty_like(h);
    auto gkc = torch::empty_like(kc);
    auto gmix = torch::empty_like(mix);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int threads = 256;

    // ----------------------------------------------------------------------------------
    // Zero grad_h
    // ----------------------------------------------------------------------------------

    int64_t gh_total = h.numel();

    fdc_n16_zero_kernel<<<
        static_cast<unsigned int>((gh_total + threads - 1) / threads),
        threads,
        0,
        stream
    >>>(
        gh.data_ptr<float>(),
        gh_total
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ----------------------------------------------------------------------------------
    // grad_h
    // ----------------------------------------------------------------------------------

    int64_t grad_h_total =
        static_cast<int64_t>(FDC_N16_D) *
        static_cast<int64_t>(T) *
        static_cast<int64_t>(FDC_N16_K);

    fdc_backward_n16_grad_h_kernel<<<
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
        static_cast<int>(off)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ----------------------------------------------------------------------------------
    // grad_kc
    // ----------------------------------------------------------------------------------

    int grad_kc_blocks =
        T *
        N *
        FDC_N16_K;

    fdc_backward_n16_grad_kc_kernel<<<
        static_cast<unsigned int>(grad_kc_blocks),
        FDC_N16_D,
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

    // ----------------------------------------------------------------------------------
    // grad_mix
    // ----------------------------------------------------------------------------------

    dim3 grad_mix_grid(
        FDC_N16_D,
        static_cast<unsigned int>(N),
        1
    );

    fdc_backward_n16_grad_mix_kernel<<<
        grad_mix_grid,
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
