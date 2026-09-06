#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>
#include <cstdint>

#include "fused_dynamic_conv_common.cuh"
#include "fused_source_conv_mixed_plans.h"

// ======================================================================================
// Fused Source Conv Mixed Backward Main
//
// Implements:
//   - backward plan name
//   - common validation
//   - heuristic plan selection
//   - explicit plan dispatch
//   - default backward entry
//   - cached-plan backward entry
//   - warmup benchmark entry
//
// Does not implement:
//   - pybind
//   - Python wrapper
//   - autograd Function
//   - dropout
//   - residual
// ======================================================================================

namespace {

constexpr int64_t FSCM_BWD_MAX_C_REDUCE = 64;
constexpr int64_t FSCM_BWD_MAX_S_REDUCE = 8;

// ======================================================================================
// Common validation
// ======================================================================================

static void fscm_backward_check_common_inputs(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    const char* fn_name
) {
    TORCH_CHECK(
        grad_out.defined(),
        fn_name,
        ": grad_out must be defined."
    );

    TORCH_CHECK(
        x_norm.defined(),
        fn_name,
        ": x_norm must be defined."
    );

    TORCH_CHECK(
        weight.defined(),
        fn_name,
        ": weight must be defined."
    );

    TORCH_CHECK(
        mix_weight.defined(),
        fn_name,
        ": mix_weight must be defined."
    );

    TORCH_CHECK(
        mix_bias.defined(),
        fn_name,
        ": mix_bias must be defined."
    );

    TORCH_CHECK(
        grad_out.is_cuda(),
        fn_name,
        ": grad_out must be CUDA tensor."
    );

    TORCH_CHECK(
        x_norm.is_cuda(),
        fn_name,
        ": x_norm must be CUDA tensor."
    );

    TORCH_CHECK(
        weight.is_cuda(),
        fn_name,
        ": weight must be CUDA tensor."
    );

    TORCH_CHECK(
        mix_weight.is_cuda(),
        fn_name,
        ": mix_weight must be CUDA tensor."
    );

    TORCH_CHECK(
        mix_bias.is_cuda(),
        fn_name,
        ": mix_bias must be CUDA tensor."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.is_cuda(),
            fn_name,
            ": bias must be CUDA tensor when defined."
        );
    }

    TORCH_CHECK(
        grad_out.is_contiguous(),
        fn_name,
        ": grad_out must be contiguous."
    );

    TORCH_CHECK(
        x_norm.is_contiguous(),
        fn_name,
        ": x_norm must be contiguous."
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        fn_name,
        ": weight must be contiguous."
    );

    TORCH_CHECK(
        mix_weight.is_contiguous(),
        fn_name,
        ": mix_weight must be contiguous."
    );

    TORCH_CHECK(
        mix_bias.is_contiguous(),
        fn_name,
        ": mix_bias must be contiguous."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.is_contiguous(),
            fn_name,
            ": bias must be contiguous when defined."
        );
    }

    TORCH_CHECK(
        grad_out.dim() == 3,
        fn_name,
        ": grad_out must be [B,D,T]."
    );

    TORCH_CHECK(
        x_norm.dim() == 3,
        fn_name,
        ": x_norm must be [B,D,L]."
    );

    TORCH_CHECK(
        weight.dim() == 3,
        fn_name,
        ": weight must be [D,C,S]."
    );

    TORCH_CHECK(
        mix_weight.dim() == 2,
        fn_name,
        ": mix_weight must be [D,C]."
    );

    TORCH_CHECK(
        mix_bias.dim() == 1,
        fn_name,
        ": mix_bias must be [D]."
    );

    int64_t B = x_norm.size(0);
    int64_t D = x_norm.size(1);
    int64_t L = x_norm.size(2);
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    TORCH_CHECK(
        B > 0,
        fn_name,
        ": B must be positive."
    );

    TORCH_CHECK(
        D > 0,
        fn_name,
        ": D must be positive."
    );

    TORCH_CHECK(
        L > 0,
        fn_name,
        ": L must be positive."
    );

    TORCH_CHECK(
        C > 0,
        fn_name,
        ": C must be positive."
    );

    TORCH_CHECK(
        S > 0,
        fn_name,
        ": S must be positive."
    );

    TORCH_CHECK(
        weight.size(0) == D,
        fn_name,
        ": weight D mismatch."
    );

    TORCH_CHECK(
        grad_out.size(0) == B,
        fn_name,
        ": grad_out B mismatch."
    );

    TORCH_CHECK(
        grad_out.size(1) == D,
        fn_name,
        ": grad_out D mismatch."
    );

    TORCH_CHECK(
        grad_out.size(2) == T,
        fn_name,
        ": grad_out T mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(0) == D,
        fn_name,
        ": mix_weight D mismatch."
    );

    TORCH_CHECK(
        mix_weight.size(1) == C,
        fn_name,
        ": mix_weight C mismatch."
    );

    TORCH_CHECK(
        mix_bias.size(0) == D,
        fn_name,
        ": mix_bias D mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            bias.dim() == 2,
            fn_name,
            ": bias must be [D,C] when defined."
        );

        TORCH_CHECK(
            bias.size(0) == D,
            fn_name,
            ": bias D mismatch."
        );

        TORCH_CHECK(
            bias.size(1) == C,
            fn_name,
            ": bias C mismatch."
        );
    }

    TORCH_CHECK(
        off >= 0,
        fn_name,
        ": off must be >= 0."
    );

    TORCH_CHECK(
        T > 0,
        fn_name,
        ": T must be positive."
    );

    TORCH_CHECK(
        off + T <= L,
        fn_name,
        ": off + T must be <= L."
    );

    TORCH_CHECK(
        dilation > 0,
        fn_name,
        ": dilation must be positive."
    );

    TORCH_CHECK(
        grad_out.scalar_type() == x_norm.scalar_type(),
        fn_name,
        ": grad_out and x_norm dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == weight.scalar_type(),
        fn_name,
        ": x_norm and weight dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == mix_weight.scalar_type(),
        fn_name,
        ": x_norm and mix_weight dtype mismatch."
    );

    TORCH_CHECK(
        x_norm.scalar_type() == mix_bias.scalar_type(),
        fn_name,
        ": x_norm and mix_bias dtype mismatch."
    );

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(
            x_norm.scalar_type() == bias.scalar_type(),
            fn_name,
            ": x_norm and bias dtype mismatch."
        );
    }

    TORCH_CHECK(
        x_norm.scalar_type() == at::ScalarType::Float ||
        x_norm.scalar_type() == at::ScalarType::Half ||
        x_norm.scalar_type() == at::ScalarType::BFloat16,
        fn_name,
        ": unsupported dtype."
    );
}

// ======================================================================================
// Plan legality
// ======================================================================================

static bool fscm_backward_reduce_plan_legal(
    torch::Tensor weight
) {
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    return C <= FSCM_BWD_MAX_C_REDUCE &&
           S <= FSCM_BWD_MAX_S_REDUCE;
}

static bool fscm_backward_plan_is_legal(
    FusedSourceConvMixedBackwardPlan plan,
    torch::Tensor weight,
    int64_t tile_t
) {
    switch (plan) {
        case FusedSourceConvMixedBackwardPlan::Atomic:
            return true;

        case FusedSourceConvMixedBackwardPlan::PerDReduce:
            return tile_t > 0 &&
                   fscm_backward_reduce_plan_legal(
                       weight
                   );

        case FusedSourceConvMixedBackwardPlan::GatherX:
            return tile_t > 0 &&
                   fscm_backward_reduce_plan_legal(
                       weight
                   );

        default:
            return false;
    }
}

static std::vector<std::pair<FusedSourceConvMixedBackwardPlan, int64_t>>
fscm_backward_candidate_plans(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight
) {
    std::vector<std::pair<FusedSourceConvMixedBackwardPlan, int64_t>> plans;

    int64_t T = grad_out.size(2);
    int64_t L = x_norm.size(2);
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    bool reduce_legal =
        fscm_backward_reduce_plan_legal(
            weight
        );

    // Always include B0.
    plans.push_back(
        {
            FusedSourceConvMixedBackwardPlan::Atomic,
            0
        }
    );

    if (reduce_legal) {
        const int64_t tile_candidates[] = {
            64,
            128,
            256
        };

        for (int i = 0; i < 3; ++i) {
            int64_t tile_t =
                tile_candidates[i];

            if (tile_t <= 0) {
                continue;
            }

            if (tile_t > T && T > 0) {
                tile_t = T;
            }

            plans.push_back(
                {
                    FusedSourceConvMixedBackwardPlan::PerDReduce,
                    tile_t
                }
            );

            // GatherX can be good when L is not much larger than affected range,
            // and when avoiding x atomic is beneficial.
            //
            // Include it in warmup candidates and let benchmark decide.
            if (C <= 64 && S <= 8 && L > 0) {
                plans.push_back(
                    {
                        FusedSourceConvMixedBackwardPlan::GatherX,
                        tile_t
                    }
                );
            }
        }
    }

    return plans;
}

// ======================================================================================
// Benchmark helper
// ======================================================================================

static float fscm_backward_time_one_plan_ms(
    FusedSourceConvMixedBackwardPlan plan,
    int64_t tile_t,
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t warmup_iters,
    int64_t bench_iters
) {
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    for (int64_t i = 0; i < warmup_iters; ++i) {
        auto grads =
            fused_source_conv_mixed_backward_plan_cuda(
                grad_out,
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation,
                static_cast<int64_t>(plan),
                tile_t
            );

        (void)grads;
    }

    C10_CUDA_CHECK(
        cudaStreamSynchronize(
            stream
        )
    );

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    C10_CUDA_CHECK(
        cudaEventCreate(
            &start_event
        )
    );

    C10_CUDA_CHECK(
        cudaEventCreate(
            &stop_event
        )
    );

    C10_CUDA_CHECK(
        cudaEventRecord(
            start_event,
            stream
        )
    );

    for (int64_t i = 0; i < bench_iters; ++i) {
        auto grads =
            fused_source_conv_mixed_backward_plan_cuda(
                grad_out,
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation,
                static_cast<int64_t>(plan),
                tile_t
            );

        (void)grads;
    }

    C10_CUDA_CHECK(
        cudaEventRecord(
            stop_event,
            stream
        )
    );

    C10_CUDA_CHECK(
        cudaEventSynchronize(
            stop_event
        )
    );

    float elapsed_ms = 0.0f;

    C10_CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsed_ms,
            start_event,
            stop_event
        )
    );

    C10_CUDA_CHECK(
        cudaEventDestroy(
            start_event
        )
    );

    C10_CUDA_CHECK(
        cudaEventDestroy(
            stop_event
        )
    );

    return elapsed_ms / static_cast<float>(
        bench_iters
    );
}

} // namespace

// ======================================================================================
// Cached plan packing
// ======================================================================================

int64_t fused_source_conv_mixed_backward_pack_cached_plan(
    int64_t plan,
    int64_t tile_t
) {
    uint64_t p =
        static_cast<uint64_t>(
            static_cast<uint32_t>(plan)
        );

    uint64_t tt =
        static_cast<uint64_t>(
            static_cast<uint32_t>(tile_t)
        );

    uint64_t packed =
        p | (tt << 32);

    return static_cast<int64_t>(
        packed
    );
}

int64_t fused_source_conv_mixed_backward_cached_plan_get_plan(
    int64_t cached_plan
) {
    uint64_t packed =
        static_cast<uint64_t>(
            cached_plan
        );

    return static_cast<int64_t>(
        packed & 0xffffffffULL
    );
}

int64_t fused_source_conv_mixed_backward_cached_plan_get_tile_t(
    int64_t cached_plan
) {
    uint64_t packed =
        static_cast<uint64_t>(
            cached_plan
        );

    return static_cast<int64_t>(
        (packed >> 32) & 0xffffffffULL
    );
}

// ======================================================================================
// Plan name
// ======================================================================================

const char* fused_source_conv_mixed_backward_plan_name(
    FusedSourceConvMixedBackwardPlan plan
) {
    switch (plan) {
        case FusedSourceConvMixedBackwardPlan::Atomic:
            return "Atomic";

        case FusedSourceConvMixedBackwardPlan::PerDReduce:
            return "PerDReduce";

        case FusedSourceConvMixedBackwardPlan::GatherX:
            return "GatherX";

        default:
            return "Unknown";
    }
}

// ======================================================================================
// Heuristic plan
// ======================================================================================

FusedSourceConvMixedBackwardPlan fused_source_conv_mixed_backward_heuristic_plan(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t tile_t
) {
    fscm_backward_check_common_inputs(
        grad_out,
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        "fused_source_conv_mixed_backward_heuristic_plan"
    );

    int64_t B = x_norm.size(0);
    int64_t D = x_norm.size(1);
    int64_t L = x_norm.size(2);
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    bool reduce_legal =
        fscm_backward_reduce_plan_legal(
            weight
        );

    if (!reduce_legal) {
        return FusedSourceConvMixedBackwardPlan::Atomic;
    }

    // Simple heuristic:
    //
    // - Small T or tiny shape: Atomic is simpler and often fine.
    // - Main target C/S small and D large: PerDReduce is usually better.
    // - GatherX may be good when avoiding grad_x atomic matters, but it scans L.
    //   Prefer it only when L is reasonably close to T + dilation*(S-1).
    int64_t affected =
        T + dilation * (S - 1);

    if (T <= 32 || B * D * T < 8192) {
        return FusedSourceConvMixedBackwardPlan::Atomic;
    }

    if (affected * 2 >= L && C <= 32 && S <= 4) {
        return FusedSourceConvMixedBackwardPlan::GatherX;
    }

    return FusedSourceConvMixedBackwardPlan::PerDReduce;
}

// ======================================================================================
// Explicit plan dispatch
// ======================================================================================

std::vector<torch::Tensor> fused_source_conv_mixed_backward_plan_cuda(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t plan_int,
    int64_t tile_t
) {
    fscm_backward_check_common_inputs(
        grad_out,
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        "fused_source_conv_mixed_backward_plan_cuda"
    );

    auto plan =
        static_cast<FusedSourceConvMixedBackwardPlan>(
            plan_int
        );

    TORCH_CHECK(
        fscm_backward_plan_is_legal(
            plan,
            weight,
            tile_t
        ),
        "fused_source_conv_mixed_backward_plan_cuda: illegal plan ",
        plan_int,
        " (",
        fused_source_conv_mixed_backward_plan_name(plan),
        "), tile_t=",
        tile_t,
        ", weight=",
        weight.sizes(),
        "."
    );

    switch (plan) {
        case FusedSourceConvMixedBackwardPlan::Atomic:
            return fused_source_conv_mixed_backward_atomic_cuda(
                grad_out,
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        case FusedSourceConvMixedBackwardPlan::PerDReduce:
            return fused_source_conv_mixed_backward_per_d_reduce_cuda(
                grad_out,
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation,
                tile_t
            );

        case FusedSourceConvMixedBackwardPlan::GatherX:
            return fused_source_conv_mixed_backward_gather_x_cuda(
                grad_out,
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation,
                tile_t
            );

        default:
            TORCH_CHECK(
                false,
                "fused_source_conv_mixed_backward_plan_cuda: unknown plan ",
                plan_int,
                "."
            );
    }
}

// ======================================================================================
// Default backward entry
// ======================================================================================

std::vector<torch::Tensor> fused_source_conv_mixed_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    int64_t default_tile_t = 64;

    if (T < default_tile_t) {
        default_tile_t = T;
    }

    auto plan =
        fused_source_conv_mixed_backward_heuristic_plan(
            grad_out,
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation,
            default_tile_t
        );

    FDC_DEBUG_PATH(
        fused_source_conv_mixed_backward_plan_name(
            plan
        )
    );

    return fused_source_conv_mixed_backward_plan_cuda(
        grad_out,
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        static_cast<int64_t>(plan),
        default_tile_t
    );
}

// ======================================================================================
// Cached plan backward entry
// ======================================================================================

std::vector<torch::Tensor> fused_source_conv_mixed_backward_cached_plan_cuda(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t cached_plan
) {
    int64_t plan =
        fused_source_conv_mixed_backward_cached_plan_get_plan(
            cached_plan
        );

    int64_t tile_t =
        fused_source_conv_mixed_backward_cached_plan_get_tile_t(
            cached_plan
        );

    if (static_cast<FusedSourceConvMixedBackwardPlan>(plan) ==
        FusedSourceConvMixedBackwardPlan::Atomic) {
        tile_t = 0;
    }

    return fused_source_conv_mixed_backward_plan_cuda(
        grad_out,
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        plan,
        tile_t
    );
}

// ======================================================================================
// Warmup benchmark entry
//
// Returns:
//   packed cached_plan
//
// cached_plan stores:
//   plan id
//   tile_t
// ======================================================================================

int64_t fused_source_conv_mixed_backward_warmup_cuda(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t warmup_iters,
    int64_t bench_iters
) {
    fscm_backward_check_common_inputs(
        grad_out,
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        "fused_source_conv_mixed_backward_warmup_cuda"
    );

    TORCH_CHECK(
        warmup_iters >= 0,
        "fused_source_conv_mixed_backward_warmup_cuda: warmup_iters must be >= 0."
    );

    TORCH_CHECK(
        bench_iters > 0,
        "fused_source_conv_mixed_backward_warmup_cuda: bench_iters must be > 0."
    );

    auto candidates =
        fscm_backward_candidate_plans(
            grad_out,
            x_norm,
            weight
        );

    TORCH_CHECK(
        !candidates.empty(),
        "fused_source_conv_mixed_backward_warmup_cuda: no candidates."
    );

    FusedSourceConvMixedBackwardPlan best_plan =
        candidates[0].first;

    int64_t best_tile_t =
        candidates[0].second;

    float best_ms =
        std::numeric_limits<float>::infinity();

    for (const auto& item : candidates) {
        auto plan =
            item.first;

        int64_t tile_t =
            item.second;

        if (!fscm_backward_plan_is_legal(
                plan,
                weight,
                tile_t
            )) {
            continue;
        }

        float ms =
            fscm_backward_time_one_plan_ms(
                plan,
                tile_t,
                grad_out,
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation,
                warmup_iters,
                bench_iters
            );

        if (ms < best_ms) {
            best_ms = ms;
            best_plan = plan;
            best_tile_t = tile_t;
        }
    }

    if (best_plan == FusedSourceConvMixedBackwardPlan::Atomic) {
        best_tile_t = 0;
    }

    return fused_source_conv_mixed_backward_pack_cached_plan(
        static_cast<int64_t>(
            best_plan
        ),
        best_tile_t
    );
}
