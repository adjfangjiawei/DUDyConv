#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>
#include <string>

#include "fused_dynamic_conv_common.cuh"
#include "fused_source_conv_mixed_plans.h"

// ======================================================================================
// Fused Source Conv Mixed Forward Main
//
// This file implements:
//
//   1. plan name
//   2. common validation
//   3. heuristic forward plan selection
//   4. explicit plan dispatch
//   5. default forward entry
//   6. cached-plan forward entry
//   7. warmup benchmark entry
//
// It does NOT implement:
//   - Python wrapper
//   - pybind registration
//   - backward
//   - dropout
//   - residual
// ======================================================================================

namespace {

// ======================================================================================
// Common validation
// ======================================================================================

static void fscm_forward_check_common_inputs(
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

    int64_t Dw = weight.size(0);
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
        Dw == D,
        fn_name,
        ": weight D mismatch."
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

static bool fscm_is_no_boundary(
    torch::Tensor weight,
    int64_t off,
    int64_t dilation
) {
    int64_t S = weight.size(2);

    return off >= dilation * (S - 1);
}

static bool fscm_plan_is_legal(
    FusedSourceConvMixedForwardPlan plan,
    torch::Tensor x_norm,
    torch::Tensor weight,
    int64_t off,
    int64_t dilation
) {
    int64_t D = x_norm.size(1);
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    bool no_boundary =
        fscm_is_no_boundary(
            weight,
            off,
            dilation
        );

    switch (plan) {
        case FusedSourceConvMixedForwardPlan::Generic:
            return true;

        case FusedSourceConvMixedForwardPlan::C8S3:
            return C == 8 && S == 3;

        case FusedSourceConvMixedForwardPlan::C32S3:
            return C == 32 && S == 3;

        case FusedSourceConvMixedForwardPlan::D256C16S3:
            return D == 256 && C == 16 && S == 3;

        case FusedSourceConvMixedForwardPlan::GenericNoBoundary:
            return no_boundary;

        case FusedSourceConvMixedForwardPlan::C8S3NoBoundary:
            return no_boundary && C == 8 && S == 3;

        case FusedSourceConvMixedForwardPlan::C32S3NoBoundary:
            return no_boundary && C == 32 && S == 3;

        case FusedSourceConvMixedForwardPlan::D256C16S3NoBoundary:
            return no_boundary && D == 256 && C == 16 && S == 3;

        default:
            return false;
    }
}

static std::vector<FusedSourceConvMixedForwardPlan> fscm_candidate_plans(
    torch::Tensor x_norm,
    torch::Tensor weight,
    int64_t off,
    int64_t dilation
) {
    std::vector<FusedSourceConvMixedForwardPlan> plans;

    int64_t D = x_norm.size(1);
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    bool no_boundary =
        fscm_is_no_boundary(
            weight,
            off,
            dilation
        );

    if (no_boundary) {
        if (D == 256 && C == 16 && S == 3) {
            plans.push_back(
                FusedSourceConvMixedForwardPlan::D256C16S3NoBoundary
            );
        }

        if (C == 8 && S == 3) {
            plans.push_back(
                FusedSourceConvMixedForwardPlan::C8S3NoBoundary
            );
        }

        if (C == 32 && S == 3) {
            plans.push_back(
                FusedSourceConvMixedForwardPlan::C32S3NoBoundary
            );
        }

        plans.push_back(
            FusedSourceConvMixedForwardPlan::GenericNoBoundary
        );
    }

    if (D == 256 && C == 16 && S == 3) {
        plans.push_back(
            FusedSourceConvMixedForwardPlan::D256C16S3
        );
    }

    if (C == 8 && S == 3) {
        plans.push_back(
            FusedSourceConvMixedForwardPlan::C8S3
        );
    }

    if (C == 32 && S == 3) {
        plans.push_back(
            FusedSourceConvMixedForwardPlan::C32S3
        );
    }

    plans.push_back(
        FusedSourceConvMixedForwardPlan::Generic
    );

    return plans;
}

// ======================================================================================
// CUDA event benchmark helper
// ======================================================================================

static float fscm_time_one_plan_ms(
    FusedSourceConvMixedForwardPlan plan,
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
        auto y = fused_source_conv_mixed_forward_plan_cuda(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation,
            static_cast<int64_t>(plan)
        );

        (void)y;
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
        auto y = fused_source_conv_mixed_forward_plan_cuda(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation,
            static_cast<int64_t>(plan)
        );

        (void)y;
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

    return elapsed_ms / static_cast<float>(bench_iters);
}

} // namespace

// ======================================================================================
// Plan name
// ======================================================================================

const char* fused_source_conv_mixed_forward_plan_name(
    FusedSourceConvMixedForwardPlan plan
) {
    switch (plan) {
        case FusedSourceConvMixedForwardPlan::Generic:
            return "Generic";

        case FusedSourceConvMixedForwardPlan::C8S3:
            return "C8S3";

        case FusedSourceConvMixedForwardPlan::C32S3:
            return "C32S3";

        case FusedSourceConvMixedForwardPlan::D256C16S3:
            return "D256C16S3";

        case FusedSourceConvMixedForwardPlan::GenericNoBoundary:
            return "GenericNoBoundary";

        case FusedSourceConvMixedForwardPlan::C8S3NoBoundary:
            return "C8S3NoBoundary";

        case FusedSourceConvMixedForwardPlan::C32S3NoBoundary:
            return "C32S3NoBoundary";

        case FusedSourceConvMixedForwardPlan::D256C16S3NoBoundary:
            return "D256C16S3NoBoundary";

        default:
            return "Unknown";
    }
}

// ======================================================================================
// Heuristic plan selection
// ======================================================================================

FusedSourceConvMixedForwardPlan fused_source_conv_mixed_forward_heuristic_plan(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    fscm_forward_check_common_inputs(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        "fused_source_conv_mixed_forward_heuristic_plan"
    );

    int64_t D = x_norm.size(1);
    int64_t C = weight.size(1);
    int64_t S = weight.size(2);

    bool no_boundary =
        fscm_is_no_boundary(
            weight,
            off,
            dilation
        );

    if (no_boundary) {
        if (D == 256 && C == 16 && S == 3) {
            return FusedSourceConvMixedForwardPlan::D256C16S3NoBoundary;
        }

        if (C == 8 && S == 3) {
            return FusedSourceConvMixedForwardPlan::C8S3NoBoundary;
        }

        if (C == 32 && S == 3) {
            return FusedSourceConvMixedForwardPlan::C32S3NoBoundary;
        }

        return FusedSourceConvMixedForwardPlan::GenericNoBoundary;
    }

    if (D == 256 && C == 16 && S == 3) {
        return FusedSourceConvMixedForwardPlan::D256C16S3;
    }

    if (C == 8 && S == 3) {
        return FusedSourceConvMixedForwardPlan::C8S3;
    }

    if (C == 32 && S == 3) {
        return FusedSourceConvMixedForwardPlan::C32S3;
    }

    return FusedSourceConvMixedForwardPlan::Generic;
}

// ======================================================================================
// Explicit plan dispatch
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_plan_cuda(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t plan_int
) {
    fscm_forward_check_common_inputs(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        "fused_source_conv_mixed_forward_plan_cuda"
    );

    auto plan =
        static_cast<FusedSourceConvMixedForwardPlan>(
            plan_int
        );

    TORCH_CHECK(
        fscm_plan_is_legal(
            plan,
            x_norm,
            weight,
            off,
            dilation
        ),
        "fused_source_conv_mixed_forward_plan_cuda: illegal plan ",
        plan_int,
        " (",
        fused_source_conv_mixed_forward_plan_name(plan),
        ") for shape x_norm=",
        x_norm.sizes(),
        ", weight=",
        weight.sizes(),
        ", off=",
        off,
        ", T=",
        T,
        ", dilation=",
        dilation,
        "."
    );

    switch (plan) {
        case FusedSourceConvMixedForwardPlan::Generic:
            return fused_source_conv_mixed_forward_direct_generic_cuda(
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        case FusedSourceConvMixedForwardPlan::C8S3:
            return fused_source_conv_mixed_forward_direct_c8s3_cuda(
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        case FusedSourceConvMixedForwardPlan::C32S3:
            return fused_source_conv_mixed_forward_direct_c32s3_cuda(
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        case FusedSourceConvMixedForwardPlan::D256C16S3:
            return fused_source_conv_mixed_forward_d256_c16s3_cuda(
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        case FusedSourceConvMixedForwardPlan::GenericNoBoundary:
            return fused_source_conv_mixed_forward_direct_generic_noboundary_cuda(
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        case FusedSourceConvMixedForwardPlan::C8S3NoBoundary:
            return fused_source_conv_mixed_forward_direct_c8s3_noboundary_cuda(
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        case FusedSourceConvMixedForwardPlan::C32S3NoBoundary:
            return fused_source_conv_mixed_forward_direct_c32s3_noboundary_cuda(
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        case FusedSourceConvMixedForwardPlan::D256C16S3NoBoundary:
            return fused_source_conv_mixed_forward_d256_c16s3_noboundary_cuda(
                x_norm,
                weight,
                bias,
                mix_weight,
                mix_bias,
                off,
                T,
                dilation
            );

        default:
            TORCH_CHECK(
                false,
                "fused_source_conv_mixed_forward_plan_cuda: unknown plan ",
                plan_int,
                "."
            );
    }
}

// ======================================================================================
// Default forward entry: heuristic plan
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_cuda(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
    auto plan =
        fused_source_conv_mixed_forward_heuristic_plan(
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            off,
            T,
            dilation
        );

    FDC_DEBUG_PATH(
        fused_source_conv_mixed_forward_plan_name(
            plan
        )
    );

    return fused_source_conv_mixed_forward_plan_cuda(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        static_cast<int64_t>(plan)
    );
}

// ======================================================================================
// Cached-plan forward entry
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_cached_plan_cuda(
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
    return fused_source_conv_mixed_forward_plan_cuda(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        cached_plan
    );
}

// ======================================================================================
// Warmup benchmark entry
//
// Returns:
//   int64_t best_plan
//
// Python side can cache this integer and later call:
//   fused_source_conv_mixed_forward_cached_plan_cuda(..., best_plan)
//
// This benchmark intentionally tests all legal candidates and picks the lowest avg ms.
// ======================================================================================

int64_t fused_source_conv_mixed_forward_warmup_cuda(
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
    fscm_forward_check_common_inputs(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        off,
        T,
        dilation,
        "fused_source_conv_mixed_forward_warmup_cuda"
    );

    TORCH_CHECK(
        warmup_iters >= 0,
        "fused_source_conv_mixed_forward_warmup_cuda: warmup_iters must be >= 0."
    );

    TORCH_CHECK(
        bench_iters > 0,
        "fused_source_conv_mixed_forward_warmup_cuda: bench_iters must be > 0."
    );

    auto candidates =
        fscm_candidate_plans(
            x_norm,
            weight,
            off,
            dilation
        );

    TORCH_CHECK(
        !candidates.empty(),
        "fused_source_conv_mixed_forward_warmup_cuda: no legal candidate plans."
    );

    FusedSourceConvMixedForwardPlan best_plan =
        candidates[0];

    float best_ms =
        std::numeric_limits<float>::infinity();

    for (auto plan : candidates) {
        if (!fscm_plan_is_legal(
                plan,
                x_norm,
                weight,
                off,
                dilation
            )) {
            continue;
        }

        float ms =
            fscm_time_one_plan_ms(
                plan,
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
        }
    }

    return static_cast<int64_t>(
        best_plan
    );
}
