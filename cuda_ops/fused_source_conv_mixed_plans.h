#pragma once

#include <torch/extension.h>

#include <vector>

// ======================================================================================
// Fused Source Conv Mixed
//
// Layout:
//
//   x_norm:
//     [B,D,L]
//
//   weight:
//     [D,C,S]
//
//   bias:
//     [D,C] or undefined/empty
//
//   mix_weight:
//     [D,C]
//
//   mix_bias:
//     [D]
//
//   output:
//     [B,D,T]
//
// Semantics:
//
//   z[b,d,c,t] = bias[d,c] +
//                sum_j x_norm[b,d, off+t-dilation*(S-1-j)] * weight[d,c,j]
//
//   out[b,d,t] = mix_bias[d] +
//                sum_c GELU(z[b,d,c,t]) * mix_weight[d,c]
//
// Notes:
//   - No source-channel dropout.
//   - No post-mix dropout.
//   - No residual add.
//   - No source nonlinear residual.
//   - Accumulation is float.
// ======================================================================================

// ======================================================================================
// Forward plan enum
// ======================================================================================

enum class FusedSourceConvMixedForwardPlan : int {
  Generic = 0,
  C8S3 = 1,
  C32S3 = 2,
  D256C16S3 = 3,

  GenericNoBoundary = 100,
  C8S3NoBoundary = 101,
  C32S3NoBoundary = 102,
  D256C16S3NoBoundary = 103
};

const char *
fused_source_conv_mixed_forward_plan_name(FusedSourceConvMixedForwardPlan plan);

FusedSourceConvMixedForwardPlan fused_source_conv_mixed_forward_heuristic_plan(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

// ======================================================================================
// Individual forward plans
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_direct_generic_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

torch::Tensor fused_source_conv_mixed_forward_direct_c8s3_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

torch::Tensor fused_source_conv_mixed_forward_direct_c32s3_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

torch::Tensor fused_source_conv_mixed_forward_d256_c16s3_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

torch::Tensor fused_source_conv_mixed_forward_direct_generic_noboundary_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

torch::Tensor fused_source_conv_mixed_forward_direct_c8s3_noboundary_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

torch::Tensor fused_source_conv_mixed_forward_direct_c32s3_noboundary_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

torch::Tensor fused_source_conv_mixed_forward_d256_c16s3_noboundary_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

// ======================================================================================
// Forward selected/cached/warmup entries
// ======================================================================================

torch::Tensor fused_source_conv_mixed_forward_plan_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation, int64_t plan);

torch::Tensor fused_source_conv_mixed_forward_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation);

int64_t fused_source_conv_mixed_forward_warmup_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation, int64_t warmup_iters, int64_t bench_iters);

torch::Tensor fused_source_conv_mixed_forward_cached_plan_cuda(
    torch::Tensor x_norm, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mix_weight, torch::Tensor mix_bias, int64_t off, int64_t T,
    int64_t dilation, int64_t cached_plan);

// ======================================================================================
// Backward plan enum
// ======================================================================================

enum class FusedSourceConvMixedBackwardPlan : int {
  Atomic = 0,
  PerDReduce = 1,
  GatherX = 2
};

const char *fused_source_conv_mixed_backward_plan_name(
    FusedSourceConvMixedBackwardPlan plan);

FusedSourceConvMixedBackwardPlan
fused_source_conv_mixed_backward_heuristic_plan(
    torch::Tensor grad_out, torch::Tensor x_norm, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mix_weight, torch::Tensor mix_bias,
    int64_t off, int64_t T, int64_t dilation, int64_t tile_t);

// ======================================================================================
// Individual backward plans
// ======================================================================================

std::vector<torch::Tensor> fused_source_conv_mixed_backward_atomic_cuda(
    torch::Tensor grad_out, torch::Tensor x_norm, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mix_weight, torch::Tensor mix_bias,
    int64_t off, int64_t T, int64_t dilation);

std::vector<torch::Tensor> fused_source_conv_mixed_backward_per_d_reduce_cuda(
    torch::Tensor grad_out, torch::Tensor x_norm, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mix_weight, torch::Tensor mix_bias,
    int64_t off, int64_t T, int64_t dilation, int64_t tile_t);

std::vector<torch::Tensor> fused_source_conv_mixed_backward_gather_x_cuda(
    torch::Tensor grad_out, torch::Tensor x_norm, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mix_weight, torch::Tensor mix_bias,
    int64_t off, int64_t T, int64_t dilation, int64_t tile_t);

// ======================================================================================
// Backward selected/cached/warmup entries
// ======================================================================================
//
// cached_plan packing:
//   low  32 bits: plan id
//   high 32 bits: tile_t
//
// For Atomic plan, tile_t may be 0.
// ======================================================================================

int64_t fused_source_conv_mixed_backward_pack_cached_plan(int64_t plan,
                                                          int64_t tile_t);

int64_t
fused_source_conv_mixed_backward_cached_plan_get_plan(int64_t cached_plan);

int64_t
fused_source_conv_mixed_backward_cached_plan_get_tile_t(int64_t cached_plan);

std::vector<torch::Tensor> fused_source_conv_mixed_backward_plan_cuda(
    torch::Tensor grad_out, torch::Tensor x_norm, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mix_weight, torch::Tensor mix_bias,
    int64_t off, int64_t T, int64_t dilation, int64_t plan, int64_t tile_t);

std::vector<torch::Tensor> fused_source_conv_mixed_backward_cuda(
    torch::Tensor grad_out, torch::Tensor x_norm, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mix_weight, torch::Tensor mix_bias,
    int64_t off, int64_t T, int64_t dilation);

std::vector<torch::Tensor> fused_source_conv_mixed_backward_cached_plan_cuda(
    torch::Tensor grad_out, torch::Tensor x_norm, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mix_weight, torch::Tensor mix_bias,
    int64_t off, int64_t T, int64_t dilation, int64_t cached_plan);

int64_t fused_source_conv_mixed_backward_warmup_cuda(
    torch::Tensor grad_out, torch::Tensor x_norm, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mix_weight, torch::Tensor mix_bias,
    int64_t off, int64_t T, int64_t dilation, int64_t warmup_iters,
    int64_t bench_iters);
