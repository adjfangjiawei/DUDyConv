#include <torch/extension.h>

#include <vector>

#include "fused_source_conv_mixed_plans.h"

// ======================================================================================
// Fused Source Conv Mixed operator registration
//
// This file intentionally does NOT define PYBIND11_MODULE.
//
// The existing extension already has its pybind module in:
//
//   cuda_ops/fused_dynamic_conv.cpp
//
// Defining another PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) in this file would
// cause duplicate module definition / linker or import problems.
//
// Instead, we register these functions through TORCH_LIBRARY. After importing
// cuda_ops_ext once, Python can call:
//
//   torch.ops.fused_source_conv_mixed.forward(...)
//   torch.ops.fused_source_conv_mixed.forward_plan(...)
//   torch.ops.fused_source_conv_mixed.forward_cached_plan(...)
//   torch.ops.fused_source_conv_mixed.forward_warmup(...)
//
//   torch.ops.fused_source_conv_mixed.backward(...)
//   torch.ops.fused_source_conv_mixed.backward_plan(...)
//   torch.ops.fused_source_conv_mixed.backward_cached_plan(...)
//   torch.ops.fused_source_conv_mixed.backward_warmup(...)
//
// Important:
//
//   Ops without Tensor arguments cannot be registered only under CUDA dispatch,
//   because PyTorch cannot infer a dispatch key when there are no Tensor inputs.
//
//   Therefore these scalar-only helper ops are registered under
//   CompositeExplicitAutograd:
//
//     backward_pack_cached_plan
//     backward_cached_plan_get_plan
//     backward_cached_plan_get_tile_t
//
// ======================================================================================

namespace {

torch::Tensor fscm_forward_op(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation
) {
  return fused_source_conv_mixed_forward_cuda(
      x_norm,
      weight,
      bias,
      mix_weight,
      mix_bias,
      off,
      T,
      dilation
  );
}

torch::Tensor fscm_forward_plan_op(
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t plan
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
      plan
  );
}

torch::Tensor fscm_forward_cached_plan_op(
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
  return fused_source_conv_mixed_forward_cached_plan_cuda(
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

int64_t fscm_forward_warmup_op(
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
  return fused_source_conv_mixed_forward_warmup_cuda(
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
}

std::vector<torch::Tensor> fscm_backward_op(
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
  return fused_source_conv_mixed_backward_cuda(
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
}

std::vector<torch::Tensor> fscm_backward_plan_op(
    torch::Tensor grad_out,
    torch::Tensor x_norm,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mix_weight,
    torch::Tensor mix_bias,
    int64_t off,
    int64_t T,
    int64_t dilation,
    int64_t plan,
    int64_t tile_t
) {
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

std::vector<torch::Tensor> fscm_backward_cached_plan_op(
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
  return fused_source_conv_mixed_backward_cached_plan_cuda(
      grad_out,
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

int64_t fscm_backward_warmup_op(
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
  return fused_source_conv_mixed_backward_warmup_cuda(
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
}

int64_t fscm_backward_pack_cached_plan_op(
    int64_t plan,
    int64_t tile_t
) {
  return fused_source_conv_mixed_backward_pack_cached_plan(
      plan,
      tile_t
  );
}

int64_t fscm_backward_cached_plan_get_plan_op(
    int64_t cached_plan
) {
  return fused_source_conv_mixed_backward_cached_plan_get_plan(
      cached_plan
  );
}

int64_t fscm_backward_cached_plan_get_tile_t_op(
    int64_t cached_plan
) {
  return fused_source_conv_mixed_backward_cached_plan_get_tile_t(
      cached_plan
  );
}

std::vector<torch::Tensor> fscm_backward_atomic_op(
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
}

std::vector<torch::Tensor> fscm_backward_per_d_reduce_op(
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
}

std::vector<torch::Tensor> fscm_backward_gather_x_op(
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
}

} // namespace

TORCH_LIBRARY(fused_source_conv_mixed, m) {
  m.def("forward("
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation"
        ") -> Tensor");

  m.def("forward_plan("
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation, "
        "int plan"
        ") -> Tensor");

  m.def("forward_cached_plan("
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation, "
        "int cached_plan"
        ") -> Tensor");

  m.def("forward_warmup("
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation, "
        "int warmup_iters, "
        "int bench_iters"
        ") -> int");

  m.def("backward("
        "Tensor grad_out, "
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation"
        ") -> Tensor[]");

  m.def("backward_plan("
        "Tensor grad_out, "
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation, "
        "int plan, "
        "int tile_t"
        ") -> Tensor[]");

  m.def("backward_cached_plan("
        "Tensor grad_out, "
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation, "
        "int cached_plan"
        ") -> Tensor[]");

  m.def("backward_warmup("
        "Tensor grad_out, "
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation, "
        "int warmup_iters, "
        "int bench_iters"
        ") -> int");

  m.def("backward_pack_cached_plan("
        "int plan, "
        "int tile_t"
        ") -> int");

  m.def("backward_cached_plan_get_plan("
        "int cached_plan"
        ") -> int");

  m.def("backward_cached_plan_get_tile_t("
        "int cached_plan"
        ") -> int");

  m.def("backward_atomic("
        "Tensor grad_out, "
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation"
        ") -> Tensor[]");

  m.def("backward_per_d_reduce("
        "Tensor grad_out, "
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation, "
        "int tile_t"
        ") -> Tensor[]");

  m.def("backward_gather_x("
        "Tensor grad_out, "
        "Tensor x_norm, "
        "Tensor weight, "
        "Tensor bias, "
        "Tensor mix_weight, "
        "Tensor mix_bias, "
        "int off, "
        "int T, "
        "int dilation, "
        "int tile_t"
        ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(fused_source_conv_mixed, CUDA, m) {
  m.impl("forward", TORCH_FN(fscm_forward_op));

  m.impl("forward_plan", TORCH_FN(fscm_forward_plan_op));

  m.impl("forward_cached_plan", TORCH_FN(fscm_forward_cached_plan_op));

  m.impl("forward_warmup", TORCH_FN(fscm_forward_warmup_op));

  m.impl("backward", TORCH_FN(fscm_backward_op));

  m.impl("backward_plan", TORCH_FN(fscm_backward_plan_op));

  m.impl("backward_cached_plan", TORCH_FN(fscm_backward_cached_plan_op));

  m.impl("backward_warmup", TORCH_FN(fscm_backward_warmup_op));

  m.impl("backward_atomic", TORCH_FN(fscm_backward_atomic_op));

  m.impl("backward_per_d_reduce", TORCH_FN(fscm_backward_per_d_reduce_op));

  m.impl("backward_gather_x", TORCH_FN(fscm_backward_gather_x_op));
}

TORCH_LIBRARY_IMPL(fused_source_conv_mixed, CompositeExplicitAutograd, m) {
  m.impl("backward_pack_cached_plan",
         TORCH_FN(fscm_backward_pack_cached_plan_op));

  m.impl("backward_cached_plan_get_plan",
         TORCH_FN(fscm_backward_cached_plan_get_plan_op));

  m.impl("backward_cached_plan_get_tile_t",
         TORCH_FN(fscm_backward_cached_plan_get_tile_t_op));
}
