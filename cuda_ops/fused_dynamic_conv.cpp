#include <torch/extension.h>

#include <string>
#include <vector>

// ======================================================================================
// CUDA functions implemented in fused_dynamic_conv_kernel.cu
// ======================================================================================

torch::Tensor fused_dynamic_conv_forward_chunk_cuda(torch::Tensor h,
                                                    torch::Tensor kc,
                                                    torch::Tensor mix,
                                                    int64_t off,
                                                    int64_t dilation);

std::vector<torch::Tensor>
fused_dynamic_conv_backward_chunk_cuda(torch::Tensor go, torch::Tensor h,
                                       torch::Tensor kc, torch::Tensor mix,
                                       int64_t off, int64_t dilation);

int64_t fused_dynamic_conv_backward_chunk_warmup_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation, int64_t repeat);

int64_t fused_dynamic_conv_backward_chunk_cached_plan_cuda(torch::Tensor h,
                                                           torch::Tensor kc,
                                                           int64_t off,
                                                           int64_t dilation);

void fused_dynamic_conv_backward_chunk_clear_warmup_cache_cuda();

std::string fused_dynamic_conv_backward_plan_name_cuda(int64_t plan_id);

// ======================================================================================
// Forward autotune functions implemented in
// fused_dynamic_conv_forward_direct.cu
// ======================================================================================

int64_t
fused_dynamic_conv_forward_direct_warmup_cuda(torch::Tensor h, torch::Tensor kc,
                                              torch::Tensor mix, int64_t off,
                                              int64_t dilation, int64_t repeat);

int64_t fused_dynamic_conv_forward_direct_cached_plan_cuda(torch::Tensor h,
                                                           torch::Tensor kc,
                                                           torch::Tensor mix,
                                                           int64_t off,
                                                           int64_t dilation);

void fused_dynamic_conv_forward_direct_clear_warmup_cache_cuda();

std::string fused_dynamic_conv_forward_direct_plan_name_cuda(int64_t plan_id);

// ======================================================================================
// Python bindings
// ======================================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_chunk", &fused_dynamic_conv_forward_chunk_cuda,
        "Fused dynamic convolution forward chunk CUDA", py::arg("h"),
        py::arg("kc"), py::arg("mix"), py::arg("off"), py::arg("dilation"));

  m.def("forward_chunk_warmup", &fused_dynamic_conv_forward_direct_warmup_cuda,
        "Fused dynamic convolution forward chunk warmup/autotune CUDA",
        py::arg("h"), py::arg("kc"), py::arg("mix"), py::arg("off"),
        py::arg("dilation"), py::arg("repeat") = 3);

  m.def("forward_chunk_cached_plan",
        &fused_dynamic_conv_forward_direct_cached_plan_cuda,
        "Get cached fused dynamic convolution forward plan CUDA", py::arg("h"),
        py::arg("kc"), py::arg("mix"), py::arg("off"), py::arg("dilation"));

  m.def("forward_chunk_clear_warmup_cache",
        &fused_dynamic_conv_forward_direct_clear_warmup_cache_cuda,
        "Clear fused dynamic convolution forward warmup/autotune cache CUDA");

  m.def("forward_plan_name", &fused_dynamic_conv_forward_direct_plan_name_cuda,
        "Get fused dynamic convolution forward plan name", py::arg("plan_id"));

  m.def("backward_chunk", &fused_dynamic_conv_backward_chunk_cuda,
        "Fused dynamic convolution backward chunk CUDA", py::arg("go"),
        py::arg("h"), py::arg("kc"), py::arg("mix"), py::arg("off"),
        py::arg("dilation"));

  m.def("backward_chunk_warmup", &fused_dynamic_conv_backward_chunk_warmup_cuda,
        "Fused dynamic convolution backward chunk warmup/autotune CUDA",
        py::arg("go"), py::arg("h"), py::arg("kc"), py::arg("mix"),
        py::arg("off"), py::arg("dilation"), py::arg("repeat") = 3);

  m.def("backward_chunk_cached_plan",
        &fused_dynamic_conv_backward_chunk_cached_plan_cuda,
        "Get cached fused dynamic convolution backward plan CUDA", py::arg("h"),
        py::arg("kc"), py::arg("off"), py::arg("dilation"));

  m.def("backward_chunk_clear_warmup_cache",
        &fused_dynamic_conv_backward_chunk_clear_warmup_cache_cuda,
        "Clear fused dynamic convolution backward warmup/autotune cache CUDA");

  m.def("backward_plan_name", &fused_dynamic_conv_backward_plan_name_cuda,
        "Get fused dynamic convolution backward plan name", py::arg("plan_id"));
}
