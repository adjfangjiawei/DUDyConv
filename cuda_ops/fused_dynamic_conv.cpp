#include <torch/extension.h>
#include <vector>

torch::Tensor fused_dynamic_conv_forward_chunk_cuda(torch::Tensor h_full,
                                                    torch::Tensor kernel_chunk,
                                                    torch::Tensor kernel_mix,
                                                    int64_t t_offset,
                                                    int64_t dilation);

std::vector<torch::Tensor> fused_dynamic_conv_backward_chunk_cuda(
    torch::Tensor grad_out, torch::Tensor h_full, torch::Tensor kernel_chunk,
    torch::Tensor kernel_mix, int64_t t_offset, int64_t dilation);

static void check_common_inputs(const torch::Tensor &h_full,
                                const torch::Tensor &kernel_chunk,
                                const torch::Tensor &kernel_mix,
                                int64_t t_offset, int64_t dilation) {
  TORCH_CHECK(h_full.is_cuda(), "h_full must be CUDA tensor.");
  TORCH_CHECK(kernel_chunk.is_cuda(), "kernel_chunk must be CUDA tensor.");
  TORCH_CHECK(kernel_mix.is_cuda(), "kernel_mix must be CUDA tensor.");

  TORCH_CHECK(h_full.is_contiguous(), "h_full must be contiguous.");
  TORCH_CHECK(kernel_chunk.is_contiguous(), "kernel_chunk must be contiguous.");
  TORCH_CHECK(kernel_mix.is_contiguous(), "kernel_mix must be contiguous.");

  TORCH_CHECK(h_full.dim() == 3, "h_full must be [B,D,L].");
  TORCH_CHECK(kernel_chunk.dim() == 4, "kernel_chunk must be [B,T,N,K].");
  TORCH_CHECK(kernel_mix.dim() == 2, "kernel_mix must be [D,N].");

  TORCH_CHECK(h_full.scalar_type() == torch::kFloat32 ||
                  h_full.scalar_type() == torch::kFloat16 ||
                  h_full.scalar_type() == torch::kBFloat16,
              "h_full dtype must be float32, float16 or bfloat16.");

  TORCH_CHECK(kernel_chunk.scalar_type() == h_full.scalar_type(),
              "kernel_chunk dtype must equal h_full dtype.");

  TORCH_CHECK(kernel_mix.scalar_type() == h_full.scalar_type(),
              "kernel_mix dtype must equal h_full dtype.");

  const auto B = h_full.size(0);
  const auto D = h_full.size(1);
  const auto L = h_full.size(2);

  const auto Bk = kernel_chunk.size(0);
  const auto T = kernel_chunk.size(1);
  const auto N = kernel_chunk.size(2);
  const auto K = kernel_chunk.size(3);

  const auto Dm = kernel_mix.size(0);
  const auto Nm = kernel_mix.size(1);

  TORCH_CHECK(B > 0, "B must be positive.");
  TORCH_CHECK(D > 0, "D must be positive.");
  TORCH_CHECK(L > 0, "L must be positive.");
  TORCH_CHECK(T > 0, "T must be positive.");
  TORCH_CHECK(N > 0, "N must be positive.");
  TORCH_CHECK(K > 0, "K must be positive.");

  TORCH_CHECK(B == Bk, "B mismatch.");
  TORCH_CHECK(D == Dm, "D mismatch.");
  TORCH_CHECK(N == Nm, "N mismatch.");

  TORCH_CHECK(t_offset >= 0, "t_offset must be non-negative.");
  TORCH_CHECK(dilation > 0, "dilation must be positive.");
  TORCH_CHECK(t_offset + T <= L, "t_offset + T must be <= L.");
}

static void check_backward_inputs(const torch::Tensor &grad_out,
                                  const torch::Tensor &h_full,
                                  const torch::Tensor &kernel_chunk,
                                  const torch::Tensor &kernel_mix,
                                  int64_t t_offset, int64_t dilation) {
  check_common_inputs(h_full, kernel_chunk, kernel_mix, t_offset, dilation);

  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA tensor.");
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous.");
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must be [B,D,T].");
  TORCH_CHECK(grad_out.scalar_type() == h_full.scalar_type(),
              "grad_out dtype must equal h_full dtype.");

  TORCH_CHECK(grad_out.size(0) == h_full.size(0), "grad_out B mismatch.");
  TORCH_CHECK(grad_out.size(1) == h_full.size(1), "grad_out D mismatch.");
  TORCH_CHECK(grad_out.size(2) == kernel_chunk.size(1), "grad_out T mismatch.");
}

torch::Tensor forward_chunk(torch::Tensor h_full, torch::Tensor kernel_chunk,
                            torch::Tensor kernel_mix, int64_t t_offset,
                            int64_t dilation) {
  check_common_inputs(h_full, kernel_chunk, kernel_mix, t_offset, dilation);

  return fused_dynamic_conv_forward_chunk_cuda(h_full, kernel_chunk, kernel_mix,
                                               t_offset, dilation);
}

std::vector<torch::Tensor> backward_chunk(torch::Tensor grad_out,
                                          torch::Tensor h_full,
                                          torch::Tensor kernel_chunk,
                                          torch::Tensor kernel_mix,
                                          int64_t t_offset, int64_t dilation) {
  check_backward_inputs(grad_out, h_full, kernel_chunk, kernel_mix, t_offset,
                        dilation);

  return fused_dynamic_conv_backward_chunk_cuda(grad_out, h_full, kernel_chunk,
                                                kernel_mix, t_offset, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_chunk", &forward_chunk,
        "Fused dynamic convolution chunk forward with offset");

  m.def("backward_chunk", &backward_chunk,
        "Fused dynamic convolution chunk backward with cuBLAS GEMM in CUDA");
}
