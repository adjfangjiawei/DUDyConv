#include <torch/extension.h>
#include <vector>

#include "fused_dynamic_conv_plans.h"
#include "fused_dynamic_conv_backward_n_k3_d256_gemm_template.cuh"

bool fdc_backward_n16_k3_d256_gemm_available_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    return fdc_backward_n_k3_d256_gemm_detail::available<16>(
        go,
        h,
        kc,
        mix,
        off,
        dilation
    );
}

std::vector<torch::Tensor> fdc_backward_n16_k3_d256_gemm_cuda(
    torch::Tensor go,
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    return fdc_backward_n_k3_d256_gemm_detail::backward_cuda<16>(
        go,
        h,
        kc,
        mix,
        off,
        "backward_n16_k3_d256_gemm"
    );
}
