#include <torch/extension.h>

#include "fused_dynamic_conv_forward_n_k3_d256_gemm_fp16_template.cuh"

bool fdc_forward_n32_k3_d256_gemm_fp16_available_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off,
    int64_t dilation
) {
    return fdc_forward_n_k3_d256_gemm_fp16_detail::available<32>(
        h,
        kc,
        mix,
        off,
        dilation
    );
}

torch::Tensor fdc_forward_n32_k3_d256_gemm_fp16_cuda(
    torch::Tensor h,
    torch::Tensor kc,
    torch::Tensor mix,
    int64_t off
) {
    return fdc_forward_n_k3_d256_gemm_fp16_detail::forward_cuda<32>(
        h,
        kc,
        mix,
        off,
        "forward_n32_k3_d256_gemm_fp16"
    );
}
