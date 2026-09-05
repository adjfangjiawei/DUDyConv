#pragma once

#include <string>
#include <torch/extension.h>
#include <vector>

// ======================================================================================
// Forward plans
// ======================================================================================

torch::Tensor fused_dynamic_conv_forward_direct_cuda(torch::Tensor h,
                                                     torch::Tensor kc,
                                                     torch::Tensor mix,
                                                     int64_t off,
                                                     int64_t dilation);

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

torch::Tensor fdc_new_forward_gemm_nk3_d256_cuda(torch::Tensor h,
                                                 torch::Tensor kc,
                                                 torch::Tensor mix,
                                                 int64_t off);

bool fdc_new_gemm_nk3_d256_available_cuda(torch::Tensor h, torch::Tensor kc,
                                          torch::Tensor mix, int64_t off,
                                          int64_t dilation);

// ======================================================================================
// Backward old plans
// ======================================================================================

std::vector<torch::Tensor>
fdc_backward_small_n6k3_cuda(torch::Tensor go, torch::Tensor h,
                             torch::Tensor kc, torch::Tensor mix, int64_t off);

std::vector<torch::Tensor>
fdc_backward_mid_n6k3_warp_cuda(torch::Tensor go, torch::Tensor h,
                                torch::Tensor kc, torch::Tensor mix,
                                int64_t off, bool full4096_offset0);

std::vector<torch::Tensor>
fdc_backward_base_n6k3_cuda(torch::Tensor go, torch::Tensor h, torch::Tensor kc,
                            torch::Tensor mix, int64_t off,
                            bool full4096_offset0);

std::vector<torch::Tensor>
fdc_backward_base_n16k3_cuda(torch::Tensor go, torch::Tensor h,
                             torch::Tensor kc, torch::Tensor mix, int64_t off);

std::vector<torch::Tensor>
fdc_backward_large_cuda(torch::Tensor go, torch::Tensor h, torch::Tensor kc,
                        torch::Tensor mix, int64_t off, int64_t dilation);

std::vector<torch::Tensor>
fdc_backward_fp16_gemmex_v3_cuda(torch::Tensor go, torch::Tensor h,
                                 torch::Tensor kc, torch::Tensor mix,
                                 int64_t off, int64_t dilation);

// ======================================================================================
// New GEMM/materialized plans
// ======================================================================================

std::vector<torch::Tensor>
fdc_new_backward_gemm_nk3_d256_cuda(torch::Tensor go, torch::Tensor h,
                                    torch::Tensor kc, torch::Tensor mix,
                                    int64_t off);
