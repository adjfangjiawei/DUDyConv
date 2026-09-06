#pragma once

#include <torch/extension.h>

#include <string>
#include <vector>

// ======================================================================================
// Layout convention
// ======================================================================================
//
// h:
//   [B, D, L]
//
// kc / kernel_chunk:
//   [B, K, N, T]
//
// mix:
//   [D, N]
//
// go:
//   [B, D, T]
//
// output:
//   [B, D, T]
//
// grad_kc:
//   [B, K, N, T]
//
// ======================================================================================

// ======================================================================================
// Forward selector entry
// ======================================================================================

torch::Tensor fused_dynamic_conv_forward_direct_cuda(torch::Tensor h,
                                                     torch::Tensor kc,
                                                     torch::Tensor mix,
                                                     int64_t off,
                                                     int64_t dilation);

// ======================================================================================
// Forward GEMM plans
// ======================================================================================

bool fdc_new_gemm_nk3_d256_available_cuda(torch::Tensor h, torch::Tensor kc,
                                          torch::Tensor mix, int64_t off,
                                          int64_t dilation);

torch::Tensor fdc_new_forward_gemm_nk3_d256_cuda(torch::Tensor h,
                                                 torch::Tensor kc,
                                                 torch::Tensor mix,
                                                 int64_t off);

bool fdc_forward_manyn_k3_d256_cublas_available_cuda(torch::Tensor h,
                                                     torch::Tensor kc,
                                                     torch::Tensor mix,
                                                     int64_t off,
                                                     int64_t dilation);

torch::Tensor fdc_forward_manyn_k3_d256_cublas_cuda(torch::Tensor h,
                                                    torch::Tensor kc,
                                                    torch::Tensor mix,
                                                    int64_t off,
                                                    int64_t dilation);

// ======================================================================================
// Forward direct execution implementations
// ======================================================================================

torch::Tensor fdc_forward_direct_generic_2d_cuda(torch::Tensor h,
                                                 torch::Tensor kc,
                                                 torch::Tensor mix, int64_t off,
                                                 int64_t dilation);

torch::Tensor fdc_forward_direct_generic_smalln_preload_cuda(torch::Tensor h,
                                                             torch::Tensor kc,
                                                             torch::Tensor mix,
                                                             int64_t off,
                                                             int64_t dilation);

torch::Tensor fdc_forward_direct_n6k3_d256_cuda(torch::Tensor h,
                                                torch::Tensor kc,
                                                torch::Tensor mix, int64_t off);

torch::Tensor fdc_forward_direct_n6k3_d512_cuda(torch::Tensor h,
                                                torch::Tensor kc,
                                                torch::Tensor mix, int64_t off);

// ======================================================================================
// Forward specialized N={16,32,64,128}, K=3, D=256 implementations
// ======================================================================================

bool fdc_forward_n16_k3_d256_available_cuda(torch::Tensor h, torch::Tensor kc,
                                            torch::Tensor mix, int64_t off,
                                            int64_t dilation);

torch::Tensor fdc_forward_n16_k3_d256_cuda(torch::Tensor h, torch::Tensor kc,
                                           torch::Tensor mix, int64_t off);

bool fdc_forward_n32_k3_d256_available_cuda(torch::Tensor h, torch::Tensor kc,
                                            torch::Tensor mix, int64_t off,
                                            int64_t dilation);

torch::Tensor fdc_forward_n32_k3_d256_cuda(torch::Tensor h, torch::Tensor kc,
                                           torch::Tensor mix, int64_t off);

bool fdc_forward_n64_k3_d256_available_cuda(torch::Tensor h, torch::Tensor kc,
                                            torch::Tensor mix, int64_t off,
                                            int64_t dilation);

torch::Tensor fdc_forward_n64_k3_d256_cuda(torch::Tensor h, torch::Tensor kc,
                                           torch::Tensor mix, int64_t off);

bool fdc_forward_n128_k3_d256_available_cuda(torch::Tensor h, torch::Tensor kc,
                                             torch::Tensor mix, int64_t off,
                                             int64_t dilation);

torch::Tensor fdc_forward_n128_k3_d256_cuda(torch::Tensor h, torch::Tensor kc,
                                            torch::Tensor mix, int64_t off);

// ======================================================================================
// Forward N=128 K=3 D=256 batched SGEMM plan
// ======================================================================================

bool fdc_forward_n128_k3_d256_batched_sgemm_available_cuda(torch::Tensor h,
                                                           torch::Tensor kc,
                                                           torch::Tensor mix,
                                                           int64_t off,
                                                           int64_t dilation);

torch::Tensor fdc_forward_n128_k3_d256_batched_sgemm_cuda(torch::Tensor h,
                                                          torch::Tensor kc,
                                                          torch::Tensor mix,
                                                          int64_t off);

// ======================================================================================
// Forward fp16 Tensor Core GEMM plans
// ======================================================================================

bool fdc_forward_n16_k3_d256_gemm_fp16_available_cuda(torch::Tensor h,
                                                      torch::Tensor kc,
                                                      torch::Tensor mix,
                                                      int64_t off,
                                                      int64_t dilation);

torch::Tensor fdc_forward_n16_k3_d256_gemm_fp16_cuda(torch::Tensor h,
                                                     torch::Tensor kc,
                                                     torch::Tensor mix,
                                                     int64_t off);

bool fdc_forward_n32_k3_d256_gemm_fp16_available_cuda(torch::Tensor h,
                                                      torch::Tensor kc,
                                                      torch::Tensor mix,
                                                      int64_t off,
                                                      int64_t dilation);

torch::Tensor fdc_forward_n32_k3_d256_gemm_fp16_cuda(torch::Tensor h,
                                                     torch::Tensor kc,
                                                     torch::Tensor mix,
                                                     int64_t off);

bool fdc_forward_n64_k3_d256_gemm_fp16_available_cuda(torch::Tensor h,
                                                      torch::Tensor kc,
                                                      torch::Tensor mix,
                                                      int64_t off,
                                                      int64_t dilation);

torch::Tensor fdc_forward_n64_k3_d256_gemm_fp16_cuda(torch::Tensor h,
                                                     torch::Tensor kc,
                                                     torch::Tensor mix,
                                                     int64_t off);

bool fdc_forward_n128_k3_d256_gemm_fp16_available_cuda(torch::Tensor h,
                                                       torch::Tensor kc,
                                                       torch::Tensor mix,
                                                       int64_t off,
                                                       int64_t dilation);

torch::Tensor fdc_forward_n128_k3_d256_gemm_fp16_cuda(torch::Tensor h,
                                                      torch::Tensor kc,
                                                      torch::Tensor mix,
                                                      int64_t off);

// ======================================================================================
// Backward basic plans
// ======================================================================================

std::vector<torch::Tensor>
fdc_backward_mid_n6k3_warp_cuda(torch::Tensor go, torch::Tensor h,
                                torch::Tensor kc, torch::Tensor mix,
                                int64_t off, bool full4096_offset0);

std::vector<torch::Tensor> fdc_backward_manyn_k3_d256_cuda(torch::Tensor go,
                                                           torch::Tensor h,
                                                           torch::Tensor kc,
                                                           torch::Tensor mix,
                                                           int64_t off);

bool fdc_backward_manyn_k3_d256_cublas_sliced_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor>
fdc_backward_manyn_k3_d256_cublas_sliced_cuda(torch::Tensor go, torch::Tensor h,
                                              torch::Tensor kc,
                                              torch::Tensor mix, int64_t off);

std::vector<torch::Tensor>
fdc_backward_large_cuda(torch::Tensor go, torch::Tensor h, torch::Tensor kc,
                        torch::Tensor mix, int64_t off, int64_t dilation);

std::vector<torch::Tensor>
fdc_backward_fp16_gemmex_v3_cuda(torch::Tensor go, torch::Tensor h,
                                 torch::Tensor kc, torch::Tensor mix,
                                 int64_t off, int64_t dilation);

// ======================================================================================
// Backward GEMM plans
// ======================================================================================

std::vector<torch::Tensor>
fdc_new_backward_gemm_nk3_d256_cuda(torch::Tensor go, torch::Tensor h,
                                    torch::Tensor kc, torch::Tensor mix,
                                    int64_t off);

bool fdc_backward_n16_k3_d256_gemm_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor> fdc_backward_n16_k3_d256_gemm_cuda(torch::Tensor go,
                                                              torch::Tensor h,
                                                              torch::Tensor kc,
                                                              torch::Tensor mix,
                                                              int64_t off);

bool fdc_backward_n32_k3_d256_gemm_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor> fdc_backward_n32_k3_d256_gemm_cuda(torch::Tensor go,
                                                              torch::Tensor h,
                                                              torch::Tensor kc,
                                                              torch::Tensor mix,
                                                              int64_t off);

bool fdc_backward_n64_k3_d256_gemm_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor> fdc_backward_n64_k3_d256_gemm_cuda(torch::Tensor go,
                                                              torch::Tensor h,
                                                              torch::Tensor kc,
                                                              torch::Tensor mix,
                                                              int64_t off);

bool fdc_backward_n128_k3_d256_gemm_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor>
fdc_backward_n128_k3_d256_gemm_cuda(torch::Tensor go, torch::Tensor h,
                                    torch::Tensor kc, torch::Tensor mix,
                                    int64_t off);

// ======================================================================================
// Backward fp16 Tensor Core GEMM plans
// ======================================================================================

bool fdc_backward_n16_k3_d256_gemm_fp16_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor>
fdc_backward_n16_k3_d256_gemm_fp16_cuda(torch::Tensor go, torch::Tensor h,
                                        torch::Tensor kc, torch::Tensor mix,
                                        int64_t off);

bool fdc_backward_n32_k3_d256_gemm_fp16_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor>
fdc_backward_n32_k3_d256_gemm_fp16_cuda(torch::Tensor go, torch::Tensor h,
                                        torch::Tensor kc, torch::Tensor mix,
                                        int64_t off);

bool fdc_backward_n64_k3_d256_gemm_fp16_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor>
fdc_backward_n64_k3_d256_gemm_fp16_cuda(torch::Tensor go, torch::Tensor h,
                                        torch::Tensor kc, torch::Tensor mix,
                                        int64_t off);

bool fdc_backward_n128_k3_d256_gemm_fp16_available_cuda(
    torch::Tensor go, torch::Tensor h, torch::Tensor kc, torch::Tensor mix,
    int64_t off, int64_t dilation);

std::vector<torch::Tensor>
fdc_backward_n128_k3_d256_gemm_fp16_cuda(torch::Tensor go, torch::Tensor h,
                                         torch::Tensor kc, torch::Tensor mix,
                                         int64_t off);
