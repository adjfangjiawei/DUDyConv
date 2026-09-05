#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cstdlib>
#include <type_traits>

template <typename T>
__device__ __forceinline__ float fdc_to_float_dev(T x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float fdc_to_float_dev<c10::Half>(c10::Half x) {
    return __half2float(static_cast<__half>(x));
}

template <>
__device__ __forceinline__ float fdc_to_float_dev<c10::BFloat16>(c10::BFloat16 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __bfloat162float(static_cast<__nv_bfloat16>(x));
#else
    return static_cast<float>(x);
#endif
}

template <typename T>
__device__ __forceinline__ T fdc_from_float_dev(float x) {
    return static_cast<T>(x);
}

template <>
__device__ __forceinline__ c10::Half fdc_from_float_dev<c10::Half>(float x) {
    return c10::Half(__float2half_rn(x));
}

template <>
__device__ __forceinline__ c10::BFloat16 fdc_from_float_dev<c10::BFloat16>(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return c10::BFloat16(__float2bfloat16_rn(x));
#else
    return c10::BFloat16(x);
#endif
}

__device__ __forceinline__ float fdc_warp_sum_float(float v) {
    unsigned mask = 0xffffffffu;

    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);

    return v;
}

static inline void fdc_check_cublas(cublasStatus_t s, const char* msg) {
    TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, msg);
}

static inline bool fdc_debug_path_enabled() {
    const char* v = std::getenv("FDC_DEBUG_PATH");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

#define FDC_DEBUG_PATH(msg)                 \
    do {                                    \
        if (fdc_debug_path_enabled()) {     \
            TORCH_WARN(msg);                \
        }                                   \
    } while (0)
