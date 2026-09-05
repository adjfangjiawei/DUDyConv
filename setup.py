import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT_DIR = Path(__file__).resolve().parent
CUDA_OPS_DIR = ROOT_DIR / "cuda_ops"


sources = [
    str(CUDA_OPS_DIR / "fused_dynamic_conv.cpp"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_kernel.cu"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_forward_direct.cu"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_backward_small_n6k3.cu"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_backward_mid_n6k3.cu"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_backward_base_n6k3.cu"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_backward_base_n16k3.cu"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_backward_large.cu"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_backward_fp16_gemmex.cu"),
    str(CUDA_OPS_DIR / "fused_dynamic_conv_gemm_plans.cu"),
]


include_dirs = [
    str(CUDA_OPS_DIR),
]


extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17",
    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-lineinfo",

        # GTX 1660 Ti = Turing, compute capability 7.5
        "-gencode=arch=compute_75,code=sm_75",

        # 防止部分 CUDA/torch 组合因为 half/bfloat16 宏出问题
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
}


setup(
    name="cuda_ops_ext",
    version="0.0.1",
    description="Fused dynamic convolution CUDA extension",
    packages=[],
    ext_modules=[
        CUDAExtension(
            name="cuda_ops_ext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    zip_safe=False,
)
