import os
import torch
import torch.nn as nn

from torch.utils.cpp_extension import load


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_EXT = None


def _load_fused_dynamic_conv_ext():
    global _EXT

    if _EXT is not None:
        return _EXT

    cpp_path = os.path.join(
        _THIS_DIR,
        "fused_dynamic_conv.cpp",
    )

    cu_path = os.path.join(
        _THIS_DIR,
        "fused_dynamic_conv_kernel.cu",
    )

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]

    extra_cflags = [
        "-O3",
    ]

    _EXT = load(
        name="fused_dynamic_conv_ext",
        sources=[
            cpp_path,
            cu_path,
        ],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )

    return _EXT


class FusedDynamicConvChunkFunction(torch.autograd.Function):
    """
    带 offset 的融合 dynamic conv chunk 算子。

    输入:
        h_full:
            [B,D,L], contiguous, cuda, fp32/fp16/bf16

        kernel_chunk:
            [B,T,N,K], contiguous, cuda, same dtype

        kernel_mix:
            [D,N], contiguous, cuda, same dtype

        t_offset:
            int。kernel_chunk 第 0 个时间位置对应 h_full 的全局时间位置。

        dilation:
            int。因果卷积采样间隔。

    输出:
        out_chunk:
            [B,D,T]

    公式:
        global_t = t_offset + i

        out[b,d,i] =
            sum_k sum_n kernel_chunk[b,i,n,k]
                    * kernel_mix[d,n]
                    * h_full[b,d,global_t - k * dilation]

        越界位置跳过。
    """

    @staticmethod
    def forward(
        ctx,
        h_full,
        kernel_chunk,
        kernel_mix,
        t_offset: int,
        dilation: int = 1,
    ):
        if not isinstance(t_offset, int):
            t_offset = int(t_offset)

        if not isinstance(dilation, int):
            dilation = int(dilation)

        if dilation <= 0:
            raise RuntimeError("dilation must be positive.")

        if not h_full.is_cuda:
            raise RuntimeError("h_full must be CUDA tensor.")
        if not kernel_chunk.is_cuda:
            raise RuntimeError("kernel_chunk must be CUDA tensor.")
        if not kernel_mix.is_cuda:
            raise RuntimeError("kernel_mix must be CUDA tensor.")

        if h_full.dim() != 3:
            raise RuntimeError(f"h_full must be [B,D,L], got {tuple(h_full.shape)}.")
        if kernel_chunk.dim() != 4:
            raise RuntimeError(f"kernel_chunk must be [B,T,N,K], got {tuple(kernel_chunk.shape)}.")
        if kernel_mix.dim() != 2:
            raise RuntimeError(f"kernel_mix must be [D,N], got {tuple(kernel_mix.shape)}.")

        B, D, L = h_full.shape
        Bk, T, N, K = kernel_chunk.shape
        Dm, Nm = kernel_mix.shape

        if B != Bk:
            raise RuntimeError(f"B mismatch: h_full={B}, kernel_chunk={Bk}.")
        if D != Dm:
            raise RuntimeError(f"D mismatch: h_full={D}, kernel_mix={Dm}.")
        if N != Nm:
            raise RuntimeError(f"N mismatch: kernel_chunk={N}, kernel_mix={Nm}.")
        if T <= 0:
            raise RuntimeError("T must be positive.")
        if K <= 0:
            raise RuntimeError("K must be positive.")
        if t_offset < 0:
            raise RuntimeError("t_offset must be non-negative.")
        if t_offset + T > L:
            raise RuntimeError(
                f"t_offset + T must be <= L, got t_offset={t_offset}, T={T}, L={L}."
            )

        if h_full.dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            raise RuntimeError(f"Unsupported dtype: {h_full.dtype}.")

        if kernel_chunk.dtype != h_full.dtype:
            raise RuntimeError("kernel_chunk dtype must equal h_full dtype.")
        if kernel_mix.dtype != h_full.dtype:
            raise RuntimeError("kernel_mix dtype must equal h_full dtype.")

        h_full_c = h_full.contiguous()
        kernel_chunk_c = kernel_chunk.contiguous()
        kernel_mix_c = kernel_mix.contiguous()

        ext = _load_fused_dynamic_conv_ext()

        out = ext.forward_chunk(
            h_full_c,
            kernel_chunk_c,
            kernel_mix_c,
            int(t_offset),
            int(dilation),
        )

        ctx.save_for_backward(
            h_full_c,
            kernel_chunk_c,
            kernel_mix_c,
        )

        ctx.t_offset = int(t_offset)
        ctx.dilation = int(dilation)

        return out

    @staticmethod
    def backward(
        ctx,
        grad_out,
    ):
        h_full, kernel_chunk, kernel_mix = ctx.saved_tensors

        if grad_out is None:
            return None, None, None, None, None

        grad_out_c = grad_out.contiguous()

        ext = _load_fused_dynamic_conv_ext()

        grad_h_full, grad_kernel_chunk, grad_kernel_mix = ext.backward_chunk(
            grad_out_c,
            h_full,
            kernel_chunk,
            kernel_mix,
            int(ctx.t_offset),
            int(ctx.dilation),
        )

        return (
            grad_h_full,
            grad_kernel_chunk,
            grad_kernel_mix,
            None,
            None,
        )


def fused_dynamic_conv_chunk(
    h_full: torch.Tensor,
    kernel_chunk: torch.Tensor,
    kernel_mix: torch.Tensor,
    t_offset: int,
    dilation: int = 1,
) -> torch.Tensor:
    return FusedDynamicConvChunkFunction.apply(
        h_full,
        kernel_chunk,
        kernel_mix,
        int(t_offset),
        int(dilation),
    )


class FusedDynamicConvFunction(torch.autograd.Function):
    """
    整段版本包装。

    等价于:
        fused_dynamic_conv_chunk(
            h_full=h,
            kernel_chunk=kernel,
            kernel_mix=kernel_mix,
            t_offset=0,
            dilation=dilation,
        )

    输入:
        h:
            [B,D,L]

        kernel:
            [B,L,N,K]

        kernel_mix:
            [D,N]

    输出:
        [B,D,L]
    """

    @staticmethod
    def forward(
        ctx,
        h,
        kernel,
        kernel_mix,
        dilation: int = 1,
    ):
        out = FusedDynamicConvChunkFunction.apply(
            h,
            kernel,
            kernel_mix,
            0,
            int(dilation),
        )

        ctx.dilation = int(dilation)

        return out

    @staticmethod
    def backward(
        ctx,
        grad_out,
    ):
        raise RuntimeError(
            "Internal error: FusedDynamicConvFunction should not receive backward directly. "
            "Use fused_dynamic_conv_chunk or fused_dynamic_conv."
        )


def fused_dynamic_conv(
    h: torch.Tensor,
    kernel: torch.Tensor,
    kernel_mix: torch.Tensor,
    dilation: int = 1,
) -> torch.Tensor:
    return fused_dynamic_conv_chunk(
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=0,
        dilation=int(dilation),
    )


class FusedDynamicConvChunk1d(nn.Module):
    """
    带 offset 的 nn.Module 包装。

    这个类持有 kernel_mix 参数。

    forward:
        h_full:
            [B,D,L]

        kernel_chunk:
            [B,T,N,K]

        t_offset:
            int

        dilation:
            int

    return:
        out_chunk:
            [B,D,T]
    """

    def __init__(
        self,
        embed_dim: int,
        num_kernels: int,
        init_scale: float = 0.02,
        use_bias: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.num_kernels = int(num_kernels)
        self.init_scale = float(init_scale)
        self.use_bias = bool(use_bias)

        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if self.num_kernels <= 0:
            raise ValueError("num_kernels must be positive.")

        factory_kwargs = {
            "dtype": dtype,
            "device": device,
        }

        self.kernel_mix = nn.Parameter(
            torch.empty(
                self.embed_dim,
                self.num_kernels,
                **factory_kwargs,
            )
        )

        nn.init.normal_(
            self.kernel_mix,
            mean=0.0,
            std=self.init_scale,
        )

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.embed_dim,
                    **factory_kwargs,
                )
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

    def forward(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        t_offset: int,
        dilation: int = 1,
    ) -> torch.Tensor:
        kernel_mix = self.kernel_mix

        if kernel_mix.dtype != h_full.dtype:
            kernel_mix = kernel_mix.to(
                dtype=h_full.dtype,
            )

        out = fused_dynamic_conv_chunk(
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            kernel_mix=kernel_mix,
            t_offset=int(t_offset),
            dilation=int(dilation),
        )

        if self.bias is not None:
            out = out + self.bias.to(
                dtype=out.dtype,
                device=out.device,
            ).view(
                1,
                -1,
                1,
            )

        return out


class FusedDynamicConv1d(FusedDynamicConvChunk1d):
    """
    整段版本 nn.Module。

    forward:
        h:
            [B,D,L]

        kernel:
            [B,L,N,K]

        dilation:
            int

    return:
        out:
            [B,D,L]
    """

    def forward(
        self,
        h: torch.Tensor,
        kernel: torch.Tensor,
        dilation: int = 1,
    ) -> torch.Tensor:
        return super().forward(
            h_full=h,
            kernel_chunk=kernel,
            t_offset=0,
            dilation=int(dilation),
        )
