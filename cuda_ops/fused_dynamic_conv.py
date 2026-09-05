import torch
import torch.nn as nn

import cuda_ops_ext as _EXT


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
            raise RuntimeError(
                f"h_full must be [B,D,L], got {tuple(h_full.shape)}."
            )

        if kernel_chunk.dim() != 4:
            raise RuntimeError(
                f"kernel_chunk must be [B,T,N,K], got {tuple(kernel_chunk.shape)}."
            )

        if kernel_mix.dim() != 2:
            raise RuntimeError(
                f"kernel_mix must be [D,N], got {tuple(kernel_mix.shape)}."
            )

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

        out = _EXT.forward_chunk(
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

        grad_h_full, grad_kernel_chunk, grad_kernel_mix = _EXT.backward_chunk(
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


def _check_forward_inputs(
    h_full: torch.Tensor,
    kernel_chunk: torch.Tensor,
    kernel_mix: torch.Tensor,
    t_offset: int,
    dilation: int,
) -> None:
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
        raise RuntimeError(
            f"h_full must be [B,D,L], got {tuple(h_full.shape)}."
        )

    if kernel_chunk.dim() != 4:
        raise RuntimeError(
            f"kernel_chunk must be [B,T,N,K], got {tuple(kernel_chunk.shape)}."
        )

    if kernel_mix.dim() != 2:
        raise RuntimeError(
            f"kernel_mix must be [D,N], got {tuple(kernel_mix.shape)}."
        )

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


def fused_dynamic_conv_forward_warmup_chunk(
    h_full: torch.Tensor,
    kernel_chunk: torch.Tensor,
    kernel_mix: torch.Tensor,
    t_offset: int,
    dilation: int = 1,
    repeat: int = 3,
) -> int:
    if not isinstance(t_offset, int):
        t_offset = int(t_offset)

    if not isinstance(dilation, int):
        dilation = int(dilation)

    if not isinstance(repeat, int):
        repeat = int(repeat)

    if repeat <= 0:
        raise RuntimeError("repeat must be positive.")

    _check_forward_inputs(
        h_full=h_full,
        kernel_chunk=kernel_chunk,
        kernel_mix=kernel_mix,
        t_offset=int(t_offset),
        dilation=int(dilation),
    )

    plan_id = _EXT.forward_chunk_warmup(
        h_full.contiguous(),
        kernel_chunk.contiguous(),
        kernel_mix.contiguous(),
        int(t_offset),
        int(dilation),
        int(repeat),
    )

    return int(plan_id)


def fused_dynamic_conv_forward_warmup(
    h: torch.Tensor,
    kernel: torch.Tensor,
    kernel_mix: torch.Tensor,
    dilation: int = 1,
    repeat: int = 3,
) -> int:
    return fused_dynamic_conv_forward_warmup_chunk(
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=0,
        dilation=int(dilation),
        repeat=int(repeat),
    )


def fused_dynamic_conv_forward_cached_plan_chunk(
    h_full: torch.Tensor,
    kernel_chunk: torch.Tensor,
    kernel_mix: torch.Tensor,
    t_offset: int,
    dilation: int = 1,
) -> int:
    if not isinstance(t_offset, int):
        t_offset = int(t_offset)

    if not isinstance(dilation, int):
        dilation = int(dilation)

    _check_forward_inputs(
        h_full=h_full,
        kernel_chunk=kernel_chunk,
        kernel_mix=kernel_mix,
        t_offset=int(t_offset),
        dilation=int(dilation),
    )

    return int(
        _EXT.forward_chunk_cached_plan(
            h_full.contiguous(),
            kernel_chunk.contiguous(),
            kernel_mix.contiguous(),
            int(t_offset),
            int(dilation),
        )
    )


def fused_dynamic_conv_forward_cached_plan(
    h: torch.Tensor,
    kernel: torch.Tensor,
    kernel_mix: torch.Tensor,
    dilation: int = 1,
) -> int:
    return fused_dynamic_conv_forward_cached_plan_chunk(
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=0,
        dilation=int(dilation),
    )


def fused_dynamic_conv_forward_clear_warmup_cache() -> None:
    _EXT.forward_chunk_clear_warmup_cache()


def fused_dynamic_conv_forward_plan_name(plan_id: int) -> str:
    plan_id = int(plan_id)

    try:
        return str(_EXT.forward_plan_name(plan_id))
    except Exception:
        names = {
            -1: "not_cached",
            0: "forward_direct_generic_2d",
            1: "forward_direct_generic_smalln_preload",
            2: "forward_direct_n6k3_d256",
            3: "forward_direct_n16k3_d256",
            4: "forward_direct_n6k7_d256",
            5: "forward_direct_n6k3_d512",
            6: "forward_new_gemm_nk3_d256",
        }

        return names.get(plan_id, f"unknown_forward_plan_{plan_id}")


def fused_dynamic_conv_backward_warmup_chunk(
    grad_out: torch.Tensor,
    h_full: torch.Tensor,
    kernel_chunk: torch.Tensor,
    kernel_mix: torch.Tensor,
    t_offset: int,
    dilation: int = 1,
    repeat: int = 3,
) -> int:
    if not isinstance(t_offset, int):
        t_offset = int(t_offset)

    if not isinstance(dilation, int):
        dilation = int(dilation)

    if not isinstance(repeat, int):
        repeat = int(repeat)

    if dilation <= 0:
        raise RuntimeError("dilation must be positive.")

    if repeat <= 0:
        raise RuntimeError("repeat must be positive.")

    if not grad_out.is_cuda:
        raise RuntimeError("grad_out must be CUDA tensor.")

    if not h_full.is_cuda:
        raise RuntimeError("h_full must be CUDA tensor.")

    if not kernel_chunk.is_cuda:
        raise RuntimeError("kernel_chunk must be CUDA tensor.")

    if not kernel_mix.is_cuda:
        raise RuntimeError("kernel_mix must be CUDA tensor.")

    if h_full.dim() != 3:
        raise RuntimeError(
            f"h_full must be [B,D,L], got {tuple(h_full.shape)}."
        )

    if kernel_chunk.dim() != 4:
        raise RuntimeError(
            f"kernel_chunk must be [B,T,N,K], got {tuple(kernel_chunk.shape)}."
        )

    if kernel_mix.dim() != 2:
        raise RuntimeError(
            f"kernel_mix must be [D,N], got {tuple(kernel_mix.shape)}."
        )

    if grad_out.dim() != 3:
        raise RuntimeError(
            f"grad_out must be [B,D,T], got {tuple(grad_out.shape)}."
        )

    B, D, L = h_full.shape
    Bk, T, N, K = kernel_chunk.shape
    Dm, Nm = kernel_mix.shape

    if B != Bk:
        raise RuntimeError(f"B mismatch: h_full={B}, kernel_chunk={Bk}.")

    if D != Dm:
        raise RuntimeError(f"D mismatch: h_full={D}, kernel_mix={Dm}.")

    if N != Nm:
        raise RuntimeError(f"N mismatch: kernel_chunk={N}, kernel_mix={Nm}.")

    if grad_out.shape[0] != B:
        raise RuntimeError(
            f"grad_out B mismatch: grad_out={grad_out.shape[0]}, h_full={B}."
        )

    if grad_out.shape[1] != D:
        raise RuntimeError(
            f"grad_out D mismatch: grad_out={grad_out.shape[1]}, h_full={D}."
        )

    if grad_out.shape[2] != T:
        raise RuntimeError(
            f"grad_out T mismatch: grad_out={grad_out.shape[2]}, kernel_chunk={T}."
        )

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

    if grad_out.dtype != h_full.dtype:
        raise RuntimeError("grad_out dtype must equal h_full dtype.")

    if kernel_chunk.dtype != h_full.dtype:
        raise RuntimeError("kernel_chunk dtype must equal h_full dtype.")

    if kernel_mix.dtype != h_full.dtype:
        raise RuntimeError("kernel_mix dtype must equal h_full dtype.")

    plan_id = _EXT.backward_chunk_warmup(
        grad_out.contiguous(),
        h_full.contiguous(),
        kernel_chunk.contiguous(),
        kernel_mix.contiguous(),
        int(t_offset),
        int(dilation),
        int(repeat),
    )

    return int(plan_id)


def fused_dynamic_conv_backward_warmup(
    grad_out: torch.Tensor,
    h: torch.Tensor,
    kernel: torch.Tensor,
    kernel_mix: torch.Tensor,
    dilation: int = 1,
    repeat: int = 3,
) -> int:
    return fused_dynamic_conv_backward_warmup_chunk(
        grad_out=grad_out,
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=0,
        dilation=int(dilation),
        repeat=int(repeat),
    )


def fused_dynamic_conv_backward_cached_plan_chunk(
    h_full: torch.Tensor,
    kernel_chunk: torch.Tensor,
    t_offset: int,
    dilation: int = 1,
) -> int:
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

    if h_full.dim() != 3:
        raise RuntimeError(
            f"h_full must be [B,D,L], got {tuple(h_full.shape)}."
        )

    if kernel_chunk.dim() != 4:
        raise RuntimeError(
            f"kernel_chunk must be [B,T,N,K], got {tuple(kernel_chunk.shape)}."
        )

    B, D, L = h_full.shape
    Bk, T, N, K = kernel_chunk.shape

    if B != Bk:
        raise RuntimeError(f"B mismatch: h_full={B}, kernel_chunk={Bk}.")

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

    return int(
        _EXT.backward_chunk_cached_plan(
            h_full.contiguous(),
            kernel_chunk.contiguous(),
            int(t_offset),
            int(dilation),
        )
    )


def fused_dynamic_conv_backward_cached_plan(
    h: torch.Tensor,
    kernel: torch.Tensor,
    dilation: int = 1,
) -> int:
    return fused_dynamic_conv_backward_cached_plan_chunk(
        h_full=h,
        kernel_chunk=kernel,
        t_offset=0,
        dilation=int(dilation),
    )


def fused_dynamic_conv_backward_clear_warmup_cache() -> None:
    _EXT.backward_chunk_clear_warmup_cache()


def fused_dynamic_conv_backward_plan_name(plan_id: int) -> str:
    plan_id = int(plan_id)

    try:
        return str(_EXT.backward_plan_name(plan_id))
    except Exception:
        names = {
            -1: "not_cached",
            0: "backward_small_n6k3",
            1: "backward_mid_n6k3_warp",
            2: "backward_base_n6k3",
            3: "backward_base_n16k3",
            4: "backward_large",
            5: "backward_fp16_gemmex_v3",
            6: "backward_new_gemm_nk3_d256",
        }

        return names.get(plan_id, f"unknown_plan_{plan_id}")


class FusedDynamicConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h,
        kernel,
        kernel_mix,
        dilation: int = 1,
    ):
        return fused_dynamic_conv_chunk(
            h_full=h,
            kernel_chunk=kernel,
            kernel_mix=kernel_mix,
            t_offset=0,
            dilation=int(dilation),
        )

    @staticmethod
    def backward(
        ctx,
        grad_out,
    ):
        raise RuntimeError(
            "Internal error: FusedDynamicConvFunction should not receive backward directly."
        )


class FusedDynamicConvChunk1d(nn.Module):
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

    def warmup_forward(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        t_offset: int,
        dilation: int = 1,
        repeat: int = 3,
    ) -> int:
        kernel_mix = self.kernel_mix

        if kernel_mix.dtype != h_full.dtype:
            kernel_mix = kernel_mix.to(
                dtype=h_full.dtype,
            )

        return fused_dynamic_conv_forward_warmup_chunk(
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            kernel_mix=kernel_mix,
            t_offset=int(t_offset),
            dilation=int(dilation),
            repeat=int(repeat),
        )

    def cached_forward_plan(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        t_offset: int,
        dilation: int = 1,
    ) -> int:
        kernel_mix = self.kernel_mix

        if kernel_mix.dtype != h_full.dtype:
            kernel_mix = kernel_mix.to(
                dtype=h_full.dtype,
            )

        return fused_dynamic_conv_forward_cached_plan_chunk(
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            kernel_mix=kernel_mix,
            t_offset=int(t_offset),
            dilation=int(dilation),
        )

    def warmup_backward(
        self,
        grad_out: torch.Tensor,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        t_offset: int,
        dilation: int = 1,
        repeat: int = 3,
    ) -> int:
        kernel_mix = self.kernel_mix

        if kernel_mix.dtype != h_full.dtype:
            kernel_mix = kernel_mix.to(
                dtype=h_full.dtype,
            )

        return fused_dynamic_conv_backward_warmup_chunk(
            grad_out=grad_out,
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            kernel_mix=kernel_mix,
            t_offset=int(t_offset),
            dilation=int(dilation),
            repeat=int(repeat),
        )

    def cached_backward_plan(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        t_offset: int,
        dilation: int = 1,
    ) -> int:
        return fused_dynamic_conv_backward_cached_plan_chunk(
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            t_offset=int(t_offset),
            dilation=int(dilation),
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
    def warmup_forward(
        self,
        h: torch.Tensor,
        kernel: torch.Tensor,
        dilation: int = 1,
        repeat: int = 3,
    ) -> int:
        return super().warmup_forward(
            h_full=h,
            kernel_chunk=kernel,
            t_offset=0,
            dilation=int(dilation),
            repeat=int(repeat),
        )

    def cached_forward_plan(
        self,
        h: torch.Tensor,
        kernel: torch.Tensor,
        dilation: int = 1,
    ) -> int:
        return super().cached_forward_plan(
            h_full=h,
            kernel_chunk=kernel,
            t_offset=0,
            dilation=int(dilation),
        )

    def warmup_backward(
        self,
        grad_out: torch.Tensor,
        h: torch.Tensor,
        kernel: torch.Tensor,
        dilation: int = 1,
        repeat: int = 3,
    ) -> int:
        return super().warmup_backward(
            grad_out=grad_out,
            h_full=h,
            kernel_chunk=kernel,
            t_offset=0,
            dilation=int(dilation),
            repeat=int(repeat),
        )

    def cached_backward_plan(
        self,
        h: torch.Tensor,
        kernel: torch.Tensor,
        dilation: int = 1,
    ) -> int:
        return super().cached_backward_plan(
            h_full=h,
            kernel_chunk=kernel,
            t_offset=0,
            dilation=int(dilation),
        )

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
