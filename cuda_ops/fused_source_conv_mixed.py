import torch
import torch.nn as nn

# Importing cuda_ops_ext is required so that TORCH_LIBRARY registrations in
# fused_source_conv_mixed_ops.cpp are executed.
import cuda_ops_ext as _CUDA_OPS_EXT  # noqa: F401


_FSCM_OPS = torch.ops.fused_source_conv_mixed


FORWARD_PLAN_GENERIC = 0
FORWARD_PLAN_C8S3 = 1
FORWARD_PLAN_C32S3 = 2
FORWARD_PLAN_D256C16S3 = 3

FORWARD_PLAN_GENERIC_NO_BOUNDARY = 100
FORWARD_PLAN_C8S3_NO_BOUNDARY = 101
FORWARD_PLAN_C32S3_NO_BOUNDARY = 102
FORWARD_PLAN_D256C16S3_NO_BOUNDARY = 103


BACKWARD_PLAN_ATOMIC = 0
BACKWARD_PLAN_PER_D_REDUCE = 1
BACKWARD_PLAN_GATHER_X = 2


def _empty_bias_like_input(
    x_norm: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        0,
        device=x_norm.device,
        dtype=x_norm.dtype,
    )


def _check_source_conv_mixed_forward_inputs(
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int,
) -> None:
    if not isinstance(off, int):
        off = int(off)

    if not isinstance(T, int):
        T = int(T)

    if not isinstance(dilation, int):
        dilation = int(dilation)

    if not x_norm.is_cuda:
        raise RuntimeError("x_norm must be CUDA tensor.")

    if not weight.is_cuda:
        raise RuntimeError("weight must be CUDA tensor.")

    if bias is not None and bias.numel() > 0 and not bias.is_cuda:
        raise RuntimeError("bias must be CUDA tensor when provided.")

    if not mix_weight.is_cuda:
        raise RuntimeError("mix_weight must be CUDA tensor.")

    if not mix_bias.is_cuda:
        raise RuntimeError("mix_bias must be CUDA tensor.")

    if x_norm.dim() != 3:
        raise RuntimeError(
            f"x_norm must be [B,D,L], got {tuple(x_norm.shape)}."
        )

    if weight.dim() != 3:
        raise RuntimeError(
            f"weight must be [D,C,S], got {tuple(weight.shape)}."
        )

    if mix_weight.dim() != 2:
        raise RuntimeError(
            f"mix_weight must be [D,C], got {tuple(mix_weight.shape)}."
        )

    if mix_bias.dim() != 1:
        raise RuntimeError(
            f"mix_bias must be [D], got {tuple(mix_bias.shape)}."
        )

    B, D, L = x_norm.shape
    Dw, C, S = weight.shape

    if B <= 0:
        raise RuntimeError("B must be positive.")

    if D <= 0:
        raise RuntimeError("D must be positive.")

    if L <= 0:
        raise RuntimeError("L must be positive.")

    if C <= 0:
        raise RuntimeError("C must be positive.")

    if S <= 0:
        raise RuntimeError("S must be positive.")

    if Dw != D:
        raise RuntimeError(f"weight D mismatch: weight={Dw}, x_norm={D}.")

    if mix_weight.shape[0] != D:
        raise RuntimeError(
            f"mix_weight D mismatch: mix_weight={mix_weight.shape[0]}, x_norm={D}."
        )

    if mix_weight.shape[1] != C:
        raise RuntimeError(
            f"mix_weight C mismatch: mix_weight={mix_weight.shape[1]}, weight={C}."
        )

    if mix_bias.shape[0] != D:
        raise RuntimeError(
            f"mix_bias D mismatch: mix_bias={mix_bias.shape[0]}, x_norm={D}."
        )

    if bias is not None and bias.numel() > 0:
        if bias.dim() != 2:
            raise RuntimeError(
                f"bias must be [D,C], got {tuple(bias.shape)}."
            )

        if bias.shape[0] != D:
            raise RuntimeError(
                f"bias D mismatch: bias={bias.shape[0]}, x_norm={D}."
            )

        if bias.shape[1] != C:
            raise RuntimeError(
                f"bias C mismatch: bias={bias.shape[1]}, weight={C}."
            )

    if off < 0:
        raise RuntimeError("off must be non-negative.")

    if T <= 0:
        raise RuntimeError("T must be positive.")

    if off + T > L:
        raise RuntimeError(
            f"off + T must be <= L, got off={off}, T={T}, L={L}."
        )

    if dilation <= 0:
        raise RuntimeError("dilation must be positive.")

    if x_norm.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise RuntimeError(f"Unsupported dtype: {x_norm.dtype}.")

    if weight.dtype != x_norm.dtype:
        raise RuntimeError("weight dtype must equal x_norm dtype.")

    if mix_weight.dtype != x_norm.dtype:
        raise RuntimeError("mix_weight dtype must equal x_norm dtype.")

    if mix_bias.dtype != x_norm.dtype:
        raise RuntimeError("mix_bias dtype must equal x_norm dtype.")

    if bias is not None and bias.numel() > 0 and bias.dtype != x_norm.dtype:
        raise RuntimeError("bias dtype must equal x_norm dtype.")


def _check_source_conv_mixed_backward_inputs(
    grad_out: torch.Tensor,
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int,
) -> None:
    _check_source_conv_mixed_forward_inputs(
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    if not grad_out.is_cuda:
        raise RuntimeError("grad_out must be CUDA tensor.")

    if grad_out.dim() != 3:
        raise RuntimeError(
            f"grad_out must be [B,D,T], got {tuple(grad_out.shape)}."
        )

    B, D, _L = x_norm.shape

    if grad_out.shape[0] != B:
        raise RuntimeError(
            f"grad_out B mismatch: grad_out={grad_out.shape[0]}, x_norm={B}."
        )

    if grad_out.shape[1] != D:
        raise RuntimeError(
            f"grad_out D mismatch: grad_out={grad_out.shape[1]}, x_norm={D}."
        )

    if grad_out.shape[2] != T:
        raise RuntimeError(
            f"grad_out T mismatch: grad_out={grad_out.shape[2]}, T={T}."
        )

    if grad_out.dtype != x_norm.dtype:
        raise RuntimeError("grad_out dtype must equal x_norm dtype.")


def _as_bias_tensor(
    x_norm: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if bias is None:
        return _empty_bias_like_input(
            x_norm
        )

    return bias


class FusedSourceConvMixedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_norm: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        mix_weight: torch.Tensor,
        mix_bias: torch.Tensor,
        off: int,
        T: int,
        dilation: int = 1,
    ) -> torch.Tensor:
        if not isinstance(off, int):
            off = int(off)

        if not isinstance(T, int):
            T = int(T)

        if not isinstance(dilation, int):
            dilation = int(dilation)

        _check_source_conv_mixed_forward_inputs(
            x_norm=x_norm,
            weight=weight,
            bias=bias,
            mix_weight=mix_weight,
            mix_bias=mix_bias,
            off=int(off),
            T=int(T),
            dilation=int(dilation),
        )

        x_norm_c = x_norm.contiguous()
        weight_c = weight.contiguous()
        bias_c = _as_bias_tensor(
            x_norm_c,
            bias,
        ).contiguous()
        mix_weight_c = mix_weight.contiguous()
        mix_bias_c = mix_bias.contiguous()

        out = _FSCM_OPS.forward(
            x_norm_c,
            weight_c,
            bias_c,
            mix_weight_c,
            mix_bias_c,
            int(off),
            int(T),
            int(dilation),
        )

        ctx.save_for_backward(
            x_norm_c,
            weight_c,
            bias_c,
            mix_weight_c,
            mix_bias_c,
        )

        ctx.has_bias = bool(
            bias is not None and bias.numel() > 0
        )
        ctx.off = int(off)
        ctx.T = int(T)
        ctx.dilation = int(dilation)

        return out

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ):
        if grad_out is None:
            return None, None, None, None, None, None, None, None

        x_norm, weight, bias, mix_weight, mix_bias = ctx.saved_tensors

        grad_out_c = grad_out.contiguous()

        grads = _FSCM_OPS.backward(
            grad_out_c,
            x_norm,
            weight,
            bias,
            mix_weight,
            mix_bias,
            int(ctx.off),
            int(ctx.T),
            int(ctx.dilation),
        )

        grad_x_norm = grads[0]
        grad_weight = grads[1]
        grad_bias = grads[2]
        grad_mix_weight = grads[3]
        grad_mix_bias = grads[4]

        if not ctx.has_bias:
            grad_bias = None

        return (
            grad_x_norm,
            grad_weight,
            grad_bias,
            grad_mix_weight,
            grad_mix_bias,
            None,
            None,
            None,
        )


def fused_source_conv_mixed(
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int = 1,
) -> torch.Tensor:
    return FusedSourceConvMixedFunction.apply(
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
        int(off),
        int(T),
        int(dilation),
    )


def fused_source_conv_mixed_forward(
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int = 1,
) -> torch.Tensor:
    _check_source_conv_mixed_forward_inputs(
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    x_norm_c = x_norm.contiguous()
    bias_c = _as_bias_tensor(
        x_norm_c,
        bias,
    ).contiguous()

    return _FSCM_OPS.forward(
        x_norm_c,
        weight.contiguous(),
        bias_c,
        mix_weight.contiguous(),
        mix_bias.contiguous(),
        int(off),
        int(T),
        int(dilation),
    )


def fused_source_conv_mixed_forward_plan(
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int,
    plan: int,
) -> torch.Tensor:
    _check_source_conv_mixed_forward_inputs(
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    x_norm_c = x_norm.contiguous()
    bias_c = _as_bias_tensor(
        x_norm_c,
        bias,
    ).contiguous()

    return _FSCM_OPS.forward_plan(
        x_norm_c,
        weight.contiguous(),
        bias_c,
        mix_weight.contiguous(),
        mix_bias.contiguous(),
        int(off),
        int(T),
        int(dilation),
        int(plan),
    )


def fused_source_conv_mixed_forward_cached_plan(
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int,
    cached_plan: int,
) -> torch.Tensor:
    _check_source_conv_mixed_forward_inputs(
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    x_norm_c = x_norm.contiguous()
    bias_c = _as_bias_tensor(
        x_norm_c,
        bias,
    ).contiguous()

    return _FSCM_OPS.forward_cached_plan(
        x_norm_c,
        weight.contiguous(),
        bias_c,
        mix_weight.contiguous(),
        mix_bias.contiguous(),
        int(off),
        int(T),
        int(dilation),
        int(cached_plan),
    )


def fused_source_conv_mixed_forward_warmup(
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int = 1,
    warmup_iters: int = 2,
    bench_iters: int = 5,
) -> int:
    if warmup_iters < 0:
        raise RuntimeError("warmup_iters must be >= 0.")

    if bench_iters <= 0:
        raise RuntimeError("bench_iters must be positive.")

    _check_source_conv_mixed_forward_inputs(
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    x_norm_c = x_norm.contiguous()
    bias_c = _as_bias_tensor(
        x_norm_c,
        bias,
    ).contiguous()

    return int(
        _FSCM_OPS.forward_warmup(
            x_norm_c,
            weight.contiguous(),
            bias_c,
            mix_weight.contiguous(),
            mix_bias.contiguous(),
            int(off),
            int(T),
            int(dilation),
            int(warmup_iters),
            int(bench_iters),
        )
    )


def fused_source_conv_mixed_backward(
    grad_out: torch.Tensor,
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int = 1,
):
    _check_source_conv_mixed_backward_inputs(
        grad_out=grad_out,
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    x_norm_c = x_norm.contiguous()
    bias_c = _as_bias_tensor(
        x_norm_c,
        bias,
    ).contiguous()

    grads = _FSCM_OPS.backward(
        grad_out.contiguous(),
        x_norm_c,
        weight.contiguous(),
        bias_c,
        mix_weight.contiguous(),
        mix_bias.contiguous(),
        int(off),
        int(T),
        int(dilation),
    )

    if bias is None:
        return grads[0], grads[1], None, grads[3], grads[4]

    return grads[0], grads[1], grads[2], grads[3], grads[4]


def fused_source_conv_mixed_backward_plan(
    grad_out: torch.Tensor,
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int,
    plan: int,
    tile_t: int,
):
    _check_source_conv_mixed_backward_inputs(
        grad_out=grad_out,
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    x_norm_c = x_norm.contiguous()
    bias_c = _as_bias_tensor(
        x_norm_c,
        bias,
    ).contiguous()

    grads = _FSCM_OPS.backward_plan(
        grad_out.contiguous(),
        x_norm_c,
        weight.contiguous(),
        bias_c,
        mix_weight.contiguous(),
        mix_bias.contiguous(),
        int(off),
        int(T),
        int(dilation),
        int(plan),
        int(tile_t),
    )

    if bias is None:
        return grads[0], grads[1], None, grads[3], grads[4]

    return grads[0], grads[1], grads[2], grads[3], grads[4]


def fused_source_conv_mixed_backward_cached_plan(
    grad_out: torch.Tensor,
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int,
    cached_plan: int,
):
    _check_source_conv_mixed_backward_inputs(
        grad_out=grad_out,
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    x_norm_c = x_norm.contiguous()
    bias_c = _as_bias_tensor(
        x_norm_c,
        bias,
    ).contiguous()

    grads = _FSCM_OPS.backward_cached_plan(
        grad_out.contiguous(),
        x_norm_c,
        weight.contiguous(),
        bias_c,
        mix_weight.contiguous(),
        mix_bias.contiguous(),
        int(off),
        int(T),
        int(dilation),
        int(cached_plan),
    )

    if bias is None:
        return grads[0], grads[1], None, grads[3], grads[4]

    return grads[0], grads[1], grads[2], grads[3], grads[4]


def fused_source_conv_mixed_backward_warmup(
    grad_out: torch.Tensor,
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int = 1,
    warmup_iters: int = 2,
    bench_iters: int = 5,
) -> int:
    if warmup_iters < 0:
        raise RuntimeError("warmup_iters must be >= 0.")

    if bench_iters <= 0:
        raise RuntimeError("bench_iters must be positive.")

    _check_source_conv_mixed_backward_inputs(
        grad_out=grad_out,
        x_norm=x_norm,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=int(off),
        T=int(T),
        dilation=int(dilation),
    )

    x_norm_c = x_norm.contiguous()
    bias_c = _as_bias_tensor(
        x_norm_c,
        bias,
    ).contiguous()

    return int(
        _FSCM_OPS.backward_warmup(
            grad_out.contiguous(),
            x_norm_c,
            weight.contiguous(),
            bias_c,
            mix_weight.contiguous(),
            mix_bias.contiguous(),
            int(off),
            int(T),
            int(dilation),
            int(warmup_iters),
            int(bench_iters),
        )
    )


def fused_source_conv_mixed_backward_pack_cached_plan(
    plan: int,
    tile_t: int,
) -> int:
    return int(
        _FSCM_OPS.backward_pack_cached_plan(
            int(plan),
            int(tile_t),
        )
    )


def fused_source_conv_mixed_backward_cached_plan_get_plan(
    cached_plan: int,
) -> int:
    return int(
        _FSCM_OPS.backward_cached_plan_get_plan(
            int(cached_plan),
        )
    )


def fused_source_conv_mixed_backward_cached_plan_get_tile_t(
    cached_plan: int,
) -> int:
    return int(
        _FSCM_OPS.backward_cached_plan_get_tile_t(
            int(cached_plan),
        )
    )


def fused_source_conv_mixed_forward_plan_name(
    plan: int,
) -> str:
    plan = int(plan)

    names = {
        FORWARD_PLAN_GENERIC: "Generic",
        FORWARD_PLAN_C8S3: "C8S3",
        FORWARD_PLAN_C32S3: "C32S3",
        FORWARD_PLAN_D256C16S3: "D256C16S3",
        FORWARD_PLAN_GENERIC_NO_BOUNDARY: "GenericNoBoundary",
        FORWARD_PLAN_C8S3_NO_BOUNDARY: "C8S3NoBoundary",
        FORWARD_PLAN_C32S3_NO_BOUNDARY: "C32S3NoBoundary",
        FORWARD_PLAN_D256C16S3_NO_BOUNDARY: "D256C16S3NoBoundary",
    }

    return names.get(
        plan,
        f"unknown_forward_plan_{plan}",
    )


def fused_source_conv_mixed_backward_plan_name(
    plan: int,
) -> str:
    plan = int(plan)

    names = {
        BACKWARD_PLAN_ATOMIC: "Atomic",
        BACKWARD_PLAN_PER_D_REDUCE: "PerDReduce",
        BACKWARD_PLAN_GATHER_X: "GatherX",
    }

    return names.get(
        plan,
        f"unknown_backward_plan_{plan}",
    )


class FusedSourceConvMixedChunk1d(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        source_channels: int,
        kernel_size: int,
        init_scale: float = 0.02,
        use_bias: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.source_channels = int(source_channels)
        self.kernel_size = int(kernel_size)
        self.init_scale = float(init_scale)
        self.use_bias = bool(use_bias)

        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")

        if self.source_channels <= 0:
            raise ValueError("source_channels must be positive.")

        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")

        factory_kwargs = {
            "dtype": dtype,
            "device": device,
        }

        self.weight = nn.Parameter(
            torch.empty(
                self.embed_dim,
                self.source_channels,
                self.kernel_size,
                **factory_kwargs,
            )
        )

        self.mix_weight = nn.Parameter(
            torch.empty(
                self.embed_dim,
                self.source_channels,
                **factory_kwargs,
            )
        )

        self.mix_bias = nn.Parameter(
            torch.zeros(
                self.embed_dim,
                **factory_kwargs,
            )
        )

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.embed_dim,
                    self.source_channels,
                    **factory_kwargs,
                )
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=self.init_scale,
        )

        nn.init.normal_(
            self.mix_weight,
            mean=0.0,
            std=self.init_scale,
        )

    def warmup_forward(
        self,
        x_norm: torch.Tensor,
        off: int,
        T: int,
        dilation: int = 1,
        warmup_iters: int = 2,
        bench_iters: int = 5,
    ) -> int:
        weight = self.weight
        bias = self.bias
        mix_weight = self.mix_weight
        mix_bias = self.mix_bias

        if weight.dtype != x_norm.dtype:
            weight = weight.to(
                dtype=x_norm.dtype,
            )

        if bias is not None and bias.dtype != x_norm.dtype:
            bias = bias.to(
                dtype=x_norm.dtype,
            )

        if mix_weight.dtype != x_norm.dtype:
            mix_weight = mix_weight.to(
                dtype=x_norm.dtype,
            )

        if mix_bias.dtype != x_norm.dtype:
            mix_bias = mix_bias.to(
                dtype=x_norm.dtype,
            )

        return fused_source_conv_mixed_forward_warmup(
            x_norm=x_norm,
            weight=weight,
            bias=bias,
            mix_weight=mix_weight,
            mix_bias=mix_bias,
            off=int(off),
            T=int(T),
            dilation=int(dilation),
            warmup_iters=int(warmup_iters),
            bench_iters=int(bench_iters),
        )

    def warmup_backward(
        self,
        grad_out: torch.Tensor,
        x_norm: torch.Tensor,
        off: int,
        T: int,
        dilation: int = 1,
        warmup_iters: int = 2,
        bench_iters: int = 5,
    ) -> int:
        weight = self.weight
        bias = self.bias
        mix_weight = self.mix_weight
        mix_bias = self.mix_bias

        if weight.dtype != x_norm.dtype:
            weight = weight.to(
                dtype=x_norm.dtype,
            )

        if bias is not None and bias.dtype != x_norm.dtype:
            bias = bias.to(
                dtype=x_norm.dtype,
            )

        if mix_weight.dtype != x_norm.dtype:
            mix_weight = mix_weight.to(
                dtype=x_norm.dtype,
            )

        if mix_bias.dtype != x_norm.dtype:
            mix_bias = mix_bias.to(
                dtype=x_norm.dtype,
            )

        return fused_source_conv_mixed_backward_warmup(
            grad_out=grad_out,
            x_norm=x_norm,
            weight=weight,
            bias=bias,
            mix_weight=mix_weight,
            mix_bias=mix_bias,
            off=int(off),
            T=int(T),
            dilation=int(dilation),
            warmup_iters=int(warmup_iters),
            bench_iters=int(bench_iters),
        )

    def forward(
        self,
        x_norm: torch.Tensor,
        off: int,
        T: int,
        dilation: int = 1,
    ) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        mix_weight = self.mix_weight
        mix_bias = self.mix_bias

        if weight.dtype != x_norm.dtype:
            weight = weight.to(
                dtype=x_norm.dtype,
            )

        if bias is not None and bias.dtype != x_norm.dtype:
            bias = bias.to(
                dtype=x_norm.dtype,
            )

        if mix_weight.dtype != x_norm.dtype:
            mix_weight = mix_weight.to(
                dtype=x_norm.dtype,
            )

        if mix_bias.dtype != x_norm.dtype:
            mix_bias = mix_bias.to(
                dtype=x_norm.dtype,
            )

        return fused_source_conv_mixed(
            x_norm=x_norm,
            weight=weight,
            bias=bias,
            mix_weight=mix_weight,
            mix_bias=mix_bias,
            off=int(off),
            T=int(T),
            dilation=int(dilation),
        )


class FusedSourceConvMixed1d(FusedSourceConvMixedChunk1d):
    def warmup_forward(
        self,
        x_norm: torch.Tensor,
        dilation: int = 1,
        warmup_iters: int = 2,
        bench_iters: int = 5,
    ) -> int:
        return super().warmup_forward(
            x_norm=x_norm,
            off=0,
            T=int(x_norm.shape[2]),
            dilation=int(dilation),
            warmup_iters=int(warmup_iters),
            bench_iters=int(bench_iters),
        )

    def warmup_backward(
        self,
        grad_out: torch.Tensor,
        x_norm: torch.Tensor,
        dilation: int = 1,
        warmup_iters: int = 2,
        bench_iters: int = 5,
    ) -> int:
        return super().warmup_backward(
            grad_out=grad_out,
            x_norm=x_norm,
            off=0,
            T=int(x_norm.shape[2]),
            dilation=int(dilation),
            warmup_iters=int(warmup_iters),
            bench_iters=int(bench_iters),
        )

    def forward(
        self,
        x_norm: torch.Tensor,
        dilation: int = 1,
    ) -> torch.Tensor:
        return super().forward(
            x_norm=x_norm,
            off=0,
            T=int(x_norm.shape[2]),
            dilation=int(dilation),
        )
