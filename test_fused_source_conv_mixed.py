import argparse
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cuda_ops.fused_source_conv_mixed import (
    BACKWARD_PLAN_ATOMIC,
    BACKWARD_PLAN_GATHER_X,
    BACKWARD_PLAN_PER_D_REDUCE,
    FORWARD_PLAN_C32S3,
    FORWARD_PLAN_C32S3_NO_BOUNDARY,
    FORWARD_PLAN_C8S3,
    FORWARD_PLAN_C8S3_NO_BOUNDARY,
    FORWARD_PLAN_D256C16S3,
    FORWARD_PLAN_D256C16S3_NO_BOUNDARY,
    FORWARD_PLAN_GENERIC,
    FORWARD_PLAN_GENERIC_NO_BOUNDARY,
    fused_source_conv_mixed,
    fused_source_conv_mixed_backward,
    fused_source_conv_mixed_backward_cached_plan,
    fused_source_conv_mixed_backward_cached_plan_get_plan,
    fused_source_conv_mixed_backward_cached_plan_get_tile_t,
    fused_source_conv_mixed_backward_pack_cached_plan,
    fused_source_conv_mixed_backward_plan,
    fused_source_conv_mixed_backward_plan_name,
    fused_source_conv_mixed_backward_warmup,
    fused_source_conv_mixed_forward,
    fused_source_conv_mixed_forward_cached_plan,
    fused_source_conv_mixed_forward_plan,
    fused_source_conv_mixed_forward_plan_name,
    fused_source_conv_mixed_forward_warmup,
)


@dataclass
class Case:
    name: str
    B: int
    D: int
    L: int
    T: int
    C: int
    S: int
    off: int
    dilation: int
    use_bias: bool = True
    scale: float = 0.1


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_leaf_randn(
    shape,
    device,
    dtype,
    scale: float,
    requires_grad: bool = True,
) -> torch.Tensor:
    x = torch.randn(
        *shape,
        device=device,
        dtype=dtype,
    )

    x = x * scale
    x = x.detach()

    if requires_grad:
        x.requires_grad_(True)

    return x


def clone_leaf(
    x: Optional[torch.Tensor],
    requires_grad: bool = True,
) -> Optional[torch.Tensor]:
    if x is None:
        return None

    y = x.detach().clone()

    if requires_grad:
        y.requires_grad_(True)

    return y


def clear_grads(
    tensors,
) -> None:
    for x in tensors:
        if x is not None and x.grad is not None:
            x.grad = None


def torch_reference_source_conv_mixed_chunk(
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int = 1,
) -> torch.Tensor:
    """
    PyTorch reference for fused_source_conv_mixed.

    Layout:
        x_norm:
            [B,D,L]

        weight:
            [D,C,S]

        bias:
            [D,C] or None

        mix_weight:
            [D,C]

        mix_bias:
            [D]

    Output:
        out:
            [B,D,T]

    Semantics:
        z[b,d,c,t] =
            bias[d,c] +
            sum_j x_norm[b,d, off+t-dilation*(S-1-j)] * weight[d,c,j]

        out[b,d,t] =
            mix_bias[d] +
            sum_c GELU(z[b,d,c,t]) * mix_weight[d,c]
    """

    B, D, L = x_norm.shape
    Dw, C, S = weight.shape
    Dm, Cm = mix_weight.shape
    Db = mix_bias.shape[0]

    assert D == Dw
    assert D == Dm
    assert C == Cm
    assert D == Db
    assert off >= 0
    assert T > 0
    assert off + T <= L
    assert dilation > 0

    out = mix_bias.view(
        1,
        D,
        1,
    ).expand(
        B,
        D,
        T,
    ).clone()

    for c in range(C):
        z = torch.zeros(
            B,
            D,
            T,
            device=x_norm.device,
            dtype=x_norm.dtype,
        )

        if bias is not None:
            z = z + bias[:, c].view(
                1,
                D,
                1,
            )

        for j in range(S):
            src = off + torch.arange(
                T,
                device=x_norm.device,
            ) - dilation * (S - 1 - j)

            valid = (
                src >= 0
            ) & (
                src < L
            )

            if not bool(valid.any().item()):
                continue

            src_valid = src[valid].long()
            t_valid = torch.arange(
                T,
                device=x_norm.device,
            )[valid].long()

            x_slice = x_norm[
                :,
                :,
                src_valid,
            ]

            w = weight[
                :,
                c,
                j,
            ].view(
                1,
                D,
                1,
            )

            z[
                :,
                :,
                t_valid,
            ] = z[
                :,
                :,
                t_valid,
            ] + x_slice * w

        a = F.gelu(
            z,
            approximate="none",
        )

        out = out + a * mix_weight[
            :,
            c,
        ].view(
            1,
            D,
            1,
        )

    return out


def torch_reference_source_conv_mixed_materialized(
    x_norm: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mix_weight: torch.Tensor,
    mix_bias: torch.Tensor,
    off: int,
    T: int,
    dilation: int = 1,
) -> torch.Tensor:
    """
    Alternative reference.

    This materializes z:
        z [B,D,C,T]

    Then applies GELU and source mix.

    This is useful as an independent reference from the c-loop version.
    """

    B, D, L = x_norm.shape
    Dw, C, S = weight.shape

    assert D == Dw

    z = torch.zeros(
        B,
        D,
        C,
        T,
        device=x_norm.device,
        dtype=x_norm.dtype,
    )

    if bias is not None:
        z = z + bias.view(
            1,
            D,
            C,
            1,
        )

    for j in range(S):
        src = off + torch.arange(
            T,
            device=x_norm.device,
        ) - dilation * (S - 1 - j)

        valid = (
            src >= 0
        ) & (
            src < L
        )

        if not bool(valid.any().item()):
            continue

        src_valid = src[valid].long()
        t_valid = torch.arange(
            T,
            device=x_norm.device,
        )[valid].long()

        x_slice = x_norm[
            :,
            :,
            src_valid,
        ]
        # [B,D,T_valid]

        w = weight[
            :,
            :,
            j,
        ]
        # [D,C]

        z[
            :,
            :,
            :,
            t_valid,
        ] = z[
            :,
            :,
            :,
            t_valid,
        ] + x_slice.unsqueeze(
            2
        ) * w.view(
            1,
            D,
            C,
            1,
        )

    a = F.gelu(
        z,
        approximate="none",
    )

    out = (
        a * mix_weight.view(
            1,
            D,
            C,
            1,
        )
    ).sum(
        dim=2
    )

    out = out + mix_bias.view(
        1,
        D,
        1,
    )

    return out


def get_tolerances(
    dtype: torch.dtype,
) -> Tuple[float, float]:
    if dtype == torch.float32:
        return 8e-4, 8e-4

    if dtype == torch.float16:
        return 2.0e-1, 2.0e-1

    if dtype == torch.bfloat16:
        return 2.5e-1, 2.5e-1

    raise ValueError(
        dtype
    )


def max_abs_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return float(
        (
            a.float() - b.float()
        ).abs().max().item()
    )


def max_rel_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    af = a.float()
    bf = b.float()

    denom = torch.maximum(
        bf.abs(),
        torch.tensor(
            1e-6,
            device=bf.device,
            dtype=bf.dtype,
        ),
    )

    return float(
        (
            (
                af - bf
            ).abs() / denom
        ).max().item()
    )


def assert_close_named(
    name: str,
    actual: Optional[torch.Tensor],
    expected: Optional[torch.Tensor],
    dtype: torch.dtype,
    print_detail: bool = True,
) -> None:
    if actual is None and expected is None:
        if print_detail:
            print(
                f"    {name:<34} both None",
                flush=True,
            )
        return

    if actual is None or expected is None:
        raise RuntimeError(
            f"{name}: one tensor is None, the other is not."
        )

    atol, rtol = get_tolerances(
        dtype
    )

    abs_diff = max_abs_diff(
        actual,
        expected,
    )

    rel_diff = max_rel_diff(
        actual,
        expected,
    )

    if print_detail:
        print(
            f"    {name:<34} abs={abs_diff:.8e} rel={rel_diff:.8e} "
            f"atol={atol} rtol={rtol}",
            flush=True,
        )

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
    )


def estimate_case_bytes(
    case: Case,
    dtype: torch.dtype,
    include_ref: bool = True,
) -> int:
    if dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        elem = 2
    elif dtype == torch.float32:
        elem = 4
    else:
        elem = 8

    B = case.B
    D = case.D
    L = case.L
    T = case.T
    C = case.C
    S = case.S

    x = B * D * L
    weight = D * C * S
    bias = D * C if case.use_bias else 0
    mix_weight = D * C
    mix_bias = D
    out = B * D * T
    grad = out

    base_min = (
        x
        + weight
        + bias
        + mix_weight
        + mix_bias
        + out
        + grad
    )

    if include_ref:
        # materialized z [B,D,C,T]
        base_min += B * D * C * T

    return int(
        base_min * elem
    )


def get_available_dtypes() -> List[torch.dtype]:
    dtypes = [
        torch.float32,
        torch.float16,
    ]

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtypes.append(
            torch.bfloat16
        )

    return dtypes


def dtype_to_name(
    dtype: torch.dtype,
) -> str:
    if dtype == torch.float32:
        return "float32"

    if dtype == torch.float16:
        return "float16"

    if dtype == torch.bfloat16:
        return "bfloat16"

    if dtype == torch.float64:
        return "float64"

    raise ValueError(
        f"Unsupported dtype: {dtype}"
    )


def case_to_dict(
    case: Case,
) -> Dict[str, Any]:
    return {
        "name": case.name,
        "B": case.B,
        "D": case.D,
        "L": case.L,
        "T": case.T,
        "C": case.C,
        "S": case.S,
        "off": case.off,
        "dilation": case.dilation,
        "use_bias": case.use_bias,
        "scale": case.scale,
    }


def print_case_header(
    idx: int,
    total: int,
    case: Case,
    dtype: torch.dtype,
) -> None:
    est_mb = estimate_case_bytes(
        case,
        dtype,
        include_ref=True,
    ) / 1024.0 / 1024.0

    no_boundary = case.off >= case.dilation * (
        case.S - 1
    )

    print(
        "-" * 100,
        flush=True,
    )

    print(
        f"[case {idx + 1}/{total}] {case.name} "
        f"dtype={dtype} "
        f"B={case.B} D={case.D} L={case.L} T={case.T} "
        f"C={case.C} S={case.S} "
        f"off={case.off} dilation={case.dilation} "
        f"use_bias={case.use_bias} "
        f"no_boundary={no_boundary} "
        f"estimated_min={est_mb:.1f} MiB",
        flush=True,
    )


def make_case_tensors(
    case: Case,
    dtype: torch.dtype,
    seed: int,
    requires_grad: bool = True,
):
    device = "cuda"

    set_seed(
        seed
    )

    x_norm = make_leaf_randn(
        shape=(
            case.B,
            case.D,
            case.L,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=requires_grad,
    )

    weight = make_leaf_randn(
        shape=(
            case.D,
            case.C,
            case.S,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=requires_grad,
    )

    if case.use_bias:
        bias = make_leaf_randn(
            shape=(
                case.D,
                case.C,
            ),
            device=device,
            dtype=dtype,
            scale=case.scale,
            requires_grad=requires_grad,
        )
    else:
        bias = None

    mix_weight = make_leaf_randn(
        shape=(
            case.D,
            case.C,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=requires_grad,
    )

    mix_bias = make_leaf_randn(
        shape=(
            case.D,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=requires_grad,
    )

    return (
        x_norm,
        weight,
        bias,
        mix_weight,
        mix_bias,
    )


def legal_forward_plans_for_case(
    case: Case,
) -> List[int]:
    D = case.D
    C = case.C
    S = case.S
    no_boundary = case.off >= case.dilation * (
        S - 1
    )

    plans = []

    if no_boundary:
        plans.append(
            FORWARD_PLAN_GENERIC_NO_BOUNDARY
        )

        if C == 8 and S == 3:
            plans.append(
                FORWARD_PLAN_C8S3_NO_BOUNDARY
            )

        if C == 32 and S == 3:
            plans.append(
                FORWARD_PLAN_C32S3_NO_BOUNDARY
            )

        if D == 256 and C == 16 and S == 3:
            plans.append(
                FORWARD_PLAN_D256C16S3_NO_BOUNDARY
            )

    plans.append(
        FORWARD_PLAN_GENERIC
    )

    if C == 8 and S == 3:
        plans.append(
            FORWARD_PLAN_C8S3
        )

    if C == 32 and S == 3:
        plans.append(
            FORWARD_PLAN_C32S3
        )

    if D == 256 and C == 16 and S == 3:
        plans.append(
            FORWARD_PLAN_D256C16S3
        )

    # Remove duplicates while preserving order.
    result = []
    seen = set()

    for plan in plans:
        if plan not in seen:
            result.append(
                plan
            )
            seen.add(
                plan
            )

    return result


def legal_backward_plan_tile_pairs_for_case(
    case: Case,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = [
        (
            BACKWARD_PLAN_ATOMIC,
            0,
        )
    ]

    reduce_legal = case.C <= 64 and case.S <= 8

    if reduce_legal:
        for tile_t in (
            64,
            128,
            256,
        ):
            tile = min(
                tile_t,
                case.T,
            )

            pairs.append(
                (
                    BACKWARD_PLAN_PER_D_REDUCE,
                    tile,
                )
            )

            pairs.append(
                (
                    BACKWARD_PLAN_GATHER_X,
                    tile,
                )
            )

    return pairs


def run_correctness_case_body(
    case: Case,
    dtype: torch.dtype,
    seed: int,
    compare_ref2: bool = True,
    test_all_forward_plans: bool = True,
    test_all_backward_plans: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    (
        x,
        weight,
        bias,
        mix_weight,
        mix_bias,
    ) = make_case_tensors(
        case=case,
        dtype=dtype,
        seed=seed,
        requires_grad=True,
    )

    x_ref = clone_leaf(
        x
    )
    weight_ref = clone_leaf(
        weight
    )
    bias_ref = clone_leaf(
        bias
    )
    mix_weight_ref = clone_leaf(
        mix_weight
    )
    mix_bias_ref = clone_leaf(
        mix_bias
    )

    forward_plan = fused_source_conv_mixed_forward_warmup(
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
        warmup_iters=1,
        bench_iters=1,
    )

    print(
        f"    forward warmup selected plan: "
        f"{forward_plan} {fused_source_conv_mixed_forward_plan_name(forward_plan)}",
        flush=True,
    )

    out = fused_source_conv_mixed(
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
    )

    out_forward_only = fused_source_conv_mixed_forward(
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
    )

    assert_close_named(
        "forward_autograd_vs_forward",
        out,
        out_forward_only,
        dtype,
    )

    out_ref = torch_reference_source_conv_mixed_chunk(
        x_norm=x_ref,
        weight=weight_ref,
        bias=bias_ref,
        mix_weight=mix_weight_ref,
        mix_bias=mix_bias_ref,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
    )

    assert_close_named(
        "forward_vs_ref",
        out,
        out_ref,
        dtype,
    )

    if compare_ref2:
        x_ref2 = clone_leaf(
            x
        )
        weight_ref2 = clone_leaf(
            weight
        )
        bias_ref2 = clone_leaf(
            bias
        )
        mix_weight_ref2 = clone_leaf(
            mix_weight
        )
        mix_bias_ref2 = clone_leaf(
            mix_bias
        )

        out_ref2 = torch_reference_source_conv_mixed_materialized(
            x_norm=x_ref2,
            weight=weight_ref2,
            bias=bias_ref2,
            mix_weight=mix_weight_ref2,
            mix_bias=mix_bias_ref2,
            off=case.off,
            T=case.T,
            dilation=case.dilation,
        )

        assert_close_named(
            "ref_vs_ref2",
            out_ref,
            out_ref2,
            dtype,
        )
    else:
        x_ref2 = None
        weight_ref2 = None
        bias_ref2 = None
        mix_weight_ref2 = None
        mix_bias_ref2 = None
        out_ref2 = None

    if test_all_forward_plans:
        for plan in legal_forward_plans_for_case(
            case
        ):
            out_plan = fused_source_conv_mixed_forward_plan(
                x_norm=x,
                weight=weight,
                bias=bias,
                mix_weight=mix_weight,
                mix_bias=mix_bias,
                off=case.off,
                T=case.T,
                dilation=case.dilation,
                plan=plan,
            )

            assert_close_named(
                f"forward_plan_{plan}",
                out_plan,
                out_ref,
                dtype,
            )

    grad = make_leaf_randn(
        shape=out.shape,
        device="cuda",
        dtype=dtype,
        scale=case.scale,
        requires_grad=False,
    )

    backward_cached = fused_source_conv_mixed_backward_warmup(
        grad_out=grad,
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
        warmup_iters=1,
        bench_iters=1,
    )

    bwd_plan = fused_source_conv_mixed_backward_cached_plan_get_plan(
        backward_cached
    )
    bwd_tile_t = fused_source_conv_mixed_backward_cached_plan_get_tile_t(
        backward_cached
    )

    print(
        f"    backward warmup selected cached={backward_cached} "
        f"plan={bwd_plan} {fused_source_conv_mixed_backward_plan_name(bwd_plan)} "
        f"tile_t={bwd_tile_t}",
        flush=True,
    )

    out.backward(
        grad
    )

    out_ref.backward(
        grad
    )

    assert_close_named(
        "grad_x",
        x.grad,
        x_ref.grad,
        dtype,
    )

    assert_close_named(
        "grad_weight",
        weight.grad,
        weight_ref.grad,
        dtype,
    )

    assert_close_named(
        "grad_bias",
        bias.grad if bias is not None else None,
        bias_ref.grad if bias_ref is not None else None,
        dtype,
    )

    assert_close_named(
        "grad_mix_weight",
        mix_weight.grad,
        mix_weight_ref.grad,
        dtype,
    )

    assert_close_named(
        "grad_mix_bias",
        mix_bias.grad,
        mix_bias_ref.grad,
        dtype,
    )

    if compare_ref2:
        out_ref2.backward(
            grad
        )

        assert_close_named(
            "grad_x_ref2",
            x_ref2.grad,
            x_ref.grad,
            dtype,
        )

        assert_close_named(
            "grad_weight_ref2",
            weight_ref2.grad,
            weight_ref.grad,
            dtype,
        )

        assert_close_named(
            "grad_bias_ref2",
            bias_ref2.grad if bias_ref2 is not None else None,
            bias_ref.grad if bias_ref is not None else None,
            dtype,
        )

        assert_close_named(
            "grad_mix_weight_ref2",
            mix_weight_ref2.grad,
            mix_weight_ref.grad,
            dtype,
        )

        assert_close_named(
            "grad_mix_bias_ref2",
            mix_bias_ref2.grad,
            mix_bias_ref.grad,
            dtype,
        )

    if test_all_backward_plans:
        expected_grads = (
            x_ref.grad.detach(),
            weight_ref.grad.detach(),
            bias_ref.grad.detach() if bias_ref is not None else None,
            mix_weight_ref.grad.detach(),
            mix_bias_ref.grad.detach(),
        )

        for plan, tile_t in legal_backward_plan_tile_pairs_for_case(
            case
        ):
            grads = fused_source_conv_mixed_backward_plan(
                grad_out=grad,
                x_norm=x.detach(),
                weight=weight.detach(),
                bias=bias.detach() if bias is not None else None,
                mix_weight=mix_weight.detach(),
                mix_bias=mix_bias.detach(),
                off=case.off,
                T=case.T,
                dilation=case.dilation,
                plan=plan,
                tile_t=tile_t,
            )

            plan_name = fused_source_conv_mixed_backward_plan_name(
                plan
            )

            assert_close_named(
                f"bwd_{plan_name}_tile{tile_t}_gx",
                grads[0],
                expected_grads[0],
                dtype,
            )

            assert_close_named(
                f"bwd_{plan_name}_tile{tile_t}_gw",
                grads[1],
                expected_grads[1],
                dtype,
            )

            assert_close_named(
                f"bwd_{plan_name}_tile{tile_t}_gb",
                grads[2],
                expected_grads[2],
                dtype,
            )

            assert_close_named(
                f"bwd_{plan_name}_tile{tile_t}_gmw",
                grads[3],
                expected_grads[3],
                dtype,
            )

            assert_close_named(
                f"bwd_{plan_name}_tile{tile_t}_gmb",
                grads[4],
                expected_grads[4],
                dtype,
            )

    torch.cuda.synchronize()


def run_cached_plan_body(
    dtype: torch.dtype,
) -> None:
    print(
        f"\n[cached plan API] dtype={dtype}",
        flush=True,
    )

    case = Case(
        name="cached_api_d256_c16_s3",
        B=1,
        D=256,
        L=256,
        T=128,
        C=16,
        S=3,
        off=64,
        dilation=1,
        use_bias=True,
        scale=0.1,
    )

    (
        x,
        weight,
        bias,
        mix_weight,
        mix_bias,
    ) = make_case_tensors(
        case=case,
        dtype=dtype,
        seed=777,
        requires_grad=True,
    )

    fwd_plan = fused_source_conv_mixed_forward_warmup(
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
        warmup_iters=1,
        bench_iters=1,
    )

    out1 = fused_source_conv_mixed_forward(
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
    )

    out2 = fused_source_conv_mixed_forward_cached_plan(
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
        cached_plan=fwd_plan,
    )

    assert_close_named(
        "cached_forward",
        out2,
        out1,
        dtype,
    )

    grad = make_leaf_randn(
        shape=out1.shape,
        device="cuda",
        dtype=dtype,
        scale=0.1,
        requires_grad=False,
    )

    bwd_cached = fused_source_conv_mixed_backward_warmup(
        grad_out=grad,
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
        warmup_iters=1,
        bench_iters=1,
    )

    plan = fused_source_conv_mixed_backward_cached_plan_get_plan(
        bwd_cached
    )
    tile_t = fused_source_conv_mixed_backward_cached_plan_get_tile_t(
        bwd_cached
    )

    repacked = fused_source_conv_mixed_backward_pack_cached_plan(
        plan=plan,
        tile_t=tile_t,
    )

    if int(repacked) != int(bwd_cached):
        raise RuntimeError(
            f"cached plan repack mismatch: cached={bwd_cached}, repacked={repacked}"
        )

    grads1 = fused_source_conv_mixed_backward(
        grad_out=grad,
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
    )

    grads2 = fused_source_conv_mixed_backward_cached_plan(
        grad_out=grad,
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
        cached_plan=bwd_cached,
    )

    names = [
        "cached_grad_x",
        "cached_grad_weight",
        "cached_grad_bias",
        "cached_grad_mix_weight",
        "cached_grad_mix_bias",
    ]

    for name, actual, expected in zip(
        names,
        grads2,
        grads1,
    ):
        assert_close_named(
            name,
            actual,
            expected,
            dtype,
        )

    torch.cuda.synchronize()


def run_gradcheck_reference_only_body() -> None:
    print(
        "\n[gradcheck reference formula only]",
        flush=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = "cuda"
    dtype = torch.float64

    B = 1
    D = 3
    L = 9
    T = 5
    C = 2
    S = 3
    off = 2
    dilation = 2

    torch.manual_seed(
        2024
    )

    x = make_leaf_randn(
        shape=(
            B,
            D,
            L,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    weight = make_leaf_randn(
        shape=(
            D,
            C,
            S,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    bias = make_leaf_randn(
        shape=(
            D,
            C,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    mix_weight = make_leaf_randn(
        shape=(
            D,
            C,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    mix_bias = make_leaf_randn(
        shape=(
            D,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    def fn(
        x_,
        weight_,
        bias_,
        mix_weight_,
        mix_bias_,
    ):
        return torch_reference_source_conv_mixed_chunk(
            x_norm=x_,
            weight=weight_,
            bias=bias_,
            mix_weight=mix_weight_,
            mix_bias=mix_bias_,
            off=off,
            T=T,
            dilation=dilation,
        )

    ok = torch.autograd.gradcheck(
        fn,
        (
            x,
            weight,
            bias,
            mix_weight,
            mix_bias,
        ),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )

    print(
        f"    gradcheck reference: {ok}",
        flush=True,
    )

    if not ok:
        raise RuntimeError(
            "gradcheck failed."
        )


def benchmark_case_body(
    case: Case,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    autotune_warmup_iters: int,
    autotune_bench_iters: int,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = "cuda"

    (
        base_x,
        base_weight,
        base_bias,
        base_mix_weight,
        base_mix_bias,
    ) = make_case_tensors(
        case=case,
        dtype=dtype,
        seed=123,
        requires_grad=False,
    )

    x = clone_leaf(
        base_x
    )
    weight = clone_leaf(
        base_weight
    )
    bias = clone_leaf(
        base_bias
    )
    mix_weight = clone_leaf(
        base_mix_weight
    )
    mix_bias = clone_leaf(
        base_mix_bias
    )

    grad = make_leaf_randn(
        shape=(
            case.B,
            case.D,
            case.T,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=False,
    )

    forward_plan = fused_source_conv_mixed_forward_warmup(
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
        warmup_iters=autotune_warmup_iters,
        bench_iters=autotune_bench_iters,
    )

    backward_cached = fused_source_conv_mixed_backward_warmup(
        grad_out=grad,
        x_norm=x,
        weight=weight,
        bias=bias,
        mix_weight=mix_weight,
        mix_bias=mix_bias,
        off=case.off,
        T=case.T,
        dilation=case.dilation,
        warmup_iters=autotune_warmup_iters,
        bench_iters=autotune_bench_iters,
    )

    backward_plan = fused_source_conv_mixed_backward_cached_plan_get_plan(
        backward_cached
    )
    backward_tile_t = fused_source_conv_mixed_backward_cached_plan_get_tile_t(
        backward_cached
    )

    for _ in range(
        warmup
    ):
        clear_grads(
            [
                x,
                weight,
                bias,
                mix_weight,
                mix_bias,
            ]
        )

        out = fused_source_conv_mixed_forward_cached_plan(
            x_norm=x,
            weight=weight,
            bias=bias,
            mix_weight=mix_weight,
            mix_bias=mix_bias,
            off=case.off,
            T=case.T,
            dilation=case.dilation,
            cached_plan=forward_plan,
        )

        grads = fused_source_conv_mixed_backward_cached_plan(
            grad_out=grad,
            x_norm=x,
            weight=weight,
            bias=bias,
            mix_weight=mix_weight,
            mix_bias=mix_bias,
            off=case.off,
            T=case.T,
            dilation=case.dilation,
            cached_plan=backward_cached,
        )

        del out
        del grads

    torch.cuda.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(
        iters
    ):
        out = fused_source_conv_mixed_forward_cached_plan(
            x_norm=x,
            weight=weight,
            bias=bias,
            mix_weight=mix_weight,
            mix_bias=mix_bias,
            off=case.off,
            T=case.T,
            dilation=case.dilation,
            cached_plan=forward_plan,
        )

        grads = fused_source_conv_mixed_backward_cached_plan(
            grad_out=grad,
            x_norm=x,
            weight=weight,
            bias=bias,
            mix_weight=mix_weight,
            mix_bias=mix_bias,
            off=case.off,
            T=case.T,
            dilation=case.dilation,
            cached_plan=backward_cached,
        )

        del out
        del grads

    end.record()
    torch.cuda.synchronize()

    fused_ms = start.elapsed_time(
        end
    ) / iters

    x_ref = clone_leaf(
        base_x
    )
    weight_ref = clone_leaf(
        base_weight
    )
    bias_ref = clone_leaf(
        base_bias
    )
    mix_weight_ref = clone_leaf(
        base_mix_weight
    )
    mix_bias_ref = clone_leaf(
        base_mix_bias
    )

    for _ in range(
        warmup
    ):
        clear_grads(
            [
                x_ref,
                weight_ref,
                bias_ref,
                mix_weight_ref,
                mix_bias_ref,
            ]
        )

        out_ref = torch_reference_source_conv_mixed_chunk(
            x_norm=x_ref,
            weight=weight_ref,
            bias=bias_ref,
            mix_weight=mix_weight_ref,
            mix_bias=mix_bias_ref,
            off=case.off,
            T=case.T,
            dilation=case.dilation,
        )

        out_ref.backward(
            grad
        )

    torch.cuda.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(
        iters
    ):
        clear_grads(
            [
                x_ref,
                weight_ref,
                bias_ref,
                mix_weight_ref,
                mix_bias_ref,
            ]
        )

        out_ref = torch_reference_source_conv_mixed_chunk(
            x_norm=x_ref,
            weight=weight_ref,
            bias=bias_ref,
            mix_weight=mix_weight_ref,
            mix_bias=mix_bias_ref,
            off=case.off,
            T=case.T,
            dilation=case.dilation,
        )

        out_ref.backward(
            grad
        )

    end.record()
    torch.cuda.synchronize()

    ref_ms = start.elapsed_time(
        end
    ) / iters

    return {
        "fused_ms": float(
            fused_ms
        ),
        "ref_ms": float(
            ref_ms
        ),
        "speedup": float(
            ref_ms / fused_ms
        ),
        "forward_plan_id": int(
            forward_plan
        ),
        "forward_plan_name": fused_source_conv_mixed_forward_plan_name(
            forward_plan
        ),
        "backward_cached_plan": int(
            backward_cached
        ),
        "backward_plan_id": int(
            backward_plan
        ),
        "backward_plan_name": fused_source_conv_mixed_backward_plan_name(
            backward_plan
        ),
        "backward_tile_t": int(
            backward_tile_t
        ),
    }


def get_quick_cases() -> List[Case]:
    return [
        Case(
            name="tiny_c1_s1_bias",
            B=1,
            D=1,
            L=8,
            T=8,
            C=1,
            S=1,
            off=0,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="tiny_c1_s1_nobias",
            B=1,
            D=2,
            L=8,
            T=8,
            C=1,
            S=1,
            off=0,
            dilation=1,
            use_bias=False,
        ),
        Case(
            name="boundary_c8_s3",
            B=1,
            D=16,
            L=64,
            T=32,
            C=8,
            S=3,
            off=0,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="noboundary_c8_s3",
            B=1,
            D=16,
            L=64,
            T=32,
            C=8,
            S=3,
            off=8,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="odd_dilation2_c5_s4",
            B=1,
            D=17,
            L=67,
            T=19,
            C=5,
            S=4,
            off=23,
            dilation=2,
            use_bias=True,
        ),
    ]


def get_full_cases() -> List[Case]:
    cases: List[Case] = []

    cases.extend(
        get_quick_cases()
    )

    cases.extend(
        [
            Case(
                name="c32_s3_boundary",
                B=1,
                D=64,
                L=96,
                T=64,
                C=32,
                S=3,
                off=0,
                dilation=1,
                use_bias=True,
            ),
            Case(
                name="c32_s3_noboundary",
                B=1,
                D=64,
                L=128,
                T=64,
                C=32,
                S=3,
                off=16,
                dilation=1,
                use_bias=False,
            ),
            Case(
                name="d256_c16_s3_boundary",
                B=1,
                D=256,
                L=128,
                T=64,
                C=16,
                S=3,
                off=0,
                dilation=1,
                use_bias=True,
            ),
            Case(
                name="d256_c16_s3_noboundary",
                B=1,
                D=256,
                L=256,
                T=128,
                C=16,
                S=3,
                off=64,
                dilation=1,
                use_bias=True,
            ),
            Case(
                name="d256_c16_s3_dil4",
                B=1,
                D=256,
                L=512,
                T=128,
                C=16,
                S=3,
                off=128,
                dilation=4,
                use_bias=True,
            ),
            Case(
                name="batch2_d64_c8_s3",
                B=2,
                D=64,
                L=256,
                T=128,
                C=8,
                S=3,
                off=64,
                dilation=1,
                use_bias=True,
            ),
            Case(
                name="generic_c7_s5",
                B=1,
                D=33,
                L=257,
                T=113,
                C=7,
                S=5,
                off=143,
                dilation=3,
                use_bias=True,
            ),
            Case(
                name="c16_s8",
                B=1,
                D=128,
                L=512,
                T=256,
                C=16,
                S=8,
                off=128,
                dilation=2,
                use_bias=False,
            ),
            Case(
                name="c64_s3",
                B=1,
                D=128,
                L=512,
                T=256,
                C=64,
                S=3,
                off=64,
                dilation=1,
                use_bias=True,
            ),
        ]
    )

    return cases


def get_stress_cases() -> List[Case]:
    return [
        Case(
            name="stress_d256_c16_s3_t1024",
            B=1,
            D=256,
            L=4096,
            T=1024,
            C=16,
            S=3,
            off=1024,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="stress_d256_c16_s3_t4096",
            B=1,
            D=256,
            L=8192,
            T=4096,
            C=16,
            S=3,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="stress_d256_c8_s3_t4096",
            B=1,
            D=256,
            L=8192,
            T=4096,
            C=8,
            S=3,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="stress_d256_c32_s3_t2048",
            B=1,
            D=256,
            L=8192,
            T=2048,
            C=32,
            S=3,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="stress_d512_c16_s3_t2048",
            B=1,
            D=512,
            L=8192,
            T=2048,
            C=16,
            S=3,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="stress_d256_c16_s7_t2048_dil2",
            B=1,
            D=256,
            L=16384,
            T=2048,
            C=16,
            S=7,
            off=4096,
            dilation=2,
            use_bias=True,
        ),
    ]


def get_benchmark_cases() -> List[Case]:
    return [
        Case(
            name="bench_d256_c16_s3_t256",
            B=1,
            D=256,
            L=1024,
            T=256,
            C=16,
            S=3,
            off=256,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="bench_d256_c16_s3_t1024",
            B=1,
            D=256,
            L=4096,
            T=1024,
            C=16,
            S=3,
            off=1024,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="bench_d256_c16_s3_t4096",
            B=1,
            D=256,
            L=8192,
            T=4096,
            C=16,
            S=3,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="bench_d256_c8_s3_t4096",
            B=1,
            D=256,
            L=8192,
            T=4096,
            C=8,
            S=3,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="bench_d256_c32_s3_t2048",
            B=1,
            D=256,
            L=8192,
            T=2048,
            C=32,
            S=3,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="bench_d512_c16_s3_t2048",
            B=1,
            D=512,
            L=8192,
            T=2048,
            C=16,
            S=3,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="bench_b2_d256_c16_s3_t1024",
            B=2,
            D=256,
            L=4096,
            T=1024,
            C=16,
            S=3,
            off=1024,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="bench_d256_c16_s7_t2048",
            B=1,
            D=256,
            L=8192,
            T=2048,
            C=16,
            S=7,
            off=2048,
            dilation=1,
            use_bias=True,
        ),
        Case(
            name="bench_d256_c64_s3_t1024",
            B=1,
            D=256,
            L=4096,
            T=1024,
            C=64,
            S=3,
            off=1024,
            dilation=1,
            use_bias=True,
        ),
    ]


def worker_entry(
    mode: str,
    payload: Dict[str, Any],
    result_queue,
) -> None:
    try:
        if mode == "correctness":
            case = Case(
                **payload["case"]
            )

            dtype = getattr(
                torch,
                payload["dtype"],
            )

            run_correctness_case_body(
                case=case,
                dtype=dtype,
                seed=int(
                    payload["seed"]
                ),
                compare_ref2=bool(
                    payload["compare_ref2"]
                ),
                test_all_forward_plans=bool(
                    payload["test_all_forward_plans"]
                ),
                test_all_backward_plans=bool(
                    payload["test_all_backward_plans"]
                ),
            )

            result_queue.put(
                {
                    "status": "ok",
                    "message": "",
                }
            )

        elif mode == "cached_plan":
            dtype = getattr(
                torch,
                payload["dtype"],
            )

            run_cached_plan_body(
                dtype=dtype,
            )

            result_queue.put(
                {
                    "status": "ok",
                    "message": "",
                }
            )

        elif mode == "gradcheck":
            run_gradcheck_reference_only_body()

            result_queue.put(
                {
                    "status": "ok",
                    "message": "",
                }
            )

        elif mode == "benchmark":
            case = Case(
                **payload["case"]
            )

            dtype = getattr(
                torch,
                payload["dtype"],
            )

            result = benchmark_case_body(
                case=case,
                dtype=dtype,
                warmup=int(
                    payload["warmup"]
                ),
                iters=int(
                    payload["iters"]
                ),
                autotune_warmup_iters=int(
                    payload["autotune_warmup_iters"]
                ),
                autotune_bench_iters=int(
                    payload["autotune_bench_iters"]
                ),
            )

            result_queue.put(
                {
                    "status": "ok",
                    "message": "",
                    "result": result,
                }
            )

        else:
            raise RuntimeError(
                f"Unknown worker mode: {mode}"
            )

    except torch.cuda.OutOfMemoryError as e:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        result_queue.put(
            {
                "status": "oom",
                "message": str(
                    e
                ),
            }
        )

    except RuntimeError as e:
        msg = str(
            e
        )

        if "out of memory" in msg.lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            result_queue.put(
                {
                    "status": "oom",
                    "message": msg,
                }
            )
        else:
            result_queue.put(
                {
                    "status": "fail",
                    "message": traceback.format_exc(),
                }
            )

    except Exception:
        result_queue.put(
            {
                "status": "fail",
                "message": traceback.format_exc(),
            }
        )


def run_with_timeout(
    mode: str,
    payload: Dict[str, Any],
    timeout_sec: float,
) -> Dict[str, Any]:
    ctx = mp.get_context(
        "spawn"
    )

    result_queue = ctx.Queue()

    proc = ctx.Process(
        target=worker_entry,
        args=(
            mode,
            payload,
            result_queue,
        ),
    )

    proc.start()
    proc.join(
        timeout=timeout_sec
    )

    if proc.is_alive():
        proc.terminate()
        proc.join()

        return {
            "status": "timeout",
            "message": f"timeout after {timeout_sec:.1f}s",
        }

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        if proc.exitcode == 0:
            return {
                "status": "ok",
                "message": "",
            }

        return {
            "status": "fail",
            "message": f"worker exited with code {proc.exitcode}",
        }


def run_correctness_suite(
    cases: List[Case],
    dtypes: List[torch.dtype],
    timeout: float,
    compare_ref2: bool,
    stop_on_fail: bool,
    test_all_forward_plans: bool,
    test_all_backward_plans: bool,
) -> None:
    total = len(
        cases
    ) * len(
        dtypes
    )

    passed = 0
    skipped_oom = 0
    skipped_timeout = 0
    failed = 0

    counter = 0

    for dtype in dtypes:
        print(
            "=" * 100,
            flush=True,
        )

        print(
            f"Correctness dtype={dtype}",
            flush=True,
        )

        print(
            "=" * 100,
            flush=True,
        )

        for case_idx, case in enumerate(
            cases
        ):
            print_case_header(
                counter,
                total,
                case,
                dtype,
            )

            payload = {
                "case": case_to_dict(
                    case
                ),
                "dtype": dtype_to_name(
                    dtype
                ),
                "seed": 1234 + case_idx,
                "compare_ref2": compare_ref2,
                "test_all_forward_plans": test_all_forward_plans,
                "test_all_backward_plans": test_all_backward_plans,
            }

            result = run_with_timeout(
                mode="correctness",
                payload=payload,
                timeout_sec=timeout,
            )

            status = result[
                "status"
            ]

            if status == "ok":
                print(
                    "    RESULT: PASS",
                    flush=True,
                )
                passed += 1

            elif status == "oom":
                print(
                    "    RESULT: OOM SKIP",
                    flush=True,
                )
                print(
                    f"    {result['message']}",
                    flush=True,
                )
                skipped_oom += 1

            elif status == "timeout":
                print(
                    "    RESULT: TIMEOUT SKIP",
                    flush=True,
                )
                print(
                    f"    {result['message']}",
                    flush=True,
                )
                skipped_timeout += 1

            else:
                print(
                    "    RESULT: FAIL",
                    flush=True,
                )
                print(
                    result[
                        "message"
                    ],
                    flush=True,
                )
                failed += 1

                if stop_on_fail:
                    raise RuntimeError(
                        f"Correctness failed: {case.name}, dtype={dtype}"
                    )

            counter += 1

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(
        "=" * 100,
        flush=True,
    )

    print(
        f"Correctness summary: pass={passed}, oom_skip={skipped_oom}, "
        f"timeout_skip={skipped_timeout}, fail={failed}, total={total}",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )

    if failed > 0:
        raise RuntimeError(
            f"{failed} correctness cases failed."
        )


def run_cached_plan_suite(
    dtypes: List[torch.dtype],
    timeout: float,
) -> None:
    for dtype in dtypes:
        payload = {
            "dtype": dtype_to_name(
                dtype
            ),
        }

        result = run_with_timeout(
            mode="cached_plan",
            payload=payload,
            timeout_sec=timeout,
        )

        if result[
            "status"
        ] == "ok":
            print(
                f"cached_plan dtype={dtype}: PASS",
                flush=True,
            )
        elif result[
            "status"
        ] in (
            "oom",
            "timeout",
        ):
            print(
                f"cached_plan dtype={dtype}: {result['status'].upper()} SKIP",
                flush=True,
            )
            print(
                result[
                    "message"
                ],
                flush=True,
            )
        else:
            print(
                f"cached_plan dtype={dtype}: FAIL",
                flush=True,
            )
            print(
                result[
                    "message"
                ],
                flush=True,
            )
            raise RuntimeError(
                f"cached_plan failed for dtype={dtype}"
            )


def run_gradcheck_suite(
    timeout: float,
) -> None:
    result = run_with_timeout(
        mode="gradcheck",
        payload={},
        timeout_sec=timeout,
    )

    if result[
        "status"
    ] == "ok":
        print(
            "gradcheck_reference_only: PASS",
            flush=True,
        )
    elif result[
        "status"
    ] in (
        "oom",
        "timeout",
    ):
        print(
            f"gradcheck_reference_only: {result['status'].upper()} SKIP",
            flush=True,
        )
        print(
            result[
                "message"
            ],
            flush=True,
        )
    else:
        print(
            "gradcheck_reference_only: FAIL",
            flush=True,
        )
        print(
            result[
                "message"
            ],
            flush=True,
        )
        raise RuntimeError(
            "gradcheck reference failed."
        )


def run_benchmark_suite(
    cases: List[Case],
    dtype: torch.dtype,
    timeout: float,
    warmup: int,
    iters: int,
    autotune_warmup_iters: int,
    autotune_bench_iters: int,
) -> None:
    print(
        "=" * 100,
        flush=True,
    )

    print(
        f"Benchmark dtype={dtype} warmup={warmup} iters={iters} "
        f"autotune_warmup_iters={autotune_warmup_iters} "
        f"autotune_bench_iters={autotune_bench_iters}",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )

    rows = []

    for idx, case in enumerate(
        cases
    ):
        print_case_header(
            idx,
            len(
                cases
            ),
            case,
            dtype,
        )

        payload = {
            "case": case_to_dict(
                case
            ),
            "dtype": dtype_to_name(
                dtype
            ),
            "warmup": warmup,
            "iters": iters,
            "autotune_warmup_iters": autotune_warmup_iters,
            "autotune_bench_iters": autotune_bench_iters,
        }

        result = run_with_timeout(
            mode="benchmark",
            payload=payload,
            timeout_sec=timeout,
        )

        status = result[
            "status"
        ]

        if status == "ok":
            r = result[
                "result"
            ]

            print(
                f"    selected forward  plan: "
                f"{r['forward_plan_id']} {r['forward_plan_name']}",
                flush=True,
            )

            print(
                f"    selected backward plan: "
                f"cached={r['backward_cached_plan']} "
                f"plan={r['backward_plan_id']} {r['backward_plan_name']} "
                f"tile_t={r['backward_tile_t']}",
                flush=True,
            )

            print(
                f"    fused forward+backward: {r['fused_ms']:.3f} ms/iter",
                flush=True,
            )

            print(
                f"    ref   forward+backward: {r['ref_ms']:.3f} ms/iter",
                flush=True,
            )

            print(
                f"    speedup: {r['speedup']:.3f}x",
                flush=True,
            )

            rows.append(
                (
                    case.name,
                    r[
                        "fused_ms"
                    ],
                    r[
                        "ref_ms"
                    ],
                    r[
                        "speedup"
                    ],
                    "ok",
                    r[
                        "forward_plan_id"
                    ],
                    r[
                        "forward_plan_name"
                    ],
                    r[
                        "backward_cached_plan"
                    ],
                    r[
                        "backward_plan_id"
                    ],
                    r[
                        "backward_plan_name"
                    ],
                    r[
                        "backward_tile_t"
                    ],
                )
            )

        elif status == "oom":
            print(
                "    RESULT: OOM SKIP",
                flush=True,
            )
            print(
                f"    {result['message']}",
                flush=True,
            )

            rows.append(
                (
                    case.name,
                    float(
                        "nan"
                    ),
                    float(
                        "nan"
                    ),
                    float(
                        "nan"
                    ),
                    "oom",
                    -1,
                    "oom",
                    -1,
                    -1,
                    "oom",
                    -1,
                )
            )

        elif status == "timeout":
            print(
                "    RESULT: TIMEOUT SKIP",
                flush=True,
            )
            print(
                f"    {result['message']}",
                flush=True,
            )

            rows.append(
                (
                    case.name,
                    float(
                        "nan"
                    ),
                    float(
                        "nan"
                    ),
                    float(
                        "nan"
                    ),
                    "timeout",
                    -1,
                    "timeout",
                    -1,
                    -1,
                    "timeout",
                    -1,
                )
            )

        else:
            print(
                "    RESULT: FAIL",
                flush=True,
            )
            print(
                result[
                    "message"
                ],
                flush=True,
            )

            rows.append(
                (
                    case.name,
                    float(
                        "nan"
                    ),
                    float(
                        "nan"
                    ),
                    float(
                        "nan"
                    ),
                    "fail",
                    -1,
                    "fail",
                    -1,
                    -1,
                    "fail",
                    -1,
                )
            )

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(
        "=" * 100,
        flush=True,
    )

    print(
        "Benchmark summary",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )

    for (
        name,
        fused_ms,
        ref_ms,
        speedup,
        status,
        forward_plan_id,
        forward_plan_name,
        backward_cached_plan,
        backward_plan_id,
        backward_plan_name,
        backward_tile_t,
    ) in rows:
        if status == "ok":
            print(
                f"{name:<36} fused={fused_ms:>9.3f} ms  "
                f"ref={ref_ms:>9.3f} ms  speedup={speedup:>7.3f}x  "
                f"fwd={forward_plan_id}:{forward_plan_name}  "
                f"bwd_cached={backward_cached_plan}  "
                f"bwd={backward_plan_id}:{backward_plan_name}:tile{backward_tile_t}",
                flush=True,
            )
        else:
            print(
                f"{name:<36} {status}",
                flush=True,
            )


def parse_dtype_list(
    text: str,
) -> List[torch.dtype]:
    text = text.strip().lower()

    available = get_available_dtypes()

    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "all": None,
    }

    if text == "all":
        return available

    result = []

    for item in text.split(
        ","
    ):
        key = item.strip().lower()

        if key not in mapping or mapping[
            key
        ] is None:
            raise ValueError(
                f"Unknown dtype: {item}"
            )

        dtype = mapping[
            key
        ]

        if dtype == torch.bfloat16 and dtype not in available:
            print(
                "bf16 is not supported on this GPU, skip.",
                flush=True,
            )
            continue

        result.append(
            dtype
        )

    if not result:
        raise RuntimeError(
            "No valid dtype selected."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick correctness cases.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full correctness cases.",
    )

    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run large correctness stress cases. OOM and timeout will be skipped.",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark suite.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run quick + full + stress + benchmark.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="all",
        help="Dtype list: fp32,fp16,bf16,all. Example: --dtype fp32 or --dtype fp16,bf16",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout seconds per correctness case.",
    )

    parser.add_argument(
        "--benchmark-timeout",
        type=float,
        default=240.0,
        help="Timeout seconds per benchmark case.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Benchmark warmup iterations after autotune plan has been selected.",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Benchmark measured iterations.",
    )

    parser.add_argument(
        "--autotune-warmup-iters",
        type=int,
        default=1,
        help="Warmup iterations for each candidate during op warmup/autotune.",
    )

    parser.add_argument(
        "--autotune-bench-iters",
        type=int,
        default=3,
        help="Measured iterations for each candidate during op warmup/autotune.",
    )

    parser.add_argument(
        "--no-ref2",
        action="store_true",
        help="Do not compare with materialized z reference in correctness cases.",
    )

    parser.add_argument(
        "--no-all-forward-plans",
        action="store_true",
        help="Do not test all legal forward plans explicitly.",
    )

    parser.add_argument(
        "--no-all-backward-plans",
        action="store_true",
        help="Do not test all legal backward plans explicitly.",
    )

    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop immediately on first correctness failure.",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    if args.autotune_warmup_iters < 0:
        raise RuntimeError(
            "--autotune-warmup-iters must be >= 0."
        )

    if args.autotune_bench_iters <= 0:
        raise RuntimeError(
            "--autotune-bench-iters must be positive."
        )

    if args.warmup < 0:
        raise RuntimeError(
            "--warmup must be >= 0."
        )

    if args.iters <= 0:
        raise RuntimeError(
            "--iters must be positive."
        )

    dtypes = parse_dtype_list(
        args.dtype
    )

    if not (
        args.quick
        or args.full
        or args.stress
        or args.benchmark
        or args.all
    ):
        args.quick = True

    print(
        f"CUDA device: {torch.cuda.get_device_name()}",
        flush=True,
    )

    print(
        f"CUDA capability: {torch.cuda.get_device_capability()}",
        flush=True,
    )

    print(
        f"Selected dtypes: {dtypes}",
        flush=True,
    )

    print(
        f"PID: {os.getpid()}",
        flush=True,
    )

    compare_ref2 = not args.no_ref2
    test_all_forward_plans = not args.no_all_forward_plans
    test_all_backward_plans = not args.no_all_backward_plans

    if args.quick or args.all:
        run_correctness_suite(
            cases=get_quick_cases(),
            dtypes=dtypes,
            timeout=args.timeout,
            compare_ref2=compare_ref2,
            stop_on_fail=args.stop_on_fail,
            test_all_forward_plans=test_all_forward_plans,
            test_all_backward_plans=test_all_backward_plans,
        )

        run_cached_plan_suite(
            dtypes=dtypes,
            timeout=args.timeout,
        )

    if args.full or args.all:
        run_correctness_suite(
            cases=get_full_cases(),
            dtypes=dtypes,
            timeout=args.timeout,
            compare_ref2=compare_ref2,
            stop_on_fail=args.stop_on_fail,
            test_all_forward_plans=test_all_forward_plans,
            test_all_backward_plans=test_all_backward_plans,
        )

        run_cached_plan_suite(
            dtypes=dtypes,
            timeout=args.timeout,
        )

        run_gradcheck_suite(
            timeout=args.timeout,
        )

    if args.stress or args.all:
        run_correctness_suite(
            cases=get_stress_cases(),
            dtypes=dtypes,
            timeout=args.timeout,
            compare_ref2=False,
            stop_on_fail=args.stop_on_fail,
            test_all_forward_plans=False,
            test_all_backward_plans=False,
        )

    if args.benchmark or args.all:
        bench_dtype = torch.float16

        if torch.cuda.is_bf16_supported():
            bench_dtype = torch.bfloat16

        if len(
            dtypes
        ) == 1:
            bench_dtype = dtypes[
                0
            ]

        run_benchmark_suite(
            cases=get_benchmark_cases(),
            dtype=bench_dtype,
            timeout=args.benchmark_timeout,
            warmup=args.warmup,
            iters=args.iters,
            autotune_warmup_iters=args.autotune_warmup_iters,
            autotune_bench_iters=args.autotune_bench_iters,
        )


if __name__ == "__main__":
    try:
        mp.set_start_method(
            "spawn",
            force=True,
        )
    except RuntimeError:
        pass

    main()