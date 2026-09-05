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

from cuda_ops import (
    fused_dynamic_conv_chunk,
    fused_dynamic_conv,
    fused_dynamic_conv_forward_warmup_chunk,
    fused_dynamic_conv_forward_cached_plan_chunk,
    fused_dynamic_conv_forward_clear_warmup_cache,
    fused_dynamic_conv_forward_plan_name,
    fused_dynamic_conv_backward_warmup_chunk,
    fused_dynamic_conv_backward_cached_plan_chunk,
    fused_dynamic_conv_backward_clear_warmup_cache,
    fused_dynamic_conv_backward_plan_name,
)


@dataclass
class Case:
    name: str
    B: int
    D: int
    L: int
    T: int
    N: int
    K: int
    t_offset: int
    dilation: int
    scale: float = 0.1


def set_seed(seed: int = 1234):
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
):
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
    x: torch.Tensor,
    requires_grad: bool = True,
) -> torch.Tensor:
    y = x.detach().clone()

    if requires_grad:
        y.requires_grad_(True)

    return y


def clear_grads(
    tensors,
):
    for x in tensors:
        if x is not None and x.grad is not None:
            x.grad = None


def torch_reference_materialized_chunk(
    h_full: torch.Tensor,
    kernel_chunk: torch.Tensor,
    kernel_mix: torch.Tensor,
    t_offset: int,
    dilation: int = 1,
) -> torch.Tensor:
    B, D, L = h_full.shape
    Bk, T, N, K = kernel_chunk.shape
    Dm, Nm = kernel_mix.shape

    assert B == Bk
    assert D == Dm
    assert N == Nm
    assert t_offset >= 0
    assert dilation > 0
    assert t_offset + T <= L

    mixed_kernel = torch.einsum(
        "btnk,dn->bdtk",
        kernel_chunk,
        kernel_mix,
    )

    out = torch.zeros(
        B,
        D,
        T,
        device=h_full.device,
        dtype=h_full.dtype,
    )

    for kk in range(K):
        src_start_global = t_offset - kk * dilation
        src_end_global = t_offset + T - kk * dilation

        dst_start = 0
        dst_end = T

        if src_start_global < 0:
            dst_start = -src_start_global
            src_start_global = 0

        if src_end_global > L:
            overflow = src_end_global - L
            dst_end = T - overflow
            src_end_global = L

        if dst_start >= dst_end:
            continue

        h_slice = h_full[
            :,
            :,
            src_start_global:src_end_global,
        ]

        out[
            :,
            :,
            dst_start:dst_end,
        ] = out[
            :,
            :,
            dst_start:dst_end,
        ] + mixed_kernel[
            :,
            :,
            dst_start:dst_end,
            kk,
        ] * h_slice

    return out


def torch_reference_no_materialize_chunk(
    h_full: torch.Tensor,
    kernel_chunk: torch.Tensor,
    kernel_mix: torch.Tensor,
    t_offset: int,
    dilation: int = 1,
) -> torch.Tensor:
    B, D, L = h_full.shape
    Bk, T, N, K = kernel_chunk.shape
    Dm, Nm = kernel_mix.shape

    assert B == Bk
    assert D == Dm
    assert N == Nm
    assert t_offset >= 0
    assert dilation > 0
    assert t_offset + T <= L

    out = torch.zeros(
        B,
        D,
        T,
        device=h_full.device,
        dtype=h_full.dtype,
    )

    for kk in range(K):
        src_start_global = t_offset - kk * dilation
        src_end_global = t_offset + T - kk * dilation

        dst_start = 0
        dst_end = T

        if src_start_global < 0:
            dst_start = -src_start_global
            src_start_global = 0

        if src_end_global > L:
            overflow = src_end_global - L
            dst_end = T - overflow
            src_end_global = L

        if dst_start >= dst_end:
            continue

        weight = torch.einsum(
            "btn,dn->bdt",
            kernel_chunk[
                :,
                dst_start:dst_end,
                :,
                kk,
            ],
            kernel_mix,
        )

        h_slice = h_full[
            :,
            :,
            src_start_global:src_end_global,
        ]

        out[
            :,
            :,
            dst_start:dst_end,
        ] = out[
            :,
            :,
            dst_start:dst_end,
        ] + weight * h_slice

    return out


def get_tolerances(dtype: torch.dtype):
    if dtype == torch.float32:
        return 4e-4, 4e-4

    if dtype == torch.float16:
        return 1.5e-1, 1.5e-1

    if dtype == torch.bfloat16:
        return 2.0e-1, 2.0e-1

    raise ValueError(dtype)


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
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    print_detail: bool = True,
):
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
            f"    {name:<30} abs={abs_diff:.8e} rel={rel_diff:.8e} "
            f"atol={atol} rtol={rtol}",
            flush=True,
        )

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
    )


def run_correctness_case_body(
    case: Case,
    dtype: torch.dtype,
    seed: int,
    compare_ref2: bool = True,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = "cuda"

    set_seed(seed)

    h = make_leaf_randn(
        shape=(
            case.B,
            case.D,
            case.L,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
    )

    kernel = make_leaf_randn(
        shape=(
            case.B,
            case.T,
            case.N,
            case.K,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
    )

    kernel_mix = make_leaf_randn(
        shape=(
            case.D,
            case.N,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
    )

    h_ref = clone_leaf(h)
    kernel_ref = clone_leaf(kernel)
    kernel_mix_ref = clone_leaf(kernel_mix)

    forward_plan_id = fused_dynamic_conv_forward_warmup_chunk(
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=case.t_offset,
        dilation=case.dilation,
        repeat=1,
    )

    cached_forward_plan_id = fused_dynamic_conv_forward_cached_plan_chunk(
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=case.t_offset,
        dilation=case.dilation,
    )

    if cached_forward_plan_id != forward_plan_id:
        raise RuntimeError(
            f"forward cached plan mismatch: selected={forward_plan_id}, "
            f"cached={cached_forward_plan_id}"
        )

    print(
        f"    forward warmup selected plan: "
        f"{forward_plan_id} {fused_dynamic_conv_forward_plan_name(forward_plan_id)}",
        flush=True,
    )

    out = fused_dynamic_conv_chunk(
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=case.t_offset,
        dilation=case.dilation,
    )

    out_ref = torch_reference_materialized_chunk(
        h_full=h_ref,
        kernel_chunk=kernel_ref,
        kernel_mix=kernel_mix_ref,
        t_offset=case.t_offset,
        dilation=case.dilation,
    )

    assert_close_named(
        "forward_vs_materialized",
        out,
        out_ref,
        dtype,
    )

    if compare_ref2:
        h_ref2 = clone_leaf(h)
        kernel_ref2 = clone_leaf(kernel)
        kernel_mix_ref2 = clone_leaf(kernel_mix)

        out_ref2 = torch_reference_no_materialize_chunk(
            h_full=h_ref2,
            kernel_chunk=kernel_ref2,
            kernel_mix=kernel_mix_ref2,
            t_offset=case.t_offset,
            dilation=case.dilation,
        )

        assert_close_named(
            "forward_ref_vs_ref2",
            out_ref2,
            out_ref,
            dtype,
        )
    else:
        h_ref2 = None
        kernel_ref2 = None
        kernel_mix_ref2 = None
        out_ref2 = None

    grad = make_leaf_randn(
        shape=out.shape,
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=False,
    )

    backward_plan_id = fused_dynamic_conv_backward_warmup_chunk(
        grad_out=grad,
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=case.t_offset,
        dilation=case.dilation,
        repeat=1,
    )

    cached_backward_plan_id = fused_dynamic_conv_backward_cached_plan_chunk(
        h_full=h,
        kernel_chunk=kernel,
        t_offset=case.t_offset,
        dilation=case.dilation,
    )

    if cached_backward_plan_id != backward_plan_id:
        raise RuntimeError(
            f"backward cached plan mismatch: selected={backward_plan_id}, "
            f"cached={cached_backward_plan_id}"
        )

    print(
        f"    backward warmup selected plan: "
        f"{backward_plan_id} {fused_dynamic_conv_backward_plan_name(backward_plan_id)}",
        flush=True,
    )

    out.backward(
        grad
    )

    out_ref.backward(
        grad
    )

    assert_close_named(
        "grad_h",
        h.grad,
        h_ref.grad,
        dtype,
    )

    assert_close_named(
        "grad_kernel",
        kernel.grad,
        kernel_ref.grad,
        dtype,
    )

    assert_close_named(
        "grad_kernel_mix",
        kernel_mix.grad,
        kernel_mix_ref.grad,
        dtype,
    )

    if compare_ref2:
        out_ref2.backward(
            grad
        )

        assert_close_named(
            "grad_h_ref2",
            h_ref2.grad,
            h_ref.grad,
            dtype,
        )

        assert_close_named(
            "grad_kernel_ref2",
            kernel_ref2.grad,
            kernel_ref.grad,
            dtype,
        )

        assert_close_named(
            "grad_mix_ref2",
            kernel_mix_ref2.grad,
            kernel_mix_ref.grad,
            dtype,
        )

    torch.cuda.synchronize()

def run_full_length_equivalence_body(
    dtype: torch.dtype,
):
    print(
        f"\n[full length equivalence] dtype={dtype}",
        flush=True,
    )

    set_seed(
        777
    )

    B = 2
    D = 17
    L = 65
    N = 5
    K = 4
    dilation = 2

    device = "cuda"

    h = make_leaf_randn(
        shape=(
            B,
            D,
            L,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    kernel = make_leaf_randn(
        shape=(
            B,
            L,
            N,
            K,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    kernel_mix = make_leaf_randn(
        shape=(
            D,
            N,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    h1 = clone_leaf(h)
    k1 = clone_leaf(kernel)
    m1 = clone_leaf(kernel_mix)

    h2 = clone_leaf(h)
    k2 = clone_leaf(kernel)
    m2 = clone_leaf(kernel_mix)

    forward_plan_id = fused_dynamic_conv_forward_warmup_chunk(
        h_full=h1,
        kernel_chunk=k1,
        kernel_mix=m1,
        t_offset=0,
        dilation=dilation,
        repeat=1,
    )

    cached_forward_plan_id = fused_dynamic_conv_forward_cached_plan_chunk(
        h_full=h1,
        kernel_chunk=k1,
        kernel_mix=m1,
        t_offset=0,
        dilation=dilation,
    )

    if cached_forward_plan_id != forward_plan_id:
        raise RuntimeError(
            f"forward cached plan mismatch: selected={forward_plan_id}, "
            f"cached={cached_forward_plan_id}"
        )

    print(
        f"    forward warmup selected plan: "
        f"{forward_plan_id} {fused_dynamic_conv_forward_plan_name(forward_plan_id)}",
        flush=True,
    )

    grad_probe = make_leaf_randn(
        shape=(
            B,
            D,
            L,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
        requires_grad=False,
    )

    backward_plan_id = fused_dynamic_conv_backward_warmup_chunk(
        grad_out=grad_probe,
        h_full=h1,
        kernel_chunk=k1,
        kernel_mix=m1,
        t_offset=0,
        dilation=dilation,
        repeat=1,
    )

    cached_backward_plan_id = fused_dynamic_conv_backward_cached_plan_chunk(
        h_full=h1,
        kernel_chunk=k1,
        t_offset=0,
        dilation=dilation,
    )

    if cached_backward_plan_id != backward_plan_id:
        raise RuntimeError(
            f"backward cached plan mismatch: selected={backward_plan_id}, "
            f"cached={cached_backward_plan_id}"
        )

    print(
        f"    backward warmup selected plan: "
        f"{backward_plan_id} {fused_dynamic_conv_backward_plan_name(backward_plan_id)}",
        flush=True,
    )

    out1 = fused_dynamic_conv(
        h1,
        k1,
        m1,
        dilation=dilation,
    )

    out2 = fused_dynamic_conv_chunk(
        h2,
        k2,
        m2,
        t_offset=0,
        dilation=dilation,
    )

    assert_close_named(
        "full_forward",
        out1,
        out2,
        dtype,
    )

    grad = make_leaf_randn(
        shape=out1.shape,
        device=device,
        dtype=dtype,
        scale=0.1,
        requires_grad=False,
    )

    out1.backward(
        grad
    )

    out2.backward(
        grad
    )

    assert_close_named(
        "full_grad_h",
        h1.grad,
        h2.grad,
        dtype,
    )

    assert_close_named(
        "full_grad_kernel",
        k1.grad,
        k2.grad,
        dtype,
    )

    assert_close_named(
        "full_grad_mix",
        m1.grad,
        m2.grad,
        dtype,
    )

    torch.cuda.synchronize()

def run_gradcheck_reference_only_body():
    print(
        "\n[gradcheck reference formula only]",
        flush=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = "cuda"
    dtype = torch.float64

    B = 1
    D = 3
    L = 7
    T = 5
    N = 2
    K = 3
    t_offset = 1
    dilation = 2

    torch.manual_seed(
        2024
    )

    h = make_leaf_randn(
        shape=(
            B,
            D,
            L,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    kernel = make_leaf_randn(
        shape=(
            B,
            T,
            N,
            K,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    kernel_mix = make_leaf_randn(
        shape=(
            D,
            N,
        ),
        device=device,
        dtype=dtype,
        scale=0.1,
    )

    def fn(
        h_,
        kernel_,
        kernel_mix_,
    ):
        return torch_reference_materialized_chunk(
            h_,
            kernel_,
            kernel_mix_,
            t_offset=t_offset,
            dilation=dilation,
        )

    ok = torch.autograd.gradcheck(
        fn,
        (
            h,
            kernel,
            kernel_mix,
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
        raise RuntimeError("gradcheck failed.")


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
    N = case.N
    K = case.K

    h = B * D * L
    out = B * D * T
    kernel = B * T * N * K
    mix = D * N
    grad = out

    base_min = h + out + kernel + mix + grad

    if include_ref:
        mixed_kernel = B * D * T * K
        base_min += mixed_kernel

    return int(base_min * elem)


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


def get_quick_cases() -> List[Case]:
    return [
        Case(
            name="tiny_k1",
            B=1,
            D=1,
            L=8,
            T=8,
            N=1,
            K=1,
            t_offset=0,
            dilation=1,
        ),
        Case(
            name="small_offset0",
            B=1,
            D=16,
            L=64,
            T=32,
            N=6,
            K=3,
            t_offset=0,
            dilation=1,
        ),
        Case(
            name="odd_dilation2",
            B=1,
            D=17,
            L=67,
            T=19,
            N=5,
            K=4,
            t_offset=23,
            dilation=2,
        ),
    ]


def get_full_cases() -> List[Case]:
    cases = []

    cases.extend(
        get_quick_cases()
    )

    cases.extend(
        [
            Case(
                name="B2_D17_mid_dil2",
                B=2,
                D=17,
                L=67,
                T=19,
                N=5,
                K=4,
                t_offset=23,
                dilation=2,
            ),
            Case(
                name="B2_D31_dil4",
                B=2,
                D=31,
                L=129,
                T=64,
                N=6,
                K=3,
                t_offset=65,
                dilation=4,
            ),
            Case(
                name="D33_K5_dil8",
                B=1,
                D=33,
                L=256,
                T=113,
                N=7,
                K=5,
                t_offset=143,
                dilation=8,
            ),
            Case(
                name="full_len_dil16",
                B=1,
                D=64,
                L=257,
                T=257,
                N=6,
                K=3,
                t_offset=0,
                dilation=16,
            ),
            Case(
                name="near_tail",
                B=2,
                D=64,
                L=511,
                T=127,
                N=8,
                K=7,
                t_offset=384,
                dilation=3,
            ),
            Case(
                name="medium_D256",
                B=1,
                D=256,
                L=1024,
                T=513,
                N=6,
                K=3,
                t_offset=511,
                dilation=1,
            ),
            Case(
                name="N1_K3",
                B=1,
                D=128,
                L=512,
                T=256,
                N=1,
                K=3,
                t_offset=128,
                dilation=1,
            ),
            Case(
                name="N2_K15_dil2",
                B=1,
                D=64,
                L=1024,
                T=256,
                N=2,
                K=15,
                t_offset=512,
                dilation=2,
            ),
            Case(
                name="N16_K3",
                B=1,
                D=128,
                L=1024,
                T=512,
                N=16,
                K=3,
                t_offset=256,
                dilation=1,
            ),
            Case(
                name="N32_K2",
                B=1,
                D=64,
                L=1024,
                T=512,
                N=32,
                K=2,
                t_offset=256,
                dilation=1,
            ),
        ]
    )

    return cases


def get_stress_cases() -> List[Case]:
    return [
        Case(
            name="bench_small_T4096",
            B=1,
            D=256,
            L=4096,
            T=4096,
            N=6,
            K=3,
            t_offset=0,
            dilation=1,
        ),
        Case(
            name="bench_mid_T16384",
            B=1,
            D=256,
            L=65536,
            T=16384,
            N=6,
            K=3,
            t_offset=32768,
            dilation=1,
        ),
        Case(
            name="bench_big_T65536",
            B=1,
            D=256,
            L=262144,
            T=65536,
            N=6,
            K=3,
            t_offset=131072,
            dilation=1,
        ),
        Case(
            name="D512_T8192",
            B=1,
            D=512,
            L=32768,
            T=8192,
            N=6,
            K=3,
            t_offset=16384,
            dilation=1,
        ),
        Case(
            name="D1024_T4096",
            B=1,
            D=1024,
            L=16384,
            T=4096,
            N=6,
            K=3,
            t_offset=8192,
            dilation=1,
        ),
        Case(
            name="N16_T8192",
            B=1,
            D=256,
            L=32768,
            T=8192,
            N=16,
            K=3,
            t_offset=16384,
            dilation=1,
        ),
        Case(
            name="N32_T4096",
            B=1,
            D=256,
            L=16384,
            T=4096,
            N=32,
            K=3,
            t_offset=8192,
            dilation=1,
        ),
        Case(
            name="K15_T4096",
            B=1,
            D=256,
            L=32768,
            T=4096,
            N=6,
            K=15,
            t_offset=16384,
            dilation=2,
        ),
    ]


def worker_entry(
    mode: str,
    payload: Dict[str, Any],
    result_queue,
):
    try:
        if mode == "correctness":
            case_dict = payload["case"]
            dtype_name = payload["dtype"]
            seed = payload["seed"]
            compare_ref2 = payload["compare_ref2"]

            dtype = getattr(
                torch,
                dtype_name,
            )

            case = Case(
                **case_dict
            )

            run_correctness_case_body(
                case=case,
                dtype=dtype,
                seed=seed,
                compare_ref2=compare_ref2,
            )

            result_queue.put(
                {
                    "status": "ok",
                    "message": "",
                }
            )

        elif mode == "full_length":
            dtype_name = payload["dtype"]

            dtype = getattr(
                torch,
                dtype_name,
            )

            run_full_length_equivalence_body(
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

            warmup = int(
                payload["warmup"]
            )

            iters = int(
                payload["iters"]
            )

            autotune_repeat = int(
                payload["autotune_repeat"]
            )

            result = benchmark_case_body(
                case=case,
                dtype=dtype,
                warmup=warmup,
                iters=iters,
                autotune_repeat=autotune_repeat,
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
                "message": str(e),
            }
        )

    except RuntimeError as e:
        msg = str(e)

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
        "N": case.N,
        "K": case.K,
        "t_offset": case.t_offset,
        "dilation": case.dilation,
        "scale": case.scale,
    }


def print_case_header(
    idx: int,
    total: int,
    case: Case,
    dtype: torch.dtype,
):
    est_mb = estimate_case_bytes(
        case,
        dtype,
        include_ref=True,
    ) / 1024.0 / 1024.0

    print(
        "-" * 100,
        flush=True,
    )

    print(
        f"[case {idx + 1}/{total}] {case.name} "
        f"dtype={dtype} "
        f"B={case.B} D={case.D} L={case.L} T={case.T} "
        f"N={case.N} K={case.K} "
        f"offset={case.t_offset} dilation={case.dilation} "
        f"estimated_min={est_mb:.1f} MiB",
        flush=True,
    )


def run_correctness_suite(
    cases: List[Case],
    dtypes: List[torch.dtype],
    timeout: float,
    compare_ref2: bool,
    stop_on_fail: bool,
):
    total = len(cases) * len(dtypes)

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

        for case_idx, case in enumerate(cases):
            print_case_header(
                counter,
                total,
                case,
                dtype,
            )

            payload = {
                "case": case_to_dict(case),
                "dtype": dtype_to_name(dtype),
                "seed": 1234 + case_idx,
                "compare_ref2": compare_ref2,
            }

            result = run_with_timeout(
                mode="correctness",
                payload=payload,
                timeout_sec=timeout,
            )

            status = result["status"]

            if status == "ok":
                print(
                    f"    RESULT: PASS",
                    flush=True,
                )
                passed += 1

            elif status == "oom":
                print(
                    f"    RESULT: OOM SKIP",
                    flush=True,
                )
                print(
                    f"    {result['message']}",
                    flush=True,
                )
                skipped_oom += 1

            elif status == "timeout":
                print(
                    f"    RESULT: TIMEOUT SKIP",
                    flush=True,
                )
                print(
                    f"    {result['message']}",
                    flush=True,
                )
                skipped_timeout += 1

            else:
                print(
                    f"    RESULT: FAIL",
                    flush=True,
                )
                print(
                    result["message"],
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


def benchmark_case_body(
    case: Case,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    autotune_repeat: int,
) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = "cuda"

    set_seed(
        123
    )

    fused_dynamic_conv_forward_clear_warmup_cache()
    fused_dynamic_conv_backward_clear_warmup_cache()

    base_h = make_leaf_randn(
        shape=(
            case.B,
            case.D,
            case.L,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=False,
    )

    base_kernel = make_leaf_randn(
        shape=(
            case.B,
            case.T,
            case.N,
            case.K,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=False,
    )

    base_kernel_mix = make_leaf_randn(
        shape=(
            case.D,
            case.N,
        ),
        device=device,
        dtype=dtype,
        scale=case.scale,
        requires_grad=False,
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

    h = clone_leaf(
        base_h
    )

    kernel = clone_leaf(
        base_kernel
    )

    kernel_mix = clone_leaf(
        base_kernel_mix
    )

    selected_forward_plan = fused_dynamic_conv_forward_warmup_chunk(
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=case.t_offset,
        dilation=case.dilation,
        repeat=autotune_repeat,
    )

    cached_forward_plan = fused_dynamic_conv_forward_cached_plan_chunk(
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=case.t_offset,
        dilation=case.dilation,
    )

    if cached_forward_plan != selected_forward_plan:
        raise RuntimeError(
            f"forward cached plan mismatch: selected={selected_forward_plan}, "
            f"cached={cached_forward_plan}"
        )

    selected_forward_plan_name = fused_dynamic_conv_forward_plan_name(
        selected_forward_plan
    )

    selected_backward_plan = fused_dynamic_conv_backward_warmup_chunk(
        grad_out=grad,
        h_full=h,
        kernel_chunk=kernel,
        kernel_mix=kernel_mix,
        t_offset=case.t_offset,
        dilation=case.dilation,
        repeat=autotune_repeat,
    )

    cached_backward_plan = fused_dynamic_conv_backward_cached_plan_chunk(
        h_full=h,
        kernel_chunk=kernel,
        t_offset=case.t_offset,
        dilation=case.dilation,
    )

    if cached_backward_plan != selected_backward_plan:
        raise RuntimeError(
            f"backward cached plan mismatch: selected={selected_backward_plan}, "
            f"cached={cached_backward_plan}"
        )

    selected_backward_plan_name = fused_dynamic_conv_backward_plan_name(
        selected_backward_plan
    )

    for _ in range(
        warmup
    ):
        clear_grads(
            [
                h,
                kernel,
                kernel_mix,
            ]
        )

        out = fused_dynamic_conv_chunk(
            h,
            kernel,
            kernel_mix,
            t_offset=case.t_offset,
            dilation=case.dilation,
        )

        out.backward(
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
                h,
                kernel,
                kernel_mix,
            ]
        )

        out = fused_dynamic_conv_chunk(
            h,
            kernel,
            kernel_mix,
            t_offset=case.t_offset,
            dilation=case.dilation,
        )

        out.backward(
            grad
        )

    end.record()
    torch.cuda.synchronize()

    fused_ms = start.elapsed_time(
        end
    ) / iters

    h_ref = clone_leaf(
        base_h
    )

    kernel_ref = clone_leaf(
        base_kernel
    )

    kernel_mix_ref = clone_leaf(
        base_kernel_mix
    )

    for _ in range(
        warmup
    ):
        clear_grads(
            [
                h_ref,
                kernel_ref,
                kernel_mix_ref,
            ]
        )

        out_ref = torch_reference_materialized_chunk(
            h_ref,
            kernel_ref,
            kernel_mix_ref,
            t_offset=case.t_offset,
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
                h_ref,
                kernel_ref,
                kernel_mix_ref,
            ]
        )

        out_ref = torch_reference_materialized_chunk(
            h_ref,
            kernel_ref,
            kernel_mix_ref,
            t_offset=case.t_offset,
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
        "fused_ms": float(fused_ms),
        "ref_ms": float(ref_ms),
        "speedup": float(ref_ms / fused_ms),
        "forward_plan_id": int(selected_forward_plan),
        "forward_plan_name": selected_forward_plan_name,
        "backward_plan_id": int(selected_backward_plan),
        "backward_plan_name": selected_backward_plan_name,
    }

def get_benchmark_cases() -> List[Case]:
    return [
        Case(
            name="original_small_T4096",
            B=1,
            D=256,
            L=4096,
            T=4096,
            N=6,
            K=3,
            t_offset=0,
            dilation=1,
        ),
        Case(
            name="original_mid_T16384",
            B=1,
            D=256,
            L=65536,
            T=16384,
            N=6,
            K=3,
            t_offset=32768,
            dilation=1,
        ),
        Case(
            name="original_big_T65536",
            B=1,
            D=256,
            L=262144,
            T=65536,
            N=6,
            K=3,
            t_offset=131072,
            dilation=1,
        ),
        Case(
            name="small_T1024",
            B=1,
            D=256,
            L=4096,
            T=1024,
            N=6,
            K=3,
            t_offset=1024,
            dilation=1,
        ),
        Case(
            name="mid_T8192",
            B=1,
            D=256,
            L=32768,
            T=8192,
            N=6,
            K=3,
            t_offset=16384,
            dilation=1,
        ),
        Case(
            name="N16_T8192",
            B=1,
            D=256,
            L=32768,
            T=8192,
            N=16,
            K=3,
            t_offset=16384,
            dilation=1,
        ),
        Case(
            name="K7_T8192",
            B=1,
            D=256,
            L=32768,
            T=8192,
            N=6,
            K=7,
            t_offset=16384,
            dilation=1,
        ),
        Case(
            name="D512_T8192",
            B=1,
            D=512,
            L=32768,
            T=8192,
            N=6,
            K=3,
            t_offset=16384,
            dilation=1,
        ),
    ]


def run_benchmark_suite(
    cases: List[Case],
    dtype: torch.dtype,
    timeout: float,
    warmup: int,
    iters: int,
    autotune_repeat: int,
):
    print(
        "=" * 100,
        flush=True,
    )

    print(
        f"Benchmark dtype={dtype} warmup={warmup} iters={iters} autotune_repeat={autotune_repeat}",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )

    rows = []

    for idx, case in enumerate(cases):
        print_case_header(
            idx,
            len(cases),
            case,
            dtype,
        )

        payload = {
            "case": case_to_dict(case),
            "dtype": dtype_to_name(dtype),
            "warmup": warmup,
            "iters": iters,
            "autotune_repeat": autotune_repeat,
        }

        result = run_with_timeout(
            mode="benchmark",
            payload=payload,
            timeout_sec=timeout,
        )

        status = result["status"]

        if status == "ok":
            r = result["result"]

            print(
                f"    selected forward  plan: "
                f"{r['forward_plan_id']} {r['forward_plan_name']}",
                flush=True,
            )

            print(
                f"    selected backward plan: "
                f"{r['backward_plan_id']} {r['backward_plan_name']}",
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
                    r["fused_ms"],
                    r["ref_ms"],
                    r["speedup"],
                    "ok",
                    r["forward_plan_id"],
                    r["forward_plan_name"],
                    r["backward_plan_id"],
                    r["backward_plan_name"],
                )
            )

        elif status == "oom":
            print(
                f"    RESULT: OOM SKIP",
                flush=True,
            )
            print(
                f"    {result['message']}",
                flush=True,
            )

            rows.append(
                (
                    case.name,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    "oom",
                    -1,
                    "oom",
                    -1,
                    "oom",
                )
            )

        elif status == "timeout":
            print(
                f"    RESULT: TIMEOUT SKIP",
                flush=True,
            )
            print(
                f"    {result['message']}",
                flush=True,
            )

            rows.append(
                (
                    case.name,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    "timeout",
                    -1,
                    "timeout",
                    -1,
                    "timeout",
                )
            )

        else:
            print(
                f"    RESULT: FAIL",
                flush=True,
            )
            print(
                result["message"],
                flush=True,
            )

            rows.append(
                (
                    case.name,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    "fail",
                    -1,
                    "fail",
                    -1,
                    "fail",
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
        backward_plan_id,
        backward_plan_name,
    ) in rows:
        if status == "ok":
            print(
                f"{name:<28} fused={fused_ms:>9.3f} ms  "
                f"ref={ref_ms:>9.3f} ms  speedup={speedup:>7.3f}x  "
                f"fwd={forward_plan_id}:{forward_plan_name}  "
                f"bwd={backward_plan_id}:{backward_plan_name}",
                flush=True,
            )
        else:
            print(
                f"{name:<28} {status}",
                flush=True,
            )

def run_full_length_suite(
    dtypes: List[torch.dtype],
    timeout: float,
):
    for dtype in dtypes:
        payload = {
            "dtype": dtype_to_name(dtype),
        }

        result = run_with_timeout(
            mode="full_length",
            payload=payload,
            timeout_sec=timeout,
        )

        if result["status"] == "ok":
            print(
                f"full_length dtype={dtype}: PASS",
                flush=True,
            )
        elif result["status"] in (
            "oom",
            "timeout",
        ):
            print(
                f"full_length dtype={dtype}: {result['status'].upper()} SKIP",
                flush=True,
            )
            print(
                result["message"],
                flush=True,
            )
        else:
            print(
                f"full_length dtype={dtype}: FAIL",
                flush=True,
            )
            print(
                result["message"],
                flush=True,
            )
            raise RuntimeError(
                f"full_length failed for dtype={dtype}"
            )


def run_gradcheck_suite(
    timeout: float,
):
    result = run_with_timeout(
        mode="gradcheck",
        payload={},
        timeout_sec=timeout,
    )

    if result["status"] == "ok":
        print(
            "gradcheck_reference_only: PASS",
            flush=True,
        )
    elif result["status"] in (
        "oom",
        "timeout",
    ):
        print(
            f"gradcheck_reference_only: {result['status'].upper()} SKIP",
            flush=True,
        )
        print(
            result["message"],
            flush=True,
        )
    else:
        print(
            "gradcheck_reference_only: FAIL",
            flush=True,
        )
        print(
            result["message"],
            flush=True,
        )
        raise RuntimeError(
            "gradcheck reference failed."
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

    for item in text.split(","):
        key = item.strip().lower()

        if key not in mapping or mapping[key] is None:
            raise ValueError(
                f"Unknown dtype: {item}"
            )

        dtype = mapping[key]

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


def main():
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
        help="Dtype list: fp32,fp16,bf16,all. Example: --dtype bf16 or --dtype fp16,bf16",
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
        "--autotune-repeat",
        type=int,
        default=3,
        help="Number of timing repeats for each backward candidate plan during warmup/autotune.",
    )

    parser.add_argument(
        "--no-ref2",
        action="store_true",
        help="Do not compare with no-materialize reference in correctness cases.",
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

    if args.autotune_repeat <= 0:
        raise RuntimeError(
            "--autotune-repeat must be positive."
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

    if args.quick or args.all:
        run_correctness_suite(
            cases=get_quick_cases(),
            dtypes=dtypes,
            timeout=args.timeout,
            compare_ref2=compare_ref2,
            stop_on_fail=args.stop_on_fail,
        )

        run_full_length_suite(
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
        )

        run_full_length_suite(
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
        )

    if args.benchmark or args.all:
        bench_dtype = torch.float16

        if torch.cuda.is_bf16_supported():
            bench_dtype = torch.bfloat16

        if len(dtypes) == 1:
            bench_dtype = dtypes[0]

        run_benchmark_suite(
            cases=get_benchmark_cases(),
            dtype=bench_dtype,
            timeout=args.benchmark_timeout,
            warmup=args.warmup,
            iters=args.iters,
            autotune_repeat=args.autotune_repeat,
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
