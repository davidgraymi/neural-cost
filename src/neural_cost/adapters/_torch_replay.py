"""Isolated replay of portable operations as PyTorch kernels.

Replaying each captured :class:`~neural_cost.operations.Operation` alone, with
synthetic tensors of the recorded shapes, measures what the kernel library
actually achieves for that exact shape.  Comparing the isolated times with the
roofline separates *kernel inefficiency* (the library is slow for this shape)
from *composition effects* (what happens only when ops run together: framework
glue, uncaptured ops, cache interference).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil, prod
from time import perf_counter_ns
from typing import Any

from ..operations import Operation


def make_sync(torch: Any, device: Any) -> Callable[[], None]:
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


def per_call(
    fn: Callable[[], Any],
    sync: Callable[[], None],
    *,
    target_seconds: float = 3e-3,
    trials: int = 5,
    reps: int | None = None,
) -> float:
    """Best per-call time of ``fn`` repeated back to back and synchronized once."""
    fn()
    sync()
    if reps is None:
        start = perf_counter_ns()
        for _ in range(10):
            fn()
        sync()
        single = max((perf_counter_ns() - start) / 1e10, 1e-7)
        reps = max(10, min(5000, int(target_seconds / single)))
    best = float("inf")
    for _ in range(trials):
        start = perf_counter_ns()
        for _ in range(reps):
            fn()
        sync()
        best = min(best, (perf_counter_ns() - start) / 1e9 / reps)
    return best


def _dtype(torch: Any, dtype_bytes: int, device: Any) -> Any:
    if dtype_bytes == 2:
        return torch.bfloat16 if device.type == "cpu" else torch.float16
    if dtype_bytes == 8:
        return torch.float64
    return torch.float32


def _builder(torch: Any, op: Operation, device: Any) -> Callable[[], Any] | None:
    """Return a zero-argument callable executing ``op`` once, or None if unsupported."""
    functional = torch.nn.functional
    dtype = _dtype(torch, op.dtype_bytes, device)

    def rand(shape: tuple[int, ...]) -> Any:
        return torch.randn(shape, device=device, dtype=dtype)

    kind, attrs = op.kind, op.attrs
    if kind == "linear" and len(op.inputs) >= 2 and len(op.inputs[1]) == 2:
        x, (k, n) = rand(op.inputs[0]), op.inputs[1]
        weight = rand((n, k))
        has_bias = int(attrs.get("parameter_bytes", 0)) > k * n * op.dtype_bytes
        bias = rand((n,)) if has_bias else None
        return lambda: functional.linear(x, weight, bias)
    if kind == "matmul" and len(op.inputs) >= 2:
        a, b = rand(op.inputs[0]), rand(op.inputs[1])
        try:
            torch.matmul(a, b)
        except RuntimeError:
            return None
        return lambda: torch.matmul(a, b)
    if kind == "conv2d" and len(op.inputs) >= 2 and len(op.output) == 4:
        (n, c, h, w), (o, ci, kh, kw) = op.inputs[0], op.inputs[1]
        ho, wo = op.output[2], op.output[3]
        sh, sw = max(1, round(h / ho)), max(1, round(w / wo))
        ph = max(0, ceil(((ho - 1) * sh + kh - h) / 2))
        pw = max(0, ceil(((wo - 1) * sw + kw - w) / 2))
        groups = int(attrs.get("groups", 1))
        x, weight = rand((n, c, h, w)), rand((o, ci, kh, kw))
        fn = lambda: functional.conv2d(x, weight, None, (sh, sw), (ph, pw), groups=groups)
        return fn if tuple(fn().shape) == tuple(op.output) else None
    if kind == "elementwise":
        x = rand(op.output)
        return lambda: torch.relu(x)
    if kind == "softmax":
        x = rand(op.output)
        return lambda: torch.softmax(x, -1)
    if kind == "layernorm":
        x = rand(op.output)
        return lambda: functional.layer_norm(x, (op.output[-1],))
    if kind == "rmsnorm" and hasattr(functional, "rms_norm"):
        x = rand(op.output)
        return lambda: functional.rms_norm(x, (op.output[-1],))
    if kind == "batchnorm" and len(op.output) >= 2:
        x = rand(op.output)
        mean = torch.zeros(op.output[1], device=device, dtype=dtype)
        var = torch.ones_like(mean)
        return lambda: functional.batch_norm(x, mean, var, training=False)
    if kind == "embedding" and op.inputs:
        dim = op.output[-1]
        vocab = max(1, int(attrs.get("parameter_bytes", 0)) // (dim * op.dtype_bytes))
        indices = torch.randint(0, vocab, op.inputs[0], device=device)
        table = rand((vocab, dim))
        return lambda: functional.embedding(indices, table)
    if kind == "pooling" and op.inputs and len(op.inputs[0]) == 4:
        x = rand(op.inputs[0])
        size = op.output[2:] if len(op.output) == 4 else (1, 1)
        return lambda: functional.adaptive_avg_pool2d(x, size)
    if kind == "attention" and len(op.output) >= 2:
        num_heads = int(attrs.get("num_heads", 1))
        if len(op.output) >= 4:
            b, heads = op.output[0], num_heads
            t = int(attrs.get("seq_len", op.output[2]))
            hd = int(attrs.get("head_dim", op.output[3]))
            d = heads * hd
        else:
            d = op.output[-1]
            t = int(attrs.get("seq_len", op.output[-2]))
            b = max(1, prod(op.output[:-1]) // t)
            heads = num_heads
            hd = int(attrs.get("head_dim", d // max(heads, 1)))
        q = rand((b, heads, t, hd))
        sdpa = functional.scaled_dot_product_attention
        if not bool(attrs.get("include_projections", True)):
            return lambda: sdpa(q, q, q)
        proj = int(attrs.get("proj_dim", d))
        x = rand((b, t, d))
        w_in, w_out = rand((3 * proj, d)), rand((d, proj))

        def full() -> Any:
            qkv = functional.linear(x, w_in)
            qh = qkv[..., :proj].reshape(b, t, heads, -1).transpose(1, 2)
            kh = qkv[..., proj : 2 * proj].reshape(b, t, heads, -1).transpose(1, 2)
            vh = qkv[..., 2 * proj :].reshape(b, t, heads, -1).transpose(1, 2)
            out = sdpa(qh, kh, vh).transpose(1, 2).reshape(b, t, proj)
            return functional.linear(out, w_out)

        return full
    return None


def replay_operations(
    operations: Sequence[Operation], device: Any = None, *, target_seconds: float = 3e-3
) -> list[float | None]:
    """Isolated per-operation seconds (× ``repeat``), or None where unsupported.

    Identical operations are measured once.  Each time is the best
    back-to-back per-call time, i.e. kernel plus steady-state dispatch.
    """
    import torch

    device = torch.device(device or "cpu")
    sync = make_sync(torch, device)
    cache: dict[tuple, float | None] = {}
    results: list[float | None] = []
    with torch.no_grad():
        for op in operations:
            key = (
                op.kind,
                op.inputs,
                op.output,
                op.dtype_bytes,
                tuple(
                    sorted(
                        (k, v) for k, v in op.attrs.items() if k not in {"parameter_id", "repeat"}
                    )
                ),
            )
            if key not in cache:
                try:
                    fn = _builder(torch, op, device)
                    cache[key] = (
                        per_call(fn, sync, target_seconds=target_seconds)
                        if fn is not None
                        else None
                    )
                except (RuntimeError, ValueError, TypeError):
                    cache[key] = None
            seconds = cache[key]
            repeat = int(op.attrs.get("repeat", 1))
            results.append(seconds * repeat if seconds is not None else None)
    return results
