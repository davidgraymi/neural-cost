"""Canonical, framework-independent FLOP and parameter accounting.

Different adapters see the same network at different granularity: a PyTorch
forward hook records one ``attention`` operation, while a JAX trace records
the individual ``dot_general``, ``exp`` and ``reduce_sum`` primitives that
implement it.  Their *total* FLOPs legitimately differ, because one counts
auxiliary point-wise work the other never sees.

This module defines one counting convention that is invariant to that
granularity:

* **dense FLOPs** — multiply-accumulate work in contractions (matmul, linear,
  convolution, attention scores/projections, expert FFNs, SSM projections),
  counted as 2 FLOPs per MAC;
* **auxiliary FLOPs** — everything else (activations, normalization, softmax,
  pooling, embedding lookups);
* **parameter count** — unique trainable elements, de-duplicated by
  ``parameter_id`` so tied or re-used weights count once.

Two adapters describing the same mathematical model must agree on dense FLOPs
and parameter count.  :func:`cross_check` verifies exactly that, turning the
"framework-neutral" claim into a testable property.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median

from .estimate import attention_flop_parts, estimate_operation
from .operations import Operation, numel

DENSE_KINDS = frozenset(
    {"matmul", "linear", "conv2d", "moe", "paged_attention", "state_space_model"}
)


@dataclass(frozen=True, slots=True)
class FlopBreakdown:
    """Dense vs auxiliary FLOPs and unique parameter elements for a workload."""

    dense_flops: int
    auxiliary_flops: int
    parameter_count: int
    dense_flops_by_kind: Mapping[str, int] = field(default_factory=dict)
    operation_count: int = 0

    @property
    def total_flops(self) -> int:
        return self.dense_flops + self.auxiliary_flops


def flop_breakdown(operations: Iterable[Operation]) -> FlopBreakdown:
    """Split a workload's FLOPs into canonical dense and auxiliary components."""
    dense = 0
    auxiliary = 0
    by_kind: dict[str, int] = {}
    parameters = 0
    seen: set[int] = set()
    count = 0
    for operation in operations:
        count += 1
        cost = estimate_operation(operation)
        repeat = int(operation.attrs.get("repeat", 1))
        if operation.kind == "attention":
            proj, scores, softmax = attention_flop_parts(operation)
            op_dense, op_aux = (proj + scores) * repeat, softmax * repeat
        elif operation.kind in DENSE_KINDS:
            op_dense, op_aux = cost.flops, 0
        else:
            op_dense, op_aux = 0, cost.flops
        dense += op_dense
        auxiliary += op_aux
        if op_dense:
            by_kind[operation.kind] = by_kind.get(operation.kind, 0) + op_dense

        declared = operation.attrs.get("parameter_bytes")
        if (
            declared is None
            and operation.kind in {"linear", "conv2d"}
            and len(operation.inputs) > 1
        ):
            declared = numel(operation.inputs[1]) * operation.dtype_bytes
        if not declared:
            continue
        parameter_id = operation.attrs.get("parameter_id")
        if isinstance(parameter_id, int):
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
        parameters += int(declared) // operation.dtype_bytes
    return FlopBreakdown(dense, auxiliary, parameters, by_kind, count)


@dataclass(frozen=True, slots=True)
class FrameworkCount:
    """One adapter's canonical counts and deviation from the consensus."""

    framework: str
    breakdown: FlopBreakdown
    dense_deviation: float
    parameter_deviation: float
    consistent: bool


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """Cross-framework agreement on dense FLOPs and parameter count."""

    reference_dense_flops: float
    reference_parameter_count: float
    rtol: float
    counts: tuple[FrameworkCount, ...]

    @property
    def consistent(self) -> bool:
        return all(count.consistent for count in self.counts)

    @property
    def outliers(self) -> tuple[str, ...]:
        return tuple(count.framework for count in self.counts if not count.consistent)

    def render(self) -> str:
        lines = [
            f"Cross-framework consistency (rtol {self.rtol:.1%}): "
            + ("OK" if self.consistent else "MISMATCH in " + ", ".join(self.outliers)),
            (
                f"  {'framework':<14} {'dense FLOPs':>14} {'Δ':>8} {'params':>12} {'Δ':>8} "
                f"{'aux FLOPs':>12}"
            ),
        ]
        for count in self.counts:
            b = count.breakdown
            lines.append(
                f"  {count.framework:<14} {b.dense_flops:>14,} {count.dense_deviation:>+8.2%} "
                f"{b.parameter_count:>12,} {count.parameter_deviation:>+8.2%} "
                f"{b.auxiliary_flops:>12,}" + ("" if count.consistent else "  <-- mismatch")
            )
        lines.append(
            "  note: auxiliary FLOPs depend on capture granularity and are not expected to match."
        )
        return "\n".join(lines)


def _deviation(value: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if value == 0 else float("inf")
    return (value - reference) / reference


def cross_check(
    workloads: Mapping[str, Sequence[Operation]],
    *,
    rtol: float = 0.01,
    check_parameters: bool = True,
) -> ConsistencyReport:
    """Verify that several adapters agree on one model's canonical counts.

    ``workloads`` maps a framework label to the operations its adapter
    captured for the *same* mathematical model.  The consensus reference is
    the median across frameworks, so a single buggy adapter is singled out
    rather than shifting the reference.
    """
    if not workloads:
        raise ValueError("cross_check requires at least one workload")
    if rtol < 0:
        raise ValueError("rtol must be non-negative")
    breakdowns = {name: flop_breakdown(ops) for name, ops in workloads.items()}
    ref_dense = median(b.dense_flops for b in breakdowns.values())
    ref_params = median(b.parameter_count for b in breakdowns.values())
    counts = []
    for name, breakdown in breakdowns.items():
        dense_dev = _deviation(breakdown.dense_flops, ref_dense)
        param_dev = _deviation(breakdown.parameter_count, ref_params)
        ok = abs(dense_dev) <= rtol and (not check_parameters or abs(param_dev) <= rtol)
        counts.append(FrameworkCount(name, breakdown, dense_dev, param_dev, ok))
    return ConsistencyReport(ref_dense, ref_params, rtol, tuple(counts))
