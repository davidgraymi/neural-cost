"""Explain *where* the gap between a roofline bound and measured runtime goes.

A roofline efficiency of 4% says a workload is slow, not why.  This module
decomposes the measured median runtime into an ordered waterfall of
non-overlapping, physically motivated components:

``roofline_floor``
    max(FLOPs / peak, fused bytes / peak bandwidth): perfect fusion, perfect
    compute/memory overlap, datasheet peaks.  Nothing can run faster.
``unfused_traffic``
    Extra DRAM round-trips for intermediates that a fusing compiler (Inductor,
    XLA) would keep on chip.
``serialization``
    Operations execute one after another, so each is bounded by its *own*
    roofline; Σᵢ max(fᵢ/P, bᵢ/BW) ≥ max(ΣF/P, ΣB/BW).  The difference is lost
    compute/memory overlap across operations.
``peak_derate``
    Hardware never reaches its datasheet peak.  Re-evaluating the per-op
    bounds at *measured* achievable FLOP/s and bandwidth (see
    :class:`HardwareCalibration`) separates "the chip can't" from "the code
    doesn't".
``dispatch``
    Per-kernel host overhead (Python, framework dispatch, driver launch) times
    the kernel count, plus any fixed per-call overhead.
``kernel_inefficiency``
    Only with ``isolated_seconds`` from an adapter's ``replay``: how much
    slower each operation runs *alone* than its calibrated size-aware
    roofline.  This is the kernel library's shortfall for that exact shape.
``timing_jitter``
    Median minus fastest sample: run-to-run noise, not a code property.
``residual``
    Everything else: kernel-level inefficiency (tiling, occupancy, cache
    misses), operations the adapter did not capture, and model error.

Each level is computed cumulatively so components never double count, and the
components sum exactly to the measured median.  A negative residual means the
model explains *more* than the measurement, i.e. some assumption (hardware
spec, calibration, FLOP count) is wrong — reported, never hidden.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from .estimate import CostEstimate, estimate_fused_operations, estimate_operation
from .hardware import HardwareSpec
from .operations import Operation
from .profiler import Measurement


@dataclass(frozen=True, slots=True)
class HardwareCalibration:
    """Empirically achievable throughput and overhead on one device/framework.

    ``dispatch_seconds`` is the per-kernel host overhead and
    ``call_overhead_seconds`` a fixed cost per model invocation (e.g. a device
    synchronization).  Produce one with an adapter's ``calibrate()``
    microbenchmarks or with :func:`neural_cost.validation.fit_cost_model` on
    your own measurements.

    Throughput depends strongly on operation size: a 64×64 GEMM reaches a few
    percent of the rate of a 4096×4096 one, and a 1 MiB copy runs from cache
    while a 256 MiB copy streams from DRAM.  ``flops_curve`` maps work (FLOPs
    per kernel) to achieved FLOP/s and ``bandwidth_curve`` maps traffic
    (bytes per kernel) to achieved bytes/s; :meth:`flops_rate` and
    :meth:`bandwidth_rate` interpolate them in log-log space.  Without curves
    the scalar ``achievable_*`` values are used for every size.
    """

    achievable_flops: float
    achievable_bandwidth: float
    dispatch_seconds: float = 0.0
    call_overhead_seconds: float = 0.0
    source: str = "manual"
    flops_curve: tuple[tuple[float, float], ...] = ()
    bandwidth_curve: tuple[tuple[float, float], ...] = ()
    async_dispatch: bool = False

    def __post_init__(self) -> None:
        if self.achievable_flops <= 0 or self.achievable_bandwidth <= 0:
            raise ValueError("achievable_flops and achievable_bandwidth must be positive")
        if self.dispatch_seconds < 0 or self.call_overhead_seconds < 0:
            raise ValueError("overheads must be non-negative")
        for curve in (self.flops_curve, self.bandwidth_curve):
            if any(size <= 0 or rate <= 0 for size, rate in curve):
                raise ValueError("curve points must be positive (size, rate) pairs")

    def flops_rate(self, flops: float) -> float:
        """Achievable FLOP/s for a kernel performing ``flops`` operations."""
        return _interpolate(self.flops_curve, flops, self.achievable_flops)

    def bandwidth_rate(self, nbytes: float) -> float:
        """Achievable bytes/s for a kernel moving ``nbytes`` bytes."""
        return _interpolate(self.bandwidth_curve, nbytes, self.achievable_bandwidth)

    def kernel_seconds(self, flops: float, nbytes: float) -> float:
        """Size-aware roofline time of one kernel at achievable throughput."""
        compute = flops / self.flops_rate(flops) if flops > 0 else 0.0
        memory = nbytes / self.bandwidth_rate(nbytes) if nbytes > 0 else 0.0
        return max(compute, memory)

    def predict(
        self,
        costs: Sequence[CostEstimate],
        kernel_count: int | None = None,
        extra_kernels: Sequence[tuple[int, bool]] | None = None,
    ) -> tuple[float, float]:
        """Return ``(execution_seconds, exposed_overhead_seconds)`` for a kernel sequence.

        Each cost may represent ``operations`` repeated kernels (e.g. a scan
        body); rates are looked up at the per-kernel size.  With synchronous
        dispatch (CPU eager) every kernel pays ``dispatch_seconds`` on top of
        its execution.  With asynchronous dispatch (CUDA, MPS) launches overlap
        execution, so a kernel only exposes ``max(0, dispatch − runtime)``.
        Kernels beyond the captured operations (``kernel_count`` larger than
        the operation count) are assumed tiny and cost one dispatch each,
        unless ``extra_kernels`` describes them as ``(bytes, pays_dispatch)``
        pairs (e.g. from ``TorchAdapter.trace_kernels``): each is then a
        memory-bound kernel moving ``bytes``, paying dispatch only when issued
        from Python.
        """
        execution = 0.0
        exposed = self.call_overhead_seconds
        captured = 0
        for cost in costs:
            repeats = max(1, cost.operations)
            captured += repeats
            per_kernel = self.kernel_seconds(cost.flops / repeats, cost.total_bytes / repeats)
            execution += repeats * per_kernel
            if self.async_dispatch:
                exposed += repeats * max(0.0, self.dispatch_seconds - per_kernel)
            else:
                exposed += repeats * self.dispatch_seconds
        if extra_kernels is not None:
            for nbytes, pays_dispatch in extra_kernels:
                seconds = self.kernel_seconds(0, nbytes)
                execution += seconds
                if pays_dispatch:
                    exposed += (
                        max(0.0, self.dispatch_seconds - seconds)
                        if self.async_dispatch
                        else self.dispatch_seconds
                    )
        elif kernel_count is not None:
            exposed += max(0, kernel_count - captured) * self.dispatch_seconds
        return execution, exposed

    def to_hardware_spec(
        self, name: str | None = None, device_type: str | None = None
    ) -> HardwareSpec:
        """A :class:`HardwareSpec` whose peaks are the measured achievable rates.

        Use it for devices with no published-spec table (TPU, jax-mps,
        future plugins) or to score a workload against what the device can
        actually reach rather than its datasheet.
        """
        return HardwareSpec(
            name or f"measured ({self.source})",
            self.achievable_flops,
            self.achievable_bandwidth,
            device_type=device_type,
        )

    def fraction_of(self, spec: HardwareSpec) -> tuple[float, float]:
        """``(flops, bandwidth)`` achievable rates as fractions of ``spec``'s peaks."""
        return (
            self.achievable_flops / spec.peak_flops,
            self.achievable_bandwidth / spec.memory_bandwidth,
        )


def _interpolate(curve: tuple[tuple[float, float], ...], size: float, default: float) -> float:
    """Log-log interpolation of ``rate(size)``; clamps outside the measured range."""
    if not curve or size <= 0:
        return default
    points = sorted(curve)
    if size <= points[0][0]:
        return points[0][1]
    if size >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in pairwise(points):
        if x0 <= size <= x1:
            t = (math.log(size) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return default  # pragma: no cover - unreachable for sorted points


@dataclass(frozen=True, slots=True)
class GapComponent:
    """One slice of the measured runtime."""

    name: str
    seconds: float
    fraction: float
    explanation: str
    recommendation: str

    @property
    def speedup_if_removed(self) -> float:
        """Projected speedup if this component were eliminated entirely."""
        remaining = 1.0 - self.fraction
        return 1.0 / remaining if remaining > 1e-12 else float("inf")


_EXPLANATIONS = {
    "roofline_floor": (
        "Compulsory work at datasheet peak with perfect fusion and overlap.",
        "Reduce work itself: lower precision, sparsity, smaller model, or better hardware.",
    ),
    "unfused_traffic": (
        "DRAM round-trips for intermediates between fusible producer/consumer ops.",
        "Fuse point-wise chains: torch.compile, jax.jit/XLA, tf.function(jit_compile=True).",
    ),
    "serialization": (
        "Lost overlap: each op is bounded by its own roofline, run back to back.",
        "Fuse heterogeneous ops or overlap independent ops (streams, larger fused kernels).",
    ),
    "peak_derate": (
        "Gap between datasheet peak and measured achievable throughput on this device.",
        (
            "Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and "
            "aligned shapes to approach achievable peak."
        ),
    ),
    "dispatch": (
        "Per-kernel host/launch overhead multiplied by the number of kernels.",
        (
            "Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile "
            "mode='reduce-overhead', jax.jit), or reduce kernel count via fusion."
        ),
    ),
    "kernel_inefficiency": (
        (
            "Isolated per-op time beyond its size-aware achievable roofline: the kernel library "
            "is slow for these exact shapes."
        ),
        (
            "Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts "
            "(channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse."
        ),
    ),
    "timing_jitter": (
        "Median minus fastest sample: run-to-run variance.",
        "Pin clocks/threads, isolate the process, add warmup, report min alongside median.",
    ),
    "residual": (
        (
            "Unexplained: kernel inefficiency (tiling, occupancy, cache misses), uncaptured "
            "ops, or model error."
        ),
        (
            "Profile kernels (Nsight Compute, torch.profiler, XLA dump) to see which kernels miss "
            "their per-op bound."
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class GapAttribution:
    """A waterfall decomposition of measured runtime."""

    measured_seconds: float
    components: tuple[GapComponent, ...]
    kernel_count: int
    calibration: HardwareCalibration | None
    overexplained: bool
    bound_violated: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def component(self, name: str) -> GapComponent:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)

    @property
    def explained_fraction(self) -> float:
        """Fraction of runtime attributed to modeled causes (everything but residual)."""
        return sum(c.fraction for c in self.components if c.name != "residual")

    @property
    def dominant(self) -> GapComponent:
        """Largest component other than the irreducible roofline floor."""
        candidates = [c for c in self.components if c.name != "roofline_floor"]
        return max(candidates, key=lambda c: c.seconds)

    def recommendations(self, min_fraction: float = 0.05) -> tuple[str, ...]:
        """Actionable next steps for components above ``min_fraction``, largest first."""
        ranked = sorted(
            (
                c
                for c in self.components
                if c.fraction >= min_fraction and c.name not in {"roofline_floor"}
            ),
            key=lambda c: c.seconds,
            reverse=True,
        )
        return tuple(
            f"{c.name} ({c.fraction:.0%}, ≤{c.speedup_if_removed:.2f}x if removed): "
            f"{c.recommendation}"
            for c in ranked
        )

    def render(self, width: int = 32) -> str:
        lines = [
            f"Gap attribution — measured {self.measured_seconds * 1e3:.3f} ms, "
            f"{self.kernel_count} kernels"
            + (f", calibration: {self.calibration.source}" if self.calibration else ""),
        ]
        for c in self.components:
            filled = max(0, min(width, round(abs(c.fraction) * width)))
            bar = ("█" if c.seconds >= 0 else "▒") * filled + "·" * (width - filled)
            lines.append(f"  {c.name:<16} {c.seconds * 1e3:>10.4f} ms {c.fraction:>7.1%} {bar}")
        lines.append(f"  explained by model: {self.explained_fraction:.1%}")
        if self.bound_violated:
            lines.append(
                "  ! measured time is below the roofline floor: the hardware spec or FLOP/byte "
                "count is wrong."
            )
        elif self.overexplained:
            lines.append(
                "  ! components exceed measured time (negative residual): calibration or "
                "kernel count is likely overestimated."
            )
        lines.extend(f"  ! {w}" for w in self.warnings)
        lines.extend(f"  next: {r}" for r in self.recommendations())
        return "\n".join(lines)


def _per_op_bound(costs: Sequence[CostEstimate], flops_rate: float, bandwidth: float) -> float:
    return sum(max(c.flops / flops_rate, c.total_bytes / bandwidth) for c in costs)


def attribute_gap(
    operations: Iterable[Operation],
    measurement: Measurement,
    hardware: HardwareSpec,
    calibration: HardwareCalibration | None = None,
    *,
    kernel_count: int | None = None,
    assume_fusion: bool = True,
    isolated_seconds: Sequence[float | None] | None = None,
    extra_kernels: Sequence[tuple[int, bool]] | None = None,
) -> GapAttribution:
    """Decompose measured runtime into the waterfall described in the module docstring.

    ``kernel_count`` overrides the number of launched kernels; by default it
    is the number of captured operations (including ``repeat``).  Hook-based
    adapters capture layers rather than every kernel, so pass the profiler's
    kernel count when you have it.  ``assume_fusion=False`` treats the unfused
    traffic as compulsory (e.g. when comparing against an eager baseline you do
    not intend to compile).  ``isolated_seconds`` (from ``adapter.replay``)
    enables the ``kernel_inefficiency`` component; the residual then measures
    only composition effects.  ``extra_kernels`` models uncaptured kernels by
    their traffic (see :meth:`HardwareCalibration.predict`).
    """
    op_list = list(operations)
    measured = measurement.median_seconds
    costs = [estimate_operation(op) for op in op_list]
    total = sum(costs, CostEstimate(0, 0, 0, 0))
    kernels = total.operations if kernel_count is None else int(kernel_count)
    if kernels < 0:
        raise ValueError("kernel_count must be non-negative")

    peak_f, peak_bw = hardware.peak_flops, hardware.memory_bandwidth
    fused_bytes = (
        estimate_fused_operations(op_list).total_bytes if assume_fusion else (total.total_bytes)
    )
    warnings: list[str] = []

    level0 = max(total.flops / peak_f, fused_bytes / peak_bw)
    level1 = max(total.flops / peak_f, total.total_bytes / peak_bw)
    level2 = max(level1, _per_op_bound(costs, peak_f, peak_bw))

    if calibration is not None:
        ach_f = min(calibration.achievable_flops, peak_f)
        ach_bw = min(calibration.achievable_bandwidth, peak_bw)
        if calibration.achievable_flops > peak_f or calibration.achievable_bandwidth > peak_bw:
            warnings.append(
                "calibrated throughput exceeds the HardwareSpec peak; clamped to peak "
                "(the spec may be too conservative)."
            )
        if calibration.flops_curve or calibration.bandwidth_curve:
            execution, dispatch = calibration.predict(costs, kernels, extra_kernels)
            level3 = max(level2, execution)
        else:
            level3 = max(level2, _per_op_bound(costs, ach_f, ach_bw))
            dispatch = kernels * calibration.dispatch_seconds + calibration.call_overhead_seconds
    else:
        level3 = level2
        dispatch = 0.0
        warnings.append(
            "no calibration supplied: peak_derate and dispatch are 0 and fold into residual; "
            "pass adapter.calibrate() or fit_cost_model() for a full attribution."
        )
    level4 = level3 + dispatch

    level5 = level4
    if isolated_seconds is not None:
        if calibration is None:
            raise ValueError("isolated_seconds requires a calibration")
        if len(isolated_seconds) != len(costs):
            raise ValueError("isolated_seconds must align with operations")
        from .validation import replay_prediction

        level5 = max(
            level4,
            replay_prediction(costs, isolated_seconds, calibration, kernels, extra_kernels),
        )

    residual_total = measured - level5
    jitter = 0.0
    if measurement.samples_seconds and residual_total > 0:
        jitter = min(residual_total, max(0.0, measured - min(measurement.samples_seconds)))
    residual = residual_total - jitter

    raw = [
        ("roofline_floor", level0),
        ("unfused_traffic", level1 - level0),
        ("serialization", level2 - level1),
        ("peak_derate", level3 - level2),
        ("dispatch", dispatch),
    ]
    if isolated_seconds is not None:
        raw.append(("kernel_inefficiency", level5 - level4))
    raw += [("timing_jitter", jitter), ("residual", residual)]
    explanations = dict(_EXPLANATIONS)
    if isolated_seconds is not None:
        explanations["residual"] = (
            (
                "Composition effects: the ops are each fast alone but slower together "
                "(framework glue, uncaptured ops, cache/TLB interference, memory allocation)."
            ),
            (
                "Look between kernels: allocator churn, host-side Python, layout conversions; "
                "capture with CUDA graphs or compile to remove glue."
            ),
        )
    components = tuple(
        GapComponent(name, seconds, seconds / measured, *explanations[name])
        for name, seconds in raw
    )
    return GapAttribution(
        measured_seconds=measured,
        components=components,
        kernel_count=kernels,
        calibration=calibration,
        overexplained=residual < -1e-12,
        bound_violated=level0 > measured,
        warnings=tuple(warnings),
    )
