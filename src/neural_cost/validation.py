"""Validate cost models against measurements, with honest out-of-sample error.

Two different claims need two different checks:

1. **The roofline is a lower bound.**  A bound is falsified by any
   measurement that runs *faster* than it.  :func:`validate_lower_bound`
   counts violations and reports tightness (bound / measured), plus the rank
   correlation between bound and measurement — whether the bound orders
   workloads correctly even when it is loose.

2. **A calibrated model predicts runtime.**  :func:`fit_cost_model` fits

       T ≈ c₀ + α·kernels + FLOPs / P_eff + bytes / BW_eff

   by non-negative least squares (every coefficient is a physical rate or
   overhead, so none may be negative).  :func:`validate_predictor` scores it
   with leave-one-out cross-validation: each workload is predicted by a model
   fit on all *other* workloads, so the reported error is generalization
   error, not goodness of fit.

:func:`run_validation_suite` runs both against real models through any
adapter, and :class:`ValidationReport` renders Markdown / JSON for
publication.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

from .attribution import HardwareCalibration
from .estimate import CostEstimate, estimate_fused_operations, estimate_operation
from .hardware import HardwareSpec
from .memory import estimate_memory
from .operations import Operation

# ----------------------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """One measured workload with its static cost."""

    name: str
    flops: int
    bytes: int
    kernels: int
    measured_seconds: float
    lower_bound_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    calibrated_seconds: float | None = None
    replay_seconds: float | None = None

    @classmethod
    def from_operations(
        cls,
        name: str,
        operations: Sequence[Operation],
        measured_seconds: float,
        hardware: HardwareSpec,
        *,
        kernels: int | None = None,
        fused: bool = False,
        metadata: dict[str, Any] | None = None,
        calibration: HardwareCalibration | None = None,
        cache_aware: bool = True,
        isolated_seconds: Sequence[float | None] | None = None,
        extra_kernels: Sequence[tuple[int, bool]] | None = None,
    ) -> ValidationCase:
        """Build a case; the lower bound is the per-op (serialized) roofline sum.

        ``fused=True`` uses fused traffic and the aggregate roofline instead,
        the right bound for compiled (jit / torch.compile) execution.  With a
        ``calibration`` the case also carries a *zero-shot* prediction —
        per-op roofline at achievable throughput plus dispatch overhead — that
        uses no fitting on measured model runtimes at all.  ``isolated_seconds``
        (from an adapter's ``replay``) adds an *op-replay* prediction: the sum
        of isolated per-op times plus per-call overhead and dispatch for
        kernels the adapter did not capture.
        """
        if measured_seconds <= 0:
            raise ValueError("measured_seconds must be positive")
        costs = [estimate_operation(op) for op in operations]
        total = sum(costs, CostEstimate(0, 0, 0, 0))
        # Repeated (warm) inference keeps a small working set on chip, so DRAM
        # bandwidth is not a valid bound for it.  When parameters plus the
        # largest activation fit a cache level, bound traffic by that level.
        memory = estimate_memory(operations)
        working_set = memory.parameter_bytes + 2 * memory.minimum_peak_activation_bytes
        resident = hardware.find_resident_cache(working_set) if cache_aware else None
        bandwidth = resident.bandwidth if resident is not None else hardware.memory_bandwidth
        if fused:
            nbytes = estimate_fused_operations(operations).total_bytes
            bound = max(total.flops / hardware.peak_flops, nbytes / bandwidth)
        else:
            nbytes = total.total_bytes
            bound = sum(
                max(c.flops / hardware.peak_flops, c.total_bytes / bandwidth) for c in costs
            )
        metadata = {
            **(metadata or {}),
            "working_set_bytes": working_set,
            "bound_memory_level": resident.name if resident is not None else "DRAM",
        }
        kernel_count = total.operations if kernels is None else kernels
        calibrated = None
        if calibration is not None:
            if fused:  # one compiled program: a single kernel with fused traffic
                execution, overhead = calibration.predict(
                    [CostEstimate(total.flops, nbytes, 0, 1)], kernel_count, extra_kernels
                )
            else:
                execution, overhead = calibration.predict(costs, kernel_count, extra_kernels)
            calibrated = execution + overhead
        replayed = None
        if isolated_seconds is not None and calibration is not None:
            replayed = replay_prediction(
                costs, isolated_seconds, calibration, kernel_count, extra_kernels
            )
        return cls(
            name,
            total.flops,
            nbytes,
            kernel_count,
            measured_seconds,
            bound,
            metadata,
            calibrated,
            replayed,
        )


# ------------------------------------------------------------------------ statistics


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (ties averaged); NaN when undefined."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy) if vx and vy else float("nan")


def _geomean(values: Sequence[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


# ------------------------------------------------------------------------------ NNLS


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-14:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                factor = a[r][col] / a[col][col]
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def nnls(rows: Sequence[Sequence[float]], target: Sequence[float]) -> list[float]:
    """Lawson–Hanson non-negative least squares for small dense problems."""
    m = len(rows)
    n = len(rows[0]) if rows else 0
    if m == 0 or n == 0:
        return [0.0] * n
    # Column scaling keeps seconds-per-FLOP and seconds-per-kernel comparable.
    scale = [math.sqrt(sum(r[j] ** 2 for r in rows)) or 1.0 for j in range(n)]
    a = [[r[j] / scale[j] for j in range(n)] for r in rows]
    b = list(target)

    def gradient(x: list[float]) -> list[float]:
        resid = [b[i] - sum(a[i][j] * x[j] for j in range(n)) for i in range(m)]
        return [sum(a[i][j] * resid[i] for i in range(m)) for j in range(n)]

    def lstsq(active: list[int]) -> list[float] | None:
        gram = [[sum(a[i][p] * a[i][q] for i in range(m)) for q in active] for p in active]
        rhs = [sum(a[i][p] * b[i] for i in range(m)) for p in active]
        return _solve(gram, rhs)

    x = [0.0] * n
    passive: list[int] = []
    for _ in range(3 * n + 10):
        w = gradient(x)
        candidates = [j for j in range(n) if j not in passive and w[j] > 1e-12]
        if not candidates:
            break
        passive.append(max(candidates, key=lambda j: w[j]))
        while True:
            z_passive = lstsq(passive)
            if z_passive is None:
                passive.pop()
                break
            z = [0.0] * n
            for idx, j in enumerate(passive):
                z[j] = z_passive[idx]
            if all(z[j] > 0 for j in passive):
                x = z
                break
            alpha = (
                min(x[j] / (x[j] - z[j]) for j in passive if z[j] <= 0 and x[j] - z[j] != 0)
                if any(z[j] <= 0 and x[j] - z[j] != 0 for j in passive)
                else 0.0
            )
            x = [x[j] + alpha * (z[j] - x[j]) for j in range(n)]
            passive = [j for j in passive if x[j] > 1e-15]
    return [x[j] / scale[j] for j in range(n)]


# ------------------------------------------------------------------------ cost model


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fitted additive model T = c₀ + α·kernels + FLOPs·s_f + bytes·s_b."""

    call_overhead_seconds: float
    dispatch_seconds: float
    seconds_per_flop: float
    seconds_per_byte: float
    fitted_cases: int

    @property
    def effective_flops(self) -> float:
        return 1.0 / self.seconds_per_flop if self.seconds_per_flop > 0 else float("inf")

    @property
    def effective_bandwidth(self) -> float:
        return 1.0 / self.seconds_per_byte if self.seconds_per_byte > 0 else float("inf")

    def predict(self, flops: float, nbytes: float, kernels: float) -> float:
        return (
            self.call_overhead_seconds
            + kernels * self.dispatch_seconds
            + flops * self.seconds_per_flop
            + nbytes * self.seconds_per_byte
        )

    def predict_case(self, case: ValidationCase) -> float:
        return self.predict(case.flops, case.bytes, case.kernels)

    def to_calibration(self, hardware: HardwareSpec) -> HardwareCalibration:
        """Express the fit as a calibration usable by :func:`attribute_gap`.

        Rates the fit could not identify (coefficient 0) fall back to peak.
        """
        flops = self.effective_flops if math.isfinite(self.effective_flops) else hardware.peak_flops
        bw = (
            self.effective_bandwidth
            if math.isfinite(self.effective_bandwidth)
            else hardware.memory_bandwidth
        )
        return HardwareCalibration(
            achievable_flops=flops,
            achievable_bandwidth=bw,
            dispatch_seconds=self.dispatch_seconds,
            call_overhead_seconds=self.call_overhead_seconds,
            source=f"NNLS fit on {self.fitted_cases} measurements",
        )


def fit_cost_model(cases: Sequence[ValidationCase], *, relative: bool = True) -> CostModel:
    """Fit :class:`CostModel` coefficients by non-negative least squares.

    ``relative=True`` (default) minimizes *relative* error by dividing each
    row by its measured time, so microsecond and second-scale workloads are
    weighted equally — the natural objective when workloads span orders of
    magnitude.
    """
    if not cases:
        raise ValueError("fit_cost_model requires at least one case")
    rows, target = [], []
    for case in cases:
        weight = 1.0 / case.measured_seconds if relative else 1.0
        rows.append([weight, case.kernels * weight, case.flops * weight, case.bytes * weight])
        target.append(case.measured_seconds * weight)
    c0, alpha, s_f, s_b = nnls(rows, target)
    return CostModel(c0, alpha, s_f, s_b, len(cases))


# --------------------------------------------------------------------------- reports


@dataclass(frozen=True, slots=True)
class CaseResult:
    name: str
    measured_seconds: float
    lower_bound_seconds: float
    tightness: float
    predicted_seconds: float | None
    relative_error: float | None
    metadata: dict[str, Any]
    zero_shot_seconds: float | None = None
    zero_shot_error: float | None = None
    replay_seconds: float | None = None
    replay_error: float | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Bound validity and out-of-sample prediction accuracy over a workload set."""

    title: str
    cases: tuple[CaseResult, ...]
    bound_violations: int
    tightness_median: float
    tightness_min: float
    tightness_max: float
    bound_rank_correlation: float
    loo_mape: float | None
    loo_median_ape: float | None
    loo_bias: float | None
    loo_within_25pct: float | None
    loo_rank_correlation: float | None
    roofline_only_mape: float
    full_model: CostModel | None
    hardware: str
    zero_shot_mape: float | None = None
    zero_shot_median_ape: float | None = None
    zero_shot_rank_correlation: float | None = None
    calibration_source: str | None = None
    replay_mape: float | None = None
    replay_median_ape: float | None = None
    replay_rank_correlation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        def default(value: Any) -> Any:
            return None if isinstance(value, float) and not math.isfinite(value) else str(value)

        return json.dumps(self.to_dict(), default=default, **kwargs)

    def render(self) -> str:
        lines = [f"Validation: {self.title} ({len(self.cases)} workloads, {self.hardware})"]
        lines.append(
            f"  lower bound: {self.bound_violations} violations; tightness median "
            f"{self.tightness_median:.1%} (min {self.tightness_min:.1%}, max "
            f"{self.tightness_max:.1%}); rank corr {self.bound_rank_correlation:.3f}"
        )
        lines.append(f"  roofline-as-predictor MAPE: {self.roofline_only_mape:.1%}")
        if self.zero_shot_mape is not None:
            lines.append(
                f"  zero-shot (microbenchmark-calibrated): MAPE {self.zero_shot_mape:.1%}, "
                f"median APE {self.zero_shot_median_ape:.1%}, rank corr "
                f"{self.zero_shot_rank_correlation:.3f}"
            )
        if self.replay_mape is not None:
            lines.append(
                f"  op-replay (isolated kernels composed): MAPE {self.replay_mape:.1%}, "
                f"median APE {self.replay_median_ape:.1%}, rank corr "
                f"{self.replay_rank_correlation:.3f}"
            )
        if self.loo_mape is not None:
            lines.append(
                f"  calibrated model (leave-one-out): MAPE {self.loo_mape:.1%}, median APE "
                f"{self.loo_median_ape:.1%}, bias ×{self.loo_bias:.3f}, within ±25%: "
                f"{self.loo_within_25pct:.0%}, rank corr {self.loo_rank_correlation:.3f}"
            )
        if self.full_model is not None:
            m = self.full_model
            lines.append(
                f"  fit (all data): overhead {m.call_overhead_seconds * 1e6:.1f} µs/call, "
                f"{m.dispatch_seconds * 1e6:.2f} µs/kernel, {m.effective_flops / 1e9:.1f} "
                f"GFLOP/s, {m.effective_bandwidth / 1e9:.1f} GB/s effective"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            f"### {self.title}",
            "",
            f"Hardware: {self.hardware}. {len(self.cases)} workloads.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Lower-bound violations | {self.bound_violations} / {len(self.cases)} |",
            (
                f"| Bound tightness (median, min–max) | {self.tightness_median:.1%} "
                f"({self.tightness_min:.1%}–{self.tightness_max:.1%}) |"
            ),
            f"| Bound rank correlation (Spearman) | {self.bound_rank_correlation:.3f} |",
            f"| Roofline-as-predictor MAPE | {self.roofline_only_mape:.1%} |",
        ]
        if self.zero_shot_mape is not None:
            lines += [
                f"| Zero-shot MAPE (microbenchmarks only, no fit) | {self.zero_shot_mape:.1%} |",
                f"| Zero-shot median APE | {self.zero_shot_median_ape:.1%} |",
                f"| Zero-shot rank correlation | {self.zero_shot_rank_correlation:.3f} |",
            ]
        if self.replay_mape is not None:
            lines += [
                f"| Op-replay MAPE (isolated kernels, no fit) | **{self.replay_mape:.1%}** |",
                f"| Op-replay median APE | {self.replay_median_ape:.1%} |",
                f"| Op-replay rank correlation | {self.replay_rank_correlation:.3f} |",
            ]
        if self.loo_mape is not None:
            lines += [
                f"| Calibrated model LOO MAPE | **{self.loo_mape:.1%}** |",
                f"| Calibrated model LOO median APE | {self.loo_median_ape:.1%} |",
                f"| Calibrated model bias (geomean pred/meas) | ×{self.loo_bias:.3f} |",
                f"| Within ±25% | {self.loo_within_25pct:.0%} |",
                f"| Calibrated rank correlation | {self.loo_rank_correlation:.3f} |",
            ]
        if self.full_model is not None:
            m = self.full_model
            lines += [
                f"| Fitted call overhead | {m.call_overhead_seconds * 1e6:.1f} µs |",
                f"| Fitted per-kernel dispatch | {m.dispatch_seconds * 1e6:.2f} µs |",
                f"| Fitted effective compute | {m.effective_flops / 1e9:.1f} GFLOP/s |",
                f"| Fitted effective bandwidth | {m.effective_bandwidth / 1e9:.1f} GB/s |",
            ]
        lines += [
            "",
            (
                "| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error "
                "| Op-replay (ms) | Error | LOO fit (ms) | Error |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

        def fmt(value: float | None, pattern: str) -> str:
            return format(value, pattern) if value is not None else "–"

        for c in self.cases:
            zs = c.zero_shot_seconds * 1e3 if c.zero_shot_seconds is not None else None
            pred = c.predicted_seconds * 1e3 if c.predicted_seconds is not None else None
            rp = c.replay_seconds * 1e3 if c.replay_seconds is not None else None
            lines.append(
                f"| {c.name} | {c.measured_seconds * 1e3:.3f} | "
                f"{c.lower_bound_seconds * 1e3:.4f} | {c.tightness:.1%} | {fmt(zs, '.3f')} | "
                f"{fmt(c.zero_shot_error, '+.1%')} | {fmt(rp, '.3f')} | "
                f"{fmt(c.replay_error, '+.1%')} | {fmt(pred, '.3f')} | "
                f"{fmt(c.relative_error, '+.1%')} |"
            )
        return "\n".join(lines)


def validate_lower_bound(cases: Sequence[ValidationCase], *, tolerance: float = 0.0) -> dict:
    """Count bound violations (measured < bound × (1 − tolerance)) and tightness stats."""
    if not cases:
        raise ValueError("no cases")
    tight = [c.lower_bound_seconds / c.measured_seconds for c in cases]
    violations = sum(c.measured_seconds < c.lower_bound_seconds * (1 - tolerance) for c in cases)
    return {
        "violations": violations,
        "tightness_median": median(tight),
        "tightness_min": min(tight),
        "tightness_max": max(tight),
        "rank_correlation": spearman(
            [c.lower_bound_seconds for c in cases], [c.measured_seconds for c in cases]
        ),
    }


def validate_predictor(
    cases: Sequence[ValidationCase],
    *,
    title: str = "workloads",
    hardware: str = "unknown",
    tolerance: float = 0.0,
    calibration_source: str | None = None,
) -> ValidationReport:
    """Score the lower bound and a leave-one-out calibrated cost model."""
    bound = validate_lower_bound(cases, tolerance=tolerance)
    predictions: list[float | None] = [None] * len(cases)
    if len(cases) >= 3:
        for i in range(len(cases)):
            train = [c for j, c in enumerate(cases) if j != i]
            predictions[i] = fit_cost_model(train).predict_case(cases[i])

    results, errors, zs_errors, rp_errors = [], [], [], []
    for case, pred in zip(cases, predictions, strict=True):
        err = (pred - case.measured_seconds) / case.measured_seconds if pred is not None else None
        if err is not None:
            errors.append(err)
        zs = case.calibrated_seconds
        zs_err = (zs - case.measured_seconds) / case.measured_seconds if zs is not None else None
        if zs_err is not None:
            zs_errors.append(zs_err)
        rp = case.replay_seconds
        rp_err = (rp - case.measured_seconds) / case.measured_seconds if rp is not None else None
        if rp_err is not None:
            rp_errors.append(rp_err)
        results.append(
            CaseResult(
                case.name,
                case.measured_seconds,
                case.lower_bound_seconds,
                case.lower_bound_seconds / case.measured_seconds,
                pred,
                err,
                case.metadata,
                zs,
                zs_err,
                rp,
                rp_err,
            )
        )
    zs_cases = [c for c in cases if c.calibrated_seconds is not None]
    rp_cases = [c for c in cases if c.replay_seconds is not None]
    roofline_mape = sum(
        abs(c.lower_bound_seconds - c.measured_seconds) / c.measured_seconds for c in cases
    ) / len(cases)
    has_loo = bool(errors)
    return ValidationReport(
        title=title,
        cases=tuple(results),
        bound_violations=bound["violations"],
        tightness_median=bound["tightness_median"],
        tightness_min=bound["tightness_min"],
        tightness_max=bound["tightness_max"],
        bound_rank_correlation=bound["rank_correlation"],
        loo_mape=sum(abs(e) for e in errors) / len(errors) if has_loo else None,
        loo_median_ape=median(abs(e) for e in errors) if has_loo else None,
        loo_bias=_geomean([1 + e for e in errors]) if has_loo else None,
        loo_within_25pct=sum(abs(e) <= 0.25 for e in errors) / len(errors) if has_loo else None,
        loo_rank_correlation=spearman(
            [p for p in predictions if p is not None],
            [c.measured_seconds for c, p in zip(cases, predictions, strict=True) if p is not None],
        )
        if has_loo
        else None,
        roofline_only_mape=roofline_mape,
        full_model=fit_cost_model(cases) if len(cases) >= 1 else None,
        hardware=hardware,
        zero_shot_mape=sum(abs(e) for e in zs_errors) / len(zs_errors) if zs_errors else None,
        zero_shot_median_ape=median(abs(e) for e in zs_errors) if zs_errors else None,
        zero_shot_rank_correlation=spearman(
            [c.calibrated_seconds for c in zs_cases], [c.measured_seconds for c in zs_cases]
        )
        if zs_errors
        else None,
        calibration_source=calibration_source,
        replay_mape=sum(abs(e) for e in rp_errors) / len(rp_errors) if rp_errors else None,
        replay_median_ape=median(abs(e) for e in rp_errors) if rp_errors else None,
        replay_rank_correlation=spearman(
            [c.replay_seconds for c in rp_cases], [c.measured_seconds for c in rp_cases]
        )
        if rp_errors
        else None,
    )


def replay_prediction(
    costs: Sequence[CostEstimate],
    isolated_seconds: Sequence[float | None],
    calibration: HardwareCalibration,
    kernel_count: int | None = None,
    extra_kernels: Sequence[tuple[int, bool]] | None = None,
) -> float:
    """Compose isolated per-op times into a whole-model prediction.

    Replayed times already include each kernel's own dispatch.  Operations
    that could not be replayed fall back to the calibrated size-aware model;
    per-call overhead and dispatch for uncaptured kernels are added on top.
    """
    if len(isolated_seconds) != len(costs):
        raise ValueError("isolated_seconds must align with operations")
    total = calibration.call_overhead_seconds
    captured = 0
    for cost, isolated in zip(costs, isolated_seconds, strict=True):
        captured += max(1, cost.operations)
        if isolated is not None:
            total += isolated
        else:
            execution, overhead = calibration.predict([cost])
            total += execution + overhead - calibration.call_overhead_seconds
    if extra_kernels is not None:
        execution, overhead = calibration.predict([], None, extra_kernels)
        total += execution + overhead - calibration.call_overhead_seconds
    elif kernel_count is not None:
        total += max(0, kernel_count - captured) * calibration.dispatch_seconds
    return total


# --------------------------------------------------------------------------- harness


@dataclass(frozen=True, slots=True)
class Workload:
    """A named model + example inputs to validate through an adapter.

    ``runner`` is what gets timed (defaults to ``model``), e.g. a jitted or
    compiled wrapper; ``kernels`` overrides the launch count (1 for a jitted
    function executed as a single XLA program).
    """

    name: str
    model: Any
    inputs: tuple[Any, ...]
    runner: Callable[..., Any] | None = None
    kernels: int | None = None
    fused: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def run_validation_suite(
    adapter: Any,
    workloads: Sequence[Workload],
    hardware: HardwareSpec,
    *,
    title: str | None = None,
    warmup: int = 5,
    repeats: int = 30,
    calibration: HardwareCalibration | None = None,
    replay: bool = True,
    warmup_seconds: float = 0.0,
) -> ValidationReport:
    """Trace, benchmark and validate every workload through ``adapter``.

    When ``calibration`` is given (e.g. from ``adapter.calibrate()``) each
    workload also gets a zero-shot prediction; kernel counts come from
    ``workload.kernels`` or, if the adapter offers it, ``adapter.count_kernels``.
    With ``replay`` and an adapter that offers ``replay(operations, device)``,
    each workload also gets an op-replay prediction.  ``warmup_seconds`` runs
    each workload for at least that long first, letting GPU clocks ramp up.
    """
    cases = []
    for workload in workloads:
        operations = adapter.operations(workload.model, workload.inputs)
        runner = workload.runner or workload.model
        kernels = workload.kernels
        extra = None
        if kernels is None and hasattr(adapter, "trace_kernels"):
            records = adapter.trace_kernels(runner, workload.inputs)
            kernels = len(records)
            extra = [(r.bytes, r.python_dispatch) for r in records if not r.heavy]
        elif kernels is None and hasattr(adapter, "count_kernels"):
            kernels = adapter.count_kernels(runner, workload.inputs)
        if warmup_seconds > 0:
            warmed = 0.0
            while warmed < warmup_seconds:
                warm = adapter.benchmark(runner, *workload.inputs, warmup=0, repeats=5)
                warmed += sum(warm.samples_seconds)
        measurement = adapter.benchmark(runner, *workload.inputs, warmup=warmup, repeats=repeats)
        isolated = None
        if replay and calibration is not None and hasattr(adapter, "replay"):
            isolated = adapter.replay(operations, workload.metadata.get("device"))
        cases.append(
            ValidationCase.from_operations(
                workload.name,
                operations,
                measurement.median_seconds,
                hardware,
                kernels=kernels,
                fused=workload.fused,
                metadata={**workload.metadata, "min_seconds": min(measurement.samples_seconds)},
                calibration=calibration,
                isolated_seconds=isolated,
                extra_kernels=extra,
            )
        )
    return validate_predictor(
        cases,
        title=title or getattr(adapter, "name", "adapter"),
        hardware=hardware.name,
        calibration_source=calibration.source if calibration else None,
    )
