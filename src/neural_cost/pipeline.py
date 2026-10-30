"""Profile a workload made of several functions called different numbers of times.

Real workloads are pipelines: one self-play search is tree bookkeeping plus
``N`` environment steps plus ``N`` network forwards.  :class:`Pipeline`
benchmarks each stage, multiplies by its call count, and compares the sum with
an end-to-end measurement.  The difference is reported as the unexplained
remainder (Python dispatch, loop overhead, host/device transfers).

>>> pipeline = Pipeline(
...     {
...         "tree": Stage(tree_fn, (state,)),
...         "env": Stage(env_step, (env_state, action), calls=16),
...         "net": Stage(net_fn, (obs,), calls=16),
...     },
...     total=Stage(search_fn, (state,)),
... )
>>> print(pipeline.run(adapter=JaxAdapter()).render())  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .analysis import GapAnalysis, analyze_gap
from .estimate import estimate_operations
from .hardware import HardwareSpec
from .operations import Operation
from .profiler import Measurement, benchmark


@dataclass(frozen=True)
class Stage:
    """One function in a pipeline, its arguments and how often it runs per pipeline call.

    ``calls`` may be fractional (e.g. a stage that runs every other step).
    ``operations`` optionally supplies the stage's operations for a per-stage
    roofline report; otherwise they are traced with the adapter passed to
    :meth:`Pipeline.run` when ``hardware`` is given.
    """

    function: Callable[..., Any]
    args: tuple[Any, ...] = ()
    calls: float = 1.0
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    operations: Sequence[Operation] | None = None

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("Stage.function must be callable")
        if self.calls < 0:
            raise ValueError("Stage.calls must be non-negative")
        object.__setattr__(self, "args", tuple(self.args))


StageLike = Stage | Callable[..., Any] | tuple[Any, ...]


def _as_stage(value: StageLike) -> Stage:
    """Accept ``Stage``, a bare callable, ``(fn, args)`` or ``(fn, args, calls)``."""
    if isinstance(value, Stage):
        return value
    if callable(value):
        return Stage(value)
    if isinstance(value, tuple) and value and callable(value[0]) and len(value) <= 3:
        function, *rest = value
        args = rest[0] if rest else ()
        calls = rest[1] if len(rest) > 1 else 1.0
        return Stage(function, args if isinstance(args, tuple) else (args,), calls)
    raise TypeError(f"cannot interpret {value!r} as a pipeline stage")


@dataclass(frozen=True, slots=True)
class StageResult:
    """Measured cost of one stage, scaled by its call count."""

    name: str
    calls: float
    measurement: Measurement
    share: float
    gap: GapAnalysis | None = None
    note: str | None = None

    @property
    def per_call_seconds(self) -> float:
        return self.measurement.median_seconds

    @property
    def total_seconds(self) -> float:
        return self.measurement.median_seconds * self.calls


@dataclass(frozen=True, slots=True)
class PipelineReport:
    """Per-stage time split and, with an end-to-end measurement, the unexplained remainder."""

    stages: tuple[StageResult, ...]
    measured_total: Measurement | None = None

    @property
    def stage_sum_seconds(self) -> float:
        return sum(stage.total_seconds for stage in self.stages)

    @property
    def reference_seconds(self) -> float:
        """The denominator for shares: the measured total when known, else the stage sum."""
        if self.measured_total is not None:
            return self.measured_total.median_seconds
        return self.stage_sum_seconds

    @property
    def unexplained_seconds(self) -> float | None:
        """Measured total minus the stage sum; negative when stages overlap or are overcounted."""
        if self.measured_total is None:
            return None
        return self.measured_total.median_seconds - self.stage_sum_seconds

    @property
    def unexplained_ratio(self) -> float | None:
        if self.unexplained_seconds is None:
            return None
        return self.unexplained_seconds / self.measured_total.median_seconds

    def dominant(self) -> StageResult | None:
        return max(self.stages, key=lambda stage: stage.total_seconds, default=None)

    def render(self) -> str:
        header = "Pipeline profile"
        if self.measured_total is not None:
            header += f" (measured total {self.measured_total.median_seconds * 1e3:.3f} ms"
            if self.measured_total.device:
                header += f" on {self.measured_total.device}"
            header += ")"
        width = max([len("unexplained"), *(len(stage.name) for stage in self.stages)])
        lines = [
            header,
            (
                f"  {'stage':<{width}}  {'calls':>7}  {'per call':>11}  {'total':>11}  "
                f"{'share':>7}  roofline"
            ),
        ]
        for stage in sorted(self.stages, key=lambda s: s.total_seconds, reverse=True):
            roofline = ""
            if stage.gap is not None:
                roofline = f"{stage.gap.efficiency:.1%} ({stage.gap.bottleneck}-bound)"
                if stage.gap.coverage is not None:
                    roofline += f", {stage.gap.coverage.fraction:.0%} coverage"
            elif stage.note:
                roofline = stage.note
            lines.append(
                f"  {stage.name:<{width}}  {stage.calls:>7g}  "
                f"{stage.per_call_seconds * 1e3:>8.3f} ms  {stage.total_seconds * 1e3:>8.3f} ms  "
                f"{stage.share:>7.1%}  {roofline}".rstrip()
            )
        unexplained = self.unexplained_seconds
        if unexplained is not None:
            lines.append(
                f"  {'unexplained':<{width}}  {'':>7}  {'':>11}  {unexplained * 1e3:>8.3f} ms  "
                f"{self.unexplained_ratio:>7.1%}  dispatch / loop overhead"
            )
            if unexplained < 0:
                lines.append(
                    "  note: stages sum to more than the measured total; stages may overlap "
                    "(asynchronous dispatch, caching) or call counts may be too high."
                )
        else:
            lines.append(
                f"  {'sum':<{width}}  {'':>7}  {'':>11}  {self.stage_sum_seconds * 1e3:>8.3f} ms"
            )
        return "\n".join(lines)


class Pipeline:
    """A named set of stages with call counts and an optional end-to-end function.

    Stages may be :class:`Stage` objects, bare callables, or ``(fn, args)`` /
    ``(fn, args, calls)`` tuples.  ``total`` is the end-to-end function (in
    the same forms) whose measurement the stage sum is compared with.
    """

    def __init__(self, stages: Mapping[str, StageLike], total: StageLike | None = None) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        self.stages = {name: _as_stage(stage) for name, stage in stages.items()}
        self.total = _as_stage(total) if total is not None else None

    def run(
        self,
        *,
        adapter: Any = None,
        hardware: HardwareSpec | None = None,
        warmup: int = 3,
        repeats: int = 10,
    ) -> PipelineReport:
        """Benchmark every stage (and the total) and build a :class:`PipelineReport`.

        ``adapter`` provides ``benchmark`` (e.g. ``JaxAdapter`` waits for async
        device work); without one, :func:`neural_cost.benchmark` is used.  With
        ``hardware``, stages that have ``operations`` (or that the adapter can
        trace) also get a roofline :class:`GapAnalysis`.
        """
        measure = adapter.benchmark if adapter is not None else benchmark
        measured: list[tuple[str, Stage, Measurement]] = []
        for name, stage in self.stages.items():
            result = measure(
                stage.function, *stage.args, warmup=warmup, repeats=repeats, **stage.kwargs
            )
            measured.append((name, stage, result))
        total = None
        if self.total is not None:
            total = measure(
                self.total.function,
                *self.total.args,
                warmup=warmup,
                repeats=repeats,
                **self.total.kwargs,
            )

        reference = (
            total.median_seconds
            if total is not None
            else sum(m.median_seconds * stage.calls for _, stage, m in measured)
        )
        results = []
        for name, stage, measurement in measured:
            gap, note = (
                self._roofline(stage, measurement, adapter, hardware)
                if hardware is not None
                else (None, None)
            )
            share = measurement.median_seconds * stage.calls / reference if reference > 0 else 0.0
            results.append(StageResult(name, stage.calls, measurement, share, gap, note))
        return PipelineReport(tuple(results), total)

    @staticmethod
    def _roofline(
        stage: Stage, measurement: Measurement, adapter: Any, hardware: HardwareSpec
    ) -> tuple[GapAnalysis | None, str | None]:
        operations = stage.operations
        coverage = None
        if operations is None:
            if adapter is None:
                return None, "no operations (pass Stage.operations or an adapter)"
            try:
                operations = list(adapter.operations(stage.function, stage.args))
            except Exception as error:  # noqa: BLE001 - tracing user code can fail arbitrarily
                return None, f"not traced: {type(error).__name__}: {error}"
            coverage = getattr(adapter, "coverage", None)
        operations = list(operations)
        if not operations:
            return None, "no modeled operations"
        estimate = estimate_operations(operations)
        return (
            analyze_gap(estimate, measurement, hardware, operations=operations, coverage=coverage),
            None,
        )
