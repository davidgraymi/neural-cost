"""Runtime measurement primitives independent of ML frameworks."""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from statistics import median, quantiles
from time import perf_counter_ns
from typing import Any


@dataclass(frozen=True, slots=True)
class Measurement:
    """A measured execution result."""

    median_seconds: float
    samples_seconds: tuple[float, ...]
    peak_memory_bytes: int | None = None
    device: str | None = None
    allocated_memory_bytes: int | None = None
    reserved_memory_bytes: int | None = None
    trace_event_count: int | None = None
    trace_device_time_seconds: float | None = None
    #: Calls timed together in each sample; every sample is already divided by it.
    calls_per_sample: int = 1
    #: Extra time a call takes when timed alone: one synchronised call minus the
    #: per-call time of back-to-back calls (device sync, completion latency,
    #: clock reads).
    sync_overhead_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.median_seconds <= 0:
            raise ValueError("median_seconds must be positive")
        if self.calls_per_sample < 1:
            raise ValueError("calls_per_sample must be at least one")
        if self.sync_overhead_seconds is not None and self.sync_overhead_seconds < 0:
            raise ValueError("sync_overhead_seconds must be non-negative")
        for name, value in (
            ("peak_memory_bytes", self.peak_memory_bytes),
            ("allocated_memory_bytes", self.allocated_memory_bytes),
            ("reserved_memory_bytes", self.reserved_memory_bytes),
            ("trace_event_count", self.trace_event_count),
            ("trace_device_time_seconds", self.trace_device_time_seconds),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def min_seconds(self) -> float:
        """Fastest sample: the run least disturbed by other work."""
        return min(self.samples_seconds, default=self.median_seconds)

    @property
    def relative_spread(self) -> float | None:
        """Interquartile range of the samples divided by their median.

        ``None`` with fewer than three samples, where a spread means little.
        """
        if len(self.samples_seconds) < 3:
            return None
        q1, _, q3 = quantiles(self.samples_seconds, n=4, method="inclusive")
        return (q3 - q1) / self.median_seconds

    @property
    def sync_overhead_ratio(self) -> float | None:
        """Sync overhead left in each per-call sample, as a fraction of the median."""
        if self.sync_overhead_seconds is None:
            return None
        return self.sync_overhead_seconds / self.calls_per_sample / self.median_seconds

    def sync_overhead_warning(self, threshold: float = 0.10) -> str | None:
        """A warning when synchronising every sample is a visible share of it."""
        ratio = self.sync_overhead_ratio
        if ratio is None or ratio <= threshold:
            return None
        return (
            f"Short samples: waiting for the device after each call adds about "
            f"{self.sync_overhead_seconds * 1e6:.1f} µs, {ratio:.0%} of the "
            f"{self.median_seconds * 1e6:.1f} µs median. That is real single-call latency, "
            f"but not kernel cost: pass min_sample_seconds (e.g. 1e-3) to benchmark() "
            f"to time back-to-back calls per sample."
        )

    def noise_warning(self, threshold: float = 0.10) -> str | None:
        """A warning when the samples vary too much to trust the median."""
        spread = self.relative_spread
        if spread is None or spread <= threshold:
            return None
        return (
            f"Noisy timing: the middle half of {len(self.samples_seconds)} samples spans "
            f"{spread:.0%} of the median (fastest {self.min_seconds * 1e3:.3f} ms, "
            f"median {self.median_seconds * 1e3:.3f} ms). Increase warmup and repeats, "
            f"close other work on the device, or fix GPU clocks before comparing."
        )


def benchmark(
    function: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    repeats: int = 10,
    min_sample_seconds: float | None = None,
    **kwargs: Any,
) -> Measurement:
    """Benchmark synchronous Python or CPU functions with warmup iterations.

    GPU frameworks should use their adapter's ``benchmark`` method so device
    synchronization and memory telemetry are handled correctly.  See
    :func:`timed_samples` for ``min_sample_seconds``.
    """
    samples, calls, overhead = timed_samples(
        lambda: function(*args, **kwargs),
        lambda output: None,
        warmup=warmup,
        repeats=repeats,
        min_sample_seconds=min_sample_seconds,
    )
    return Measurement(
        median(samples),
        tuple(samples),
        calls_per_sample=calls,
        sync_overhead_seconds=overhead,
    )


#: Upper bound on calls batched into one sample.
MAX_CALLS_PER_SAMPLE = 100_000


def timed_samples(
    call: Callable[[], Any],
    wait: Callable[[Any], None],
    *,
    warmup: int,
    repeats: int,
    min_sample_seconds: float | None = None,
    hooks: Any = None,
) -> tuple[list[float], int, float]:
    """Time ``call``; return ``(per-call samples, calls per sample, sync overhead)``.

    ``wait(output)`` must return once the work behind ``output`` (and all work
    queued before it) has finished.  By default each sample times one call
    and one wait: the end-to-end latency of a single invocation.

    Short calls are dominated by that wait: a device sync and, on GPUs, the
    latency of learning that the last kernel finished.  With
    ``min_sample_seconds`` each sample instead times ``k`` back-to-back calls
    and one wait, with ``k`` chosen so a sample lasts at least that long, and
    reports the elapsed time divided by ``k``.  That amortizes the wait and,
    on asynchronous devices, lets host dispatch overlap execution, so it
    measures steady-state per-call cost rather than isolated latency.

    The sync overhead ``w`` comes from the median single synchronised call
    (``c + w``: the timed samples themselves in the default mode, else five
    probe calls) and the median of three back-to-back batches of ``n`` calls
    (``n·c + w``) run before timing.  ``hooks`` may define ``start()``, ``stop()`` (after
    the calls, before the wait) and ``done()`` (after the wait).
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be at least one")
    if min_sample_seconds is not None and min_sample_seconds < 0:
        raise ValueError("min_sample_seconds must be non-negative")
    for _ in range(warmup):
        wait(call())
    # Model: a lone call costs c + w, a batch of n back-to-back calls n·c + w,
    # where w is the wait.  Medians keep one slow probe from skewing either.
    # Both probes use the same loop as the timed samples.
    once = median(_time_batch(call, wait, 1) for _ in range(5))
    probe_calls = max(2, min(1_000, ceil(_PROBE_SECONDS / max(once, 1e-9))))
    batch = median(_time_batch(call, wait, probe_calls) for _ in range(3))
    overhead = max(0.0, (probe_calls * once - batch) / (probe_calls - 1))
    batched = max(batch / probe_calls, 1e-9)
    calls = 1
    if min_sample_seconds:
        calls = max(1, min(MAX_CALLS_PER_SAMPLE, ceil(min_sample_seconds / max(batched, 1e-9))))
    start_hook = getattr(hooks, "start", None)
    stop_hook = getattr(hooks, "stop", None)
    done_hook = getattr(hooks, "done", None)
    samples: list[float] = []
    for _ in range(repeats):
        if start_hook:
            start_hook()
        output = None
        start = perf_counter_ns()
        for _ in range(calls):
            output = call()
        if stop_hook:
            stop_hook()
        wait(output)
        samples.append((perf_counter_ns() - start) / 1_000_000_000 / calls)
        if done_hook:
            done_hook()
    if calls == 1:
        # The timed samples are many more single calls than the probe: use
        # their median for ``c + w``, which is far less noisy for tiny calls.
        overhead = max(0.0, (probe_calls * median(samples) - batch) / (probe_calls - 1))
    return samples, calls, overhead


#: Target duration of the back-to-back batch used to estimate sync overhead.
_PROBE_SECONDS = 1e-3


def _time_batch(call: Callable[[], Any], wait: Callable[[Any], None], calls: int) -> float:
    output = None
    start = perf_counter_ns()
    for _ in range(calls):
        output = call()
    wait(output)
    return (perf_counter_ns() - start) / 1_000_000_000
