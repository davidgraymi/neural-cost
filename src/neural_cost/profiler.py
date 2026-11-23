"""Runtime measurement primitives independent of ML frameworks."""

from collections.abc import Callable
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.median_seconds <= 0:
            raise ValueError("median_seconds must be positive")
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
    function: Callable[..., Any], *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
) -> Measurement:
    """Benchmark synchronous Python or CPU functions with warmup iterations.

    GPU frameworks should use their adapter's ``benchmark`` method so device
    synchronization and memory telemetry are handled correctly.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be at least one")
    for _ in range(warmup):
        function(*args, **kwargs)
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        function(*args, **kwargs)
        samples.append((perf_counter_ns() - start) / 1_000_000_000)
    return Measurement(median(samples), tuple(samples))
