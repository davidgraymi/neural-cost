"""Hardware descriptions used by the roofline model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheSpec:
    """A cache level's performance and capacity limits."""

    name: str
    bandwidth: float  # bytes/s
    capacity: int  # bytes

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """A device's relevant performance limits.

    Values are expressed in SI units: FLOP/s, bytes/s, and bytes.  Use a
    precision-specific compute peak (for example, FP16 tensor-core peak) when
    analyzing a workload at that precision.
    """

    name: str
    peak_flops: float
    memory_bandwidth: float
    memory_capacity: int | None = None
    caches: tuple[CacheSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.peak_flops <= 0 or self.memory_bandwidth <= 0:
            raise ValueError("peak_flops and memory_bandwidth must be positive")
        if not isinstance(self.caches, tuple):
            object.__setattr__(self, "caches", tuple(self.caches))

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) at the compute/memory boundary."""
        return self.peak_flops / self.memory_bandwidth

    @property
    def device_name(self) -> str:
        """Alias for name."""
        return self.name

    def get_cache(self, name: str) -> CacheSpec | None:
        """Find a cache level by name."""
        for cache in self.caches:
            if cache.name.lower() == name.lower():
                return cache
        return None

    def find_resident_cache(self, working_set_bytes: int) -> CacheSpec | None:
        """Return the fastest (smallest fitting) cache level that holds working_set_bytes."""
        fitting = [c for c in self.caches if c.capacity >= working_set_bytes]
        if not fitting:
            return None
        return min(fitting, key=lambda c: c.capacity)
