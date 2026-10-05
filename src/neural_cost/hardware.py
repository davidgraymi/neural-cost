from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterconnectSpec:
    """Interconnect bandwidth, latency, and transmission limits (e.g. NVLink, PCIe, InfiniBand)."""

    name: str
    bandwidth: float  # bytes/s (effective transfer rate)
    latency_seconds: float = 1e-6  # seconds (default 1.0 microsecond)

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")

    def transfer_time_seconds(self, bytes_to_send: int) -> float:
        """Hockney alpha-beta network model: T = alpha + beta * S."""
        if bytes_to_send <= 0:
            return 0.0
        return self.latency_seconds + (bytes_to_send / self.bandwidth)


INTERCONNECT_PRESETS: dict[str, InterconnectSpec] = {
    # Intra-node NVLink
    "nvlink3": InterconnectSpec("NVLink 3 (A100)", bandwidth=600e9, latency_seconds=1.0e-6),
    "nvlink4": InterconnectSpec("NVLink 4 (H100)", bandwidth=900e9, latency_seconds=0.8e-6),
    "nvlink5": InterconnectSpec("NVLink 5 (B200)", bandwidth=1800e9, latency_seconds=0.5e-6),
    # Host-accelerator PCIe
    "pcie_gen4": InterconnectSpec("PCIe Gen4 x16", bandwidth=31.5e9, latency_seconds=2.0e-6),
    "pcie_gen5": InterconnectSpec("PCIe Gen5 x16", bandwidth=63.0e9, latency_seconds=1.5e-6),
    # Inter-node InfiniBand & Ethernet
    "infiniband_hdr": InterconnectSpec(
        "InfiniBand HDR (200Gb/s)", bandwidth=25e9, latency_seconds=2.0e-6
    ),
    "infiniband_ndr": InterconnectSpec(
        "InfiniBand NDR (400Gb/s)", bandwidth=50e9, latency_seconds=1.5e-6
    ),
    "infiniband_ndr800": InterconnectSpec(
        "InfiniBand NDR800 (800Gb/s)", bandwidth=100e9, latency_seconds=1.2e-6
    ),
    "ethernet_100g": InterconnectSpec("100GbE RoCEv2", bandwidth=12.5e9, latency_seconds=5.0e-6),
    "ethernet_400g": InterconnectSpec("400GbE RoCEv2", bandwidth=50e9, latency_seconds=3.0e-6),
}


def get_interconnect_preset(name: str) -> InterconnectSpec:
    """Retrieve an interconnect specification preset by name (case-insensitive)."""
    norm = name.strip().lower().replace("-", "_").replace(" ", "_")
    if norm in INTERCONNECT_PRESETS:
        return INTERCONNECT_PRESETS[norm]
    available = ", ".join(sorted(INTERCONNECT_PRESETS.keys()))
    raise ValueError(f"Unknown interconnect preset '{name}'. Available: {available}")


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


@dataclass(frozen=True, slots=True)
class ClusterTopology:
    """Multi-node / multi-accelerator cluster configuration."""

    device: HardwareSpec
    num_nodes: int = 1
    devices_per_node: int = 8
    intra_node: InterconnectSpec | None = None
    inter_node: InterconnectSpec | None = None

    def __post_init__(self) -> None:
        if self.num_nodes <= 0 or self.devices_per_node <= 0:
            raise ValueError("num_nodes and devices_per_node must be positive")

    @property
    def total_devices(self) -> int:
        """Total accelerator count in cluster."""
        return self.num_nodes * self.devices_per_node

    @property
    def total_peak_flops(self) -> float:
        """Aggregate theoretical peak compute across all cluster accelerators."""
        return self.total_devices * self.device.peak_flops

    @property
    def total_memory_capacity(self) -> int | None:
        """Aggregate memory capacity across all cluster accelerators in bytes."""
        if self.device.memory_capacity is None:
            return None
        return self.total_devices * self.device.memory_capacity
