"""CLI helpers for parsing sizes, numbers, formatting reports, and hardware resolution."""

from __future__ import annotations

import re

from neural_cost.hardware import HardwareSpec
from neural_cost.hardware_detect import DetectionResult, detect_hardware

_BYTE_UNITS: dict[str, int] = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
    "p": 1024**5,
    "pb": 1024**5,
    "pib": 1024**5,
}

_DTYPE_BYTES: dict[str, int] = {
    "fp32": 4,
    "float32": 4,
    "float": 4,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "bf16": 2,
    "bfloat16": 2,
    "int8": 1,
    "fp8": 1,
    "int4": 1,  # stored byte-aligned or packed
}


def parse_bytes(val: str | float) -> int:
    """Parse human byte string (e.g. '16GB', '512MB', '1.5TB', '1024') to integer bytes."""
    if isinstance(val, (int, float)):
        return int(val)

    s = val.strip().lower()
    if not s:
        raise ValueError("Empty byte size string")

    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-z]+)?$", s)
    if not match:
        raise ValueError(f"Cannot parse byte size: {val!r}")

    num_str, unit = match.groups()
    num = float(num_str)
    if unit is None:
        return int(num)

    if unit not in _BYTE_UNITS:
        raise ValueError(f"Unknown byte unit: {unit!r} in {val!r}")

    return int(num * _BYTE_UNITS[unit])


def parse_dtype_bytes(dtype_str: str) -> int:
    """Convert dtype string to byte size."""
    k = dtype_str.strip().lower()
    if k not in _DTYPE_BYTES:
        raise ValueError(
            f"Unsupported dtype {dtype_str!r}; choose from {sorted(_DTYPE_BYTES.keys())}"
        )
    return _DTYPE_BYTES[k]


def format_bytes(b: float) -> str:
    """Format bytes as human-readable string (e.g. 16.00 GB)."""
    val = float(b)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(val) < 1024.0 or unit == "PB":
            if unit == "B":
                return f"{int(val)} B"
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{b} B"


def format_flops(f: float) -> str:
    """Format FLOPs or FLOP/s as human-readable string (e.g. 989.00 TFLOP/s)."""
    val = float(f)
    for unit in ["FLOP", "KFLOP", "MFLOP", "GFLOP", "TFLOP", "PFLOP", "EFLOP"]:
        if abs(val) < 1e3 or unit == "EFLOP":
            return f"{val:.2f} {unit}"
        val /= 1e3
    return f"{f:.2e} FLOP"


def resolve_hardware(
    peak_flops: float | None = None,
    memory_bandwidth: float | None = None,
    device_name: str | None = None,
    bw_bench_mb: int = 256,
    benchmark_memory: bool = True,
) -> tuple[HardwareSpec, DetectionResult | None]:
    """Resolve HardwareSpec from explicit overrides or dynamic hardware detection."""
    detection: DetectionResult | None = None
    if peak_flops is None or memory_bandwidth is None:
        hw, det = detect_hardware(
            bandwidth_benchmark_mb=bw_bench_mb,
            benchmark_memory=benchmark_memory,
        )
        detection = det
        name = device_name or hw.device_name
        flops = peak_flops if peak_flops is not None else hw.peak_flops
        bw = memory_bandwidth if memory_bandwidth is not None else hw.memory_bandwidth
        return HardwareSpec(name, peak_flops=flops, memory_bandwidth=bw), detection

    name = device_name or "Custom_Hardware"
    return HardwareSpec(name, peak_flops=peak_flops, memory_bandwidth=memory_bandwidth), None
