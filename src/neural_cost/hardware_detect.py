"""Automatic hardware detection and memory-bandwidth measurement.

Provides :func:`detect_hardware` which returns a :class:`~.hardware.HardwareSpec`
populated with values derived from:

1. A STREAM-style NumPy copy benchmark (always available) to measure
   *effective* memory bandwidth.
2. ``system_profiler SPHardwareDataType`` on macOS to identify Apple Silicon
   chips and look up their published peak FP32 compute throughput.
3. ``nvidia-smi`` on systems with NVIDIA GPUs (CUDA peak FLOPs).
4. Sensible CPU-level fallbacks (logical cores × clock × scalar FMA factor).

All values are best-effort estimates for roofline purposes.  Pass explicit
``--peak-flops`` / ``--memory-bandwidth`` arguments to the comparison script
to override them with manufacturer data for your exact SKU.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from .hardware import CacheSpec, HardwareSpec

# ---------------------------------------------------------------------------
# Apple Silicon chip table
# Published single-chip FP32 peak TFLOP/s, memory bandwidth (GB/s), and SLC caches.
# Sources: Apple Developer documentation and AnandTech / Chips and Cheese.
# ---------------------------------------------------------------------------
_APPLE_CHIP_TABLE: dict[str, tuple[float, float, tuple[CacheSpec, ...]]] = {
    # (peak_tflops_fp32, bandwidth_gb_s, (CacheSpec(...), ...))
    "M1": (2.6, 68.25, (CacheSpec("SLC", 200e9, 8 * 1024 * 1024),)),
    "M1 Pro": (5.2, 200.0, (CacheSpec("SLC", 400e9, 24 * 1024 * 1024),)),
    "M1 Max": (10.4, 400.0, (CacheSpec("SLC", 800e9, 48 * 1024 * 1024),)),
    "M1 Ultra": (21.2, 800.0, (CacheSpec("SLC", 1600e9, 96 * 1024 * 1024),)),
    "M2": (3.6, 100.0, (CacheSpec("SLC", 250e9, 8 * 1024 * 1024),)),
    "M2 Pro": (6.8, 200.0, (CacheSpec("SLC", 450e9, 24 * 1024 * 1024),)),
    "M2 Max": (13.6, 400.0, (CacheSpec("SLC", 900e9, 48 * 1024 * 1024),)),
    "M2 Ultra": (27.2, 800.0, (CacheSpec("SLC", 1800e9, 96 * 1024 * 1024),)),
    "M3": (3.6, 100.0, (CacheSpec("SLC", 250e9, 8 * 1024 * 1024),)),
    "M3 Pro": (7.4, 150.0, (CacheSpec("SLC", 400e9, 24 * 1024 * 1024),)),
    "M3 Max": (14.2, 300.0, (CacheSpec("SLC", 800e9, 48 * 1024 * 1024),)),
    "M4": (4.6, 120.0, (CacheSpec("SLC", 300e9, 8 * 1024 * 1024),)),
    "M4 Pro": (9.2, 273.0, (CacheSpec("SLC", 600e9, 24 * 1024 * 1024),)),
    "M4 Max": (18.4, 546.0, (CacheSpec("SLC", 1200e9, 48 * 1024 * 1024),)),
}

# ---------------------------------------------------------------------------
# NVIDIA Datacenter and Desktop GPU Chip Table
# Published peak TFLOP/s (FP32, TF32 Tensor Core, FP16 Tensor Core), bandwidth (GB/s),
# and L2 Cache specs across Ampere, Ada Lovelace, and Hopper architectures.
# ---------------------------------------------------------------------------
_NVIDIA_CHIP_TABLE: dict[str, tuple[float, float, float, float, tuple[CacheSpec, ...]]] = {
    # (peak_tflops_fp32, peak_tflops_tf32, peak_tflops_fp16, bandwidth_gb_s, (CacheSpec(...), ...))
    "A100": (19.5, 312.0, 624.0, 1935.0, (CacheSpec("L2", 4000e9, 40 * 1024 * 1024),)),
    "A100-SXM4-80GB": (19.5, 312.0, 624.0, 2039.0, (CacheSpec("L2", 4000e9, 40 * 1024 * 1024),)),
    "A100-PCIe-80GB": (19.5, 312.0, 624.0, 1935.0, (CacheSpec("L2", 4000e9, 40 * 1024 * 1024),)),
    "L40S": (91.6, 183.0, 366.0, 864.0, (CacheSpec("L2", 2000e9, 96 * 1024 * 1024),)),
    "L40": (90.5, 181.0, 362.0, 864.0, (CacheSpec("L2", 2000e9, 96 * 1024 * 1024),)),
    "H100": (60.0, 756.0, 1513.0, 3350.0, (CacheSpec("L2", 6000e9, 50 * 1024 * 1024),)),
    "H100-SXM": (60.0, 756.0, 1513.0, 3350.0, (CacheSpec("L2", 6000e9, 50 * 1024 * 1024),)),
    "H100-PCIe": (51.0, 640.0, 1280.0, 2000.0, (CacheSpec("L2", 6000e9, 50 * 1024 * 1024),)),
    "L4": (30.3, 60.0, 120.0, 300.0, (CacheSpec("L2", 1000e9, 48 * 1024 * 1024),)),
    "T4": (8.1, 8.1, 65.0, 320.0, (CacheSpec("L2", 500e9, 4 * 1024 * 1024),)),
}


def _precision_multiplier(precision: str) -> float:
    """Return compute acceleration multiplier for precision relative to FP32."""
    p = precision.lower().strip()
    if p in ("fp16", "bf16"):
        return 2.0
    if p in ("int8", "fp8"):
        return 4.0
    return 1.0


@dataclass
class DetectionResult:
    """Raw values collected during hardware probing."""

    chip_name: str | None
    logical_cores: int
    clock_hz: float | None
    measured_bandwidth_gb_s: float | None
    peak_flops: float
    memory_bandwidth: float
    source: str  # human-readable description of where values came from


# ---------------------------------------------------------------------------
# Bandwidth measurement
# ---------------------------------------------------------------------------


def _measure_bandwidth_gb_s(size_mb: int = 256, repeats: int = 5) -> float:
    """Return effective memory bandwidth in GB/s via a NumPy array copy.

    Uses a *STREAM Triad*-style kernel (c = a + scalar * b) over a buffer
    large enough to exceed typical L3 caches, measured with
    :func:`time.perf_counter_ns`.
    """
    import time

    import numpy as np

    n = (size_mb * 1024 * 1024) // 8  # float64 elements
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    c = np.empty_like(a)
    scalar = 3.0

    # Warmup
    np.add(a, b, out=c)
    np.add(a, b, out=c)

    samples: list[float] = []
    bytes_moved = 3 * a.nbytes  # read a, read b, write c

    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        np.add(a * scalar, b, out=c)
        elapsed = (time.perf_counter_ns() - t0) / 1e9
        samples.append(bytes_moved / elapsed / 1e9)

    samples.sort()
    # Use the median of the top half to reduce OS scheduling noise.
    top_half = samples[len(samples) // 2 :]
    return sum(top_half) / len(top_half)


# ---------------------------------------------------------------------------
# Platform probes
# ---------------------------------------------------------------------------


def _probe_apple_silicon() -> tuple[str | None, float | None, float | None, tuple[CacheSpec, ...]]:
    """Return (chip_name, peak_tflops, bandwidth_gb_s, caches) from system_profiler."""
    if not shutil.which("system_profiler"):
        return None, None, None, ()
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"],
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None, None, None, ()

    chip_match = re.search(r"Chip:\s+(Apple[^\n\r]+)", out)
    if not chip_match:
        return None, None, None, ()

    chip_raw = chip_match.group(1).strip()
    # Table keys are bare names like "M3 Pro"; chip_raw includes "Apple " prefix.
    chip_bare = chip_raw.removeprefix("Apple ").strip()
    matched_key = None
    for key in sorted(_APPLE_CHIP_TABLE, key=len, reverse=True):
        if chip_bare.startswith(key):
            matched_key = key
            break

    if matched_key is None:
        return chip_raw, None, None, ()

    tflops, bw, caches = _APPLE_CHIP_TABLE[matched_key]
    return chip_raw, tflops * 1e12, bw * 1e9, caches


def _probe_nvidia(
    precision: str = "fp32",
) -> tuple[str | None, float | None, float | None, tuple[CacheSpec, ...]]:
    """Return (chip_name, peak_flops, bandwidth_bytes_s, caches) from nvidia-smi if present."""
    if not shutil.which("nvidia-smi"):
        return None, None, None, ()
    try:
        try:
            gpu_name_out = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                    text=True,
                    timeout=10,
                )
                .strip()
                .splitlines()
            )
            detected_name = gpu_name_out[0].strip() if gpu_name_out else None
        except (subprocess.SubprocessError, OSError, ValueError):
            detected_name = None

        if detected_name:
            for chip_key, entry in _NVIDIA_CHIP_TABLE.items():
                if chip_key.upper() in detected_name.upper():
                    p_fp32, p_tf32, p_fp16, bw_gb, caches = entry
                    p = precision.lower().strip()
                    if p in ("fp16", "bf16"):
                        tflops = p_fp16
                    elif p == "tf32":
                        tflops = p_tf32
                    elif p in ("int8", "fp8"):
                        tflops = p_fp16 * 2.0
                    else:
                        tflops = p_fp32
                    return detected_name, tflops * 1e12, bw_gb * 1e9, caches

        # Compute clock (MHz) and memory bandwidth (GB/s) for device 0.
        clock_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.max.sm,memory.total,clocks.max.mem",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        parts = [p.strip() for p in clock_out.strip().split(",")]
        if len(parts) < 3:
            return None, None, None, ()
        sm_mhz = float(parts[0])
        mem_mhz = float(parts[2])
        mult = _precision_multiplier(precision)
        peak_flops = sm_mhz * 1e6 * 2 * 5120 * mult
        bandwidth = mem_mhz * 1e6 * 2 * 256 / 8
        caches = (CacheSpec("L2", bandwidth * 3.0, 40 * 1024 * 1024),)
        return detected_name or "NVIDIA GPU", peak_flops, bandwidth, caches
    except (subprocess.SubprocessError, OSError, ValueError):
        return None, None, None, ()


def _probe_cpu_flops() -> tuple[int, float | None]:
    """Return (logical_core_count, clock_hz_or_None) from sysctl / /proc/cpuinfo."""
    import os

    cores = os.cpu_count() or 1
    clock_hz: float | None = None

    if shutil.which("sysctl"):
        # Try keys in priority order; Apple Silicon uses perflevel0 (P-core).
        for key in (
            "hw.cpufrequency_max",  # Intel macOS
            "hw.perflevel0.cpufrequency_max",  # Apple Silicon P-core
            "hw.cpufrequency",  # some Linux/BSD
        ):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", key],
                    text=True,
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                )
                val = out.strip()
                if val:
                    clock_hz = float(val)
                    break
            except (subprocess.SubprocessError, OSError, ValueError):
                continue

    # Linux /proc/cpuinfo fallback
    if clock_hz is None:
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if "cpu MHz" in line:
                        clock_hz = float(line.split(":")[1].strip()) * 1e6
                        break
        except OSError:
            pass

    return cores, clock_hz


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_hardware(
    bandwidth_benchmark_mb: int = 256,
    precision: str = "fp32",
    benchmark_memory: bool = True,
) -> tuple[HardwareSpec, DetectionResult]:
    """Probe this machine and return a :class:`HardwareSpec` for roofline analysis.

    Parameters
    ----------
    bandwidth_benchmark_mb:
        Working-set size in MiB for the NumPy bandwidth benchmark.  Increase
        for machines with very large L3 caches (e.g. 512 for server CPUs).
    precision:
        Target precision ('fp32', 'fp16', 'bf16', 'int8'). Accelerators report
        higher peak compute for lower precisions.
    benchmark_memory:
        Whether to execute the live NumPy STREAM benchmark (default True).

    Returns
    -------
    spec:
        :class:`HardwareSpec` suitable for passing to :func:`analyze_gap`.
    result:
        :class:`DetectionResult` with all raw probed values for display.
    """
    measured_bw = (
        _measure_bandwidth_gb_s(size_mb=bandwidth_benchmark_mb) if benchmark_memory else None
    )
    mult = _precision_multiplier(precision)

    # --- Apple Silicon ---
    chip_name, apple_peak_flops, apple_bw, apple_caches = _probe_apple_silicon()
    if apple_peak_flops is not None:
        peak_flops = apple_peak_flops * mult
        # Prefer manufacturer bandwidth; measured value is a lower bound.
        memory_bandwidth = (
            max(apple_bw or 0.0, measured_bw * 1e9)
            if measured_bw is not None
            else (apple_bw or 0.0)
        )
        source = f"Apple Silicon table ({chip_name}) [{precision.upper()}]"
        if measured_bw is not None:
            source += " + NumPy STREAM triad"
        cores, clock_hz = _probe_cpu_flops()
        return (
            HardwareSpec(
                chip_name or "Apple Silicon", peak_flops, memory_bandwidth, caches=apple_caches
            ),
            DetectionResult(
                chip_name,
                cores,
                clock_hz,
                measured_bw,
                peak_flops,
                memory_bandwidth,
                source,
            ),
        )

    # --- NVIDIA GPU ---
    gpu_name, nvidia_flops, nvidia_bw, nvidia_caches = _probe_nvidia(precision=precision)
    if nvidia_flops is not None and nvidia_bw is not None:
        memory_bandwidth = (
            max(nvidia_bw, measured_bw * 1e9) if measured_bw is not None else nvidia_bw
        )
        source = f"NVIDIA profile ({gpu_name}) [{precision.upper()}]"
        if measured_bw is not None:
            source += " + NumPy STREAM triad"
        cores, clock_hz = _probe_cpu_flops()
        return (
            HardwareSpec(
                gpu_name or "NVIDIA GPU", nvidia_flops, memory_bandwidth, caches=nvidia_caches
            ),
            DetectionResult(
                gpu_name,
                cores,
                clock_hz,
                measured_bw,
                nvidia_flops,
                memory_bandwidth,
                source,
            ),
        )

    # --- CPU fallback ---
    cores, clock_hz = _probe_cpu_flops()
    if clock_hz:
        # scalar FP32 FMA = 2 FLOP/cycle/core; use conservative 2× factor
        peak_flops = cores * clock_hz * 2.0 * mult
    else:
        # Very conservative: assume 2 GFLOP/s per core at unknown speed
        peak_flops = cores * 2e9 * mult

    memory_bandwidth = (measured_bw * 1e9) if measured_bw is not None else 30e9
    source = (
        f"CPU estimate ({cores} cores"
        + (f" @ {clock_hz / 1e9:.2f} GHz" if clock_hz else "")
        + f" [{precision.upper()}])"
    )
    if measured_bw is not None:
        source += " + NumPy STREAM triad"

    return (
        HardwareSpec("CPU", peak_flops, memory_bandwidth),
        DetectionResult(
            chip_name,
            cores,
            clock_hz,
            measured_bw,
            peak_flops,
            memory_bandwidth,
            source,
        ),
    )
