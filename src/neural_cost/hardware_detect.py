"""Automatic hardware detection and memory-bandwidth measurement.

Provides :func:`detect_hardware` which returns a :class:`~.hardware.HardwareSpec`
populated with values derived from:

1. A STREAM-style NumPy copy benchmark (always available) to measure
   *effective* memory bandwidth.
2. ``system_profiler SPHardwareDataType`` on macOS to identify Apple Silicon
   chips and look up their published peak FP32 compute throughput.
3. ``nvidia-smi`` on systems with NVIDIA GPUs (CUDA peak FLOPs).
4. Sensible CPU-level fallbacks (logical cores × clock × scalar FMA factor).

Pass ``device=`` (a string such as ``"cpu"``/``"gpu"``, or a framework device
object such as ``jax.devices()[0]`` or ``torch.device("mps")``) to describe the
device the workload actually runs on.  Without it the fastest detected
accelerator is assumed, which misreports efficiency for CPU runs on a machine
that also has a GPU.

All values are best-effort estimates for roofline purposes.  Pass explicit
``--peak-flops`` / ``--memory-bandwidth`` arguments to the comparison script
to override them with manufacturer data for your exact SKU.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

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
    "M3 Max": (14.2, 400.0, (CacheSpec("SLC", 800e9, 48 * 1024 * 1024),)),
    "M3 Ultra": (28.4, 819.0, (CacheSpec("SLC", 1600e9, 96 * 1024 * 1024),)),
    "M4": (4.6, 120.0, (CacheSpec("SLC", 300e9, 8 * 1024 * 1024),)),
    "M4 Pro": (9.2, 273.0, (CacheSpec("SLC", 600e9, 24 * 1024 * 1024),)),
    "M4 Max": (18.4, 546.0, (CacheSpec("SLC", 1200e9, 48 * 1024 * 1024),)),
}

# GPU core count each table row describes (the full, unbinned configuration).
# When system_profiler reports a different core count, the FP32 peak is scaled
# linearly: every Apple GPU core has 128 FP32 ALUs at the generation's clock.
_APPLE_GPU_CORES: dict[str, int] = {
    "M1": 8,
    "M1 Pro": 16,
    "M1 Max": 32,
    "M1 Ultra": 64,
    "M2": 10,
    "M2 Pro": 19,
    "M2 Max": 38,
    "M2 Ultra": 76,
    "M3": 10,
    "M3 Pro": 18,
    "M3 Max": 40,
    "M3 Ultra": 80,
    "M4": 10,
    "M4 Pro": 20,
    "M4 Max": 40,
}

# Binned variants whose memory bus is narrower than the full chip
# (GB/s by GPU core count).  Other variants share one bandwidth.
_APPLE_BINNED_BANDWIDTH: dict[str, dict[int, float]] = {
    "M3 Max": {30: 300.0, 40: 400.0},
    "M4 Max": {32: 410.0, 40: 546.0},
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


_GPU_TOKENS = ("gpu", "cuda", "rocm", "mps", "metal", "xpu")


def normalize_device_type(device: Any) -> str | None:
    """Map a device description to ``"cpu"``, ``"gpu"``, ``"tpu"`` or ``None``.

    Accepts strings (``"cpu"``, ``"cuda:0"``, ``"mps"``, ``"METAL"``,
    ``"TFRT_CPU_0"``), JAX devices (``.platform``), torch devices (``.type``)
    and arrays/tensors exposing a ``.device`` attribute.  Returns ``None`` when
    the kind cannot be determined.
    """
    if device is None:
        return None
    if not isinstance(device, str):
        for attribute in ("platform", "type"):
            value = getattr(device, attribute, None)
            if isinstance(value, str):
                return normalize_device_type(value)
        inner = getattr(device, "device", None)
        if inner is not None and inner is not device and not callable(inner):
            return normalize_device_type(inner)
        device = str(device)
    text = device.strip().lower()
    if "tpu" in text:
        return "tpu"
    if "cpu" in text:
        return "cpu"
    if any(token in text for token in _GPU_TOKENS):
        return "gpu"
    return None


# ---------------------------------------------------------------------------
# Bandwidth measurement
# ---------------------------------------------------------------------------


def _measure_bandwidth_gb_s(size_mb: int = 256, repeats: int = 5) -> float | None:
    """Return effective memory bandwidth in GB/s via a NumPy array copy.

    Uses a *STREAM Triad*-style kernel (c = a + scalar * b) over a buffer
    large enough to exceed typical L3 caches, measured with
    :func:`time.perf_counter_ns`.  Returns ``None`` when NumPy (an optional
    dependency) is not installed.
    """
    import time

    try:
        import numpy as np
    except ImportError:
        return None

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
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
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
    cores_match = re.search(r"Total Number of Cores:\s+(\d+)", out)
    if cores_match:
        gpu_cores = int(cores_match.group(1))
        reference = _APPLE_GPU_CORES.get(matched_key)
        if reference and gpu_cores != reference:
            tflops *= gpu_cores / reference
        binned = _APPLE_BINNED_BANDWIDTH.get(matched_key, {})
        if binned:
            # Nearest listed configuration at or below the reported core count.
            fitting = [cores for cores in binned if cores <= gpu_cores]
            bw = binned[max(fitting) if fitting else min(binned)]
        chip_raw = f"{chip_raw} ({gpu_cores}-core GPU)"
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


# Apple Silicon CPU clocks (P-core, E-core) per generation.  Each P-core issues
# four 128-bit FP32 FMAs per cycle (32 FLOP/cycle) and each E-core two
# (16 FLOP/cycle).  This is the NEON peak an XLA/Eigen CPU backend can reach;
# Accelerate's AMX matmul units are not counted.
_APPLE_CPU_CLOCKS: dict[str, tuple[float, float]] = {
    "M1": (3.2e9, 2.06e9),
    "M2": (3.5e9, 2.42e9),
    "M3": (4.05e9, 2.75e9),
    "M4": (4.4e9, 2.9e9),
}


def _sysctl_int(key: str) -> int | None:
    if not shutil.which("sysctl"):
        return None
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", key], text=True, timeout=5, stderr=subprocess.DEVNULL
        )
        return int(out.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _apple_cpu_peak_flops(chip_name: str | None) -> float | None:
    """FP32 NEON peak of an Apple Silicon CPU complex, or ``None`` if unknown."""
    if not chip_name:
        return None
    match = re.search(r"\b(M\d+)\b", chip_name)
    if match is None or match.group(1) not in _APPLE_CPU_CLOCKS:
        return None
    p_cores = _sysctl_int("hw.perflevel0.physicalcpu")
    e_cores = _sysctl_int("hw.perflevel1.physicalcpu") or 0
    if not p_cores:
        return None
    p_clock, e_clock = _APPLE_CPU_CLOCKS[match.group(1)]
    return p_cores * p_clock * 32.0 + e_cores * e_clock * 16.0


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
    device: Any = None,
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
    device:
        The device the workload runs on: ``"cpu"``, ``"gpu"`` or a framework
        device object (``jax.devices()[0]``, ``torch.device("mps")``, an
        array).  ``None`` (default) assumes the fastest detected accelerator.
        With ``"cpu"`` the GPU tables are skipped even when a GPU is present.
        TPUs and other devices without a spec table raise ``ValueError``;
        measure them with :func:`neural_cost.adapters.JaxAdapter.calibrate`
        or pass an explicit :class:`HardwareSpec`.

    Returns
    -------
    spec:
        :class:`HardwareSpec` suitable for passing to :func:`analyze_gap`.
    result:
        :class:`DetectionResult` with all raw probed values for display.
    """
    kind = normalize_device_type(device)
    if device is not None and kind is None:
        raise ValueError(f"cannot determine the device type of {device!r}; use 'cpu' or 'gpu'")
    if kind == "tpu":
        raise ValueError(
            "no published-spec table for TPU devices; pass an explicit HardwareSpec "
            "or measure the device with JaxAdapter().calibrate()"
        )
    measured_bw = (
        _measure_bandwidth_gb_s(size_mb=bandwidth_benchmark_mb) if benchmark_memory else None
    )
    mult = _precision_multiplier(precision)

    # --- Apple Silicon ---
    chip_name, apple_peak_flops, apple_bw, apple_caches = _probe_apple_silicon()
    if apple_peak_flops is not None and kind != "cpu":
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
                chip_name or "Apple Silicon",
                peak_flops,
                memory_bandwidth,
                caches=apple_caches,
                device_type="gpu",
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
    gpu_name, nvidia_flops, nvidia_bw, nvidia_caches = (
        _probe_nvidia(precision=precision) if kind != "cpu" else (None, None, None, ())
    )
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
                gpu_name or "NVIDIA GPU",
                nvidia_flops,
                memory_bandwidth,
                caches=nvidia_caches,
                device_type="gpu",
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

    if kind == "gpu":
        raise ValueError(
            "device is a GPU but no GPU could be identified (Apple Silicon or nvidia-smi); "
            "pass an explicit HardwareSpec or measure the device with an adapter's calibrate()"
        )

    # --- CPU fallback ---
    cores, clock_hz = _probe_cpu_flops()
    apple_cpu_flops = _apple_cpu_peak_flops(chip_name)
    if apple_cpu_flops is not None:
        peak_flops = apple_cpu_flops * mult
    elif clock_hz:
        # scalar FP32 FMA = 2 FLOP/cycle/core; use conservative 2× factor
        peak_flops = cores * clock_hz * 2.0 * mult
    else:
        # Very conservative: assume 2 GFLOP/s per core at unknown speed
        peak_flops = cores * 2e9 * mult

    # Apple Silicon CPU and GPU share unified memory: without a live
    # measurement the chip's published bandwidth is the best available bound.
    fallback_bw = apple_bw if apple_bw else 30e9
    memory_bandwidth = (measured_bw * 1e9) if measured_bw is not None else fallback_bw
    source = (
        f"CPU estimate ({cores} cores"
        + (f" @ {clock_hz / 1e9:.2f} GHz" if clock_hz else "")
        + (", Apple NEON P/E-core table" if apple_cpu_flops is not None else "")
        + f" [{precision.upper()}])"
    )
    if measured_bw is not None:
        source += " + NumPy STREAM triad"

    cpu_name = f"{chip_name.split(' (')[0]} (CPU)" if chip_name else "CPU"
    return (
        HardwareSpec(cpu_name, peak_flops, memory_bandwidth, device_type="cpu"),
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
