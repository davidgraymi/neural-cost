---
title: "Hardware detection"
description: "Detect or measure peak FLOP/s and memory bandwidth for the device a workload runs on."
section: "Guides"
order: 2
---

Auto-detection via `detect_hardware()` returns a `(HardwareSpec, DetectionResult)` tuple
to determine hardware peak compute and memory bandwidth:

- **Apple Silicon lookup from chip table**: Identifies Apple Silicon chips (M1–M4 series) and looks up published peak FP32 throughput and memory bandwidth.
- **NumPy STREAM-triad bandwidth benchmark**: Measures live effective memory bandwidth using a STREAM Triad kernel (`c = a + scalar * b`).
- **NVIDIA GPU probe via nvidia-smi**: Probes GPU models, clock rates, and bus specs on systems with NVIDIA GPUs.
- **CPU fallback**: Without an accelerator (or with `device="cpu"`), the peak is logical cores × maximum clock × the FP32 FLOP/cycle of the widest SIMD unit the CPU reports: 64 for AVX-512, 32 for AVX2+FMA, 16 for AVX, 8 for SSE2, 32 for SVE and 16 for NEON (two vector FMA pipes per core), or 2 for scalar FMA. Flags and the CPU's model name come from `/proc/cpuinfo` on Linux and `sysctl machdep.cpu` on Intel macOS. This is a datasheet-style upper bound, so roofline lower bounds built on it hold; Apple Silicon uses its NEON/AMX table instead.

```python
from neural_cost import detect_hardware

hardware, info = detect_hardware()
print(f"Device: {hardware.device_name} ({info.source})")
print(f"Peak FLOP/s: {hardware.peak_flops / 1e12:.1f} TFLOP/s")
print(f"Bandwidth: {hardware.memory_bandwidth / 1e9:.1f} GB/s")
```

## Device-aware specs

By default `detect_hardware()` describes the fastest detected accelerator.  If
the workload runs on a different device (for example JAX with
`JAX_PLATFORMS=cpu` on an Apple Silicon Mac), pass `device=` so the run is
scored against the right peak:

```python
cpu_spec, _ = detect_hardware(device="cpu")             # skip GPU tables
spec, _ = detect_hardware(device=jax.devices()[0])      # JAX device object
spec, _ = detect_hardware(device=torch.device("mps"))   # torch device
spec, _ = JaxAdapter().hardware()                       # active JAX device
```

Every detected `HardwareSpec` records a `device_type` (`"cpu"` or `"gpu"`).
When an adapter benchmark records the device it ran on, `analyze_gap` adds a
**Device mismatch** finding if the two disagree.  On Apple Silicon the CPU
peak is the higher of the NEON P/E-core peak and the per-cluster AMX/SME matrix
units that Accelerate (PyTorch's CPU BLAS) drives, and the CPU is bounded by the
chip's unified memory bandwidth (a single-threaded NumPy triad reaches well
under half of it).  The CLI accepts the same option:
`neural-cost hardware --device cpu`.

Apple variants are told apart by GPU core count (from `system_profiler`): the
FP32 peak scales with cores, and binned chips get their narrower memory bus
(M3 Max 30-core 300 GB/s vs 40-core 400 GB/s; M4 Max 32-core 410 GB/s vs
40-core 546 GB/s).

## Measuring the peak on the active device

Spec sheets do not cover every backend (jax-mps, jax-metal, TPU, future
plugins), and the achievable peak can sit well below the datasheet. JAX can
measure both peaks on the device itself with a matmul sweep for FLOP/s and a
streaming `x + 1` sweep for bandwidth. Results are cached per device for the
process:

```python
adapter = JaxAdapter()
calibration = adapter.calibrate(device=jax.devices()[0])     # HardwareCalibration
spec = calibration.to_hardware_spec(device_type="gpu")       # score against achievable peaks
calibration.fraction_of(adapter.hardware()[0])               # e.g. (0.81, 0.84) of spec sheet

spec, _ = adapter.hardware()                  # tables, measured fallback if no table entry
spec, _ = adapter.hardware(measure="always")  # always measure on the device
```

```bash
neural-cost hardware --calibrate jax --quick
#   Achievable Compute: 0.440 TFLOP/s (81% of spec)
#   Achievable BW:      57.60 GB/s (84% of spec)
```
