---
title: "Getting started"
description: "Install neural-cost and run your first roofline gap analysis."
section: "Start"
order: 1
---

## Installation

Install the core package from [PyPI](https://pypi.org/project/neural-cost/):

```bash
pip install neural-cost
```

Or using [`uv`](https://docs.astral.sh/uv/):

```bash
uv add neural-cost
```

## Framework adapters

Framework adapters are optional extras, keeping `neural-cost` lightweight with zero mandatory external dependencies:

| Framework | Installation Command | Description |
|---|---|---|
| **PyTorch** | `pip install "neural-cost[torch]"` | PyTorch eager execution & FX graph tracer adapter |
| **JAX** | `pip install "neural-cost[jax]"` | JAX primitives and jaxpr tracing |
| **JAX on Apple Silicon (Metal)** | `pip install "neural-cost[jax-metal]"` | Apple's official Metal PJRT plugin (`JAX_PLATFORMS=METAL`); pins jax/jaxlib 0.4.34, Python ≤ 3.13 |
| **JAX on Apple Silicon (MPS)** | `pip install "neural-cost[jax-mps]"` | Community PJRT plugin that runs JAX on the Apple GPU through MLX (`JAX_PLATFORMS=mps`); pins jax/jaxlib 0.11.x, Python ≥ 3.12 |
| **TensorFlow** | `pip install "neural-cost[tensorflow]"` | TensorFlow / Keras layer adapter |
| **All Frameworks** | `pip install "neural-cost[torch,jax,tensorflow]"` | All supported deep learning framework adapters |
| **Development** | `pip install "neural-cost[dev]"` | Test suite (`pytest`) & code style tools (`ruff`) |

## Analyze portable operations

```python
from neural_cost import HardwareSpec, Operation, analyze_gap, benchmark, estimate_operations, detect_hardware

ops = [Operation("classifier", "linear", ((32, 768), (768, 1000)), (32, 1000), 2)]
estimate = estimate_operations(ops)
measurement = benchmark(lambda: run_inference(), warmup=5, repeats=20)

# Auto-detect or specify manually:
hardware, info = detect_hardware()
# Or: hardware = HardwareSpec("GPU", peak_flops=312e12, memory_bandwidth=1.6e12)
report = analyze_gap(estimate, measurement, hardware)

print(report.render())
```

The theoretical model reports FLOPs, tensor reads/writes, arithmetic intensity,
and a compute/bandwidth lower bound.  The measured gap is expected: it captures
launch overhead, synchronization, framework behavior, unfused intermediates,
workspaces, caches, and imperfect kernel utilization.
