---
name: neural-cost
description: Explain why a neural network or JAX/PyTorch/TensorFlow function is slow with neural-cost. Use it for roofline analysis (memory- vs compute-bound), efficiency against the hardware peak, per-op and per-module breakdowns, CPU vs GPU/MPS comparisons, multi-stage pipelines (search plus env plus net), and choosing a device, batch size or net size.
---

# neural-cost

neural-cost compares a measured runtime with a roofline lower bound:
max(FLOPs / peak FLOP/s, compulsory bytes / peak bandwidth). It then explains
the gap. Use the library rather than ad-hoc timing when the question is
"why is this slow", "is it memory- or compute-bound", "which device or net
size", or "CPU vs GPU".

## Install

| Need | Command |
|---|---|
| PyTorch models | `pip install "neural-cost[torch]"` |
| JAX / Flax functions | `pip install "neural-cost[jax]"` |
| TensorFlow / Keras | `pip install "neural-cost[tensorflow]"` |
| JAX on the Apple GPU via jax-mps | `pip install "neural-cost[jax,jax-mps]"`. jax-mps 0.11.x needs jax/jaxlib 0.11.x and Python ≥ 3.12, and the extra pins jax to match. Run with `JAX_PLATFORMS=mps`. |
| JAX on the Apple GPU via jax-metal | `pip install "neural-cost[jax-metal]"`. It pins jax/jaxlib 0.4.34, the only versions jax-metal 0.1.1 works with, so use Python ≤ 3.13. Run with `JAX_PLATFORMS=METAL`. Never install both plugin extras. |

The core package has no dependencies. Without a GPU plugin, JAX on a Mac
silently runs on the CPU, so check `jax.devices()[0].platform`.

## Recipe: JAX / Flax

```python
import jax
import jax.numpy as jnp

from neural_cost import analyze_gap, estimate_operations
from neural_cost.adapters import JaxAdapter


def fn(x, w):  # any jittable function; weights as later arguments
    return jnp.tanh(x @ w)


fn = jax.jit(fn)
x, w = jnp.ones((128, 256)), jnp.ones((256, 256))

adapter = JaxAdapter()  # args after the first are parameters by default
hardware, _ = adapter.hardware(benchmark_memory=False)  # spec of the ACTIVE JAX device
operations = adapter.operations(fn, (x, w))
measurement = adapter.benchmark(fn, x, w, warmup=3, repeats=10)
gap = analyze_gap(
    estimate_operations(operations),
    measurement,
    hardware,
    operations=operations,
    coverage=adapter.coverage,
)
print(gap.render(top=5))
```

- **Flax:** jit `model.apply` and pass `variables` as argument 0 with
  `JaxAdapter(parameter_argnums=(0,))`, or close over them:
  `jax.jit(lambda obs: model.apply(variables, obs))` and then
  `operations(fn, (obs,))`. Arrays a function closes over count as weights.
- **Per-module view:** `gap.render(top=8, group_by_scope=True)` sums
  operations per Flax module or `jax.named_scope`, such as
  `ResNet/Block_0/GroupNorm_0` vs `ResNet/Block_0/Conv_0`.
- **`while_loop` with a data-dependent trip count** (tree search, rollouts):
  pass `JaxAdapter(trip_counts={0: 16})` or `default_trip_count=...`.
  Without either, the body is counted once with a warning. Check
  `adapter.while_trip_counts` and `adapter.warnings`. A counted
  `i < N` loop with literal bounds is inferred.
- **Devices without a spec-table entry** (TPU, new plugins):
  `adapter.hardware(measure="auto")` measures peaks on the device.
  `adapter.calibrate()` returns achievable FLOP/s and bandwidth.

## Recipe: PyTorch

```python
import torch

from neural_cost import analyze_gap, detect_hardware, estimate_operations
from neural_cost.adapters import TorchAdapter

model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
x = torch.randn(64, 256)

adapter = TorchAdapter()
operations = adapter.operations(model, (x,))
measurement = adapter.benchmark(model, x, warmup=3, repeats=10)  # synchronizes CUDA/MPS
hardware, _ = detect_hardware(device=x.device, benchmark_memory=False)  # match the device!
gap = analyze_gap(
    estimate_operations(operations),
    measurement,
    hardware,
    operations=operations,
    coverage=adapter.coverage,
)
print(gap.render(top=5))
```

`TorchAdapter` hooks modules (Linear, Conv2d, Embedding, RNN/GRU/LSTM,
MultiheadAttention, norms) and `scaled_dot_product_attention`.
Untracked modules such as ReLU and GELU show up in `adapter.coverage`.
TensorFlow: `TensorFlowAdapter()` follows the same pattern.

## Pipelines and comparisons

- **Several functions with call counts** (search = tree + N env steps + N net
  forwards): `Pipeline({"tree": Stage(f, args), "env": Stage(g, args, calls=16),
  "net": Stage(h, args, calls=16)}, total=Stage(search, args)).run(adapter=JaxAdapter())`.
  `.render()` gives the time split plus the unexplained remainder (dispatch and
  loop overhead). With `hardware=` it adds a roofline per stage.
- **CPU vs GPU for one JAX function:** write a zero-arg factory
  `build() -> (fn, args)` in `bench.py`, then run
  `neural-cost compare bench.py:build --platforms cpu,mps`. Each platform runs
  in a subprocess with its own hardware spec and speedups are relative to the
  first one. In Python, use `compare_backends(...)`.

## CLI

| Command | Use |
|---|---|
| `neural-cost hardware [--device cpu] [--calibrate jax]` | Detected peaks; `--calibrate` measures achievable peaks on the device |
| `neural-cost profile --arch transformer ...` | Static FLOPs, bytes and memory for standard architectures |
| `neural-cost benchmark --framework jax` | Measured runtime vs roofline for built-in workloads |
| `neural-cost compare [bench.py:build --platforms cpu,mps]` | Framework comparison, or one function across platforms |
| `neural-cost pipeline pipe.py:build --framework jax --roofline` | Pipeline stage split |
| `neural-cost audit --max-latency-ms ... --max-vram ...` | CI budget gate |

Add `--json` for machine-readable output.

## Interpreting the report

- **bound**: the fastest the modeled work could possibly run.
  **roofline efficiency** = bound / observed (100% = at the roofline).
- **memory-bound** (arithmetic intensity, FLOP/B, below the ridge point =
  peak FLOP/s / bandwidth): bytes moved dominate. Next steps are fusion
  (compare the fused bound in the report), bf16/fp16, fewer materialized
  intermediates, and larger batches to reuse weights.
- **compute-bound**: try faster kernels, tensor cores, lower precision, or
  more parallelism.
- **Low efficiency on small workloads** usually means dispatch/launch overhead
  rather than the math. Try larger batches, `jax.jit`, or fewer kernels.
  `attribute_gap(...)` splits the gap into components.
- The per-op table ranks operations by share of the lower bound. Use
  `sort="bytes"` to find where the traffic goes.

## Caveats you must check

1. **Coverage.** Only modeled operations count. Read the `coverage:` line,
   or `adapter.coverage.fraction` and `.unmodeled_primitives`. Below 90%,
   `render()` prints `WARNING: Low coverage`, and the bound and efficiency then
   describe only part of the workload. JAX models dense, conv, elementwise,
   reductions, gather/scatter, dynamic slices and control flow. Primitives such
   as `sort` and RNG (`threefry2x32`) are not modeled.
2. **Device.** Score a run against the device it ran on. `JaxAdapter().hardware()`
   and `detect_hardware(device=...)` handle this. If the findings include
   `Device mismatch`, the efficiency is meaningless; fix the spec. Spec-sheet
   peaks are optimistic, so `adapter.calibrate()` shows the achievable peak.
3. **Loop trip counts.** Data-dependent `while` loops are counted once unless
   `trip_counts` or `default_trip_count` is given. Check `adapter.warnings`.
4. **Timing.** Use the adapter's `benchmark`, which waits for asynchronous
   JAX/CUDA/MPS work, and warm up so compilation is excluded.

## Reporting rule

When you report results, always state:

- the device (`measurement.device`, `hardware.name` and `hardware.device_type`),
- the batch size and input shapes,
- the number of timed runs (`len(measurement.samples_seconds)`) and that the
  figure is a median,
- **the coverage**. Never quote a roofline efficiency without saying how much
  of the workload it covers, e.g. "59% roofline efficiency over 100% of 257
  traced equations".
