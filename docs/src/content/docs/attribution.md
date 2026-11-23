---
title: "Gap attribution"
description: "Split measured runtime into ordered physical components that sum to the median."
section: "Guides"
order: 4
---

`attribute_gap` splits measured runtime into ordered, non-overlapping
components that sum exactly to the median:

| Component | What it is |
|---|---|
| `roofline_floor` | FLOPs and fused bytes at datasheet peak with perfect overlap. Nothing can run faster. |
| `unfused_traffic` | Extra DRAM round-trips for intermediates a fusing compiler would keep on chip |
| `serialization` | Lost compute/memory overlap: each op is bounded by its own roofline |
| `peak_derate` | Datasheet peak vs. *measured achievable* throughput, at each op's size |
| `dispatch` | Per-kernel launch overhead (only the part not hidden by async execution) plus per-call sync |
| `kernel_inefficiency` | How much slower each op runs *alone* than its achievable roofline (op replay) |
| `timing_jitter` | Median minus fastest sample |
| `residual` | Composition effects: framework glue, uncaptured ops, cache interference |

```python
from neural_cost import attribute_gap, detect_hardware
from neural_cost.adapters import TorchAdapter

adapter = TorchAdapter(use_profiler=False)            # no profiler overhead in timings
hardware, _ = detect_hardware()
calibration = adapter.calibrate("mps")                 # size-dependent FLOP/s + bandwidth curves,
                                                       # dispatch and sync latency
ops = adapter.operations(model, inputs)
measurement = adapter.benchmark(model, *inputs, repeats=30)
kernels = adapter.trace_kernels(model, inputs)         # every ATen kernel, incl. uncaptured ones

attribution = attribute_gap(
    ops, measurement, hardware, calibration,
    kernel_count=len(kernels),
    extra_kernels=[(k.bytes, k.python_dispatch) for k in kernels if not k.heavy],
    isolated_seconds=adapter.replay(ops, "mps"),       # each op timed alone at its exact shape
    assume_fusion=False,                               # eager execution
)
print(attribution.render())
for step in attribution.recommendations():
    print(step)
```

Calibration measures throughput as a function of size: matmuls from 32² to
4096², and copies from 64 KiB (cache-resident) to 256 MiB (DRAM). Small
kernels and cache-resident data are therefore modeled at the rate they
actually reach. On asynchronous devices (CUDA, MPS) a kernel only exposes
`max(0, dispatch − runtime)` of launch overhead. Kernels issued inside a C++
composite op, such as the per-timestep ops of an eager CPU LSTM, pay no Python
dispatch.
