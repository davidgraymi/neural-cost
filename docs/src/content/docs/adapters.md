---
title: "Framework adapters"
description: "PyTorch, JAX and TensorFlow adapters, coverage reporting and per-operation breakdowns."
section: "Guides"
order: 1
---

```python
import torch
from neural_cost import estimate_model
from neural_cost.adapters import TorchAdapter

model = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))
inputs = (torch.randn(16, 128),)
estimate = estimate_model(model, inputs, TorchAdapter())
measurement = TorchAdapter().benchmark(model, *inputs)
```

`TorchAdapter` captures `Linear`, 1-D to 3-D (transposed) convolution and pooling,
`Embedding`, `RNN`, `GRU`, `LSTM`, `MultiheadAttention`, `LayerNorm`, and
`BatchNorm` modules via forward hooks and
synchronizes CUDA benchmarks.  Its `trace` method uses `torch.profiler` and
returns aggregate profiler-event and CUDA allocator statistics.
`TensorFlowAdapter` captures the equivalent Keras layers and returns supported
TensorFlow GPU allocator statistics.  `JaxAdapter` traces `dot_general`,
`conv_general_dilated`, elementwise, comparison and reduction primitives, and
indexing primitives (`gather`, `scatter*`, `dynamic_slice`,
`dynamic_update_slice`), and waits for asynchronous device work during
benchmarks.  All adapters are optional imports:

```python
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter
```

`JaxAdapter` recurses into `scan` (scaled by its static length), `cond` (most
expensive branch) and `while_loop`.  A counted `while i < N` loop with literal
bounds is inferred; for data-dependent loops (tree search, environment
rollouts) supply the trip count:

```python
adapter = JaxAdapter(trip_counts={0: 16}, default_trip_count=8)
ops = adapter.operations(search_fn, (obs, params))
adapter.while_trip_counts  # [(16, "user"), (8, "default"), ...] in trace order
```

### Coverage: how much of the workload the roofline describes

Bounds and efficiencies only describe the operations an adapter understood.
After `operations()`, `JaxAdapter` (jaxpr equations) and `TorchAdapter`
(executed leaf modules) record a `Coverage`. Pass it to `analyze_gap`, or use
`profile_model` + `analyze_model_gap`, which pass it for you:

```python
adapter = JaxAdapter()
ops = adapter.operations(fn, (obs, params))
gap = analyze_gap(estimate_operations(ops), adapter.benchmark(fn, obs, params),
                  hardware, operations=ops, coverage=adapter.coverage)
print(gap.render())
# Neural cost gap analysis
#   coverage: 41.2% of 340 equations modeled (unmodeled: sort x12, threefry2x32 x8, ...)
#   WARNING: Low coverage: only 41.2% of the traced equations were modeled; ...
```

`Coverage.unmodeled_primitives` lists skipped primitives by frequency, and
`unmodeled_output_bytes` sums their output sizes as a rough measure of the
traffic left out. The warning threshold defaults to 90%
(`analyze_gap(..., coverage_threshold=0.8)`).

### Per-operation breakdown

When `analyze_gap` receives `operations=`, `render()` includes a table of
the top operations. Choose how many rows (`top`) and the sort key
(`"share"` of the lower bound, `"bytes"` or `"flops"`). `JaxAdapter` prefixes
operation names with their name scope (`jax.named_scope`, Flax module
names), and `group_by_scope=True` sums each scope, which shows whether
normalization layers or convolutions drive the traffic:

```python
print(gap.render(top=5, group_by_scope=True))
#   top 5 of 6 scopes by share of lower bound:
#      #  operation                  kind         FLOPs        bytes   FLOP/B      bound   share  regime
#      1  model/block_0/Conv_0       conv2d       604.0 MFLOP  4.3 MB  139.11   1.116 ms   36.1%  compute
#      3  model/block_0/GroupNorm_0  elementwise  3.7 MFLOP   25.2 MB    0.15   0.369 ms   11.9%  memory
```

Group and layer norms lowered to reductions plus elementwise ops are traced in
full. Reductions count one FLOP per input element.

## Custom frameworks

Subclass `FrameworkAdapter` and implement `operations(model, example_inputs)`
to return portable `Operation` records.  The adapter can also override
`benchmark` to synchronize an accelerator or collect framework-specific memory
statistics.  This contract keeps model extraction separate from the framework-
independent estimator and analyzer.
