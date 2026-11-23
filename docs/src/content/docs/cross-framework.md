---
title: "Cross-framework consistency"
description: "Check that PyTorch, JAX and TensorFlow count the same model the same way."
section: "Guides"
order: 3
---

A roofline efficiency of 4% tells you a workload is slow, not why. Before the gap can be
explained, every framework has to count the same model the same way. This page covers
that check; [gap attribution](../attribution/) splits the measured time into causes, and
[validation](../validation/) publishes out-of-sample error on real hardware.

## How it works

Adapters see a model at different granularity. A PyTorch hook records one
`attention` op, while a JAX trace records the `dot_general`, `exp` and
`reduce_sum` primitives that implement it. `flop_breakdown` separates
granularity-invariant **dense FLOPs** (contractions, 2 FLOPs per MAC) and
**unique parameters** from **auxiliary FLOPs** (point-wise work, which depends
on what the adapter captures). `cross_check` then compares adapters against
their median:

```python
from neural_cost import cross_check
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter

report = cross_check({
    "torch": TorchAdapter().operations(torch_model, torch_inputs),
    "jax": JaxAdapter().operations(jax_fn, (x, params)),
    "tensorflow": TensorFlowAdapter().operations(keras_model, tf_inputs),
}, rtol=0.0)
print(report.render())
```

The test suite builds an MLP, an NHWC CNN, a 2-layer `lax.scan` LSTM and
multi-head attention identically in all three frameworks. It asserts that the
dense FLOPs and parameter counts match each other exactly and match
closed-form values (`tests/test_cross_framework.py`). The five reference
architectures in `examples/architecture_comparison.py` also agree exactly. The
JAX tracer recurses into `jit`, `scan` (scaled by trip count), `while`,
`cond` and `custom_jvp`, handles arbitrary `dot_general` contractions and
NHWC convolutions, and tracks parameters through transposes and broadcasts.
