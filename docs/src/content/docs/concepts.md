---
title: "Concepts"
description: "How neural-cost models a workload: portable operations, static profiles, measurement and gap analysis."
section: "Start"
order: 2
---

1. **Model or function** plus example inputs goes into an **adapter** (PyTorch, JAX, TensorFlow, or a custom `FrameworkAdapter`).
2. The adapter emits portable **`Operation` records**: `linear`, `conv2d`, `embedding`, `attention`, RNN/LSTM, `layernorm`, pooling and more.
3. The **static profile** turns operations into FLOPs, tensor traffic, parameter bytes and activation bounds.
4. The **dynamic tracer** measures runtime, allocator telemetry and (for PyTorch) profiler events.
5. **Hardware detection** (chip table, STREAM triad, `nvidia-smi`, CPU fallback) produces a `HardwareSpec` with peak FLOP/s and bandwidth.
6. **Gap analysis** combines all three into roofline efficiency, memory overhead and findings.

## Supported operation kinds

| Kind | Description |
|---|---|
| `linear` | Dense / fully-connected projection |
| `conv2d` | 2-D convolution (NCHW / OIHW) |
| `conv` | 1-D, 2-D or 3-D convolution, channels-first; `transposed`, grouped and depthwise variants |
| `matmul` | Raw matrix multiply |
| `embedding` | Token embedding lookup (read-only traffic model) |
| `attention` | Multi-head self-attention (QKV projections + softmax + output) |
| `elementwise` | Point-wise ops (ReLU, exp, tanh, …) |
| `softmax` / `layernorm` / `batchnorm` | Normalisation ops |
| `rmsnorm` | Root Mean Square normalisation |
| `swiglu` | Gated linear unit with Swish activation |
| `pooling` | Max / average / adaptive / global pooling over 1 to 3 spatial dimensions |
| `moe` | Mixture-of-Experts sparse routing and expert FFN execution |
| `gather` / `scatter` | Indexed reads and in-place indexed updates; only touched elements count as traffic |
| `custom` | Caller-supplied explicit FLOP count |

A transposed `conv` counts `C_out / groups × prod(k)` MACs for every input
element (the fvcore/ptflops convention), so the zeros a strided transposed
convolution inserts are not counted. Plain 2-D convolutions keep the
`conv2d` kind. `estimate_conv` estimates any `conv` record.

Adapters automatically emit the right kind for each layer type:

| Framework | Captured layer types |
|---|---|
| **PyTorch** | `Linear`, `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`, `Max/Avg/AdaptiveMax/AdaptiveAvgPool1d/2d/3d`, `Embedding`, `RNN`, `GRU`, `LSTM`, `MultiheadAttention`, `LayerNorm`, `BatchNorm1d/2d/3d` |
| **TensorFlow** | `Dense`, `Conv1D/2D/3D`, `Conv1D/2D/3DTranspose`, `DepthwiseConv1D/2D`, `SeparableConv1D/2D` (as depthwise + pointwise), 1-D/2-D/3-D max, average and global pooling, `Embedding`, `GRU`, `LSTM`, `MultiHeadAttention`, `BatchNormalization`, `LayerNormalization` |
| **JAX** | `dot_general` (matmul), `conv_general_dilated` (1-D to 3-D conv; `conv_transpose` as a transposed conv), elementwise/comparison/reduction primitives, `gather`/`scatter`/`dynamic_slice`/`dynamic_update_slice`, and `scan`/`while`/`cond` bodies |

## Current scope

The package profiles concrete-shape dense, matrix-multiply, convolution,
embedding lookup, multi-head attention, and common elementwise inference graphs,
alongside `softmax`, `layernorm`, `batchnorm`, and `pooling`.  Recurrent layers
(RNN, GRU, LSTM) are modelled as one input→hidden GEMM over all timesteps plus
T sequential hidden→hidden GEMMs.  The published validation covers single-device
eager PyTorch (CPU, MPS) and jitted JAX on one Apple M3; CUDA results come from
running `benchmarks/validate.py` on such a machine, and the analytical
LLM-serving and distributed models (MoE, paged attention, continuous batching,
3D parallelism) are not yet validated against measured systems.  Static training storage includes gradients
and optimizer state but does not yet trace a full backward graph.  Activation
checkpointing, distributed communication, dynamic shapes, fusion details,
complete graph coverage, and non-PyTorch kernel-level traces remain deliberate
next increments rather than silently approximated.
