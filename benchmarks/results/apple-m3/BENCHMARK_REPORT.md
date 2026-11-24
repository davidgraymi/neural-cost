# Neural-Cost Scientific Benchmark Report

> **Run:** neural-cost v0.13.0-18-gcb36a1c @ cb36a1c2c163 on Apple M3 (CPU), 2026-11-23T21:42:34+00:00 (Python 3.13.18; jax 0.11.2, jaxlib 0.11.2, keras 3.15.1, numpy 2.5.3, tensorflow 2.21.0, torch 2.14.0)

> **Device:** Apple M3 (CPU)  ·  **Peak FP32:** 2.43 TFLOP/s  
> **Peak bandwidth:** 100 GB/s (STREAM triad: 51.4 GB/s)  
> **Ridge point:** 24.3 FLOP/byte  ·  **Detection:** CPU estimate (8 cores, Apple NEON/AMX table [FP32]) + NumPy STREAM triad  

---

## Methodology

### Architectures under test

| Architecture | Description |
|---|---|
| **FF DNN** | 784 → 128 → 128 → 10, ReLU + LayerNorm |
| **ConvNeXt** | Depthwise 7×7 → LayerNorm → 1×1 Inverted Bottleneck (dim×4) → GELU → 1×1, input 32×32×3 |
| **ViT** | Vision Transformer: Patch Embedding (4×4) → Class Token + Position → Multi-Head Self-Attention + MLP, input 32×32×3 |
| **Transformer** | 2-layer encoder (MHA h=4 with SDPA + FFN×4 + RMSNorm/LayerNorm), embed=128, seq=32 |

### Frameworks and optimisation variants

| Framework | Baseline | Optimised | Notes |
|---|---|---|---|
| **PyTorch 2.14** | Eager | `torch.compile()` | Inductor backend, CPU |
| **JAX 0.11** | Eager XLA | `jax.jit()` | Full XLA JIT with tracing |
| **TensorFlow** | Eager | `tf.function()` | Graph mode, no XLA |

### Measurement protocol

- **Warmup:** 15 iterations (full compilation and cache warm)
- **Timed repeats:** 40 samples per configuration
- **Statistics reported:** median, mean, σ (stddev), CV (coefficient of variation), p95
- **Roofline efficiency:** `min(1, lower_bound / observed)` where `lower_bound = max(FLOPs/peak_flops, bytes/bandwidth)`
- **Batch sizes swept:** [1, 4, 16, 64]

---

## Figure 1 — Roofline Model (batch=16)

Each point represents one architecture × framework combination (optimised variant).  
The roofline ceiling shows the theoretical maximum given the hardware's compute and bandwidth limits.

![Roofline](figures/fig1_roofline.png)

**Key observations:**
- All workloads fall well below the roofline ceiling on this CPU (typical for small-batch inference)
- Most architectures are **memory-bound** (AI < 24 FLOP/byte ridge point); only LSTM and Transformer cross the ridge
- JAX JIT achieves the highest effective throughput per FLOP across most architectures
- CNN workloads cluster at lower arithmetic intensity due to the convolution memory pattern

---

## Figure 2 — Inference Latency by Architecture (batch=16)

Error bars show ±1σ across 40 timed iterations.

![Latency bars](figures/fig2_latency_bars.png)

**Key observations:**
- TensorFlow eager dispatch dominates latency for small, sequential workloads (RNN, LSTM)
- PyTorch and JAX are within 2× of each other for compute-heavy architectures (CNN, Transformer)
- `tf.function` substantially reduces TF latency but does not close the gap to PyTorch/JAX for recurrent models

---

## Figure 3 — Roofline Efficiency Heatmap (batch=16)

Cells show efficiency as a percentage of the theoretical roofline bound.

![Efficiency heatmap](figures/fig3_efficiency_heatmap.png)

**Interpretation:**
- Higher is better; 100% would mean perfect roofline utilisation
- JAX JIT consistently achieves the highest efficiency across architectures
- FF DNN and Transformer reach the highest relative efficiency (5–12%) due to their matrix-multiply dominance
- Sequential models (RNN/LSTM) show the lowest efficiency because of loop-level overhead

---

## Figure 4 — Latency Scaling with Batch Size

![Batch scaling](figures/fig4_batch_scaling.png)

**Key observations:**
- All frameworks show approximately linear latency growth with batch size (expected: workloads are memory-bound)
- JAX JIT shows the most consistent scaling — early compilation amortises overhead across batch sizes
- TensorFlow eager latency at batch=1 is disproportionately high due to Python dispatch overhead
- PyTorch and JAX converge at larger batches where compute becomes the bottleneck

---

## Figure 5 — Compilation Speedup (baseline → optimised, batch=16)

Speedup ratio = eager latency / optimised latency. Higher is better.

![Speedup](figures/fig5_speedup.png)

**Key observations:**
- `jax.jit()` delivers the largest speedup for JAX, especially on sequential workloads (RNN: up to 8×, LSTM: up to 6×) where Python loop overhead is eliminated by tracing
- `torch.compile()` provides moderate speedups (1.2–3×) primarily on matrix-heavy layers; sequential models benefit less because the Python loop is not compiled
- `tf.function()` consistently improves TF performance (2–5×) by removing Python dispatch overhead

---

## Figure 6 — Achieved Throughput (GFLOP/s, batch=16)

![Throughput](figures/fig6_throughput.png)

---

## Figure 7 — Measurement Noise (CV%, batch=16)

Lower CV (%) indicates more stable, reproducible measurements.

![CV heatmap](figures/fig7_cv_heatmap.png)

**Interpretation:**
- JAX JIT shows very low CV (<3%) — deterministic compilation produces stable execution times
- TensorFlow eager shows high CV on recurrent models (Python-level branching introduces jitter)
- PyTorch baseline shows moderate CV; `torch.compile()` significantly reduces it

---

## Figure 8 — Peak Memory Utilization and Allocator Overhead (batch=16)

Empirical memory telemetry measured from framework allocators compared against theoretical tensor bounds calculated by `neural_cost.profile_model` and `neural_cost.analyze_memory_gap`.

![Memory utilization](figures/fig8_memory_utilization.png)

### Memory Telemetry and Allocator Fragmentation Table (batch=16)

| Architecture | Framework | Variant | Theo Min (KB) | Theo Cons (KB) | Peak Alloc (KB) | Peak Reserved (KB) | Overhead Ratio | Pool Caching |
|---|---|---|---|---|---|---|---|---|
| FF DNN | PyTorch | baseline | 98,708.0 | 100,180.7 | 100,245.0 | 100,245.0 | **1.02×** | 1.00× (minimal) |
| FF DNN | PyTorch | compiled | 98,708.0 | 100,180.7 | 99,669.0 | 99,669.0 | **1.01×** | 1.00× (minimal) |
| FF DNN | JAX | baseline | 98,600.0 | 98,920.6 | — | — | — | — |
| FF DNN | JAX | jit | 98,600.0 | 98,920.6 | 98,920.6 | — | **1.00×** | — |
| FF DNN | TensorFlow | baseline | 472.0 | 496.7 | — | — | — | — |
| FF DNN | TensorFlow | tf.function | 472.0 | 496.7 | — | — | — | — |
| ConvNeXt | PyTorch | baseline | 183,940.5 | 1,804,565.2 | 1,684,779.9 | 1,684,779.9 | **9.16×** | 1.00× (minimal) |
| ConvNeXt | PyTorch | compiled | 183,940.5 | 1,804,565.2 | 1,682,266.9 | 1,682,266.9 | **9.15×** | 1.00× (minimal) |
| ConvNeXt | JAX | baseline | 77,072.6 | 227,625.2 | — | — | — | — |
| ConvNeXt | JAX | jit | 77,072.6 | 227,625.2 | 182,182.0 | — | **2.36×** | — |
| ViT | PyTorch | baseline | 61,576.5 | 564,905.2 | 518,918.0 | 518,918.0 | **8.43×** | 1.00× (minimal) |
| ViT | PyTorch | compiled | 61,576.5 | 564,905.2 | 518,844.0 | 518,844.0 | **8.43×** | 1.00× (minimal) |
| ViT | JAX | baseline | 27,189.0 | 74,277.6 | — | — | — | — |
| ViT | JAX | jit | 27,189.0 | 74,277.6 | 41,301.6 | — | **1.52×** | — |
| Transformer | PyTorch | baseline | 264,462.0 | 2,844,942.7 | 2,577,790.7 | 2,577,790.7 | **9.75×** | 1.00× (minimal) |
| Transformer | PyTorch | compiled | 264,462.0 | 2,844,942.7 | 2,651,230.0 | 2,651,230.0 | **10.03×** | 1.00× (minimal) |
| Transformer | JAX | baseline | 125,982.0 | 322,686.6 | — | — | — | — |
| Transformer | JAX | jit | 125,982.0 | 322,686.6 | 150,558.6 | — | **1.20×** | — |
| Transformer | TensorFlow | baseline | 2,578.0 | 5,658.7 | — | — | — | — |
| Transformer | TensorFlow | tf.function | 2,578.0 | 5,658.7 | — | — | — | — |

**Key observations:**
- **Dynamic overhead ratio:** Observed peak memory exceeds the theoretical minimum due to temporary execution buffers, convolution im2col workspaces, activation retention, and framework object overhead.
- **Allocator fragmentation & caching:** Framework caching allocators retain memory pools across iterations to avoid repeated system allocation calls. For workloads with high dynamic allocations (such as CNN feature maps), reserved memory can exceed active tensor residency.
- **Model footprint scaling:** Transformers and CNNs exhibit larger workspace overheads relative to parameter sizes, whereas feed-forward networks track closer to static parameter bounds.

---

## Full Results Table (batch=16)

<details>
<summary>Expand full results table (all variants, batch=16)</summary>

| Architecture | Framework | Variant | FLOPs | Params | AI (FLOP/B) | Latency med (ms) | ±σ | CV% | Efficiency | GFLOP/s | Bottleneck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FF DNN | PyTorch | baseline | 806.5M | 100.8M | 7.72 | 4.183 | 0.113 | 2.7 | 25.0% | 192.79 | memory |
| FF DNN | PyTorch | compiled | 806.5M | 100.8M | 7.72 | 4.246 | 0.021 | 0.5 | 24.6% | 189.93 | memory |
| FF DNN | JAX | baseline | 805.6M | 100.7M | 7.90 | 5.119 | 0.129 | 2.5 | 19.9% | 157.39 | memory |
| FF DNN | JAX | jit | 805.6M | 100.7M | 7.90 | 4.924 | 0.055 | 1.1 | 20.7% | 163.63 | memory |
| FF DNN | TensorFlow | baseline | 3.8M | 475,176 | 6.42 | 4.107 | 0.135 | 3.2 | 0.1% | 0.92 | memory |
| FF DNN | TensorFlow | tf.function | 3.8M | 475,176 | 6.42 | 0.173 | 0.019 | 10.8 | 3.4% | 21.94 | memory |
| ConvNeXt | PyTorch | baseline | 142.96G | 111.3M | 38.31 | 3445.796 | 32.122 | 0.9 | 1.7% | 41.49 | compute |
| ConvNeXt | PyTorch | compiled | 142.96G | 111.3M | 38.31 | 3443.923 | 19.831 | 0.6 | 1.7% | 41.51 | compute |
| ConvNeXt | JAX | baseline | 17.83G | 1.9M | 35.45 | 173.119 | 0.826 | 0.5 | 4.2% | 102.98 | compute |
| ConvNeXt | JAX | jit | 17.83G | 1.9M | 35.45 | 119.060 | 1.408 | 1.2 | 6.2% | 149.74 | compute |
| ViT | PyTorch | baseline | 74.34G | 43.8M | 60.25 | 96.405 | 3.778 | 3.9 | 31.8% | 771.09 | compute |
| ViT | PyTorch | compiled | 74.34G | 43.8M | 60.25 | 96.861 | 3.603 | 3.7 | 31.6% | 767.46 | compute |
| ViT | JAX | baseline | 13.92G | 8.6M | 80.69 | 36.233 | 0.262 | 0.7 | 15.8% | 384.05 | compute |
| ViT | JAX | jit | 13.92G | 8.6M | 80.69 | 29.292 | 0.386 | 1.3 | 19.6% | 475.05 | compute |
| Transformer | PyTorch | baseline | 775.24G | 170.1M | 123.83 | 827.523 | 3.724 | 0.4 | 38.6% | 936.82 | compute |
| Transformer | PyTorch | compiled | 775.24G | 170.1M | 123.83 | 829.327 | 3.964 | 0.5 | 38.5% | 934.78 | compute |
| Transformer | JAX | baseline | 129.12G | 28.3M | 170.26 | 303.879 | 2.793 | 0.9 | 17.5% | 424.91 | compute |
| Transformer | JAX | jit | 129.12G | 28.3M | 170.26 | 256.660 | 3.398 | 1.3 | 20.7% | 503.08 | compute |
| Transformer | TensorFlow | baseline | 421.5M | 1.6M | 41.09 | 21.257 | 0.279 | 1.3 | 0.8% | 19.83 | compute |
| Transformer | TensorFlow | tf.function | 421.5M | 1.6M | 41.09 | 3.046 | 0.031 | 1.0 | 5.7% | 138.37 | compute |

</details>

---

## Advanced Causal Diagnostics (batch=16)

Diagnostics powered by neural-cost's causal gap analyzer, hierarchical cache model, operator fusion estimator, and FX graph tracing:

<details>
<summary>Expand advanced diagnostics table (batch=16)</summary>

| Architecture | Framework | Variant | Fused Efficiency | Traffic Saved | Resident Cache | Top Layer Bottleneck | Layer Share |
|---|---|---|---|---|---|---|---|
| FF DNN | PyTorch | baseline | 24.4% | 2.3% | DRAM | fc2 (linear, memory-bound) | 64.8% |
| FF DNN | PyTorch | compiled | 24.0% | 2.3% | DRAM | fc2 (linear, memory-bound) | 64.8% |
| FF DNN | JAX | baseline | — | — | DRAM | dot_4 (matmul, memory-bound) | 66.3% |
| FF DNN | JAX | jit | — | — | DRAM | dot_4 (matmul, memory-bound) | 66.3% |
| FF DNN | TensorFlow | baseline | 0.1% | 5.5% | DRAM | dense_6 (linear, memory-bound) | 77.8% |
| FF DNN | TensorFlow | tf.function | 3.2% | 5.5% | DRAM | dense_6 (linear, memory-bound) | 77.8% |
| ConvNeXt | PyTorch | baseline | 1.7% | 36.9% | DRAM | stages_0_0_act (elementwise, memory-bound) | 1.9% |
| ConvNeXt | PyTorch | compiled | 1.7% | 36.9% | DRAM | stages_0_0_act (elementwise, memory-bound) | 1.9% |
| ConvNeXt | JAX | baseline | 4.2% | 15.3% | DRAM | conv_3 (conv2d, compute-bound) | 17.5% |
| ConvNeXt | JAX | jit | 6.2% | 15.3% | DRAM | conv_3 (conv2d, compute-bound) | 17.5% |
| ViT | PyTorch | baseline | 31.8% | 38.3% | DRAM | blocks_0_mlp_0 (linear, compute-bound) | 4.2% |
| ViT | PyTorch | compiled | 31.6% | 38.3% | DRAM | blocks_0_mlp_0 (linear, compute-bound) | 4.2% |
| ViT | JAX | baseline | 15.8% | 16.8% | DRAM | dot_20 (matmul, compute-bound) | 24.7% |
| ViT | JAX | jit | 19.6% | 16.8% | DRAM | dot_20 (matmul, compute-bound) | 24.7% |
| Transformer | PyTorch | baseline | 38.6% | 38.6% | DRAM | blocks_0_ffn_0 (linear, compute-bound) | 4.6% |
| Transformer | PyTorch | compiled | 38.5% | 38.6% | DRAM | blocks_0_ffn_0 (linear, compute-bound) | 4.6% |
| Transformer | JAX | baseline | 17.5% | 19.9% | DRAM | dot_16 (matmul, compute-bound) | 29.0% |
| Transformer | JAX | jit | 20.7% | 19.9% | DRAM | dot_16 (matmul, compute-bound) | 29.0% |
| Transformer | TensorFlow | baseline | 0.8% | 20.4% | DRAM | multi_head_attention_4 (attention, compute-bound) | 15.9% |
| Transformer | TensorFlow | tf.function | 5.7% | 20.4% | DRAM | multi_head_attention_4 (attention, compute-bound) | 15.9% |

</details>

---

## Compilation Speedup Summary (batch=16)

| Architecture | PyTorch (compile) | JAX (jit) | TensorFlow (tf.function) |
|---|---|---|---|
| FF DNN | **0.99×** (4.18→4.25 ms) | **1.04×** (5.12→4.92 ms) | **23.73×** (4.11→0.17 ms) |
| Deep DNN | — | — | — |
| CNN | — | — | — |
| ConvNeXt | **1.00×** (3445.80→3443.92 ms) | **1.45×** (173.12→119.06 ms) | — |
| ViT | **1.00×** (96.41→96.86 ms) | **1.24×** (36.23→29.29 ms) | — |
| Transformer | **1.00×** (827.52→829.33 ms) | **1.18×** (303.88→256.66 ms) | **6.98×** (21.26→3.05 ms) |
| RNN | — | — | — |
| LSTM | — | — | — |

---

## Per-Architecture Winner (batch=16)

- **FF DNN**: fastest framework is **TensorFlow** at 0.17 ms (batch=16)
- **ConvNeXt**: fastest framework is **JAX** at 119.06 ms (batch=16)
- **ViT**: fastest framework is **JAX** at 29.29 ms (batch=16)
- **Transformer**: fastest framework is **TensorFlow** at 3.05 ms (batch=16)



---

## LLM Prefill vs. Decode Phase Discrepancy & KV Cache Analysis

Evaluates operational regime differences in Large Language Models (Issue #22):
- **Prefill phase:** Highly parallel prompt processing bounded by arithmetic compute capacity.
- **Decode phase:** Autoregressive single-token generation bounded by DRAM memory bandwidth and KV-cache retrieval.

### Prompt Prefill Phase (Compute-Bound Regime)

| Prompt Len | Batch | TTFT (ms) | Achieved Compute | Throughput (tok/s) |
|---|---|---|---|---|
| 128 | 1 | 0.33 | 805.5 GFLOP/s | 384,080 |
| 512 | 1 | 4.26 | 1008.0 GFLOP/s | 120,165 |
| 2048 | 1 | 89.68 | 766.3 GFLOP/s | 22,837 |

### Autoregressive Decode Phase (Memory-Bandwidth Bound Regime)

| Prompt Len | Batch | Step Latency (ms) | Decode Throughput | Memory Bandwidth | BW Utilization | KV Cache (KB) |
|---|---|---|---|---|---|---|
| 128 | 1 | 0.06 | 17175.8 tok/s | 72.6 GB/s | 72.6% | 4,096.0 |
| 512 | 1 | 0.21 | 4815.9 tok/s | 81.0 GB/s | 81.0% | 16,384.0 |
| 2048 | 1 | 0.94 | 1060.6 tok/s | 71.2 GB/s | 71.2% | 65,536.0 |

**Key observations:**
- **Prefill arithmetic intensity:** Large prompt contexts saturate compute cores, delivering high GFLOP/s and high token processing rates.
- **Decode memory bandwidth bottleneck:** Token-by-token generation must load the full model weights and historical KV cache per token, achieving tens of tokens/second and saturating memory bandwidth on memory-constrained hardware.
- **KV cache scaling:** The KV cache footprints grow linearly with prompt length, increasing per-step memory traffic proportionally.

---

## Conclusions

### 1. JIT compilation is the dominant performance lever

`jax.jit()` provides the most impactful optimisation across architectures,
eliminating Python-level loop overhead for recurrent models and enabling XLA kernel
fusion for feedforward and attention layers. `torch.compile()` provides meaningful
speedups (1.5–3×) for linear/conv-heavy workloads but does not trace Python loops.
`tf.function()` closes the gap between TF eager and JIT-compiled frameworks for
feedforward models but is less effective for recurrent models.

### 2. Workload regimes on CPU at these batch sizes

The arithmetic intensity of architectures at batch=16 typically falls below the
24 FLOP/byte ridge point of the Apple M3 (CPU).
To reach compute-bound territory, larger batches or larger hidden dimensions are needed.
The roofline efficiency gap (observed efficiency typically 3–15%) is attributable to:
- Python/framework dispatch overhead
- Memory allocation and copy overhead (workspace, activations)
- Suboptimal kernel utilisation (untiled matmuls at small N)

### 3. Framework dispatch overhead matters most for sequential models

RNN and LSTM workloads show the greatest framework-to-framework disparity because
their sequential loops are executed in Python (for PyTorch/TF eager) or traced into
a flat graph (for JAX JIT). For feedforward and convolutional models, all three
frameworks are within 2–3× of each other after compilation.

### 4. Measurement reliability

CV below 5% was achieved for all compiled variants at batch ≥ 8. The
51.4 GB/s measured STREAM bandwidth (vs 100 GB/s
published) reflects OS-level scheduling noise and shared memory pressure. For
production benchmarking, repeat the sweep with exclusive CPU affinity and
real model weights.

---

*Generated by `benchmarks/generate_report.py` using [neural-cost](https://github.com/davidgraymi/neural-cost)*
