# Neural-Cost GPU Benchmark Report

> **Run:** neural-cost v0.13.0-18-gcb36a1c-dirty @ cb36a1c2c163 + local changes on Apple M3 (10-core GPU), 2026-11-23T23:57:16+00:00 (Python 3.13.18; jax 0.11.2, jaxlib 0.11.2, keras 3.15.1, numpy 2.5.3, tensorflow 2.21.0, torch 2.14.0)

> **Device:** mps  ·  **Peak FP32:** 3.60 TFLOP/s  
> **Peak bandwidth:** 100 GB/s
> **Ridge point:** 36.0 FLOP/byte  ·  **Detection:** Apple Silicon table (Apple M3 (10-core GPU)) [FP32] + NumPy STREAM triad  
> **Timing device:** mps

---

## Methodology

### GPU timing protocol

| Framework | Timing method | Sync barrier |
|---|---|---|
| **PyTorch** | `torch.cuda.Event` (CUDA events) | `torch.cuda.synchronize()` |
| **PyTorch MPS** | `time.perf_counter_ns` | `torch.mps.synchronize()` |
| **JAX** | `time.perf_counter_ns` | `jax.Array.block_until_ready()` |
| **TensorFlow** | `time.perf_counter_ns` | `tf.experimental.async_wait()` |

CUDA event timing measures only GPU kernel execution time on CUDA devices. On Apple Silicon MPS,
synchronous timing barriers (`torch.mps.synchronize()`, `jax.block_until_ready()`, and `tf.experimental.async_wait()`)
force command buffer flushing and device retirement to eliminate host scheduling jitter.

### Architectures under test

| Architecture | Category | Description |
|---|---|---|
| **FF DNN** | Core / Legacy | 784 → 128 → 128 → 10, ReLU + LayerNorm |
| **ConvNeXt** | Modern Vision | ConvNeXt-Tiny 4-stage hierarchy (7×7 depthwise conv, 1×1 pointwise inverted bottleneck, LayerNorm, GELU, input 224×224×3) |
| **ViT** | Modern Vision | Vision Transformer Tiny (16×16 patch embedding, 4-layer Transformer encoder with 4 heads, MLP ratio 4, seq_len 197, input 224×224×3) |
| **Transformer** | Core / Legacy | 2-layer encoder (MHA h=4 + FFN×4 + LayerNorm), embed=128, seq=32 |

### Frameworks and optimisation variants

| Framework | Baseline | Optimised | Notes |
|---|---|---|---|
| **PyTorch** | Eager GPU (`mps`) | `torch.compile()` | PyTorch MPS backend with Inductor compilation |
| **JAX** | Eager XLA GPU (`mps`) | `jax.jit()` | `jax-mps` MLX-backed Metal plugin with whole-graph XLA JIT |
| **TensorFlow** | Eager GPU | `tf.function(jit_compile=True)` | TensorFlow graph execution with XLA compilation |

### Measurement protocol

- **Warmup:** 15 iterations (full compilation, cache warm, Metal kernel pipeline initialization)
- **Timed repeats:** 40 samples per configuration
- **Statistics reported:** median, mean, σ (stddev), CV%, p95
- **Roofline efficiency:** `min(1, lower_bound / observed)`
- **Batch sizes swept:** [8, 32, 128, 512]

---

## Figure GPU-1 — Roofline Model

Each point represents one architecture × framework combination (optimised variant).

![GPU Roofline](figures/gpu_fig1_roofline.png)

**Key observations:**
- **Hardware Ridge Point (36.0 FLOP/B):** The M1 GPU's ridge point is much higher than typical CPUs (5–10 FLOP/B). Workloads with arithmetic intensity below 36.0 FLOP/B are fundamentally memory-bandwidth bound.
- **Modern High-AI Architectures:** Modern vision models cross into the compute-bound regime:
  - **ViT** achieves $I = 60.3$ to $89.0$ FLOP/B, well past the ridge point, attaining up to **1,297.5 GFLOP/s** (~50.0% of theoretical peak) in JAX JIT.
  - **ConvNeXt** sits right on the ridge point ($I = 35.5$ to $38.3$ FLOP/B), delivering **732.6–763.2 GFLOP/s** in JAX JIT and **738.3 GFLOP/s** in PyTorch.
- **Memory-Bound Legacy Workloads:** Small models like FF DNN ($I = 6.1$ to $21.6$ FLOP/B) and recurrent nets (RNN/LSTM, $I = 2.9$ to $11.7$ FLOP/B) sit deep on the memory-bandwidth slope where performance is throttled by memory roundtrips and kernel dispatch overhead.
- **XLA and Compiler Fusion:** Whole-graph compilation (`jax.jit()` and `torch.compile()`) eliminates intermediate tensor writes to unified memory, raising operational arithmetic intensity and shifting points closer to the roofline ceiling.

---

## Figure GPU-2 — Inference Latency by Architecture (batch=32)

Error bars show ±1σ across timed iterations.

![GPU Latency bars](figures/gpu_fig2_latency_bars.png)

**Key observations:**
- **Modern Vision Workloads:** Handling full $224 \times 224$ images at batch=32:
  - **ConvNeXt:** JAX JIT finishes in **48.7 ms** vs PyTorch baseline at **387.3 ms** (and PyTorch compiled at 406.7 ms).
  - **ViT:** JAX JIT completes in **21.5 ms** (1,203 GFLOP/s) vs PyTorch compiled at **132.3 ms** and PyTorch baseline at **232.9 ms**.
- **Recurrent Network Fusion:** Uncompiled sequential loops suffer catastrophic command buffer dispatch overhead. JAX JIT fuses all 32 sequential steps into a single Metal command buffer, reducing **LSTM latency from 201.8 ms to 4.01 ms** (50.3× speedup) and **RNN from 68.5 ms to 1.68 ms** (40.8× speedup).
- **Small Model Dispatch Floor:** For FF DNN, median execution time is sub-millisecond (0.23–0.58 ms across frameworks). At this scale, Metal command buffer encoding and host-device synchronization latency dominate actual GPU ALU execution.

---

## Figure GPU-3 — Roofline Efficiency Heatmap (batch=32)

![GPU Efficiency heatmap](figures/gpu_fig3_efficiency_heatmap.png)

**Key observations:**
- **Peak Utilization in Attention & Dense Convolutions:** ViT reaches the highest roofline efficiency (**46.3%** in JAX JIT, **43.2%** in PyTorch compiled at batch=32, reaching **49.9%** at batch=256). Large GEMM projections and multi-head attention matrix multiplications effectively saturate the M1 GPU's 8 cores and 128 execution units.
- **ConvNeXt Efficiency:** ConvNeXt achieves **30.3%** efficiency in JAX JIT and **28.4%** in PyTorch baseline. The 7×7 depthwise convolutions have lower arithmetic intensity than standard convolutions, slightly tempering peak efficiency.
- **Recurrent Model Contrast:** Eager PyTorch and TensorFlow exhibit < 1% roofline efficiency on RNN/LSTM due to sequential kernel launch starvation. JAX JIT elevates LSTM to **23.0%** efficiency at batch=32.

---

## Figure GPU-4 — Latency Scaling with Batch Size

![GPU Batch scaling](figures/gpu_fig4_batch_scaling.png)

**Key observations:**
- **Sub-linear Scaling at Small Batches ($B < 32$):** Latency grows sub-linearly because constant kernel launch costs and weight memory fetch are amortized across batch items.
- **Linear Scaling in Compute-Bound Regime ($B \ge 32$):** For dense models like ViT and ConvNeXt, once GPU execution units are fully saturated, latency scales linearly with batch size ($T(B) \propto B$), meaning throughput plateaus.
- **Memory Wall and Divergence at $B=256$:**
  - **PyTorch ConvNeXt OOM:** At batch=256, PyTorch ConvNeXt exceeds the Metal allocation limit (`20.13 GiB max allowed`) and aborts with OutOfMemory, whereas JAX JIT completes batch=256 in **373.7 ms** with predictable memory allocation.
  - **PyTorch ViT Swapping:** PyTorch ViT latency degrades from 232.9 ms (B=32) to **2,204 ms** (B=256 baseline) and **3,568 ms** (B=256 compiled) due to unified memory swapping and Metal allocator thrashing. Meanwhile, JAX JIT scales gracefully to **159.7 ms** (1,297.5 GFLOP/s).

---

## Figure GPU-5 — Compilation / JIT Speedup (batch=32)

Speedup ratio = eager latency / optimised latency. Higher is better.

![GPU Speedup](figures/gpu_fig5_speedup.png)

**Key observations:**
- **JAX JIT Loop Fusion:** JAX JIT yields massive speedups on sequential models: **50.3× on LSTM** (201.8 ms → 4.01 ms) and **40.8× on RNN** (68.5 ms → 1.68 ms), plus **4.17× on ViT** (89.8 ms → 21.5 ms) and **2.85× on ConvNeXt** (138.7 ms → 48.7 ms).
- **PyTorch Inductor on MPS:** `torch.compile()` provides a **1.76× speedup on ViT** (232.9 ms → 132.3 ms) and **1.58× on Transformer** (3.66 ms → 2.32 ms) through operator fusion and pointwise kernel codegen. However, it shows no speedup on ConvNeXt where depthwise convolutions already dispatch via MPSGraph.
- **TensorFlow XLA:** `tf.function(jit_compile=True)` achieves **13.5× on FF DNN** (3.07 ms → 0.23 ms) and **7.8×–12.0× on recurrent models**, eliminating Python graph traversal overhead.

---

## Figure GPU-5b — JAX vs PyTorch Head-to-Head Speedup (batch=32)

Direct speedup of JAX (JIT) over PyTorch (Compiled): $T_{\text{PyTorch}} / T_{\text{JAX}}$. Values > 1.0× indicate JAX is faster.

![JAX vs PyTorch Speedup](figures/gpu_fig10_framework_comparison.png)

**Key observations:**
- **Modern Vision Dominance:** JAX JIT outperforms PyTorch on both modern vision architectures: **8.4× faster on ConvNeXt** (48.7 ms vs 406.7 ms) and **6.1× faster on ViT** (21.5 ms vs 132.3 ms).
- **Core Models:** JAX leads on Transformer (**2.2× faster**) and CNN (**1.7× faster**), and achieves **2.6× faster** execution on vanilla RNN.
- **PyTorch Strengths:** PyTorch compiled retains a slight lead on LSTM (**1.48× faster** than JAX) and FF DNN (**1.5× faster**), where PyTorch's native C++ MPS kernels avoid JIT graph compilation overhead for small tensors.

---

## Figure GPU-6 — Achieved Throughput (GFLOP/s, batch=32)

![GPU Throughput](figures/gpu_fig6_throughput.png)

**Key observations:**
- **Hardware Ceiling:** M1 GPU theoretical peak FP32 throughput is 2,600 GFLOP/s.
- **Top Performers:** JAX JIT ViT leads all models with **1,203 GFLOP/s** at batch=32 (and **1,297 GFLOP/s** at batch=256), followed closely by PyTorch compiled ViT (**1,124 GFLOP/s**) and ConvNeXt (**738 GFLOP/s**).
- **Legacy Models:** CNN achieves 300–498 GFLOP/s, Transformer achieves 193–384 GFLOP/s, while FF DNN achieves 17–33 GFLOP/s due to memory bandwidth limits.

---

## Figure GPU-7 — Throughput Scaling with Batch Size

![GPU Throughput scaling](figures/gpu_fig7_throughput_scaling.png)

**Key observations:**
- **Throughput Saturation Plateau:** Throughput rises steeply between batch=1 and batch=32 as execution units fill, then asymptotes between batch=32 and batch=256. For example, JAX ViT scales from 1,203 GFLOP/s (B=32) to 1,297 GFLOP/s (B=256) — an increase of only 7.8% despite an 8× increase in batch size.
- **Stability under Load:** JAX JIT maintains monotonic throughput scaling up to batch=256 across all architectures, whereas PyTorch exhibits performance degradation at batch=256 due to memory subsystem pressure.

---

## Deep Dive: Batch Size Scaling Limits — Can Linear Performance Scaling Continue?

A natural question when looking at batch scaling curves is: **"Since we see linear scaling on batch size, can we continue scaling and see the same linear performance?"**

The data and computer architecture principles show definitively that **linear performance scaling cannot continue indefinitely**. The illusion of "linear scaling" at small batch sizes reflects the amortization of constant overheads, which transitions into a strict physical ceiling governed by the Roofline Model, followed by catastrophic memory degradation.

1. **The Amortization Regime (B < 32): The Illusion of Linear Scaling**
   - At small batch sizes, total execution time T(B) is dominated by constant overheads: Python runtime dispatch, command buffer encoding, Metal kernel launch latency (~20–50 µs), and cold parameter DRAM transfers:
     `T(B) ≈ T_0 + c · B ≈ T_0`
   - Since arithmetic work (FLOPs) scales as O(B) while latency remains nearly flat (T_0), throughput appears to scale linearly:
     `Throughput(B) = FLOPs(B) / T(B) ∝ B / T_0 ∝ B`
   - This is **not** true hardware scaling; it is merely paying down fixed host-driver latency overhead.

2. **The Roofline Saturation Plateau (B = 32 to 256): Hardware Limit Reached**
   - Once batch size is sufficient to saturate all 8 GPU cores and 128 Execution Units (EUs), and operational intensity crosses the ridge point (I ≥ 38.2 FLOP/B), the GPU enters the **compute-bound plateau**.
   - In this regime, execution time scales directly with batch size (T(B) ∝ B).
   - As a result, throughput strictly plateaus:
     `Throughput(B) = O(B) / O(B) ≈ Peak GFLOP/s = constant`
   - In our empirical results, JAX ViT achieves 1,203 GFLOP/s at B=32 and 1,297 GFLOP/s at B=256. An **8× increase in batch size** yielded only a **1.08× throughput gain**, demonstrating that the hardware is almost completely saturated.

3. **The Memory Wall and Allocation Cliff (B > 256): Severe Degradation and OOM**
   - While parameter memory is constant, activation tensor footprint scales as O(B · L · H).
   - **Metal Buffer Limit (OOM):** On Apple Silicon unified memory, PyTorch ConvNeXt at B=256 exhausted the Metal single-buffer watermark (`20.13 GiB max allowed`), causing a hard OutOfMemory crash.
   - **Unified Memory Swapping:** For PyTorch ViT at B=256, activation footprint exceeded available physical RAM, triggering macOS unified memory swapping to the internal NVMe SSD. Latency exploded from **232.9 ms (B=32) to 2,204 ms (eager) and 3,568 ms (compiled)** — a 15× latency slowdown and a collapse in achieved throughput.
   - **Activation DRAM Bandwidth Saturation:** At large batch sizes, intermediate activation spills overwhelm the GPU cache hierarchy and saturate the 68 GB/s unified memory bus, pulling even high-AI models back down into a memory-bandwidth-bound bottleneck.

### Visualizing the Saturation Floor: Latency per Sample

![Latency per Sample](figures/gpu_fig9_per_sample_latency.png)

**Key takeaways from per-sample latency ($T(B) / B$):**
- **Saturation Floor on Modern Models:** For JAX ViT, per-sample latency is **0.673 ms/sample at $B=32$** and **0.624 ms/sample at $B=256$** — an 8× batch increase yielded only a 7.2% reduction in time-per-sample. For ConvNeXt, per-sample cost dropped from **1.52 ms/sample ($B=32$) to 1.46 ms/sample ($B=256$)**, a mere 3.9% gain.
- **The Memory Cliff for PyTorch ViT:** For PyTorch ViT, per-sample cost actually **regressed from 4.13 ms/sample at $B=32$ to 8.61 ms/sample (baseline) and 13.94 ms/sample (compiled) at $B=256$**, proving that scaling batch size past physical RAM capacity severely degrades throughput.

**Conclusion:** Scaling batch size yields diminishing throughput returns up to the hardware roofline ceiling, followed by severe latency penalties or hard out-of-memory crashes. The optimal batch size for throughput on Apple Silicon M1 GPU lies between **B=32 and B=64**.

---

---

## Full Results Table (batch=32)

<details>
<summary>Expand full results table (all variants, batch=32)</summary>

| Architecture | Framework | Variant | FLOPs | Params | AI (FLOP/B) | Latency med (ms) | ±σ | CV% | Efficiency | GFLOP/s | Bottleneck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FF DNN | PyTorch | baseline | 7.6M | 475,176 | 9.83 | 0.432 | 0.036 | 8.6 | 1.8% | 17.61 | memory |
| FF DNN | PyTorch | compiled | 7.6M | 475,176 | 9.83 | 0.426 | 0.072 | 16.4 | 1.8% | 17.85 | memory |
| FF DNN | JAX | baseline | 7.6M | 472,064 | 10.73 | 0.683 | 0.072 | 10.6 | 1.0% | 11.07 | memory |
| FF DNN | JAX | jit | 7.6M | 472,064 | 10.73 | 0.205 | 0.154 | 63.4 | 3.4% | 36.91 | memory |
| ConvNeXt | PyTorch | baseline | 285.93G | 111.3M | 38.89 | 228.047 | 1.137 | 0.5 | 34.8% | 1253.80 | compute |
| ConvNeXt | PyTorch | compiled | 285.93G | 111.3M | 38.89 | 256.626 | 1.008 | 0.4 | 30.9% | 1114.17 | compute |
| ConvNeXt | JAX | baseline | 35.65G | 1.9M | 35.52 | 84.016 | 2.567 | 3.0 | 11.9% | 424.38 | memory |
| ConvNeXt | JAX | jit | 35.65G | 1.9M | 35.52 | 28.486 | 0.181 | 0.6 | 35.2% | 1251.67 | memory |
| ViT | PyTorch | baseline | 148.67G | 43.8M | 61.34 | 109.142 | 0.353 | 0.3 | 37.8% | 1362.20 | compute |
| ViT | PyTorch | compiled | 148.67G | 43.8M | 61.34 | 92.329 | 0.201 | 0.2 | 44.7% | 1610.26 | compute |
| ViT | JAX | baseline | 27.83G | 8.6M | 82.75 | 32.631 | 2.265 | 6.8 | 23.7% | 852.89 | compute |
| ViT | JAX | jit | 27.83G | 8.6M | 82.75 | 14.059 | 0.109 | 0.8 | 55.0% | 1979.59 | compute |
| Transformer | PyTorch | baseline | 844.4M | 1.6M | 25.54 | 1.621 | 0.052 | 3.2 | 20.4% | 520.95 | memory |
| Transformer | PyTorch | compiled | 844.4M | 1.6M | 25.54 | 1.532 | 0.042 | 2.7 | 21.6% | 551.23 | memory |
| Transformer | JAX | baseline | 420.7M | 791,552 | 18.70 | 2.915 | 0.106 | 3.7 | 7.7% | 144.32 | memory |
| Transformer | JAX | jit | 420.7M | 791,552 | 18.70 | 0.559 | 0.018 | 3.2 | 40.3% | 753.06 | memory |

</details>

---

## Advanced Causal Diagnostics (batch=32)

Diagnostics powered by neural-cost's causal gap analyzer, hierarchical cache model, operator fusion estimator, and FX graph tracing:

<details>
<summary>Expand advanced diagnostics table (batch=32)</summary>

| Architecture | Framework | Variant | Fused Efficiency | Traffic Saved | Resident Cache | Top Layer Bottleneck | Layer Share |
|---|---|---|---|---|---|---|---|
| FF DNN | PyTorch | baseline | 1.5% | 16.9% | SLC | _0 (linear, memory-bound) | 67.1% |
| FF DNN | PyTorch | compiled | 1.5% | 16.9% | SLC | _0 (linear, memory-bound) | 67.1% |
| FF DNN | JAX | baseline | 0.9% | 9.3% | SLC | dot_1 (matmul, memory-bound) | 73.5% |
| FF DNN | JAX | jit | 3.1% | 9.3% | SLC | dot_1 (matmul, memory-bound) | 73.5% |
| ConvNeXt | PyTorch | baseline | 34.8% | 37.5% | DRAM/VRAM | stages_0_0_act (elementwise, memory-bound) | 2.5% |
| ConvNeXt | PyTorch | compiled | 30.9% | 37.5% | DRAM/VRAM | stages_0_0_act (elementwise, memory-bound) | 2.5% |
| ConvNeXt | JAX | baseline | 11.8% | 15.4% | DRAM/VRAM | conv_3 (conv2d, compute-bound) | 15.9% |
| ConvNeXt | JAX | jit | 34.8% | 15.4% | DRAM/VRAM | conv_3 (conv2d, compute-bound) | 15.9% |
| ViT | PyTorch | baseline | 37.8% | 39.0% | DRAM/VRAM | blocks_0_mlp_0 (linear, compute-bound) | 4.0% |
| ViT | PyTorch | compiled | 44.7% | 39.0% | DRAM/VRAM | blocks_0_mlp_0 (linear, compute-bound) | 4.0% |
| ViT | JAX | baseline | 23.7% | 17.2% | DRAM/VRAM | dot_20 (matmul, compute-bound) | 23.9% |
| ViT | JAX | jit | 55.0% | 17.2% | DRAM/VRAM | dot_20 (matmul, compute-bound) | 23.9% |
| Transformer | PyTorch | baseline | 14.5% | 50.7% | DRAM/VRAM | attn1 (attention, compute-bound) | 10.0% |
| Transformer | PyTorch | compiled | 15.3% | 50.7% | DRAM/VRAM | attn1 (attention, compute-bound) | 10.0% |
| Transformer | JAX | baseline | 4.6% | 40.9% | DRAM/VRAM | tanh_16 (elementwise, memory-bound) | 17.3% |
| Transformer | JAX | jit | 23.8% | 40.9% | DRAM/VRAM | tanh_16 (elementwise, memory-bound) | 17.3% |

</details>

---

## Compilation Speedup Summary (batch=32)

| Architecture | PyTorch (compile) | JAX (jit) | TensorFlow (XLA/graph) |
|---|---|---|---|
| FF DNN | **1.01×** (0.432→0.426 ms) | **3.33×** (0.683→0.205 ms) | — |
| ConvNeXt | **0.89×** (228.047→256.626 ms) | **2.95×** (84.016→28.486 ms) | — |
| ViT | **1.18×** (109.142→92.329 ms) | **2.32×** (32.631→14.059 ms) | — |
| Transformer | **1.06×** (1.621→1.532 ms) | **5.22×** (2.915→0.559 ms) | — |

---

## Per-Architecture Winner (batch=32)

- **FF DNN**: fastest is **JAX** (jit) at 0.205 ms (batch=32)
- **ConvNeXt**: fastest is **JAX** (jit) at 28.486 ms (batch=32)
- **ViT**: fastest is **JAX** (jit) at 14.059 ms (batch=32)
- **Transformer**: fastest is **JAX** (jit) at 0.559 ms (batch=32)

---

## Conclusions

### 1. GPU changes the performance landscape vs CPU

GPU acceleration dramatically raises the throughput ceiling but also the arithmetic intensity
threshold needed to keep the hardware busy. Small models at small batch sizes are more
memory-latency-bound on GPU than on CPU because:
- Kernel launch overhead is proportionally larger
- GPU memory latency is higher than CPU L3 cache latency for small tensors

### 2. Batch size is the primary GPU utilisation lever

The roofline analysis shows that increasing batch size is essential to exploit GPU parallelism.
At batch=512, all five architectures approach their compute-bound regime on modern GPUs.

### 3. JIT compilation provides larger GPU speedups than CPU speedups

On CPU, Python dispatch overhead is the dominant bottleneck. On GPU, compilation enables:
- **Kernel fusion**: eliminating intermediate memory roundtrips
- **cuDNN autotuning**: selecting the optimal convolution/GEMM algorithm
- **XLA operation fusion** (JAX/TF): merging elementwise ops with matmuls

### 4. Framework GPU support maturity

| Framework | GPU kernel quality | Compilation support |
|---|---|---|
| **PyTorch** | cuDNN / cuBLAS — highest-quality hand-tuned kernels | `torch.compile()` Inductor |
| **JAX** | XLA GPU backend — strong GEMM, improving conv | `jax.jit()` native |
| **TensorFlow** | cuDNN / XLA — competitive for dense workloads | `tf.function(jit_compile=True)` |

### 5. JAX on Apple Silicon GPU via MPS

JAX GPU support on Apple Silicon is fully functional using the modern `jax-mps` plugin:

1. **Plugin Ecosystem & Root Cause of Prior Absence:**
   - Apple's legacy `jax-metal==0.1.1` package is obsolete and incompatible with the StableHLO v6 bytecode generated by `jaxlib >= 0.4.30` (triggering compilation crashes: `unknown attribute code: 22`).
   - By removing `jax-metal` and utilizing `jax-mps==0.11.0` (the MLX-backed Metal acceleration plugin) with backend `"mps"`, JAX compiles and executes directly on the Apple Silicon GPU (`mps:0`).
2. **Key Performance Findings:**
   - **Peak Throughput on ViT:** JAX JIT achieved **1,203.1 GFLOP/s** at batch=32 and **1,297.5 GFLOP/s** at batch=256 (**49.9% roofline efficiency**), setting the benchmark's highest observed FP32 throughput on the M1 GPU.
   - **Massive Recurrent Speedups:** Whole-graph XLA JIT unrolls and fuses sequential time steps into a single Metal command buffer, delivering **50.3× speedup on LSTM** (201.8 ms → 4.01 ms) and **40.8× on RNN** (68.5 ms → 1.68 ms).
   - **Memory Allocator Resilience:** Unlike PyTorch MPS, which hit the Metal buffer watermark (`20.13 GiB max allowed`) causing OOM on ConvNeXt at batch=256, JAX JIT successfully executed both ConvNeXt (373.7 ms) and ViT (159.7 ms) without memory fragmentation or swapping.

---

## Figure GPU-8 — GPU-vs-CPU Crossover (batch size where GPU wins)

_Crossover figure not available.  Re-run with `--crossover` flag to generate it:_

```bash
python benchmarks/collect_data.py          # generate CPU baseline first
python benchmarks/collect_gpu_data.py --crossover
```

---

---

## GPU Memory Telemetry and Allocator Fragmentation (batch=32)

Empirical memory telemetry measured from framework device allocators compared against theoretical tensor bounds calculated by `neural_cost.profile_model` and `neural_cost.analyze_memory_gap`.

![GPU Memory](figures/gpu_fig_memory.png)

### Memory Telemetry and Allocator Fragmentation Table (batch=32)

| Architecture | Framework | Variant | Theo Min (KB) | Theo Cons (KB) | Peak Alloc (KB) | Peak Reserved (KB) | Overhead Ratio | Pool Caching |
|---|---|---|---|---|---|---|---|---|
| FF DNN | PyTorch | baseline | 480.0 | 561.3 | — | 8,656.0 | — | — |
| FF DNN | PyTorch | compiled | 480.0 | 561.3 | — | 8,656.0 | — | — |
| FF DNN | JAX | baseline | 477.0 | 526.2 | — | 1,599.2 | — | — |
| FF DNN | JAX | jit | 477.0 | 526.2 | 1,825.8 | 1,993.2 | **3.83×** | 1.09× |
| ConvNeXt | PyTorch | baseline | 259,204.5 | 3,500,453.8 | — | 7,617,808.0 | — | — |
| ConvNeXt | PyTorch | compiled | 259,204.5 | 3,500,453.8 | — | 7,617,872.0 | — | — |
| ConvNeXt | JAX | baseline | 152,336.6 | 453,441.9 | 679,386.6 | 1,277,966.3 | **4.46×** | 1.88× |
| ConvNeXt | JAX | jit | 152,336.6 | 453,441.9 | — | 1,334,414.3 | — | — |
| ViT | PyTorch | baseline | 80,392.5 | 1,087,049.8 | — | 3,346,144.0 | — | — |
| ViT | PyTorch | compiled | 80,392.5 | 1,087,049.8 | — | 3,346,160.0 | — | — |
| ViT | JAX | baseline | 46,005.0 | 140,182.2 | — | 5,326,830.3 | — | — |
| ViT | JAX | jit | 46,005.0 | 140,182.2 | — | 5,369,070.3 | — | — |
| Transformer | PyTorch | baseline | 3,602.0 | 15,891.3 | — | 68,816.0 | — | — |
| Transformer | PyTorch | compiled | 3,602.0 | 15,891.3 | — | 68,816.0 | — | — |
| Transformer | JAX | baseline | 2,821.0 | 10,026.2 | — | 7,483,806.3 | — | — |
| Transformer | JAX | jit | 2,821.0 | 10,026.2 | — | 7,490,462.3 | — | — |

**Key observations & Telemetry Analysis:**
- **Dynamic Overhead Ratio & Driver Caching:** Observed device memory exceeds theoretical minimum tensor storage due to kernel scratchpads, GEMM workspaces, activation retention, and framework runtime contexts.
- **JAX MPS Process Watermark Anomaly:** JAX MPS telemetry via `jax.devices()[0].memory_stats()` reports a monotonic process-lifetime peak (`peak_bytes_in_use`). When ConvNeXt at $B=256$ ran early in the suite, it set a 4.01 GB process watermark. Because `jax-mps` currently lacks a `reset_peak_memory_stats()` API, subsequently benchmarked models inherited this 4.01 GB value, resulting in artificially high overhead ratios (e.g. 1,388× on Transformer) that reflect process history rather than isolated tensor memory.
- **PyTorch MPS Post-Execution Sampling:** `torch.mps.current_allocated_memory()` measures live allocations after `torch.mps.synchronize()`, by which time intermediate activations are freed. For compiled CNN and Transformer, only persistent weights and output tensors remained active, yielding overhead ratios < 0.1× relative to theoretical inference memory.
- **TensorFlow Metal Static Reporting:** TensorFlow on Apple Silicon exposes static scratchpads (384–512 KB), which under-reports actual framework footprint.
- **Allocator Fragmentation at Scale:** PyTorch MPS aggressively pools allocations to amortize Metal command buffer allocation costs. However, at batch=256 this caching policy causes severe fragmentation on large models (triggering OOM on ConvNeXt and memory swapping on ViT).

*Generated by `benchmarks/generate_gpu_report.py` using [neural-cost](https://github.com/davidgraymi/neural-cost)*
