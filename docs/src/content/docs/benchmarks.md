---
title: "Benchmarks"
description: "Architecture comparisons, GPU benchmarks and Apple Silicon results."
section: "Reference"
order: 3
---

Run the architecture comparison script to benchmark five canonical neural
network families side-by-side across all installed frameworks:

```bash
pip install "neural-cost[torch,jax,tensorflow]"
# Or from local editable checkout:
# pip install -e '.[torch,jax,tensorflow]'
python examples/architecture_comparison.py
```

The script evaluates **FF DNN**, **CNN**, **RNN**, **LSTM**, and **Transformer**
architectures using a shared hidden dimension (128) and batch size (16).
Hardware is auto-detected; pass `--peak-flops` / `--memory-bandwidth` to override.

## Results layout and provenance

Every benchmark script times through `adapter.benchmark()`, the same code path
users run, so published numbers carry the library's device sync, sync-overhead
measurement, short-kernel batching and profiler device time. Results are
written per machine:

```text
benchmarks/results/<device>/          e.g. apple-m3/, tesla-t4/
  benchmark_data.json                 collect_data.py      (CPU)
  benchmark_gpu_data.json             collect_gpu_data.py  (GPU)
  llm_benchmark_data.json             llm_bench.py
  validation_data.json                validate.py
  *_REPORT.md, figures/               generate_*report.py, validate.py
```

Each JSON file starts with a `run` block (neural-cost version, `git describe`,
the full git commit and whether the tree had local changes, framework
versions, Python, platform, UTC time), and every record repeats the commit.
Reports print it on their first line. Pass `--results-dir` to any script to
write or read elsewhere; the report generators default to the most recently
written directory.

## Free CUDA runs (Colab or Kaggle)

[`benchmarks/cloud_gpu_benchmark.ipynb`](https://github.com/davidgraymi/neural-cost/blob/main/benchmarks/cloud_gpu_benchmark.ipynb)
runs the suite on a free NVIDIA GPU. Open it in Google Colab (*Runtime → Change
runtime type → T4 GPU*) or Kaggle (*Accelerator → GPU T4 x2 / P100*), set
`REF` to a release tag or commit, and run all cells. It clones that commit,
installs it, runs `validate.py` and `collect_gpu_data.py --device cuda`, builds
the reports and downloads a zip to unpack into `benchmarks/results/`.

## Release validation

`.github/workflows/benchmark-validation.yml` runs after every successful
Publish on `main` (and on demand). It runs
`validate.py --quick --fail-on-violation` on a hosted CPU runner and fails if
any measured runtime beats its roofline lower bound. Hosted runners are shared
VMs, so their timings are kept as a build artifact, never published.

## GPU benchmark

Run the GPU benchmark to evaluate the same five architectures on available GPU
accelerators (CUDA, ROCm, MPS, or CPU fallback):

```bash
# Auto-detect GPU (CUDA → MPS → CPU fallback)
python benchmarks/collect_gpu_data.py

# Quick run (fewer batch sizes / repeats)
python benchmarks/collect_gpu_data.py --quick

# Explicit CUDA device
python benchmarks/collect_gpu_data.py --device cuda

# Apple Silicon GPU
python benchmarks/collect_gpu_data.py --device mps

# Override hardware specs (e.g. NVIDIA A100 80 GB)
python benchmarks/collect_gpu_data.py \
  --peak-flops 312e12 --memory-bandwidth 2.0e12
```

Results are written to `benchmarks/results/<device>/benchmark_gpu_data.json`
(for example `nvidia-tesla-t4/` or `apple-m3/`), stamped with the neural-cost
version, git commit and framework versions.

Configurations that would not fit in accelerator memory are skipped up front:
the sweep compares neural-cost's own memory estimate (with a 3× allowance for
allocator pools) against the device budget (`torch.mps.recommended_max_memory()`
on Apple Silicon, total memory on CUDA) and prints which batch sizes it skipped.
On a 16 GB Mac, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` and
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8` so PyTorch raises an out-of-memory error
before macOS kills the process. TensorFlow is skipped when it has no GPU device;
its CPU numbers come from `collect_data.py`.

Generate figures and the full markdown report:

```bash
python benchmarks/generate_gpu_report.py
# → benchmarks/results/<device>/GPU_BENCHMARK_REPORT.md + figures/gpu_fig*.png
```

### GPU-vs-CPU crossover analysis

Find the exact batch size at which each architecture first runs faster on GPU than CPU:

```bash
# Collect CPU baseline first (if not already done)
python benchmarks/collect_data.py

# Run GPU benchmark with the fine batch-size grid (1, 4, 8 … 1024)
python benchmarks/collect_gpu_data.py --crossover

# Regenerate report — Figure GPU-8 (crossover plot) will now be included
python benchmarks/generate_gpu_report.py
```

The crossover analysis sweeps `CROSSOVER_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]`
and prints a table showing the first batch at which GPU latency drops below CPU latency.

### JAX on Apple Silicon (MPS / Metal)

JAX requires an explicit GPU plugin to run on Apple Silicon GPUs.
**Without a plugin, JAX silently falls back to CPU** even on MPS-capable machines.
Install one of:

```bash
# Official Apple plugin via package extra:
pip install "neural-cost[jax-metal]"
# or directly:
pip install jax-metal

# Community jax-mps plugin via package extra (set JAX_PLATFORMS=mps):
pip install "neural-cost[jax-mps]"
# or directly:
pip install jax-mps && JAX_PLATFORMS=mps python benchmarks/collect_gpu_data.py
```

**What the plugins are.** `jax-metal` is Apple's Metal PJRT plugin.
[`jax-mps`](https://github.com/tillahoffmann/jax-mps) is a community PJRT plugin.
It maps the StableHLO that JAX emits onto MLX operations, which run on the
Apple GPU. Both are experimental. jax-mps has no float64 and no sub-byte or fp8
dtypes, and it is single-device.

**Version pins.** jax-mps is compiled against one jaxlib minor version's
StableHLO bytecode. jax-mps 0.11.x requires `jax>=0.11,<0.12`,
`jaxlib>=0.11,<0.12` and Python ≥ 3.12. With a mismatched jaxlib, JIT
compilation fails with deserialization errors. The `neural-cost[jax-mps]`
extra carries the same pin (`jax>=0.11,<0.12`), so installing it with
`neural-cost[jax]` resolves to a compatible jax. The plain `jax` extra only
requires `jax>=0.4`. If you upgrade jax yourself, upgrade jax-mps to the
matching minor release. On Python 3.11 the jax-mps extra installs nothing.
jax-metal 0.1.1, its latest release, only works with jax/jaxlib **0.4.34**.
With newer jaxlib the `METAL` device is listed, but every computation fails
with `unknown attribute code` (a StableHLO bytecode mismatch). The
`neural-cost[jax-metal]` extra pins `jax==0.4.34`/`jaxlib==0.4.34`. jaxlib
0.4.34 has no wheels for Python ≥ 3.14, where the extra installs nothing. The
two plugin extras need different jax versions, so install one or the other,
never both.

**Selecting a backend.** JAX uses one platform per process, chosen by
`JAX_PLATFORMS` before `import jax`. Use `cpu`, `mps` (jax-mps) or `METAL`
(jax-metal). Check with `jax.devices()[0].platform`. Score each run against
its own device with `JaxAdapter().hardware()`, and compare platforms with
`neural-cost compare bench.py:build --platforms cpu,mps`.

**Worked example.** [`examples/jax_flax_apple_silicon.py`](https://github.com/davidgraymi/neural-cost/blob/main/examples/jax_flax_apple_silicon.py)
profiles an AlphaZero-style Flax ResNet with GroupNorm (batch 128, 8×8×119)
end to end. It shows the exact `JaxAdapter` call for a jitted Flax `apply`
(weights as argument 0 with `parameter_argnums=(0,)`), device-aware specs,
coverage, and the per-module breakdown:

```bash
pip install "neural-cost[jax]" flax
JAX_PLATFORMS=cpu python examples/jax_flax_apple_silicon.py
python examples/jax_flax_apple_silicon.py --compare cpu,mps
```

For a closure such as `jax.jit(lambda x: model.apply(variables, x))`, use
`JaxAdapter().operations(fn, (obs,))`. Closed-over arrays are treated as
parameters automatically.

The benchmark now detects whether a plugin is installed and emits a clear diagnostic
when JAX is running on CPU instead of the GPU.

Known JAX MPS limitations (tracked upstream):
- `jax.jit()` **regresses CNN latency 3×** on MPS — XLA's Metal conv lowering inserts
  extra memory-layout transposes for statically-shaped graphs (known bug).
- `jax.jit()` gains are large for LSTM (+1.79×) and Transformer (+1.12×) where XLA
  eliminates intermediate tensor roundtrips.

### GPU vs CPU benchmark differences

| Aspect | CPU benchmark | GPU benchmark |
|---|---|---|
| Timing | `time.perf_counter_ns` | CUDA events (`torch.cuda.Event`) |
| Sync barrier | None (CPU executes synchronously) | `torch.cuda.synchronize()` / `block_until_ready()` |
| TF optimised variant | `tf.function` (no XLA) | `tf.function(jit_compile=True)` (XLA GPU) |
| Batch sizes | 1, 8, 32, 128 | 8, 32, 128, 512 |
| Device placement | CPU tensors | `.to(device)` / `jax.device_put` / `tf.device` |

<details>
<summary>Sample output (Apple M3, 3.6 TFLOP/s · 100 GB/s, batch=16, seq=32)</summary>

```
┌─ Hardware ──────────────────────────────────────────────────────────────────
│  Device          : Apple M3
│  Peak FP32       : 3.60 TFLOP/s
│  Peak bandwidth  : 100.0 GB/s  (STREAM triad: 51.6 GB/s)
│  Ridge point     : 36.0 FLOP/byte
│  Detection source: Apple Silicon table (Apple M3) [FP32] + NumPy STREAM triad
└────────────────────────────────────────────────────────────────────────────

Benchmarking PyTorch…
    FF DNN         0.05 ms  11.8% eff
    CNN            4.83 ms  8.4% eff
    RNN            0.78 ms  8.2% eff
    LSTM           2.43 ms  9.3% eff
    Transformer    1.30 ms  13.3% eff
Benchmarking JAX…
    FF DNN         0.08 ms  11.3% eff
    CNN            1.89 ms  31.9% eff
    RNN            1.00 ms  13.1% eff
    LSTM           3.71 ms  15.3% eff
    Transformer    2.41 ms  21.3% eff
Benchmarking TensorFlow…
    FF DNN         1.89 ms  0.3% eff
    CNN            4.21 ms  6.6% eff
    RNN            19.94 ms  0.3% eff
    LSTM           34.28 ms  0.7% eff
    Transformer    11.91 ms  1.0% eff


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FF DNN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Framework         FLOPs    Param B    I/O MB      AI   ms(med)     ±ms   effic.       roofline         GFLOP/s     GB/s bound   
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
PyTorch            3.8M    475,176       0.6    6.09     0.053   0.010    11.8% [██░░░░░░░░░░░░░░░░]     71.61    11.76 memory  
JAX                3.8M    475,176       0.9    4.45     0.076   0.005    11.3% [██░░░░░░░░░░░░░░░░]     50.03    11.25 memory  
TensorFlow         3.8M    475,176       0.6    6.42     1.894   0.101     0.3% [░░░░░░░░░░░░░░░░░░]      2.00     0.31 memory  
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Diagnostics (top layer bottleneck & fusion):
    • PyTorch: Top Bottleneck: _0 (linear, memory-bound) (73.7%), Fused Eff: 10.5%, Resident in SLC
    • JAX: Top Bottleneck: dot_1 (matmul, memory-bound) (53.7%), Fused Eff: 8.2%, Resident in SLC
    • TensorFlow: Top Bottleneck: dense (linear, memory-bound) (77.8%), Fused Eff: 0.3%, Resident in SLC
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  AI = arithmetic intensity (FLOP/byte).  Ridge point = 36.0 FLOP/byte  (above → compute-bound, below → memory-bound)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CNN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Framework         FLOPs    Param B    I/O MB      AI   ms(med)     ±ms   effic.       roofline         GFLOP/s     GB/s bound   
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
PyTorch          671.1M    309,288      40.4   16.63     4.829   0.094     8.4% [██░░░░░░░░░░░░░░░░]    138.97     8.36 memory  
JAX              666.9M    309,288      60.3   11.06     1.890   0.017    31.9% [██████░░░░░░░░░░░░]    352.90    31.91 memory  
TensorFlow       670.1M    309,288      27.8   24.12     4.207   0.108     6.6% [█░░░░░░░░░░░░░░░░░]    159.26     6.60 memory  
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Diagnostics (top layer bottleneck & fusion):
    • PyTorch: Top Bottleneck: _4 (conv2d, compute-bound) (31.2%), Fused Eff: 3.9%
    • JAX: Top Bottleneck: conv_10 (conv2d, compute-bound) (22.8%), Fused Eff: 9.8%
    • TensorFlow: Top Bottleneck: conv2d_1 (conv2d, compute-bound) (40.8%), Fused Eff: 4.4%
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  AI = arithmetic intensity (FLOP/byte).  Ridge point = 36.0 FLOP/byte  (above → compute-bound, below → memory-bound)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RNN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Framework         FLOPs    Param B    I/O MB      AI   ms(med)     ±ms   effic.       roofline         GFLOP/s     GB/s bound   
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
PyTorch           67.1M    267,304       6.4   10.43     0.784   0.007     8.2% [█░░░░░░░░░░░░░░░░░]     85.63     8.21 memory  
JAX               67.4M    267,304      13.1    5.14     1.003   0.013    13.1% [██░░░░░░░░░░░░░░░░]     67.22    13.09 memory  
TensorFlow        67.1M    267,304       6.6   10.22    19.936   0.280     0.3% [░░░░░░░░░░░░░░░░░░]      3.37     0.33 memory  
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Diagnostics (top layer bottleneck & fusion):
    • PyTorch: Top Bottleneck: rnn.layer0.hh (linear, memory-bound) (40.7%), Resident in SLC
    • JAX: Top Bottleneck: dot_4 (matmul, memory-bound) (20.0%), Fused Eff: 13.0%
    • TensorFlow: Top Bottleneck: simple_rnn.hh (linear, memory-bound) (39.9%), Resident in SLC
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  AI = arithmetic intensity (FLOP/byte).  Ridge point = 36.0 FLOP/byte  (above → compute-bound, below → memory-bound)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LSTM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Framework         FLOPs    Param B    I/O MB      AI   ms(med)     ±ms   effic.       roofline         GFLOP/s     GB/s bound   
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
PyTorch          268.5M       1.1M      22.6   11.90     2.432   0.007     9.3% [██░░░░░░░░░░░░░░░░]    110.38     9.27 memory  
JAX              270.2M       1.1M      56.6    4.77     3.705   0.024    15.3% [███░░░░░░░░░░░░░░░]     72.92    15.29 memory  
TensorFlow       268.5M       1.1M      23.1   11.63    34.282   0.362     0.7% [░░░░░░░░░░░░░░░░░░]      7.83     0.67 memory  
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Diagnostics (top layer bottleneck & fusion):
    • PyTorch: Top Bottleneck: lstm.layer0.hh (linear, memory-bound) (41.9%)
    • JAX: Top Bottleneck: dot_4 (matmul, memory-bound) (17.1%), Fused Eff: 15.2%
    • TensorFlow: Top Bottleneck: lstm.hh (linear, memory-bound) (41.9%)
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  AI = arithmetic intensity (FLOP/byte).  Ridge point = 36.0 FLOP/byte  (above → compute-bound, below → memory-bound)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Transformer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Framework         FLOPs    Param B    I/O MB      AI   ms(med)     ±ms   effic.       roofline         GFLOP/s     GB/s bound   
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
PyTorch          422.2M       1.6M      17.3   24.37     1.302   0.014    13.3% [██░░░░░░░░░░░░░░░░]    324.40    13.31 memory  
JAX              423.6M       1.6M      51.4    8.24     2.407   0.021    21.3% [████░░░░░░░░░░░░░░]    175.95    21.34 memory  
TensorFlow       421.4M       1.6M      10.0   42.19    11.910   0.054     1.0% [░░░░░░░░░░░░░░░░░░]     35.38     0.84 compute 
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Diagnostics (top layer bottleneck & fusion):
    • PyTorch: Top Bottleneck: attn1 (attention, compute-bound) (10.0%), Fused Eff: 9.0%
    • JAX: Top Bottleneck: add_54 (elementwise, memory-bound) (4.0%), Fused Eff: 8.5%
    • TensorFlow: Top Bottleneck: multi_head_attention (attention, compute-bound) (15.3%), Fused Eff: 1.0%
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  AI = arithmetic intensity (FLOP/byte).  Ridge point = 36.0 FLOP/byte  (above → compute-bound, below → memory-bound)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Per-Architecture × Per-Framework Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Architecture    PyTorch                       JAX                           TensorFlow                  
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  FF DNN          11.8% eff  0.05ms  3.8M FLOPs  11.3% eff  0.08ms  3.8M FLOPs  0.3% eff  1.89ms  3.8M FLOPs
  CNN             8.4% eff  4.83ms  671.1M FLOPs  31.9% eff  1.89ms  666.9M FLOPs  6.6% eff  4.21ms  670.1M FLOPs
  RNN             8.2% eff  0.78ms  67.1M FLOPs  13.1% eff  1.00ms  67.4M FLOPs  0.3% eff  19.94ms  67.1M FLOPs
  LSTM            9.3% eff  2.43ms  268.5M FLOPs  15.3% eff  3.71ms  270.2M FLOPs  0.7% eff  34.28ms  268.5M FLOPs
  Transformer     13.3% eff  1.30ms  422.2M FLOPs  21.3% eff  2.41ms  423.6M FLOPs  1.0% eff  11.91ms  421.4M FLOPs

  *Efficiency >100% means hardware spec is conservative; pass --peak-flops to calibrate.

  Canonical cross-framework consistency (dense FLOPs, unique parameters)
  FF DNN             3.8M dense FLOPs    118,794 params  identical
  CNN              660.6M dense FLOPs     77,322 params  identical
  RNN               67.1M dense FLOPs     66,826 params  identical
  LSTM             268.5M dense FLOPs    263,434 params  identical
  Transformer      419.5M dense FLOPs    397,834 params  identical
```

> **Reading the table**
> Every framework now reports identical dense FLOPs and parameters for each
> architecture (last block); the small differences in the total FLOPs column are
> auxiliary point-wise work that each adapter captures at different granularity.
> All architectures are memory-bound on this device (AI < ridge point of 36 FLOP/byte).
> JAX runs under `jax.jit` (one fused XLA program) and PyTorch in eager mode;
> TensorFlow runs eagerly, so Python dispatch dominates at this batch size,
> especially for sequential (RNN/LSTM) workloads. Use `attribute_gap` to see how
> much of each gap is dispatch, unfused traffic, or kernel inefficiency.

</details>

## Matrix-multiply workload comparison (original)

For a quick cross-framework sanity check on plain matmul shapes, run:

```bash
pip install "neural-cost[torch,jax,tensorflow]"
# Or from local editable checkout:
# pip install -e '.[torch,jax,tensorflow]'
python examples/compare_frameworks.py
```

Optional overrides with known hardware specs:

```bash
python examples/compare_frameworks.py --peak-flops 312e12 --memory-bandwidth 1.6e12
```
