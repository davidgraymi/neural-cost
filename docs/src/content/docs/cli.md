---
title: "CLI reference"
description: "The neural-cost command-line tools: hardware, profile, benchmark, audit, llm, compare and distributed."
section: "Reference"
order: 1
---

`neural-cost` provides a unified command-line suite for hardware detection, theoretical profiling, empirical benchmarking, LLM serving analysis, and CI/CD budget assertions:

```bash
# General help and subcommands:
neural-cost --help
```

## 1. Hardware Detection & STREAM Benchmark

Probe attached accelerators, lookup Apple Silicon / NVIDIA published specs, and measure live NumPy STREAM memory bandwidth:

```bash
# Live probe with memory bandwidth benchmark:
neural-cost hardware

# Instant detection without STREAM benchmark:
neural-cost hardware --no-bench

# Machine-readable JSON output:
neural-cost hardware --json

# Describe the CPU even when a GPU is present:
neural-cost hardware --device cpu
```

## 2. Static Architecture Profiling

Compute forward FLOPs, parameter counts, KV cache size, activation memory, arithmetic intensity, and theoretical latency floor on detected or target hardware:

```bash
# Zero-code profiling directly by Hugging Face Hub Model ID:
neural-cost profile meta-llama/Meta-Llama-3-8B

# Ingest Mixtral 8x7B MoE and export publication-ready SVG roofline:
neural-cost profile mistralai/Mixtral-8x7B-v0.1 --plot mixtral_roofline.svg --plot-theme dark

# Profile a 7B Transformer (Llama-style) at batch size 8:
neural-cost profile --arch transformer --batch-size 8 --seq-len 4096 --embed-dim 4096 --num-heads 32 --num-kv-heads 8 --num-layers 32 --dtype fp16

# Profile a quantized 70B Transformer with W4A16 AWQ and launch floor:
neural-cost profile --arch transformer --num-layers 80 --embed-dim 8192 --num-heads 64 --quantization w4a16_awq --num-kernels 640

# Profile training pass with selective activation checkpointing and Adam moments:
neural-cost profile --arch transformer --batch-size 4 --seq-len 4096 --embed-dim 4096 --num-layers 32 --checkpointing selective --training

# Profile an 8x7B Mixture-of-Experts layer:
neural-cost profile --arch moe --num-experts 8 --top-k 2 --embed-dim 4096 --expert-hidden-dim 14336 --decode

# Profile PagedAttention KV cache with shared prefix caching:
neural-cost profile --arch paged-attention --batch-size 32 --seq-len 2048 --block-size 16 --shared-prefix-tokens 128

# Profile State Space Model (Mamba) recurrent state and constant memory:
neural-cost profile --arch ssm --batch-size 1 --seq-len 4096 --embed-dim 4096 --num-layers 32
```

## 3. Empirical Benchmarking

Benchmark real execution latency, achieved compute throughput (GFLOP/s), achieved DRAM bandwidth (GB/s), and roofline efficiency on your local machine:

```bash
# Microbenchmark matrix multiplication:
neural-cost benchmark --arch linear --batch-size 64 --in-features 1024 --out-features 1024

# Benchmark MoE decode step:
neural-cost benchmark --arch moe --batch-size 1 --num-experts 8 --top-k 2

# Benchmark PagedAttention kernel:
neural-cost benchmark --arch paged-attention --batch-size 32 --seq-len 1024 --block-size 16

# Benchmark live NumPy STREAM memory bandwidth:
neural-cost benchmark --arch stream
```

## 4. CI Budget Gate (`audit`)

Enforce automated performance budgets and VRAM ceilings in your CI/CD pipelines. Exits with status `0` if all assertions pass, or `1` if any budget is exceeded:

```bash
# Audit any Hugging Face model directly in CI against strict VRAM and latency bounds:
neural-cost audit meta-llama/Meta-Llama-3-8B --max-vram 16GB --max-latency-ms 25.0

# Fail CI if model exceeds 16GB VRAM or 50ms latency floor:
neural-cost audit \
    --arch transformer \
    --batch-size 1 \
    --seq-len 2048 \
    --embed-dim 4096 \
    --num-heads 32 \
    --num-layers 32 \
    --max-vram 16GB \
    --max-latency-ms 50.0

# JSON output for CI pipeline reporting:
neural-cost audit --arch mlp --in-features 1024 --out-features 1024 --max-vram 1GB --json
```

## 5. Modern LLM Serving Primitives (`llm`)

Dedicated analytical calculators for production serving architectures:

```bash
# Mixture-of-Experts routing & memory traffic:
neural-cost llm moe --batch-size 1 --num-experts 8 --top-k 2 --decode

# Speculative decoding analytical breakeven solver:
neural-cost llm speculative --gamma 4 --acceptance-rate 0.75

# PagedAttention KV cache capacity and fragmentation:
neural-cost llm paged --batch-size 32 --context-len 2048 --block-size 16

# Continuous batching mixed iteration cost simulator:
neural-cost llm continuous --decode-streams 64 --prefill-tokens 512

# State Space Model (Mamba/S6/SSD) recurrent cost and KV elimination analyzer:
neural-cost llm ssm --batch-size 1 --seq-len 32768 --embed-dim 4096 --num-layers 32 --decode

# Sub-byte quantization and dequantization ALU unpacking tax (AWQ, GPTQ, FP8, NVFP4):
neural-cost llm quant --batch-size 1 --in-features 4096 --out-features 4096 --quantization w4a16_awq

# Kernel launch floor & CUDA Graph / torch.compile speedup analyzer:
neural-cost llm launch-floor --num-kernels 640 --arithmetic-floor-ms 0.5 --launch-overhead-us 5.0 --measured-latency-ms 3.5

# Activation checkpointing & selective recomputation (none, full, selective) analyzer:
neural-cost llm checkpointing --strategy selective --batch-size 4 --seq-len 4096 --embed-dim 4096 --vram-gb 80.0
```

## 6. Multi-Framework Comparison (`compare`)

Run head-to-head empirical benchmarks across PyTorch, JAX, and TensorFlow (backward-compatible with `neural-cost-compare`):

```bash
# Multi-framework benchmark runner:
neural-cost compare
# Or using the legacy binary:
neural-cost-compare --warmup 10 --repeats 30
```

### Comparing one JAX function across platforms

JAX uses one platform per process (`JAX_PLATFORMS`). Point `compare` at a
zero-argument factory that returns `(function, args)` and list the platforms.
Each platform runs in its own subprocess with its own hardware spec:

```python
# bench.py
def build():
    import jax, jax.numpy as jnp
    net = jax.jit(lambda x, w: jnp.tanh(x @ w))
    return net, (jnp.ones((128, 512)), jnp.ones((512, 512)))
```

```bash
neural-cost compare bench.py:build --platforms cpu,mps
# Backend comparison (speedup vs cpu)
#   platform  device  median    speedup  hardware spec                                  roofline               coverage
#   cpu       cpu:0   0.251 ms  1.00x    Apple M3 (CPU) (0.84 TFLOP/s, 100 GB/s)        49.4% (compute-bound)  100%
#   mps       mps:0   ...
```

From Python: `compare_backends("bench.py:build", ["cpu", "mps"]).render()`.
A platform that fails (plugin missing, timeout) is listed with its error, and
the other platforms still run. `--baseline` picks the speedup reference and
`--json` emits machine-readable results.

## 7. Distributed Cluster & 3D Parallelism (`distributed`)

Model cluster scaling efficiency, network communication volume, and MFU for frontier models:

```bash
# Model Llama-3 70B training across 16 nodes of 8x H100 with TP=8, PP=4, DP=4:
neural-cost distributed \
    --parameters 70e9 \
    --batch-size 128 \
    --seq-len 4096 \
    --embed-dim 8192 \
    --num-layers 80 \
    --num-nodes 16 \
    --devices-per-node 8 \
    --tp 8 \
    --pp 4 \
    --dp 4 \
    --microbatches 8 \
    --dp-mode zero3_fsdp \
    --intra-node nvlink4 \
    --inter-node infiniband_ndr

# Output cluster performance and communication breakdown as JSON:
neural-cost distributed --parameters 7e9 --batch-size 32 --tp 2 --dp 4 --json
```
