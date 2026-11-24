# neural-cost validation report

Generated 2026-11-24 by `benchmarks/validate.py` on **Tesla T4**
(Linux-6.6.159+-x86_64-with-glibc2.39), hardware spec from NVIDIA profile (Tesla T4) [FP32] + NumPy STREAM triad.

Run: neural-cost v0.19.0-1-gfa0e978 @ fa0e978173c1 (clean state unknown) on Tesla T4, 2026-11-24T13:14:47+00:00 (Python 3.13.16; jax 0.12.1, jaxlib 0.12.1, keras 3.13.2, numpy 2.4.6, tensorflow 2.21.0, torch 2.14.1+cu130).

**How to read this.** *Bound* is the per-operation roofline at datasheet peak
(fused aggregate roofline for jit). It must never exceed the measurement; any
violation would falsify the model. *Zero-shot* predictions use only
microbenchmark calibration (size-dependent FLOP/s and bandwidth curves,
per-kernel dispatch, per-call sync latency) and never see a model runtime.
*Op-replay* times every captured operation alone with synthetic tensors of the
recorded shape, then composes them; it never runs the model either, so its
error isolates *composition effects*. *LOO fit* is a 4-parameter non-negative cost
model `T = c0 + a*kernels + FLOPs/P + bytes/BW` where each workload is
predicted by a fit on all the others, so its error is out-of-sample.
PyTorch kernels are traced with a TorchDispatchMode: kernels the adapter did not
capture (activations, residual adds, copies) are modeled as memory-bound kernels
of their traced traffic, paying Python dispatch only when issued from Python. A
jitted JAX program is one dispatch.

### PyTorch eager (cuda)

Hardware: Tesla T4. 10 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 10 |
| Bound tightness (median, min–max) | 9.0% (0.7%–24.1%) |
| Bound rank correlation (Spearman) | 0.624 |
| Roofline-as-predictor MAPE | 90.6% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 52.1% |
| Zero-shot median APE | 45.6% |
| Zero-shot rank correlation | 0.855 |
| Op-replay MAPE (isolated kernels, no fit) | **54.9%** |
| Op-replay median APE | 36.3% |
| Op-replay rank correlation | 0.988 |
| Calibrated model LOO MAPE | **33.5%** |
| Calibrated model LOO median APE | 31.9% |
| Calibrated model bias (geomean pred/meas) | ×0.861 |
| Within ±25% | 40% |
| Calibrated rank correlation | 0.661 |
| Fitted call overhead | 0.0 µs |
| Fitted per-kernel dispatch | 26.13 µs |
| Fitted effective compute | inf GFLOP/s |
| Fitted effective bandwidth | 137.3 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.116 | 0.0011 | 0.9% | 0.070 | -39.6% | 0.084 | -27.8% | 0.146 | +25.6% |
| mlp w=256 b=64 | 0.157 | 0.0022 | 1.4% | 0.085 | -45.7% | 0.102 | -35.1% | 0.132 | -15.9% |
| mlp w=1024 b=1 | 0.150 | 0.0169 | 11.2% | 0.075 | -50.4% | 0.094 | -37.5% | 0.208 | +38.2% |
| mlp w=1024 b=64 | 0.155 | 0.0338 | 21.8% | 0.209 | +35.0% | 0.159 | +2.7% | 0.226 | +45.9% |
| cnn 32px b=1 | 0.379 | 0.0027 | 0.7% | 0.121 | -68.0% | 0.213 | -43.8% | 0.199 | -47.6% |
| cnn 32px b=16 | 0.347 | 0.0430 | 12.4% | 0.293 | -15.5% | 0.198 | -42.8% | 0.261 | -24.7% |
| lstm d=128 b=1 | 0.518 | 0.0355 | 6.9% | 1.032 | +99.1% | 1.375 | +165.3% | 0.255 | -50.9% |
| lstm d=128 b=16 | 0.493 | 0.0557 | 11.3% | 1.040 | +111.0% | 1.267 | +157.1% | 0.269 | -45.4% |
| transformer d=256 T=64 b=1 | 0.381 | 0.0135 | 3.5% | 0.208 | -45.5% | 0.273 | -28.4% | 0.309 | -19.0% |
| transformer d=256 T=64 b=8 | 0.447 | 0.1079 | 24.1% | 0.497 | +11.0% | 0.411 | -8.1% | 0.546 | +21.9% |

Calibration: torch microbenchmarks (cuda, float32) — 4271 GFLOP/s achievable, 233.7 GB/s, 8.45 µs/kernel dispatch, 15.1 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.116 ms, 5 kernels, calibration: torch microbenchmarks (cuda, float32)
  roofline_floor       0.0017 ms    1.5% ································
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0294 ms   25.3% ████████························
  dispatch             0.0390 ms   33.6% ███████████·····················
  kernel_inefficiency     0.0139 ms   12.0% ████····························
  timing_jitter        0.0070 ms    6.0% ██······························
  residual             0.0252 ms   21.7% ███████·························
  explained by model: 78.3%
  next: dispatch (34%, ≤1.51x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: peak_derate (25%, ≤1.34x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: residual (22%, ≤1.28x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
  next: kernel_inefficiency (12%, ≤1.14x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
  next: timing_jitter (6%, ≤1.06x if removed): Pin clocks/threads, isolate the process, add warmup, report min alongside median.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=64 b=8</summary>

```
Gap attribution — measured 0.447 ms, 17 kernels, calibration: torch microbenchmarks (cuda, float32)
  roofline_floor       0.1039 ms   23.2% ███████·························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0064 ms    1.4% ································
  peak_derate          0.3683 ms   82.3% ██████████████████████████······
  dispatch             0.0181 ms    4.0% █·······························
  kernel_inefficiency     0.0000 ms    0.0% ································
  timing_jitter        0.0000 ms    0.0% ································
  residual            -0.0492 ms  -11.0% ▒▒▒▒····························
  explained by model: 111.0%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: peak_derate (82%, ≤5.65x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>


### JAX jit (gpu)

Hardware: Tesla T4. 6 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 6 |
| Bound tightness (median, min–max) | 5.7% (0.6%–27.6%) |
| Bound rank correlation (Spearman) | 0.943 |
| Roofline-as-predictor MAPE | 91.0% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 21.0% |
| Zero-shot median APE | 19.2% |
| Zero-shot rank correlation | 0.943 |
| Calibrated model LOO MAPE | **10.4%** |
| Calibrated model LOO median APE | 9.4% |
| Calibrated model bias (geomean pred/meas) | ×1.005 |
| Within ±25% | 83% |
| Calibrated rank correlation | 0.771 |
| Fitted call overhead | 203.7 µs |
| Fitted per-kernel dispatch | 0.00 µs |
| Fitted effective compute | inf GFLOP/s |
| Fitted effective bandwidth | 192.6 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.187 | 0.0011 | 0.6% | 0.171 | -8.4% | – | – | 0.236 | +26.2% |
| mlp w=256 b=64 | 0.243 | 0.0044 | 1.8% | 0.180 | -26.0% | – | – | 0.206 | -15.1% |
| mlp w=1024 b=1 | 0.243 | 0.0136 | 5.6% | 0.209 | -14.0% | – | – | 0.249 | +2.4% |
| mlp w=1024 b=64 | 0.267 | 0.0337 | 12.6% | 0.324 | +21.2% | – | – | 0.286 | +6.8% |
| transformer d=256 T=64 b=1 | 0.273 | 0.0158 | 5.8% | 0.225 | -17.3% | – | – | 0.240 | -11.9% |
| transformer d=256 T=64 b=8 | 0.416 | 0.1150 | 27.6% | 0.581 | +39.4% | – | – | 0.417 | +0.0% |

Calibration: jax microbenchmarks (gpu, jit) — 5958 GFLOP/s achievable, 234.8 GB/s, 79.53 µs/kernel dispatch, 49.4 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.187 ms, 1 kernels, calibration: jax microbenchmarks (gpu, jit)
  roofline_floor       0.0018 ms    0.9% ································
  unfused_traffic      0.0001 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0816 ms   43.7% ██████████████··················
  dispatch             1.7990 ms  963.4% ████████████████████████████████
  timing_jitter        0.0000 ms    0.0% ································
  residual            -1.6957 ms -908.1% ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  explained by model: 1008.1%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: dispatch (963%, ≤infx if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: peak_derate (44%, ≤1.78x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=64 b=8</summary>

```
Gap attribution — measured 0.416 ms, 1 kernels, calibration: jax microbenchmarks (gpu, jit)
  roofline_floor       0.1281 ms   30.8% ██████████······················
  unfused_traffic      0.0852 ms   20.5% ███████·························
  serialization        0.0601 ms   14.4% █████···························
  peak_derate          1.1847 ms  284.5% ████████████████████████████████
  dispatch             2.0375 ms  489.2% ████████████████████████████████
  timing_jitter        0.0000 ms    0.0% ································
  residual            -3.0791 ms -739.3% ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  explained by model: 839.3%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: dispatch (489%, ≤infx if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: peak_derate (284%, ≤infx if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: unfused_traffic (20%, ≤1.26x if removed): Fuse point-wise chains: torch.compile, jax.jit/XLA, tf.function(jit_compile=True).
  next: serialization (14%, ≤1.17x if removed): Fuse heterogeneous ops or overlap independent ops (streams, larger fused kernels).
```

</details>


### TensorFlow graph (gpu)

Hardware: Tesla T4. 10 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 10 |
| Bound tightness (median, min–max) | 1.8% (0.1%–6.7%) |
| Bound rank correlation (Spearman) | 0.661 |
| Roofline-as-predictor MAPE | 97.8% |
| Calibrated model LOO MAPE | **12.7%** |
| Calibrated model LOO median APE | 6.1% |
| Calibrated model bias (geomean pred/meas) | ×0.947 |
| Within ±25% | 80% |
| Calibrated rank correlation | 0.842 |
| Fitted call overhead | 860.1 µs |
| Fitted per-kernel dispatch | 16.92 µs |
| Fitted effective compute | 1747.8 GFLOP/s |
| Fitted effective bandwidth | inf GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.907 | 0.0011 | 0.1% | – | – | – | – | 0.912 | +0.5% |
| mlp w=256 b=64 | 0.902 | 0.0022 | 0.2% | – | – | – | – | 0.926 | +2.7% |
| mlp w=1024 b=1 | 0.902 | 0.0169 | 1.9% | – | – | – | – | 0.916 | +1.6% |
| mlp w=1024 b=64 | 0.934 | 0.0338 | 3.6% | – | – | – | – | 1.102 | +18.0% |
| cnn 32px b=1 | 0.970 | 0.0027 | 0.3% | – | – | – | – | 0.954 | -1.7% |
| cnn 32px b=16 | 1.005 | 0.0430 | 4.3% | – | – | – | – | 1.174 | +16.9% |
| lstm d=128 b=1 | 2.051 | 0.0363 | 1.8% | – | – | – | – | 1.952 | -4.8% |
| lstm d=128 b=16 | 2.072 | 0.0554 | 2.7% | – | – | – | – | 2.223 | +7.3% |
| transformer d=256 T=64 b=1 | 1.656 | 0.0135 | 0.8% | – | – | – | – | 0.973 | -41.3% |
| transformer d=256 T=64 b=8 | 1.622 | 0.1079 | 6.7% | – | – | – | – | 1.104 | -31.9% |

No calibration microbenchmarks for this backend; no gap attribution.


### PyTorch eager (cpu)

Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz. 10 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 10 |
| Bound tightness (median, min–max) | 16.8% (0.8%–34.4%) |
| Bound rank correlation (Spearman) | 0.903 |
| Roofline-as-predictor MAPE | 84.5% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 37.9% |
| Zero-shot median APE | 33.4% |
| Zero-shot rank correlation | 0.976 |
| Op-replay MAPE (isolated kernels, no fit) | **22.4%** |
| Op-replay median APE | 17.0% |
| Op-replay rank correlation | 1.000 |
| Calibrated model LOO MAPE | **34.4%** |
| Calibrated model LOO median APE | 28.9% |
| Calibrated model bias (geomean pred/meas) | ×0.977 |
| Within ±25% | 40% |
| Calibrated rank correlation | 0.988 |
| Fitted call overhead | 0.0 µs |
| Fitted per-kernel dispatch | 20.41 µs |
| Fitted effective compute | 77.4 GFLOP/s |
| Fitted effective bandwidth | 16.8 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.124 | 0.0010 | 0.8% | 0.035 | -71.7% | 0.050 | -59.4% | 0.256 | +106.5% |
| mlp w=256 b=64 | 0.378 | 0.0668 | 17.7% | 0.268 | -29.0% | 0.280 | -25.9% | 0.374 | -1.1% |
| mlp w=1024 b=1 | 0.577 | 0.0165 | 2.9% | 0.359 | -37.8% | 0.414 | -28.2% | 0.814 | +41.1% |
| mlp w=1024 b=64 | 3.605 | 1.0536 | 29.2% | 3.343 | -7.3% | 3.461 | -4.0% | 4.347 | +20.6% |
| cnn 32px b=1 | 1.106 | 0.0807 | 7.3% | 0.346 | -68.8% | 0.513 | -53.6% | 0.456 | -58.8% |
| cnn 32px b=16 | 5.235 | 1.2908 | 24.7% | 4.364 | -16.6% | 4.764 | -9.0% | 4.837 | -7.6% |
| lstm d=128 b=1 | 1.727 | 0.0655 | 3.8% | 0.620 | -64.1% | 1.738 | +0.6% | 1.271 | -26.4% |
| lstm d=128 b=16 | 5.668 | 1.0486 | 18.5% | 4.041 | -28.7% | 5.022 | -11.4% | 4.896 | -13.6% |
| transformer d=256 T=64 b=1 | 2.592 | 0.4108 | 15.9% | 1.438 | -44.5% | 2.006 | -22.6% | 1.776 | -31.5% |
| transformer d=256 T=64 b=8 | 9.558 | 3.2867 | 34.4% | 10.516 | +10.0% | 8.711 | -8.9% | 13.085 | +36.9% |

Calibration: torch microbenchmarks (cpu, float32) — 1033 GFLOP/s achievable, 11.3 GB/s, 6.86 µs/kernel dispatch, 0.1 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.124 ms, 5 kernels, calibration: torch microbenchmarks (cpu, float32)
  roofline_floor       0.0548 ms   44.3% ██████████████··················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0000 ms    0.0% ································
  dispatch             0.0344 ms   27.8% █████████·······················
  kernel_inefficiency     0.0000 ms    0.0% ································
  timing_jitter        0.0118 ms    9.5% ███·····························
  residual             0.0228 ms   18.4% ██████··························
  explained by model: 81.6%
  ! calibrated throughput exceeds the HardwareSpec peak; clamped to peak (the spec may be too conservative).
  next: dispatch (28%, ≤1.38x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: residual (18%, ≤1.23x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
  next: timing_jitter (10%, ≤1.11x if removed): Pin clocks/threads, isolate the process, add warmup, report min alongside median.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=64 b=8</summary>

```
Gap attribution — measured 9.558 ms, 15 kernels, calibration: torch microbenchmarks (cpu, float32)
  roofline_floor       3.2867 ms   34.4% ███████████·····················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.2076 ms    2.2% █·······························
  peak_derate          6.9602 ms   72.8% ███████████████████████·········
  dispatch             0.0618 ms    0.6% ································
  kernel_inefficiency     0.0000 ms    0.0% ································
  timing_jitter        0.0000 ms    0.0% ································
  residual            -0.9585 ms  -10.0% ▒▒▒·····························
  explained by model: 110.0%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  ! calibrated throughput exceeds the HardwareSpec peak; clamped to peak (the spec may be too conservative).
  next: peak_derate (73%, ≤3.68x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>


### JAX jit (cpu)

Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz. 6 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 6 |
| Bound tightness (median, min–max) | 17.5% (0.6%–34.8%) |
| Bound rank correlation (Spearman) | 0.943 |
| Roofline-as-predictor MAPE | 82.7% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 36.1% |
| Zero-shot median APE | 31.4% |
| Zero-shot rank correlation | 1.000 |
| Calibrated model LOO MAPE | **71.3%** |
| Calibrated model LOO median APE | 24.9% |
| Calibrated model bias (geomean pred/meas) | ×1.347 |
| Within ±25% | 50% |
| Calibrated rank correlation | 0.943 |
| Fitted call overhead | 154.3 µs |
| Fitted per-kernel dispatch | 0.00 µs |
| Fitted effective compute | 100.8 GFLOP/s |
| Fitted effective bandwidth | 20.2 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.172 | 0.0011 | 0.6% | 0.063 | -63.4% | – | – | 0.389 | +125.4% |
| mlp w=256 b=64 | 0.588 | 0.0680 | 11.6% | 0.344 | -41.6% | – | – | 0.417 | -29.1% |
| mlp w=1024 b=1 | 0.598 | 0.0165 | 2.8% | 0.454 | -24.1% | – | – | 2.001 | +234.8% |
| mlp w=1024 b=64 | 3.455 | 1.0582 | 30.6% | 2.340 | -32.3% | – | – | 3.649 | +5.6% |
| transformer d=256 T=64 b=1 | 1.764 | 0.4125 | 23.4% | 1.224 | -30.6% | – | – | 1.550 | -12.1% |
| transformer d=256 T=64 b=8 | 9.474 | 3.3001 | 34.8% | 7.116 | -24.9% | – | – | 11.422 | +20.6% |

Calibration: jax microbenchmarks (cpu, jit) — 121 GFLOP/s achievable, 4.9 GB/s, 9.55 µs/kernel dispatch, 0.3 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.172 ms, 1 kernels, calibration: jax microbenchmarks (cpu, jit)
  roofline_floor       0.0639 ms   37.1% ████████████····················
  unfused_traffic      0.0023 ms    1.4% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0000 ms    0.0% ································
  dispatch             0.2105 ms  122.1% ████████████████████████████████
  timing_jitter        0.0000 ms    0.0% ································
  residual            -0.1043 ms  -60.5% ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒·············
  explained by model: 160.5%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: dispatch (122%, ≤infx if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=64 b=8</summary>

```
Gap attribution — measured 9.474 ms, 1 kernels, calibration: jax microbenchmarks (cpu, jit)
  roofline_floor       4.6587 ms   49.2% ████████████████················
  unfused_traffic      3.0983 ms   32.7% ██████████······················
  serialization        1.7154 ms   18.1% ██████··························
  peak_derate          2.4795 ms   26.2% ████████························
  dispatch             0.2391 ms    2.5% █·······························
  timing_jitter        0.0000 ms    0.0% ································
  residual            -2.7171 ms  -28.7% ▒▒▒▒▒▒▒▒▒·······················
  explained by model: 128.7%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: unfused_traffic (33%, ≤1.49x if removed): Fuse point-wise chains: torch.compile, jax.jit/XLA, tf.function(jit_compile=True).
  next: peak_derate (26%, ≤1.35x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: serialization (18%, ≤1.22x if removed): Fuse heterogeneous ops or overlap independent ops (streams, larger fused kernels).
```

</details>

