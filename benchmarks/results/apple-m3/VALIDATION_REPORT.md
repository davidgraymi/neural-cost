# neural-cost validation report

Generated 2026-10-18 by `benchmarks/validate.py` on **Apple M3**
(macOS-27.0.1-arm64-arm-64bit), hardware spec from Apple Silicon table (Apple M3) [FP32] + NumPy STREAM triad.

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

### PyTorch eager (cpu)

Hardware: Apple M3. 23 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 23 |
| Bound tightness (median, min–max) | 10.6% (1.9%–66.7%) |
| Bound rank correlation (Spearman) | 0.918 |
| Roofline-as-predictor MAPE | 85.2% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 61.4% |
| Zero-shot median APE | 65.7% |
| Zero-shot rank correlation | 0.952 |
| Op-replay MAPE (isolated kernels, no fit) | **26.3%** |
| Op-replay median APE | 21.8% |
| Op-replay rank correlation | 0.987 |
| Calibrated model LOO MAPE | **51.3%** |
| Calibrated model LOO median APE | 52.0% |
| Calibrated model bias (geomean pred/meas) | ×0.717 |
| Within ±25% | 26% |
| Calibrated rank correlation | 0.938 |
| Fitted call overhead | 10.8 µs |
| Fitted per-kernel dispatch | 0.64 µs |
| Fitted effective compute | 599.8 GFLOP/s |
| Fitted effective bandwidth | 217.8 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.015 | 0.0022 | 14.7% | 0.008 | -44.5% | 0.009 | -40.7% | 0.025 | +71.9% |
| mlp w=256 b=16 | 0.033 | 0.0025 | 7.6% | 0.009 | -71.0% | 0.018 | -45.1% | 0.022 | -31.6% |
| mlp w=256 b=128 | 0.094 | 0.0099 | 10.6% | 0.033 | -64.9% | 0.046 | -51.3% | 0.076 | -19.3% |
| mlp w=1024 b=1 | 0.044 | 0.0012 | 2.7% | 0.072 | +63.4% | 0.036 | -19.0% | 0.122 | +177.0% |
| mlp w=1024 b=16 | 0.216 | 0.0187 | 8.7% | 0.074 | -65.7% | 0.181 | -16.4% | 0.164 | -24.2% |
| mlp w=1024 b=128 | 0.590 | 0.1499 | 25.4% | 0.361 | -38.8% | 0.507 | -14.0% | 1.083 | +83.5% |
| mlp w=4096 b=1 | 2.016 | 1.3450 | 66.7% | 1.354 | -32.8% | 1.995 | -1.0% | 0.578 | -71.3% |
| mlp w=4096 b=16 | 5.457 | 1.3573 | 24.9% | 1.385 | -74.6% | 5.327 | -2.4% | 2.362 | -56.7% |
| mlp w=4096 b=128 | 10.260 | 2.4088 | 23.5% | 5.697 | -44.5% | 9.854 | -4.0% | 16.238 | +58.3% |
| cnn 32px b=1 | 0.127 | 0.0059 | 4.6% | 0.026 | -79.7% | 0.097 | -23.9% | 0.053 | -58.4% |
| cnn 32px b=8 | 0.521 | 0.0466 | 8.9% | 0.160 | -69.3% | 0.451 | -13.5% | 0.303 | -41.9% |
| cnn 32px b=32 | 9.752 | 0.1835 | 1.9% | 0.533 | -94.5% | 9.461 | -3.0% | 1.172 | -88.0% |
| cnn 64px b=1 | 0.319 | 0.0233 | 7.3% | 0.089 | -72.2% | 0.249 | -21.8% | 0.160 | -49.7% |
| cnn 64px b=8 | 1.903 | 0.1835 | 9.6% | 0.532 | -72.0% | 1.450 | -23.8% | 1.158 | -39.2% |
| cnn 64px b=32 | 22.670 | 0.8549 | 3.8% | 2.234 | -90.1% | 22.133 | -2.4% | 4.610 | -79.7% |
| lstm d=128 b=1 | 0.531 | 0.0711 | 13.4% | 0.187 | -64.8% | 0.222 | -58.1% | 0.851 | +60.2% |
| lstm d=128 b=16 | 2.202 | 0.1155 | 5.2% | 0.436 | -80.2% | 0.721 | -67.2% | 1.010 | -54.1% |
| lstm d=256 b=1 | 0.904 | 0.2806 | 31.0% | 0.894 | -1.2% | 0.521 | -42.4% | 0.960 | +6.2% |
| lstm d=256 b=16 | 5.463 | 0.2983 | 5.5% | 1.487 | -72.8% | 2.086 | -61.8% | 2.621 | -52.0% |
| transformer d=256 T=64 b=1 | 0.273 | 0.0302 | 11.1% | 0.093 | -66.0% | 0.204 | -25.1% | 0.210 | -23.1% |
| transformer d=256 T=64 b=8 | 1.191 | 0.2418 | 20.3% | 0.712 | -40.2% | 0.991 | -16.8% | 1.525 | +28.0% |
| transformer d=256 T=256 b=1 | 0.818 | 0.1354 | 16.6% | 0.373 | -54.4% | 0.664 | -18.9% | 0.843 | +3.0% |
| transformer d=256 T=256 b=8 | 6.670 | 1.1337 | 17.0% | 3.065 | -54.1% | 4.552 | -31.8% | 6.480 | -2.8% |

Calibration: torch microbenchmarks (cpu, float32) — 1658 GFLOP/s achievable, 92.2 GB/s, 0.92 µs/kernel dispatch, 0.1 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.015 ms, 5 kernels, calibration: torch microbenchmarks (cpu, float32)
  roofline_floor       0.0054 ms   36.7% ████████████····················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0000 ms    0.0% ································
  dispatch             0.0047 ms   31.7% ██████████······················
  kernel_inefficiency     0.0000 ms    0.0% ································
  timing_jitter        0.0009 ms    5.9% ██······························
  residual             0.0038 ms   25.6% ████████························
  explained by model: 74.4%
  next: dispatch (32%, ≤1.46x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: residual (26%, ≤1.34x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
  next: timing_jitter (6%, ≤1.06x if removed): Pin clocks/threads, isolate the process, add warmup, report min alongside median.
```

</details>

<details><summary>Gap attribution: mlp w=4096 b=128</summary>

```
Gap attribution — measured 10.260 ms, 5 kernels, calibration: torch microbenchmarks (cpu, float32)
  roofline_floor       2.3890 ms   23.3% ███████·························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0197 ms    0.2% ································
  peak_derate          3.2835 ms   32.0% ██████████······················
  dispatch             0.0047 ms    0.0% ································
  kernel_inefficiency     4.1677 ms   40.6% █████████████···················
  timing_jitter        0.0349 ms    0.3% ································
  residual             0.3603 ms    3.5% █·······························
  explained by model: 96.5%
  next: kernel_inefficiency (41%, ≤1.68x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
  next: peak_derate (32%, ≤1.47x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=256 b=8</summary>

```
Gap attribution — measured 6.670 ms, 15 kernels, calibration: torch microbenchmarks (cpu, float32)
  roofline_floor       1.0512 ms   15.8% █████···························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0825 ms    1.2% ································
  peak_derate          1.9225 ms   28.8% █████████·······················
  dispatch             0.0084 ms    0.1% ································
  kernel_inefficiency     1.4946 ms   22.4% ███████·························
  timing_jitter        0.0317 ms    0.5% ································
  residual             2.0794 ms   31.2% ██████████······················
  explained by model: 68.8%
  next: residual (31%, ≤1.45x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
  next: peak_derate (29%, ≤1.40x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: kernel_inefficiency (22%, ≤1.29x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
```

</details>


### PyTorch eager (mps)

Hardware: Apple M3. 23 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 23 |
| Bound tightness (median, min–max) | 20.6% (0.8%–74.3%) |
| Bound rank correlation (Spearman) | 0.917 |
| Roofline-as-predictor MAPE | 76.0% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 28.9% |
| Zero-shot median APE | 19.0% |
| Zero-shot rank correlation | 0.929 |
| Op-replay MAPE (isolated kernels, no fit) | **29.5%** |
| Op-replay median APE | 15.2% |
| Op-replay rank correlation | 0.907 |
| Calibrated model LOO MAPE | **21.7%** |
| Calibrated model LOO median APE | 13.3% |
| Calibrated model bias (geomean pred/meas) | ×0.944 |
| Within ±25% | 70% |
| Calibrated rank correlation | 0.948 |
| Fitted call overhead | 228.0 µs |
| Fitted per-kernel dispatch | 16.70 µs |
| Fitted effective compute | 2631.7 GFLOP/s |
| Fitted effective bandwidth | 88.2 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.277 | 0.0022 | 0.8% | 0.224 | -19.0% | 0.260 | -6.2% | 0.331 | +19.6% |
| mlp w=256 b=16 | 0.296 | 0.0025 | 0.8% | 0.252 | -15.0% | 0.269 | -9.1% | 0.327 | +10.2% |
| mlp w=256 b=128 | 0.319 | 0.0099 | 3.1% | 0.284 | -11.2% | 0.288 | -9.8% | 0.342 | +7.1% |
| mlp w=1024 b=1 | 0.367 | 0.0846 | 23.1% | 0.234 | -36.2% | 0.344 | -6.3% | 0.415 | +13.3% |
| mlp w=1024 b=16 | 0.397 | 0.0877 | 22.1% | 0.301 | -24.1% | 0.370 | -6.9% | 0.441 | +11.2% |
| mlp w=1024 b=128 | 0.604 | 0.1548 | 25.6% | 0.692 | +14.7% | 0.570 | -5.6% | 0.647 | +7.1% |
| mlp w=4096 b=1 | 1.810 | 1.3450 | 74.3% | 1.725 | -4.7% | 1.723 | -4.8% | 1.892 | +4.5% |
| mlp w=4096 b=16 | 1.953 | 1.3573 | 69.5% | 1.739 | -11.0% | 1.873 | -4.1% | 2.361 | +20.9% |
| mlp w=4096 b=128 | 3.768 | 2.4088 | 63.9% | 3.342 | -11.3% | 3.711 | -1.5% | 6.790 | +80.2% |
| cnn 32px b=1 | 0.414 | 0.0059 | 1.4% | 0.282 | -31.8% | 0.359 | -13.5% | 0.374 | -9.8% |
| cnn 32px b=8 | 0.569 | 0.0466 | 8.2% | 0.425 | -25.3% | 0.434 | -23.7% | 0.463 | -18.7% |
| cnn 32px b=32 | 1.205 | 0.2139 | 17.8% | 0.871 | -27.7% | 0.756 | -37.2% | 0.764 | -36.6% |
| cnn 64px b=1 | 0.463 | 0.0233 | 5.0% | 0.332 | -28.3% | 0.391 | -15.5% | 0.413 | -10.9% |
| cnn 64px b=8 | 1.040 | 0.2138 | 20.6% | 0.853 | -17.9% | 0.756 | -27.3% | 0.765 | -26.4% |
| cnn 64px b=32 | 2.745 | 0.8549 | 31.1% | 2.153 | -21.6% | 2.147 | -21.8% | 1.940 | -29.3% |
| lstm d=128 b=1 | 0.969 | 0.0711 | 7.3% | 1.896 | +95.7% | 2.231 | +130.2% | 0.486 | -49.8% |
| lstm d=128 b=16 | 1.195 | 0.1155 | 9.7% | 2.173 | +81.8% | 2.441 | +104.2% | 0.640 | -46.5% |
| lstm d=256 b=1 | 1.070 | 0.2806 | 26.2% | 1.900 | +77.7% | 2.361 | +120.7% | 1.132 | +5.9% |
| lstm d=256 b=16 | 1.783 | 0.8755 | 49.1% | 2.983 | +67.3% | 2.917 | +63.6% | 1.591 | -10.8% |
| transformer d=256 T=64 b=1 | 0.446 | 0.0302 | 6.8% | 0.401 | -10.0% | 0.374 | -16.1% | 0.625 | +40.2% |
| transformer d=256 T=64 b=8 | 0.986 | 0.2418 | 24.5% | 1.024 | +3.9% | 0.783 | -20.6% | 0.883 | -10.4% |
| transformer d=256 T=256 b=1 | 0.658 | 0.1354 | 20.6% | 0.739 | +12.3% | 0.563 | -14.4% | 0.723 | +9.8% |
| transformer d=256 T=256 b=8 | 2.732 | 1.1337 | 41.5% | 2.310 | -15.4% | 2.316 | -15.2% | 2.201 | -19.4% |

Calibration: torch microbenchmarks (mps, float32) — 2954 GFLOP/s achievable, 90.9 GB/s, 3.11 µs/kernel dispatch, 163.0 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.277 ms, 5 kernels, calibration: torch microbenchmarks (mps, float32)
  roofline_floor       0.0054 ms    2.0% █·······························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0488 ms   17.6% ██████··························
  dispatch             0.1701 ms   61.5% ████████████████████············
  kernel_inefficiency     0.0373 ms   13.5% ████····························
  timing_jitter        0.0046 ms    1.7% █·······························
  residual             0.0105 ms    3.8% █·······························
  explained by model: 96.2%
  next: dispatch (61%, ≤2.59x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: peak_derate (18%, ≤1.21x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: kernel_inefficiency (13%, ≤1.16x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
```

</details>

<details><summary>Gap attribution: mlp w=4096 b=128</summary>

```
Gap attribution — measured 3.768 ms, 5 kernels, calibration: torch microbenchmarks (mps, float32)
  roofline_floor       2.3890 ms   63.4% ████████████████████············
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0197 ms    0.5% ································
  peak_derate          0.7705 ms   20.4% ███████·························
  dispatch             0.1630 ms    4.3% █·······························
  kernel_inefficiency     0.3633 ms    9.6% ███·····························
  timing_jitter        0.0623 ms    1.7% █·······························
  residual             0.0007 ms    0.0% ································
  explained by model: 100.0%
  next: peak_derate (20%, ≤1.26x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: kernel_inefficiency (10%, ≤1.11x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=256 b=8</summary>

```
Gap attribution — measured 2.732 ms, 13 kernels, calibration: torch microbenchmarks (mps, float32)
  roofline_floor       1.0512 ms   38.5% ████████████····················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0825 ms    3.0% █·······························
  peak_derate          1.0138 ms   37.1% ████████████····················
  dispatch             0.1630 ms    6.0% ██······························
  kernel_inefficiency     0.0067 ms    0.2% ································
  timing_jitter        0.0375 ms    1.4% ································
  residual             0.3775 ms   13.8% ████····························
  explained by model: 86.2%
  next: peak_derate (37%, ≤1.59x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: residual (14%, ≤1.16x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
  next: dispatch (6%, ≤1.06x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
```

</details>


### JAX jit (cpu)

Hardware: Apple M3. 13 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 13 |
| Bound tightness (median, min–max) | 11.4% (1.5%–65.0%) |
| Bound rank correlation (Spearman) | 0.978 |
| Roofline-as-predictor MAPE | 85.7% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 38.8% |
| Zero-shot median APE | 32.8% |
| Zero-shot rank correlation | 0.989 |
| Calibrated model LOO MAPE | **45.4%** |
| Calibrated model LOO median APE | 41.0% |
| Calibrated model bias (geomean pred/meas) | ×0.915 |
| Within ±25% | 38% |
| Calibrated rank correlation | 0.951 |
| Fitted call overhead | 15.9 µs |
| Fitted per-kernel dispatch | 0.00 µs |
| Fitted effective compute | 420.9 GFLOP/s |
| Fitted effective bandwidth | 104.0 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.020 | 0.0022 | 11.4% | 0.010 | -50.1% | – | – | 0.047 | +139.8% |
| mlp w=256 b=16 | 0.076 | 0.0038 | 5.0% | 0.053 | -29.7% | – | – | 0.033 | -57.1% |
| mlp w=256 b=128 | 0.247 | 0.0153 | 6.2% | 0.115 | -53.7% | – | – | 0.132 | -46.5% |
| mlp w=1024 b=1 | 0.080 | 0.0012 | 1.5% | 0.106 | +32.8% | – | – | 0.165 | +107.1% |
| mlp w=1024 b=16 | 0.319 | 0.0188 | 5.9% | 0.186 | -41.6% | – | – | 0.270 | -15.2% |
| mlp w=1024 b=128 | 1.224 | 0.1505 | 12.3% | 1.008 | -17.7% | – | – | 1.590 | +29.9% |
| mlp w=4096 b=1 | 2.076 | 1.3482 | 65.0% | 1.494 | -28.0% | – | – | 1.225 | -41.0% |
| mlp w=4096 b=16 | 6.776 | 1.4097 | 20.8% | 1.953 | -71.2% | – | – | 3.818 | -43.7% |
| mlp w=4096 b=128 | 20.391 | 2.3916 | 11.7% | 15.534 | -23.8% | – | – | 22.770 | +11.7% |
| transformer d=256 T=64 b=1 | 0.662 | 0.0315 | 4.8% | 0.258 | -61.1% | – | – | 0.333 | -49.7% |
| transformer d=256 T=64 b=8 | 2.026 | 0.2345 | 11.6% | 1.545 | -23.7% | – | – | 2.508 | +23.8% |
| transformer d=256 T=256 b=1 | 1.521 | 0.1317 | 8.7% | 0.886 | -41.7% | – | – | 1.402 | -7.8% |
| transformer d=256 T=256 b=8 | 9.615 | 2.0487 | 21.3% | 6.766 | -29.6% | – | – | 11.279 | +17.3% |

Calibration: jax microbenchmarks (cpu, jit) — 565 GFLOP/s achievable, 91.9 GB/s, 2.38 µs/kernel dispatch, 0.3 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.020 ms, 1 kernels, calibration: jax microbenchmarks (cpu, jit)
  roofline_floor       0.0056 ms   28.5% █████████·······················
  unfused_traffic      0.0002 ms    1.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0022 ms   11.0% ████····························
  dispatch             0.0526 ms  266.2% ████████████████████████████████
  timing_jitter        0.0000 ms    0.0% ································
  residual            -0.0408 ms -206.7% ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  explained by model: 306.7%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: dispatch (266%, ≤infx if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: peak_derate (11%, ≤1.12x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>

<details><summary>Gap attribution: mlp w=4096 b=128</summary>

```
Gap attribution — measured 20.391 ms, 1 kernels, calibration: jax microbenchmarks (cpu, jit)
  roofline_floor       2.3916 ms   11.7% ████····························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.8564 ms    4.2% █·······························
  peak_derate         13.1688 ms   64.6% █████████████████████···········
  dispatch             0.0526 ms    0.3% ································
  timing_jitter        0.1643 ms    0.8% ································
  residual             3.7567 ms   18.4% ██████··························
  explained by model: 81.6%
  next: peak_derate (65%, ≤2.82x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: residual (18%, ≤1.23x if removed): Profile kernels (Nsight Compute, torch.profiler, XLA dump) to see which kernels miss their per-op bound.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=256 b=8</summary>

```
Gap attribution — measured 9.615 ms, 1 kernels, calibration: jax microbenchmarks (cpu, jit)
  roofline_floor       2.0487 ms   21.3% ███████·························
  unfused_traffic      2.0972 ms   21.8% ███████·························
  serialization        0.4858 ms    5.1% ██······························
  peak_derate          5.2987 ms   55.1% ██████████████████··············
  dispatch             0.0597 ms    0.6% ································
  timing_jitter        0.0000 ms    0.0% ································
  residual            -0.3755 ms   -3.9% ▒·······························
  explained by model: 103.9%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: peak_derate (55%, ≤2.23x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: unfused_traffic (22%, ≤1.28x if removed): Fuse point-wise chains: torch.compile, jax.jit/XLA, tf.function(jit_compile=True).
  next: serialization (5%, ≤1.05x if removed): Fuse heterogeneous ops or overlap independent ops (streams, larger fused kernels).
```

</details>

