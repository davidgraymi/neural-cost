# neural-cost validation report

Generated 2026-11-23 by `benchmarks/validate.py` on **Apple M3 (10-core GPU)**
(macOS-27.2.1-arm64-arm-64bit-Mach-O), hardware spec from Apple Silicon table (Apple M3 (10-core GPU)) [FP32] + NumPy STREAM triad.

Run: neural-cost v0.13.0-18-gcb36a1c @ cb36a1c2c163 on Apple M3 (10-core GPU), 2026-11-23T19:43:47+00:00 (Python 3.13.18; jax 0.11.2, jaxlib 0.11.2, keras 3.15.1, numpy 2.5.3, tensorflow 2.21.0, torch 2.14.0).

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

Hardware: Apple M3 (CPU). 23 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 23 |
| Bound tightness (median, min–max) | 12.8% (0.7%–66.7%) |
| Bound rank correlation (Spearman) | 0.895 |
| Roofline-as-predictor MAPE | 84.1% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 62.5% |
| Zero-shot median APE | 67.5% |
| Zero-shot rank correlation | 0.953 |
| Op-replay MAPE (isolated kernels, no fit) | **25.4%** |
| Op-replay median APE | 18.3% |
| Op-replay rank correlation | 0.987 |
| Calibrated model LOO MAPE | **51.4%** |
| Calibrated model LOO median APE | 52.0% |
| Calibrated model bias (geomean pred/meas) | ×0.718 |
| Within ±25% | 26% |
| Calibrated rank correlation | 0.938 |
| Fitted call overhead | 11.1 µs |
| Fitted per-kernel dispatch | 0.65 µs |
| Fitted effective compute | 598.5 GFLOP/s |
| Fitted effective bandwidth | 222.7 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.015 | 0.0001 | 0.7% | 0.008 | -47.6% | 0.009 | -39.1% | 0.026 | +69.5% |
| mlp w=256 b=16 | 0.033 | 0.0018 | 5.3% | 0.009 | -72.0% | 0.018 | -45.5% | 0.023 | -31.5% |
| mlp w=256 b=128 | 0.095 | 0.0141 | 14.8% | 0.033 | -65.5% | 0.045 | -52.5% | 0.076 | -20.2% |
| mlp w=1024 b=1 | 0.043 | 0.0017 | 4.0% | 0.079 | +82.1% | 0.036 | -17.8% | 0.123 | +183.2% |
| mlp w=1024 b=16 | 0.217 | 0.0278 | 12.8% | 0.082 | -62.4% | 0.181 | -16.6% | 0.164 | -24.5% |
| mlp w=1024 b=128 | 0.592 | 0.2224 | 37.6% | 0.368 | -37.8% | 0.514 | -13.2% | 1.083 | +82.9% |
| mlp w=4096 b=1 | 2.016 | 1.3450 | 66.7% | 1.303 | -35.3% | 1.992 | -1.2% | 0.566 | -71.9% |
| mlp w=4096 b=16 | 5.431 | 1.3573 | 25.0% | 1.340 | -75.3% | 5.350 | -1.5% | 2.352 | -56.7% |
| mlp w=4096 b=128 | 10.277 | 3.5640 | 34.7% | 5.678 | -44.8% | 9.928 | -3.4% | 16.257 | +58.2% |
| cnn 32px b=1 | 0.125 | 0.0085 | 6.8% | 0.027 | -78.5% | 0.099 | -21.1% | 0.053 | -57.3% |
| cnn 32px b=8 | 0.524 | 0.0681 | 13.0% | 0.170 | -67.5% | 0.471 | -10.0% | 0.305 | -41.8% |
| cnn 32px b=32 | 9.694 | 0.2725 | 2.8% | 0.567 | -94.1% | 9.677 | -0.2% | 1.179 | -87.8% |
| cnn 64px b=1 | 0.319 | 0.0341 | 10.7% | 0.097 | -69.6% | 0.267 | -16.3% | 0.161 | -49.4% |
| cnn 64px b=8 | 1.898 | 0.2725 | 14.4% | 0.567 | -70.1% | 1.486 | -21.7% | 1.164 | -38.6% |
| cnn 64px b=32 | 22.764 | 1.2219 | 5.4% | 2.265 | -90.0% | 22.040 | -3.2% | 4.635 | -79.6% |
| lstm d=128 b=1 | 0.531 | 0.0069 | 1.3% | 0.170 | -67.9% | 0.218 | -58.9% | 0.854 | +60.6% |
| lstm d=128 b=16 | 2.195 | 0.1107 | 5.0% | 0.407 | -81.4% | 0.703 | -68.0% | 1.012 | -53.9% |
| lstm d=256 b=1 | 0.904 | 0.0277 | 3.1% | 1.034 | +14.5% | 0.517 | -42.8% | 0.955 | +5.7% |
| lstm d=256 b=16 | 5.458 | 0.4427 | 8.1% | 1.572 | -71.2% | 2.035 | -62.7% | 2.619 | -52.0% |
| transformer d=256 T=64 b=1 | 0.274 | 0.0434 | 15.9% | 0.095 | -65.1% | 0.207 | -24.3% | 0.210 | -23.2% |
| transformer d=256 T=64 b=8 | 1.190 | 0.3469 | 29.2% | 0.733 | -38.4% | 1.008 | -15.3% | 1.528 | +28.4% |
| transformer d=256 T=256 b=1 | 0.818 | 0.1950 | 23.8% | 0.385 | -53.0% | 0.668 | -18.3% | 0.844 | +3.2% |
| transformer d=256 T=256 b=8 | 6.669 | 1.6419 | 24.6% | 3.112 | -53.3% | 4.606 | -30.9% | 6.492 | -2.7% |

Calibration: torch microbenchmarks (cpu, float32) — 1660 GFLOP/s achievable, 92.2 GB/s, 1.00 µs/kernel dispatch, 0.0 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.015 ms, 5 kernels, calibration: torch microbenchmarks (cpu, float32)
  roofline_floor       0.0054 ms   35.9% ███████████·····················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0000 ms    0.0% ································
  dispatch             0.0050 ms   33.4% ███████████·····················
  kernel_inefficiency     0.0000 ms    0.0% ································
  timing_jitter        0.0010 ms    6.4% ██······························
  residual             0.0037 ms   24.4% ████████························
  explained by model: 75.6%
  next: dispatch (33%, ≤1.50x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: residual (24%, ≤1.32x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
  next: timing_jitter (6%, ≤1.07x if removed): Pin clocks/threads, isolate the process, add warmup, report min alongside median.
```

</details>

<details><summary>Gap attribution: mlp w=4096 b=128</summary>

```
Gap attribution — measured 10.277 ms, 5 kernels, calibration: torch microbenchmarks (cpu, float32)
  roofline_floor       3.5457 ms   34.5% ███████████·····················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0183 ms    0.2% ································
  peak_derate          2.1087 ms   20.5% ███████·························
  dispatch             0.0050 ms    0.0% ································
  kernel_inefficiency     4.2255 ms   41.1% █████████████···················
  timing_jitter        0.0429 ms    0.4% ································
  residual             0.3305 ms    3.2% █·······························
  explained by model: 96.8%
  next: kernel_inefficiency (41%, ≤1.70x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
  next: peak_derate (21%, ≤1.26x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=256 b=8</summary>

```
Gap attribution — measured 6.669 ms, 15 kernels, calibration: torch microbenchmarks (cpu, float32)
  roofline_floor       1.5602 ms   23.4% ███████·························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0818 ms    1.2% ································
  peak_derate          1.4606 ms   21.9% ███████·························
  dispatch             0.0090 ms    0.1% ································
  kernel_inefficiency     1.5011 ms   22.5% ███████·························
  timing_jitter        0.0305 ms    0.5% ································
  residual             2.0261 ms   30.4% ██████████······················
  explained by model: 69.6%
  next: residual (30%, ≤1.44x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
  next: kernel_inefficiency (23%, ≤1.29x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
  next: peak_derate (22%, ≤1.28x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>


### PyTorch eager (mps)

Hardware: Apple M3 (10-core GPU). 23 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 23 |
| Bound tightness (median, min–max) | 20.5% (0.7%–73.7%) |
| Bound rank correlation (Spearman) | 0.917 |
| Roofline-as-predictor MAPE | 76.4% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 29.3% |
| Zero-shot median APE | 20.3% |
| Zero-shot rank correlation | 0.939 |
| Op-replay MAPE (isolated kernels, no fit) | **28.8%** |
| Op-replay median APE | 14.4% |
| Op-replay rank correlation | 0.940 |
| Calibrated model LOO MAPE | **21.2%** |
| Calibrated model LOO median APE | 12.3% |
| Calibrated model bias (geomean pred/meas) | ×0.949 |
| Within ±25% | 70% |
| Calibrated rank correlation | 0.936 |
| Fitted call overhead | 246.6 µs |
| Fitted per-kernel dispatch | 18.26 µs |
| Fitted effective compute | 2678.5 GFLOP/s |
| Fitted effective bandwidth | 87.2 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.300 | 0.0022 | 0.7% | 0.226 | -24.6% | 0.263 | -12.3% | 0.358 | +19.4% |
| mlp w=256 b=16 | 0.322 | 0.0025 | 0.8% | 0.257 | -20.3% | 0.273 | -15.1% | 0.353 | +9.6% |
| mlp w=256 b=128 | 0.340 | 0.0099 | 2.9% | 0.292 | -14.1% | 0.292 | -14.1% | 0.370 | +8.7% |
| mlp w=1024 b=1 | 0.401 | 0.0846 | 21.1% | 0.239 | -40.3% | 0.344 | -14.2% | 0.442 | +10.2% |
| mlp w=1024 b=16 | 0.418 | 0.0877 | 21.0% | 0.310 | -25.8% | 0.368 | -11.9% | 0.469 | +12.3% |
| mlp w=1024 b=128 | 0.636 | 0.1548 | 24.3% | 0.710 | +11.6% | 0.577 | -9.4% | 0.670 | +5.3% |
| mlp w=4096 b=1 | 1.824 | 1.3450 | 73.7% | 1.734 | -5.0% | 1.728 | -5.3% | 1.952 | +7.0% |
| mlp w=4096 b=16 | 1.976 | 1.3573 | 68.7% | 1.750 | -11.5% | 1.882 | -4.8% | 2.403 | +21.6% |
| mlp w=4096 b=128 | 3.775 | 2.4088 | 63.8% | 3.357 | -11.1% | 3.706 | -1.8% | 6.837 | +81.1% |
| cnn 32px b=1 | 0.441 | 0.0060 | 1.4% | 0.293 | -33.5% | 0.376 | -14.7% | 0.406 | -7.9% |
| cnn 32px b=8 | 0.594 | 0.0476 | 8.0% | 0.463 | -22.0% | 0.525 | -11.7% | 0.496 | -16.5% |
| cnn 32px b=32 | 1.239 | 0.2246 | 18.1% | 0.916 | -26.1% | 1.060 | -14.4% | 0.804 | -35.1% |
| cnn 64px b=1 | 0.491 | 0.0238 | 4.9% | 0.355 | -27.6% | 0.421 | -14.2% | 0.445 | -9.2% |
| cnn 64px b=8 | 1.093 | 0.2243 | 20.5% | 0.898 | -17.8% | 0.891 | -18.4% | 0.806 | -26.2% |
| cnn 64px b=32 | 2.761 | 0.8970 | 32.5% | 2.204 | -20.2% | 2.635 | -4.6% | 2.012 | -27.1% |
| lstm d=128 b=1 | 0.959 | 0.0711 | 7.4% | 1.876 | +95.5% | 2.285 | +138.2% | 0.512 | -46.6% |
| lstm d=128 b=16 | 1.260 | 0.1155 | 9.2% | 2.274 | +80.6% | 2.496 | +98.1% | 0.665 | -47.2% |
| lstm d=256 b=1 | 1.110 | 0.2806 | 25.3% | 1.971 | +77.5% | 2.403 | +116.4% | 1.162 | +4.6% |
| lstm d=256 b=16 | 1.814 | 0.8755 | 48.3% | 3.130 | +72.5% | 2.886 | +59.1% | 1.618 | -10.8% |
| transformer d=256 T=64 b=1 | 0.474 | 0.0302 | 6.4% | 0.410 | -13.5% | 0.368 | -22.5% | 0.674 | +42.1% |
| transformer d=256 T=64 b=8 | 1.042 | 0.2418 | 23.2% | 1.049 | +0.6% | 0.792 | -24.0% | 0.915 | -12.2% |
| transformer d=256 T=256 b=1 | 0.711 | 0.1354 | 19.1% | 0.760 | +6.9% | 0.565 | -20.5% | 0.754 | +6.1% |
| transformer d=256 T=256 b=8 | 2.755 | 1.1337 | 41.1% | 2.333 | -15.3% | 2.324 | -15.6% | 2.216 | -19.6% |

Calibration: torch microbenchmarks (mps, float32) — 2955 GFLOP/s achievable, 91.5 GB/s, 3.12 µs/kernel dispatch, 165.7 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.300 ms, 5 kernels, calibration: torch microbenchmarks (mps, float32)
  roofline_floor       0.0054 ms    1.8% █·······························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0478 ms   15.9% █████···························
  dispatch             0.1730 ms   57.7% ██████████████████··············
  kernel_inefficiency     0.0382 ms   12.7% ████····························
  timing_jitter        0.0087 ms    2.9% █·······························
  residual             0.0268 ms    8.9% ███·····························
  explained by model: 91.1%
  next: dispatch (58%, ≤2.36x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: peak_derate (16%, ≤1.19x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: kernel_inefficiency (13%, ≤1.15x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
  next: residual (9%, ≤1.10x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
```

</details>

<details><summary>Gap attribution: mlp w=4096 b=128</summary>

```
Gap attribution — measured 3.775 ms, 5 kernels, calibration: torch microbenchmarks (mps, float32)
  roofline_floor       2.3890 ms   63.3% ████████████████████············
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0197 ms    0.5% ································
  peak_derate          0.7828 ms   20.7% ███████·························
  dispatch             0.1657 ms    4.4% █·······························
  kernel_inefficiency     0.3808 ms   10.1% ███·····························
  timing_jitter        0.0211 ms    0.6% ································
  residual             0.0161 ms    0.4% ································
  explained by model: 99.6%
  next: peak_derate (21%, ≤1.26x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: kernel_inefficiency (10%, ≤1.11x if removed): Pad or reshape to library-friendly sizes (multiples of 8/64), switch layouts (channels_last), try another backend (oneDNN, cuDNN autotune, Triton), or fuse.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=256 b=8</summary>

```
Gap attribution — measured 2.755 ms, 13 kernels, calibration: torch microbenchmarks (mps, float32)
  roofline_floor       1.0512 ms   38.1% ████████████····················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0825 ms    3.0% █·······························
  peak_derate          1.0334 ms   37.5% ████████████····················
  dispatch             0.1657 ms    6.0% ██······························
  kernel_inefficiency     0.0020 ms    0.1% ································
  timing_jitter        0.0405 ms    1.5% ································
  residual             0.3801 ms   13.8% ████····························
  explained by model: 86.2%
  next: peak_derate (38%, ≤1.60x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: residual (14%, ≤1.16x if removed): Look between kernels: allocator churn, host-side Python, layout conversions; capture with CUDA graphs or compile to remove glue.
  next: dispatch (6%, ≤1.06x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
```

</details>


### JAX jit (cpu)

Hardware: Apple M3 (CPU). 13 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 13 |
| Bound tightness (median, min–max) | 15.8% (0.0%–44.8%) |
| Bound rank correlation (Spearman) | 0.934 |
| Roofline-as-predictor MAPE | 84.3% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 62.9% |
| Zero-shot median APE | 76.0% |
| Zero-shot rank correlation | 0.934 |
| Calibrated model LOO MAPE | **5.2%** |
| Calibrated model LOO median APE | 2.1% |
| Calibrated model bias (geomean pred/meas) | ×0.993 |
| Within ±25% | 92% |
| Calibrated rank correlation | 0.995 |
| Fitted call overhead | 268.9 µs |
| Fitted per-kernel dispatch | 0.00 µs |
| Fitted effective compute | inf GFLOP/s |
| Fitted effective bandwidth | 16.4 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.304 | 0.0001 | 0.0% | 0.022 | -92.6% | – | – | 0.301 | -0.8% |
| mlp w=256 b=16 | 0.311 | 0.0018 | 0.6% | 0.053 | -83.0% | – | – | 0.305 | -1.9% |
| mlp w=256 b=128 | 0.337 | 0.0141 | 4.2% | 0.118 | -65.0% | – | – | 0.344 | +2.1% |
| mlp w=1024 b=1 | 0.788 | 0.0017 | 0.2% | 0.071 | -91.0% | – | – | 0.785 | -0.5% |
| mlp w=1024 b=16 | 0.797 | 0.0278 | 3.5% | 0.191 | -76.0% | – | – | 0.804 | +1.0% |
| mlp w=1024 b=128 | 1.034 | 0.2225 | 21.5% | 1.010 | -2.4% | – | – | 0.937 | -9.4% |
| mlp w=4096 b=1 | 8.430 | 1.3450 | 16.0% | 1.482 | -82.4% | – | – | 8.487 | +0.7% |
| mlp w=4096 b=16 | 8.580 | 1.3573 | 15.8% | 1.956 | -77.2% | – | – | 8.544 | -0.4% |
| mlp w=4096 b=128 | 10.459 | 3.5461 | 33.9% | 15.531 | +48.5% | – | – | 8.908 | -14.8% |
| transformer d=256 T=64 b=1 | 0.569 | 0.0433 | 7.6% | 0.263 | -53.8% | – | – | 0.556 | -2.3% |
| transformer d=256 T=64 b=8 | 1.022 | 0.3465 | 33.9% | 1.544 | +51.0% | – | – | 1.285 | +25.7% |
| transformer d=256 T=256 b=1 | 0.894 | 0.1948 | 21.8% | 0.889 | -0.5% | – | – | 0.841 | -5.9% |
| transformer d=256 T=256 b=8 | 3.477 | 1.5584 | 44.8% | 6.743 | +93.9% | – | – | 3.567 | +2.6% |

Calibration: jax microbenchmarks (cpu, jit) — 566 GFLOP/s achievable, 92.0 GB/s, 2.40 µs/kernel dispatch, 0.2 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.304 ms, 1 kernels, calibration: jax microbenchmarks (cpu, jit)
  roofline_floor       0.0054 ms    1.8% █·······························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0064 ms    2.1% █·······························
  dispatch             0.0146 ms    4.8% ██······························
  timing_jitter        0.0187 ms    6.2% ██······························
  residual             0.2584 ms   85.1% ███████████████████████████·····
  explained by model: 14.9%
  next: residual (85%, ≤6.71x if removed): Profile kernels (Nsight Compute, torch.profiler, XLA dump) to see which kernels miss their per-op bound.
  next: timing_jitter (6%, ≤1.07x if removed): Pin clocks/threads, isolate the process, add warmup, report min alongside median.
```

</details>

<details><summary>Gap attribution: mlp w=4096 b=128</summary>

```
Gap attribution — measured 10.459 ms, 1 kernels, calibration: jax microbenchmarks (cpu, jit)
  roofline_floor       3.5461 ms   33.9% ███████████·····················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.1022 ms    1.0% ································
  peak_derate         11.8577 ms  113.4% ████████████████████████████████
  dispatch             0.0146 ms    0.1% ································
  timing_jitter        0.0000 ms    0.0% ································
  residual            -5.0620 ms  -48.4% ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒·················
  explained by model: 148.4%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: peak_derate (113%, ≤infx if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=256 b=8</summary>

```
Gap attribution — measured 3.477 ms, 1 kernels, calibration: jax microbenchmarks (cpu, jit)
  roofline_floor       1.5584 ms   44.8% ██████████████··················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.1254 ms    3.6% █·······························
  peak_derate          5.4296 ms  156.1% ████████████████████████████████
  dispatch             0.0218 ms    0.6% ································
  timing_jitter        0.0000 ms    0.0% ································
  residual            -3.6578 ms -105.2% ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  explained by model: 205.2%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: peak_derate (156%, ≤infx if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>


### JAX jit (mps)

Hardware: Apple M3 (10-core GPU). 13 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 13 |
| Bound tightness (median, min–max) | 29.5% (0.9%–73.8%) |
| Bound rank correlation (Spearman) | 0.945 |
| Roofline-as-predictor MAPE | 69.1% |
| Zero-shot MAPE (microbenchmarks only, no fit) | 25.9% |
| Zero-shot median APE | 20.1% |
| Zero-shot rank correlation | 0.945 |
| Calibrated model LOO MAPE | **20.7%** |
| Calibrated model LOO median APE | 15.5% |
| Calibrated model bias (geomean pred/meas) | ×0.958 |
| Within ±25% | 85% |
| Calibrated rank correlation | 0.951 |
| Fitted call overhead | 216.9 µs |
| Fitted per-kernel dispatch | 0.00 µs |
| Fitted effective compute | 2899.7 GFLOP/s |
| Fitted effective bandwidth | 98.6 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.230 | 0.0022 | 0.9% | 0.136 | -41.0% | – | – | 0.220 | -4.3% |
| mlp w=256 b=16 | 0.249 | 0.0025 | 1.0% | 0.138 | -44.4% | – | – | 0.219 | -11.9% |
| mlp w=256 b=128 | 0.205 | 0.0095 | 4.6% | 0.186 | -9.4% | – | – | 0.254 | +23.7% |
| mlp w=1024 b=1 | 0.287 | 0.0846 | 29.5% | 0.247 | -14.0% | – | – | 0.308 | +7.3% |
| mlp w=1024 b=16 | 0.292 | 0.0877 | 30.1% | 0.251 | -14.0% | – | – | 0.335 | +15.0% |
| mlp w=1024 b=128 | 0.498 | 0.1499 | 30.1% | 0.398 | -20.1% | – | – | 0.518 | +4.0% |
| mlp w=4096 b=1 | 1.822 | 1.3450 | 73.8% | 1.615 | -11.4% | – | – | 1.370 | -24.8% |
| mlp w=4096 b=16 | 1.844 | 1.3573 | 73.6% | 1.628 | -11.7% | – | – | 2.045 | +10.9% |
| mlp w=4096 b=128 | 3.615 | 2.3893 | 66.1% | 3.296 | -8.8% | – | – | 6.071 | +67.9% |
| transformer d=256 T=64 b=1 | 0.351 | 0.0292 | 8.3% | 0.226 | -35.6% | – | – | 0.297 | -15.5% |
| transformer d=256 T=64 b=8 | 0.808 | 0.2335 | 28.9% | 0.518 | -35.9% | – | – | 0.649 | -19.7% |
| transformer d=256 T=256 b=1 | 0.610 | 0.1313 | 21.5% | 0.369 | -39.5% | – | – | 0.465 | -23.7% |
| transformer d=256 T=256 b=8 | 3.169 | 1.0500 | 33.1% | 1.573 | -50.4% | – | – | 1.873 | -40.9% |

Calibration: jax microbenchmarks (mps, jit) — 2803 GFLOP/s achievable, 90.7 GB/s, 113.31 µs/kernel dispatch, 0.0 µs/call sync latency.

<details><summary>Gap attribution: mlp w=256 b=1</summary>

```
Gap attribution — measured 0.230 ms, 1 kernels, calibration: jax microbenchmarks (mps, jit)
  roofline_floor       0.0054 ms    2.4% █·······························
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.0000 ms    0.0% ································
  peak_derate          0.0272 ms   11.8% ████····························
  dispatch             0.6799 ms  295.2% ████████████████████████████████
  timing_jitter        0.0000 ms    0.0% ································
  residual            -0.4822 ms -209.4% ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  explained by model: 309.4%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: dispatch (295%, ≤infx if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: peak_derate (12%, ≤1.13x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>

<details><summary>Gap attribution: mlp w=4096 b=128</summary>

```
Gap attribution — measured 3.615 ms, 1 kernels, calibration: jax microbenchmarks (mps, jit)
  roofline_floor       2.3893 ms   66.1% █████████████████████···········
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.1038 ms    2.9% █·······························
  peak_derate          1.0566 ms   29.2% █████████·······················
  dispatch             0.6799 ms   18.8% ██████··························
  timing_jitter        0.0000 ms    0.0% ································
  residual            -0.6141 ms  -17.0% ▒▒▒▒▒···························
  explained by model: 117.0%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: peak_derate (29%, ≤1.41x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
  next: dispatch (19%, ≤1.23x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
```

</details>

<details><summary>Gap attribution: transformer d=256 T=256 b=8</summary>

```
Gap attribution — measured 3.169 ms, 1 kernels, calibration: jax microbenchmarks (mps, jit)
  roofline_floor       1.0500 ms   33.1% ███████████·····················
  unfused_traffic      0.0000 ms    0.0% ································
  serialization        0.1255 ms    4.0% █·······························
  peak_derate          0.9809 ms   30.9% ██████████······················
  dispatch             1.0198 ms   32.2% ██████████······················
  timing_jitter        0.0000 ms    0.0% ································
  residual            -0.0068 ms   -0.2% ································
  explained by model: 100.2%
  ! components exceed measured time (negative residual): calibration or kernel count is likely overestimated.
  next: dispatch (32%, ≤1.47x if removed): Amortize: increase batch size, capture with CUDA graphs, compile (torch.compile mode='reduce-overhead', jax.jit), or reduce kernel count via fusion.
  next: peak_derate (31%, ≤1.45x if removed): Largely a hardware property; use tuned libraries (cuBLAS/oneDNN/Accelerate) and aligned shapes to approach achievable peak.
```

</details>


### TensorFlow graph (cpu)

Hardware: Apple M3 (CPU). 23 workloads.

| Metric | Value |
|---|---|
| Lower-bound violations | 0 / 23 |
| Bound tightness (median, min–max) | 6.8% (0.1%–37.2%) |
| Bound rank correlation (Spearman) | 0.907 |
| Roofline-as-predictor MAPE | 90.8% |
| Calibrated model LOO MAPE | **22.1%** |
| Calibrated model LOO median APE | 22.4% |
| Calibrated model bias (geomean pred/meas) | ×0.940 |
| Within ±25% | 61% |
| Calibrated rank correlation | 0.954 |
| Fitted call overhead | 64.3 µs |
| Fitted per-kernel dispatch | 21.66 µs |
| Fitted effective compute | 350.9 GFLOP/s |
| Fitted effective bandwidth | 61.9 GB/s |

| Workload | Measured (ms) | Bound (ms) | Tightness | Zero-shot (ms) | Error | Op-replay (ms) | Error | LOO fit (ms) | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp w=256 b=1 | 0.124 | 0.0001 | 0.1% | – | – | – | – | 0.152 | +22.4% |
| mlp w=256 b=16 | 0.177 | 0.0018 | 1.0% | – | – | – | – | 0.144 | -18.9% |
| mlp w=256 b=128 | 0.323 | 0.0141 | 4.4% | – | – | – | – | 0.240 | -25.6% |
| mlp w=1024 b=1 | 0.210 | 0.0017 | 0.8% | – | – | – | – | 0.331 | +57.5% |
| mlp w=1024 b=16 | 0.456 | 0.0278 | 6.1% | – | – | – | – | 0.464 | +1.7% |
| mlp w=1024 b=128 | 1.761 | 0.2224 | 12.6% | – | – | – | – | 1.855 | +5.3% |
| mlp w=4096 b=1 | 3.612 | 1.3450 | 37.2% | – | – | – | – | 1.771 | -51.0% |
| mlp w=4096 b=16 | 6.182 | 1.3573 | 22.0% | – | – | – | – | 5.269 | -14.8% |
| mlp w=4096 b=128 | 21.080 | 3.5640 | 16.9% | – | – | – | – | 28.302 | +34.3% |
| cnn 32px b=1 | 0.261 | 0.0085 | 3.3% | – | – | – | – | 0.243 | -7.0% |
| cnn 32px b=8 | 0.768 | 0.0681 | 8.9% | – | – | – | – | 0.707 | -7.9% |
| cnn 32px b=32 | 1.922 | 0.2725 | 14.2% | – | – | – | – | 2.360 | +22.8% |
| cnn 64px b=1 | 0.498 | 0.0341 | 6.8% | – | – | – | – | 0.442 | -11.2% |
| cnn 64px b=8 | 1.962 | 0.2725 | 13.9% | – | – | – | – | 2.352 | +19.8% |
| cnn 64px b=32 | 7.106 | 1.2219 | 17.2% | – | – | – | – | 8.964 | +26.1% |
| lstm d=128 b=1 | 1.653 | 0.0069 | 0.4% | – | – | – | – | 2.143 | +29.6% |
| lstm d=128 b=16 | 3.036 | 0.1107 | 3.6% | – | – | – | – | 2.574 | -15.2% |
| lstm d=256 b=1 | 2.755 | 0.0277 | 1.0% | – | – | – | – | 2.913 | +5.7% |
| lstm d=256 b=16 | 6.764 | 0.4427 | 6.5% | – | – | – | – | 5.842 | -13.6% |
| transformer d=256 T=64 b=1 | 0.766 | 0.0434 | 5.7% | – | – | – | – | 0.534 | -30.3% |
| transformer d=256 T=64 b=8 | 3.987 | 0.3469 | 8.7% | – | – | – | – | 2.694 | -32.4% |
| transformer d=256 T=256 b=1 | 2.380 | 0.1950 | 8.2% | – | – | – | – | 1.606 | -32.5% |
| transformer d=256 T=256 b=8 | 14.775 | 1.6419 | 11.1% | – | – | – | – | 11.289 | -23.6% |

No calibration microbenchmarks for this backend; no gap attribution.

