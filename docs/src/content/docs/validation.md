---
title: "Validation"
description: "Out-of-sample prediction error and lower-bound checks on real hardware."
section: "Reference"
order: 2
---

`benchmarks/validate.py` runs a grid of MLP, CNN, LSTM and Transformer
workloads on every installed backend and writes
[`VALIDATION_REPORT.md`](https://github.com/davidgraymi/neural-cost/blob/main/benchmarks/results/apple-m3/VALIDATION_REPORT.md). It scores three predictors,
and none of them is ever fit to the workload it predicts:

* **Zero-shot**: microbenchmark calibration only. It never sees a model runtime.
* **Op-replay**: each captured op is timed alone with synthetic tensors, then
  the times are composed. It never runs the model.
* **LOO fit**: a 4-parameter non-negative model
  (`T = c₀ + α·kernels + FLOPs/P + bytes/BW`), with each workload predicted
  by a fit on all the others.

It also checks that the roofline is a valid **lower bound**. Any measurement
faster than the bound would falsify the model.

Results on an Apple M3 (`VALIDATION_REPORT.md`, 95 workloads, 5 backends; TensorFlow has no calibration, so only the LOO fit applies):

| Backend | Bound violations | Zero-shot MAPE (median) | Op-replay MAPE (median) | LOO fit MAPE (median) | Rank corr. (best predictor) |
|---|---:|---:|---:|---:|---:|
| PyTorch eager, CPU | 0 / 23 | 63% (68%) | **25% (18%)** | 51% (52%) | 0.987 |
| PyTorch eager, MPS | 0 / 23 | 29% (20%) | 29% (14%) | **21% (12%)** | 0.936 |
| JAX jit, CPU | 0 / 13 | 63% (76%) | n/a | **5% (2%)** | 0.995 |
| JAX jit, MPS | 0 / 13 | 26% (20%) | n/a | **21% (16%)** | 0.951 |
| TensorFlow graph, CPU | 0 / 23 | n/a | n/a | **22% (22%)** | 0.954 |

What this shows, and what it doesn't. The bound was never violated, and both
bound and predictions rank workloads correctly (Spearman ρ 0.89–0.995). The
best predictor still has a typical (median) error of 2–22%. Most of what remains is the eager-LSTM
decomposition and composition effects that only appear when ops run together.
That is exactly what the `residual` component isolates. The report contains
every per-workload number. Run it on your hardware with:

```bash
python benchmarks/validate.py
```

Building the validation harness surfaced and fixed real bugs: weights were
double-counted in linear/conv traffic, the FX adapter silently dropped
RNN/LSTM/attention layers, and Keras parameter counts included BatchNorm
moving statistics. Recurrent layers also used to be recorded as one big GEMM
instead of T sequential ones. These caused the up-to-6× cross-framework FLOP
discrepancies earlier versions of this README showed.
