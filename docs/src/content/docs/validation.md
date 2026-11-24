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

Results on an Apple M3 (`VALIDATION_REPORT.md`, 59 workloads, 3 backends):

| Backend | Bound violations | Zero-shot MAPE (median) | Op-replay MAPE (median) | LOO fit MAPE (median) | Rank corr. (best predictor) |
|---|---:|---:|---:|---:|---:|
| PyTorch eager, CPU | 0 / 23 | 61% (66%) | **26% (22%)** | 51% (52%) | 0.987 |
| PyTorch eager, MPS | 0 / 23 | 29% (19%) | 30% (15%) | **22% (13%)** | 0.948 |
| JAX jit, CPU | 0 / 13 | **39% (33%)** | n/a | 45% (41%) | 0.989 |

What this shows, and what it doesn't. The bound was never violated, and both
bound and predictions rank workloads correctly (Spearman ρ 0.91–0.99). The
best predictor still has a typical (median) error of 13–33%. Most of what remains is the eager-LSTM
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
