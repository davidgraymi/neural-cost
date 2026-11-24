# Benchmark results

One directory per machine (`apple-m3/`, `tesla-t4/`, ...). Each JSON file
starts with a `run` block naming the neural-cost version, git commit and
framework versions that produced it; each report shows it on its first line.

| Directory | Contents | Produced |
|---|---|---|
| `apple-m3/` | Validation (torch CPU/MPS, JAX CPU/MPS, TF CPU), CPU sweep (torch, JAX, TF), MPS sweep (torch, JAX; batches that do not fit in 16 GB skipped) and the LLM benchmark, with reports and figures | 2026-11-23: validation, CPU and LLM at `cb36a1c`; MPS sweep with the memory fix in `5b8cbb1` |

Results without a `run` block predate the current measurement code; rerun them
before comparing against newer results. See `docs/src/content/docs/benchmarks.md`
for how to produce new ones, including free CUDA runs on Colab or Kaggle.
