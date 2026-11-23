# Benchmark results

One directory per machine (`apple-m3/`, `tesla-t4/`, ...). Each JSON file
starts with a `run` block naming the neural-cost version, git commit and
framework versions that produced it; each report shows it on its first line.

| Directory | Contents | Produced |
|---|---|---|
| `apple-m1/` | CPU, MPS and LLM sweeps with their reports and figures | 2026-10-04, before run metadata and before the scripts timed through `adapter.benchmark()` |
| `apple-m3/` | Validation report | 2026-10-18, before run metadata |

Results without a `run` block predate the current measurement code; rerun them
before comparing against newer results. See `docs/src/content/docs/benchmarks.md`
for how to produce new ones, including free CUDA runs on Colab or Kaggle.
