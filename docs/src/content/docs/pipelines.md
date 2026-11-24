---
title: "Pipeline profiling"
description: "Benchmark multi-stage workloads and find unexplained time between stages."
section: "Guides"
order: 6
---

Real workloads are pipelines. One self-play search is tree bookkeeping plus
N environment steps plus N network forwards. `Pipeline` benchmarks each stage,
scales it by its call count, and compares the sum with an end-to-end
measurement. The difference is reported as unexplained time (dispatch, loop
overhead, transfers):

```python
from neural_cost import Pipeline, Stage
from neural_cost.adapters import JaxAdapter

pipeline = Pipeline(
    {
        "tree": Stage(tree_update, (tree, stats)),
        "env": Stage(env_step, (state, action), calls=16),
        "net": (net_apply, (params, obs), 16),        # tuple shorthand
    },
    total=Stage(search, (params, state)),
)
adapter = JaxAdapter()
report = pipeline.run(adapter=adapter, hardware=adapter.hardware()[0])
print(report.render())
# Pipeline profile (measured total 614.000 ms on mps:0)
#   stage         calls     per call        total    share  roofline
#   tree              1   300.000 ms   300.000 ms    48.9%  12.0% (memory-bound), 38% coverage
#   env              16    11.625 ms   186.000 ms    30.3%  ...
#   net              16     5.688 ms    91.000 ms    14.8%  68.9% (memory-bound), 100% coverage
#   unexplained                         37.000 ms     6.0%  dispatch / loop overhead
```

With `hardware=`, each stage also gets its own roofline `GapAnalysis`
(`report.stages[i].gap`), traced by the adapter or taken from
`Stage(operations=...)`. From the shell, point the CLI at a `Pipeline` or a
function that returns one:

```bash
neural-cost pipeline selfplay.py:build_pipeline --framework jax --roofline
neural-cost pipeline selfplay.py:build_pipeline --framework jax --json
```
