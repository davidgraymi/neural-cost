#!/usr/bin/env python3
"""Validate neural-cost predictions against measured runtimes on this machine.

For each available backend (PyTorch CPU / CUDA / MPS eager, JAX jit) this:

1. calibrates achievable FLOP/s, bandwidth and dispatch overhead with
   microbenchmarks that never touch the evaluated models;
2. traces and benchmarks a grid of MLP, CNN, LSTM and Transformer workloads;
3. reports lower-bound validity, a *zero-shot* prediction (calibration only,
   no fitting on model runtimes), an *op-replay* prediction (each captured op
   timed in isolation, then composed) and a leave-one-out calibrated cost model;
4. attributes the measured gap for representative workloads.

Usage::

    python benchmarks/validate.py                # all available backends
    python benchmarks/validate.py --quick        # smaller grid
    python benchmarks/validate.py --backends torch-cpu jax-cpu

Writes ``VALIDATION_REPORT.md`` and ``benchmarks/results/validation_data.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_cost import (
    CacheSpec,
    Workload,
    attribute_gap,
    detect_hardware,
    run_validation_suite,
)
from neural_cost.profiler import Measurement

# ------------------------------------------------------------------ torch models


def torch_workloads(device: str, quick: bool) -> list[Workload]:
    import torch

    torch.manual_seed(0)
    dev = torch.device(device)
    out: list[Workload] = []

    def add(name: str, model: torch.nn.Module, x: torch.Tensor, **meta) -> None:
        model = model.to(dev).eval()
        x = x.to(dev)

        def run(inp, _m=model):
            with torch.no_grad():
                return _m(inp)

        out.append(Workload(name, model, (x,), runner=run, metadata={**meta, "device": device}))

    widths = (256, 1024) if quick else (256, 1024, 4096)
    batches = (1, 64) if quick else (1, 16, 128)
    for w in widths:
        for b in batches:
            mlp = torch.nn.Sequential(
                torch.nn.Linear(w, w),
                torch.nn.GELU(),
                torch.nn.Linear(w, w),
                torch.nn.GELU(),
                torch.nn.Linear(w, 10),
            )
            add(f"mlp w={w} b={b}", mlp, torch.randn(b, w), family="mlp")

    for img in (32,) if quick else (32, 64):
        for b in (1, 16) if quick else (1, 8, 32):
            cnn = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 10),
            )
            add(f"cnn {img}px b={b}", cnn, torch.randn(b, 3, img, img), family="cnn")

    class LSTMHead(torch.nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.lstm = torch.nn.LSTM(d, d, num_layers=2, batch_first=True)
            self.head = torch.nn.Linear(d, 10)

        def forward(self, x):
            return self.head(self.lstm(x)[0][:, -1])

    for d in (128,) if quick else (128, 256):
        for b in (1, 16):
            add(f"lstm d={d} b={b}", LSTMHead(d), torch.randn(b, 32, d), family="lstm")

    class Block(torch.nn.Module):
        def __init__(self, d: int, h: int):
            super().__init__()
            self.ln1, self.ln2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
            self.attn = torch.nn.MultiheadAttention(d, h, batch_first=True)
            self.fc1, self.fc2 = torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d)

        def forward(self, x):
            y = self.ln1(x)
            x = x + self.attn(y, y, y, need_weights=False)[0]
            return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))

    for t in (64,) if quick else (64, 256):
        for b in (1, 8):
            add(
                f"transformer d=256 T={t} b={b}",
                Block(256, 8),
                torch.randn(b, t, 256),
                family="transformer",
            )
    return out


# -------------------------------------------------------------------- jax models


def jax_workloads(quick: bool) -> list[Workload]:
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    out: list[Workload] = []

    def mlp(x, w1, b1, w2, b2, w3, b3):
        x = jax.nn.gelu(x @ w1 + b1)
        x = jax.nn.gelu(x @ w2 + b2)
        return x @ w3 + b3

    widths = (256, 1024) if quick else (256, 1024, 4096)
    batches = (1, 64) if quick else (1, 16, 128)
    for w in widths:
        params = (
            jax.random.normal(key, (w, w)),
            jnp.zeros(w),
            jax.random.normal(key, (w, w)),
            jnp.zeros(w),
            jax.random.normal(key, (w, 10)),
            jnp.zeros(10),
        )
        for b in batches:
            x = jax.random.normal(key, (b, w))
            out.append(
                Workload(
                    f"mlp w={w} b={b}",
                    mlp,
                    (x, *params),
                    runner=jax.jit(mlp),
                    kernels=1,
                    fused=True,
                    metadata={"family": "mlp"},
                )
            )

    def block(x, wq, wk, wv, wo, w1, w2):
        b, t, d = x.shape
        h = 8

        def heads(z):
            return z.reshape(b, t, h, d // h)

        a = jax.nn.dot_product_attention(heads(x @ wq), heads(x @ wk), heads(x @ wv))
        x = x + a.reshape(b, t, d) @ wo
        return x + jax.nn.gelu(x @ w1) @ w2

    d = 256
    params = tuple(
        jax.random.normal(key, s) * 0.02 for s in [(d, d)] * 4 + [(d, 4 * d), (4 * d, d)]
    )
    for t in (64,) if quick else (64, 256):
        for b in (1, 8):
            x = jax.random.normal(key, (b, t, d))
            out.append(
                Workload(
                    f"transformer d=256 T={t} b={b}",
                    block,
                    (x, *params),
                    runner=jax.jit(block),
                    kernels=1,
                    fused=True,
                    metadata={"family": "transformer"},
                )
            )
    return out


# ------------------------------------------------------------------------ driver


def _flops(adapter, workload: Workload) -> int:
    from neural_cost import estimate_operations

    return estimate_operations(adapter.operations(workload.model, workload.inputs)).flops


def cpu_caches() -> tuple[CacheSpec, ...]:
    """Largest private/cluster CPU cache from the OS, with unknown (unbounded) bandwidth.

    Datasheets rarely publish peak cache bandwidth, so a cache-resident working set is
    bounded by compute alone: weaker, but never invalid.
    """
    size = 0
    if sys.platform == "darwin":
        for key in ("hw.perflevel0.l2cachesize", "hw.l2cachesize"):
            try:
                size = int(subprocess.check_output(["sysctl", "-n", key], text=True).strip())
                break
            except (OSError, ValueError, subprocess.CalledProcessError):
                continue
    else:
        for index in Path("/sys/devices/system/cpu/cpu0/cache").glob("index*"):
            try:
                text = (index / "size").read_text().strip().upper()
                value = int(text.rstrip("KM")) * (1024 if text.endswith("K") else 1024**2)
                size = max(size, value)
            except (OSError, ValueError):
                continue
    return (CacheSpec("CPU cache", float("inf"), size),) if size > 0 else ()


def available_backends() -> list[str]:
    found = []
    if importlib.util.find_spec("torch"):
        import torch

        found.append("torch-cpu")
        if torch.cuda.is_available():
            found.append("torch-cuda")
        if torch.backends.mps.is_available():
            found.append("torch-mps")
    if importlib.util.find_spec("jax"):
        found.append("jax-cpu")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backends", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "VALIDATION_REPORT.md")
    args = parser.parse_args()

    hardware, info = detect_hardware()
    backends = args.backends or available_backends()
    print(f"Hardware: {hardware.name} ({info.source}); backends: {', '.join(backends)}")

    sections, payload = [], {"hardware": hardware.name, "detection": info.source, "backends": {}}
    for backend in backends:
        if backend.startswith("torch"):
            from neural_cost.adapters import TorchAdapter

            adapter = TorchAdapter(use_profiler=False)  # profiler overhead would skew timings
            device = backend.split("-", 1)[1]
            calibration = adapter.calibrate(device)
            workloads = torch_workloads(device, args.quick)
            title = f"PyTorch eager ({device})"
        else:
            from neural_cost.adapters import JaxAdapter

            adapter = JaxAdapter()
            calibration = adapter.calibrate()
            workloads = jax_workloads(args.quick)
            title = "JAX jit (cpu)"
        print(f"\n[{backend}] calibration: {calibration}")
        spec = hardware
        if backend.endswith("cpu"):
            spec = replace(hardware, caches=hardware.caches + cpu_caches())
        report = run_validation_suite(
            adapter,
            workloads,
            spec,
            title=title,
            repeats=args.repeats,
            calibration=calibration,
            warmup_seconds=0.25,
        )
        print(report.render())

        # Attribute the gap for the smallest, the most expensive, and the last workload.
        picks = [workloads[0], max(workloads, key=lambda w: _flops(adapter, w)), workloads[-1]]
        attributions = []
        seen: set[str] = set()
        for workload in picks:
            if workload.name in seen:
                continue
            seen.add(workload.name)
            case = next(c for c in report.cases if c.name == workload.name)
            ops = adapter.operations(workload.model, workload.inputs)
            kernels, extra = workload.kernels, None
            if kernels is None and hasattr(adapter, "trace_kernels"):
                records = adapter.trace_kernels(workload.runner or workload.model, workload.inputs)
                kernels = len(records)
                extra = [(r.bytes, r.python_dispatch) for r in records if not r.heavy]
            isolated = (
                adapter.replay(ops, workload.metadata.get("device"))
                if hasattr(adapter, "replay")
                else None
            )
            attribution = attribute_gap(
                ops,
                Measurement(
                    case.measured_seconds, (case.metadata["min_seconds"], case.measured_seconds)
                ),
                spec,
                calibration,
                kernel_count=kernels,
                assume_fusion=workload.fused,
                isolated_seconds=isolated,
                extra_kernels=extra,
            )
            attributions.append((workload.name, attribution))
            print(attribution.render())

        md = [
            report.to_markdown(),
            "",
            f"Calibration: {calibration.source} — "
            f"{calibration.achievable_flops / 1e9:.0f} GFLOP/s achievable, "
            f"{calibration.achievable_bandwidth / 1e9:.1f} GB/s, "
            f"{calibration.dispatch_seconds * 1e6:.2f} µs/kernel dispatch, "
            f"{calibration.call_overhead_seconds * 1e6:.1f} µs/call sync latency.",
            "",
        ]
        for name, attribution in attributions:
            md += [
                f"<details><summary>Gap attribution: {name}</summary>",
                "",
                "```",
                attribution.render(),
                "```",
                "",
                "</details>",
                "",
            ]
        sections.append("\n".join(md))
        payload["backends"][backend] = {
            "calibration": {
                "achievable_flops": calibration.achievable_flops,
                "achievable_bandwidth": calibration.achievable_bandwidth,
                "dispatch_seconds": calibration.dispatch_seconds,
                "call_overhead_seconds": calibration.call_overhead_seconds,
                "flops_curve": calibration.flops_curve,
                "bandwidth_curve": calibration.bandwidth_curve,
                "source": calibration.source,
            },
            "report": json.loads(report.to_json()),
        }

    header = f"""# neural-cost validation report

Generated {date.today().isoformat()} by `benchmarks/validate.py` on **{hardware.name}**
({platform.platform()}), hardware spec from {info.source}.

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

"""
    args.output.write_text(header + "\n\n".join(sections) + "\n")
    data_path = ROOT / "benchmarks" / "results" / "validation_data.json"
    data_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {args.output} and {data_path}")


if __name__ == "__main__":
    main()
