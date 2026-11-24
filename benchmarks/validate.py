#!/usr/bin/env python3
"""Validate neural-cost predictions against measured runtimes on this machine.

For each available backend (PyTorch CPU / CUDA / MPS eager, JAX jit on CPU / MPS,
TensorFlow graph on CPU) this:

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
    python benchmarks/validate.py --quick --fail-on-violation   # CI gate

Writes ``VALIDATION_REPORT.md`` and ``validation_data.json`` to
``benchmarks/results/<device>/``, stamped with the neural-cost version and git
commit.  ``--fail-on-violation`` exits 1 if any measurement beat its lower bound.
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
sys.path.insert(0, str(ROOT))

from neural_cost import (
    CacheSpec,
    Workload,
    attribute_gap,
    detect_hardware,
    run_validation_suite,
)
from benchmarks.run_info import describe_run, results_dir, run_metadata
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


def jax_workloads(quick: bool, device=None) -> list[Workload]:
    import jax

    with jax.default_device(device):
        return _jax_workloads(quick)


def _jax_workloads(quick: bool) -> list[Workload]:
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


# -------------------------------------------------------------- tensorflow models


def tf_workloads(quick: bool) -> list[Workload]:
    """Keras models run as ``tf.function`` graphs on the CPU (no Metal plugin)."""
    import tensorflow as tf

    keras = tf.keras
    tf.random.set_seed(0)
    out: list[Workload] = []

    def add(name: str, model, x, **meta) -> None:
        graph = tf.function(lambda inp, _m=model: _m(inp, training=False))
        out.append(Workload(name, model, (x,), runner=graph, metadata=meta))

    widths = (256, 1024) if quick else (256, 1024, 4096)
    batches = (1, 64) if quick else (1, 16, 128)
    for w in widths:
        for b in batches:
            mlp = keras.Sequential(
                [
                    keras.Input((w,)),
                    keras.layers.Dense(w, activation="gelu"),
                    keras.layers.Dense(w, activation="gelu"),
                    keras.layers.Dense(10),
                ]
            )
            add(f"mlp w={w} b={b}", mlp, tf.random.normal((b, w)), family="mlp")

    for img in (32,) if quick else (32, 64):
        for b in (1, 16) if quick else (1, 8, 32):
            cnn = keras.Sequential(
                [
                    keras.Input((img, img, 3)),
                    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                    keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
                    keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
                    keras.layers.GlobalAveragePooling2D(),
                    keras.layers.Dense(10),
                ]
            )
            add(f"cnn {img}px b={b}", cnn, tf.random.normal((b, img, img, 3)), family="cnn")

    for d in (128,) if quick else (128, 256):
        for b in (1, 16):
            lstm = keras.Sequential(
                [
                    keras.Input((32, d)),
                    keras.layers.LSTM(d, return_sequences=True),
                    keras.layers.LSTM(d),
                    keras.layers.Dense(10),
                ]
            )
            add(f"lstm d={d} b={b}", lstm, tf.random.normal((b, 32, d)), family="lstm")

    def block(t: int, d: int = 256, h: int = 8):
        x = inp = keras.Input((t, d))
        y = keras.layers.LayerNormalization()(x)
        x = x + keras.layers.MultiHeadAttention(h, d // h)(y, y)
        y = keras.layers.LayerNormalization()(x)
        y = keras.layers.Dense(4 * d, activation="gelu")(y)
        return keras.Model(inp, x + keras.layers.Dense(d)(y))

    for t in (64,) if quick else (64, 256):
        for b in (1, 8):
            add(
                f"transformer d=256 T={t} b={b}",
                block(t),
                tf.random.normal((b, t, 256)),
                family="transformer",
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
        import jax

        found.append("jax-cpu")
        if _jax_device("mps") is not None:
            found.append("jax-mps")
    if importlib.util.find_spec("tensorflow"):
        found.append("tf-cpu")
    return found


def _jax_device(platform: str):
    """First JAX device of ``platform`` (``"cpu"``, ``"mps"``, ``"gpu"``), or None."""
    import jax

    try:
        return jax.devices(platform)[0]
    except RuntimeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backends", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory (default: benchmarks/results/<device>/)",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 if any measured runtime beat its lower bound",
    )
    args = parser.parse_args()

    hardware, info = detect_hardware()
    backends = args.backends or available_backends()
    print(f"Hardware: {hardware.name} ({info.source}); backends: {', '.join(backends)}")

    run = run_metadata(hardware.name, script="validate.py", backends=backends, quick=args.quick)
    sections = []
    payload = {"run": run, "hardware": hardware.name, "detection": info.source, "backends": {}}
    violations = 0
    for backend in backends:
        if backend.startswith("torch"):
            from neural_cost.adapters import TorchAdapter

            adapter = TorchAdapter(use_profiler=False)  # profiler overhead would skew timings
            device = backend.split("-", 1)[1]
            calibration = adapter.calibrate(device)
            workloads = torch_workloads(device, args.quick)
            title = f"PyTorch eager ({device})"
        elif backend == "tf-cpu":
            from neural_cost.adapters import TensorFlowAdapter

            # TensorFlow has no calibration microbenchmarks: its section reports
            # lower-bound validity and the fitted model, without zero-shot or
            # gap attribution.
            adapter = TensorFlowAdapter(trace_device_time=False)
            calibration = None
            workloads = tf_workloads(args.quick)
            title = "TensorFlow graph (cpu)"
        else:
            from neural_cost.adapters import JaxAdapter

            jax_platform = backend.split("-", 1)[1]
            device = _jax_device(jax_platform)
            if device is None:
                raise SystemExit(f"{backend}: JAX has no {jax_platform} device")
            adapter = JaxAdapter(trace_device_time=False)  # as for torch: no profiler
            calibration = adapter.calibrate(device=device)
            workloads = jax_workloads(args.quick, device)
            title = f"JAX jit ({jax_platform})"
        print(f"\n[{backend}] calibration: {calibration}")
        # Each backend is bounded by its own device: on a CUDA host the default
        # spec is the GPU's, which would make every CPU bound far too loose.
        spec, _ = detect_hardware(device="cpu" if backend.endswith("cpu") else "gpu")
        if backend.endswith("cpu"):
            spec = replace(spec, caches=spec.caches + cpu_caches())
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
        violations += report.bound_violations

        # Attribute the gap for the smallest, the most expensive, and the last workload.
        picks = [workloads[0], max(workloads, key=lambda w: _flops(adapter, w)), workloads[-1]]
        if calibration is None:
            picks = []  # attribution needs measured achievable peaks
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

        md = [report.to_markdown(), ""]
        if calibration is None:
            md += ["No calibration microbenchmarks for this backend; no gap attribution.", ""]
        else:
            md += [
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
            "calibration": None
            if calibration is None
            else {
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

Run: {describe_run(run)}.

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
    out_dir = results_dir(hardware.name, args.results_dir)
    report_path = out_dir / "VALIDATION_REPORT.md"
    data_path = out_dir / "validation_data.json"
    report_path.write_text(header + "\n\n".join(sections) + "\n")
    data_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {report_path} and {data_path}")
    if args.fail_on_violation and violations:
        print(f"FAIL: {violations} measurement(s) beat their lower bound", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
