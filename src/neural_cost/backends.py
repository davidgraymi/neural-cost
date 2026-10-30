"""Run one JAX function on several platforms and compare them in one table.

JAX picks a single platform per process (``JAX_PLATFORMS``), so a CPU vs
MPS/Metal/CUDA comparison needs a separate process per platform.
:func:`compare_backends` starts one worker subprocess per platform.  Each
worker imports the target only after ``JAX_PLATFORMS`` is set, benchmarks it,
detects that platform's hardware spec and, when it can, computes the roofline
efficiency and coverage.  The results are collected into a
:class:`BackendComparison`.

The target is ``"script.py:name"`` or ``"package.module:name"`` naming a
zero-argument factory that returns ``(function, args)`` or
``(function, args, kwargs)``.  It is a factory so arrays are created on the
platform under test::

    # bench.py
    def build():
        import jax, jax.numpy as jnp
        net = jax.jit(lambda x, w: jnp.tanh(x @ w))
        return net, (jnp.ones((128, 512)), jnp.ones((512, 512)))

    compare_backends("bench.py:build", ["cpu", "mps"]).render()
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BackendResult:
    """Measurement and hardware spec for one platform (or the error that stopped it)."""

    platform: str
    device: str | None = None
    median_seconds: float | None = None
    samples_seconds: tuple[float, ...] = ()
    hardware_name: str | None = None
    hardware_device_type: str | None = None
    peak_flops: float | None = None
    memory_bandwidth: float | None = None
    efficiency: float | None = None
    bottleneck: str | None = None
    coverage: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.median_seconds is not None


@dataclass(frozen=True, slots=True)
class BackendComparison:
    """Per-platform results; speedups are relative to ``baseline``."""

    results: tuple[BackendResult, ...]
    baseline: str

    def speedup(self, platform: str) -> float | None:
        """How many times faster ``platform`` ran than the baseline (``None`` if either failed)."""
        by_name = {result.platform: result for result in self.results}
        base, other = by_name.get(self.baseline), by_name.get(platform)
        if base is None or other is None or not base.ok or not other.ok:
            return None
        return base.median_seconds / other.median_seconds

    def fastest(self) -> BackendResult | None:
        return min((r for r in self.results if r.ok), key=lambda r: r.median_seconds, default=None)

    def render(self) -> str:
        rows = [
            ("platform", "device", "median", "speedup", "hardware spec", "roofline", "coverage")
        ]
        failed = []
        for result in self.results:
            if not result.ok:
                failed.append(result)
                rows.append((result.platform, "-", "-", "-", "-", "failed (see below)", "-"))
                continue
            speedup = self.speedup(result.platform)
            spec = result.hardware_name or "-"
            if result.peak_flops and result.memory_bandwidth:
                spec += (
                    f" ({result.peak_flops / 1e12:.2f} TFLOP/s, "
                    f"{result.memory_bandwidth / 1e9:.0f} GB/s)"
                )
            roofline = (
                f"{result.efficiency:.1%} ({result.bottleneck}-bound)"
                if result.efficiency is not None
                else "-"
            )
            rows.append(
                (
                    result.platform,
                    result.device or "-",
                    f"{result.median_seconds * 1e3:.3f} ms",
                    f"{speedup:.2f}x" if speedup is not None else "-",
                    spec,
                    roofline,
                    f"{result.coverage:.0%}" if result.coverage is not None else "-",
                )
            )
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines = [f"Backend comparison (speedup vs {self.baseline})"]
        for row in rows:
            lines.append("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
        lines.extend(f"  {result.platform}: {result.error}" for result in failed)
        return "\n".join(line.rstrip() for line in lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "results": [
                {
                    **{name: getattr(result, name) for name in BackendResult.__slots__},
                    "speedup": self.speedup(result.platform),
                }
                for result in self.results
            ],
        }


def compare_backends(
    target: str,
    platforms: Sequence[str],
    *,
    baseline: str | None = None,
    warmup: int = 3,
    repeats: int = 10,
    roofline: bool = True,
    timeout: float | None = 600.0,
    python: str = sys.executable,
) -> BackendComparison:
    """Benchmark ``target`` once per JAX platform, each in a fresh subprocess.

    ``platforms`` are ``JAX_PLATFORMS`` values (``"cpu"``, ``"cuda"``,
    ``"mps"``, ``"METAL"``, ``"tpu"``).  ``baseline`` defaults to the first
    platform.  A platform that fails (plugin missing, out of memory, timeout)
    becomes a result with ``error`` set; the other platforms still run.
    """
    platforms = [platform.strip() for platform in platforms if platform.strip()]
    if not platforms:
        raise ValueError("at least one platform is required")
    results = [
        _run_worker(target, platform, warmup, repeats, roofline, timeout, python)
        for platform in platforms
    ]
    return BackendComparison(tuple(results), baseline or platforms[0])


def _run_worker(
    target: str,
    platform: str,
    warmup: int,
    repeats: int,
    roofline: bool,
    timeout: float | None,
    python: str,
) -> BackendResult:
    env = {**os.environ, "JAX_PLATFORMS": platform}
    command = [
        python,
        "-m",
        "neural_cost.backends",
        target,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    if not roofline:
        command.append("--no-roofline")
    try:
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return BackendResult(platform, error=f"timed out after {timeout:g} s")
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        return BackendResult(
            platform,
            error=f"worker failed ({completed.returncode}): {_error_line(completed.stderr)}",
        )
    payload["platform"] = platform
    payload["samples_seconds"] = tuple(payload.get("samples_seconds", ()))
    known = set(BackendResult.__slots__)
    return BackendResult(**{key: value for key, value in payload.items() if key in known})


def _error_line(stderr: str, limit: int = 160) -> str:
    """The exception line of a traceback (``SomeError: message``), else the last line."""
    lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
    if not lines:
        return "no output"
    errors = [line for line in lines if re.match(r"^[\w.]+(Error|Exception|Exit)\b", line)]
    line = errors[-1] if errors else lines[-1]
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _worker(target: str, warmup: int, repeats: int, roofline: bool) -> dict[str, Any]:
    """Run inside the subprocess, after ``JAX_PLATFORMS`` was set by the parent."""
    from .adapters import JaxAdapter
    from .cli.helpers import load_target

    factory = load_target(target)
    built = factory()
    if not isinstance(built, tuple) or len(built) not in (2, 3):
        raise TypeError(f"{target} must return (function, args) or (function, args, kwargs)")
    function, args = built[0], tuple(built[1])
    kwargs = dict(built[2]) if len(built) == 3 else {}

    import jax

    adapter = JaxAdapter()
    measurement = adapter.benchmark(function, *args, warmup=warmup, repeats=repeats, **kwargs)
    device = jax.devices()[0]
    payload: dict[str, Any] = {
        "device": measurement.device or str(device),
        "median_seconds": measurement.median_seconds,
        "samples_seconds": list(measurement.samples_seconds),
    }
    try:
        spec, _ = adapter.hardware(device, benchmark_memory=False, quick=True)
    except Exception as error:  # noqa: BLE001 - report, keep the measurement
        payload["error"] = f"hardware detection failed: {error}"
        return payload
    payload.update(
        hardware_name=spec.name,
        hardware_device_type=spec.device_type,
        peak_flops=spec.peak_flops,
        memory_bandwidth=spec.memory_bandwidth,
    )
    if roofline and not kwargs:
        from .analysis import analyze_gap
        from .estimate import estimate_operations

        try:
            operations = list(adapter.operations(function, args))
        except Exception:  # noqa: BLE001 - tracing is best effort; timing still counts
            operations = []
        if operations:
            gap = analyze_gap(
                estimate_operations(operations),
                measurement,
                spec,
                operations=operations,
                coverage=adapter.coverage,
            )
            payload.update(efficiency=gap.efficiency, bottleneck=gap.bottleneck)
            if adapter.coverage is not None:
                payload["coverage"] = adapter.coverage.fraction
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m neural_cost.backends")
    parser.add_argument("target")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--no-roofline", action="store_true")
    args = parser.parse_args(argv)
    payload = _worker(args.target, args.warmup, args.repeats, not args.no_roofline)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
