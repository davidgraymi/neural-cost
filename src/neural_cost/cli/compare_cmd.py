"""Framework comparison subcommand and runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import mean, stdev
from typing import Any

from neural_cost.adapters import available_adapters, get_adapter
from neural_cost.adapters.base import FrameworkAdapter
from neural_cost.analysis import analyze_gap
from neural_cost.api import estimate_model
from neural_cost.hardware import HardwareSpec
from neural_cost.hardware_detect import detect_hardware


@dataclass(frozen=True)
class Result:
    framework: str
    shape_label: str  # e.g. "64×1024×1024"
    flops: int
    bytes_moved: int
    arith_intensity: float  # FLOP/byte
    median_ms: float
    stddev_ms: float
    efficiency: float
    achieved_gflops: float
    achieved_gbw: float
    bottleneck: str


def evaluate(
    name: str,
    shape_label: str,
    model: Callable[..., Any],
    inputs: tuple[Any, ...],
    adapter: FrameworkAdapter,
    hardware: HardwareSpec,
    warmup: int = 10,
    repeats: int = 30,
) -> Result:
    estimate = estimate_model(model, inputs, adapter)
    measurement = adapter.benchmark(model, *inputs, warmup=warmup, repeats=repeats)
    gap = analyze_gap(estimate, measurement, hardware)

    observed = measurement.median_seconds
    samples_ms = [s * 1e3 for s in measurement.samples_seconds]
    sd = stdev(samples_ms) if len(samples_ms) > 1 else 0.0

    return Result(
        framework=name,
        shape_label=shape_label,
        flops=estimate.flops,
        bytes_moved=estimate.total_bytes,
        arith_intensity=estimate.arithmetic_intensity,
        median_ms=observed * 1e3,
        stddev_ms=sd,
        efficiency=gap.efficiency,
        achieved_gflops=gap.achieved_flops / 1e9,
        achieved_gbw=gap.achieved_bandwidth / 1e9,
        bottleneck=gap.bottleneck,
    )


# Workload shapes: (batch_size, in_features, out_features)
SHAPES: list[tuple[int, int, int]] = [
    (1, 1024, 1024),
    (16, 1024, 1024),
    (64, 1024, 1024),
    (256, 1024, 1024),
]


def _label(batch: int, k: int, n: int) -> str:
    return f"{batch}×{k}→{n}"


def run_torch(hardware: HardwareSpec, warmup: int, repeats: int) -> list[Result]:
    import torch

    results = []
    adapter = get_adapter("torch")
    for batch, k, n in SHAPES:
        model = torch.nn.Linear(k, n, bias=False).eval()
        inputs = (torch.ones((batch, k)),)
        results.append(
            evaluate(
                "PyTorch", _label(batch, k, n), model, inputs, adapter, hardware, warmup, repeats
            )
        )
    return results


def run_jax(hardware: HardwareSpec, warmup: int, repeats: int) -> list[Result]:
    import jax.numpy as jnp

    results = []
    adapter = get_adapter("jax")
    for batch, k, n in SHAPES:
        weight = jnp.ones((k, n))

        def model(x: Any, w: Any = weight) -> Any:
            return jnp.matmul(x, w)

        inputs = (jnp.ones((batch, k)), weight)
        results.append(
            evaluate("JAX", _label(batch, k, n), model, inputs, adapter, hardware, warmup, repeats)
        )
    return results


def run_tensorflow(hardware: HardwareSpec, warmup: int, repeats: int) -> list[Result]:
    import tensorflow as tf

    results = []
    adapter = get_adapter("tensorflow")
    for batch, k, n in SHAPES:
        model = tf.keras.Sequential([tf.keras.layers.Dense(n, use_bias=False)])
        inputs = (tf.ones((batch, k)),)
        results.append(
            evaluate(
                "TensorFlow", _label(batch, k, n), model, inputs, adapter, hardware, warmup, repeats
            )
        )
    return results


_SEP = "─" * 120


def _bar(fraction: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def print_hardware_header(
    hardware: HardwareSpec, detection_source: str, measured_bw: float | None
) -> None:
    print()
    print("┌─ Hardware ─────────────────────────────────────────────────────────────────")
    print(f"│  Chip / device   : {hardware.name}")
    print(f"│  Peak FP32       : {hardware.peak_flops / 1e12:.2f} TFLOP/s")
    bw_str = f"{hardware.memory_bandwidth / 1e9:.1f} GB/s"
    if measured_bw is not None:
        bw_str += f"  (measured NumPy STREAM: {measured_bw:.1f} GB/s)"
    print(f"│  Peak bandwidth  : {bw_str}")
    print(f"│  Ridge point     : {hardware.ridge_point:.1f} FLOP/byte")
    print(f"│  Source          : {detection_source}")
    print("└────────────────────────────────────────────────────────────────────────────")
    print()


def print_results(results: list[Result], hardware: HardwareSpec) -> None:
    col = {
        "fw": 12,
        "shape": 14,
        "flops": 14,
        "bw": 10,
        "ai": 8,
        "med": 10,
        "sd": 8,
        "eff": 8,
        "bar": 22,
        "gflops": 10,
        "gbw": 10,
        "bot": 8,
    }

    header = (
        f"{'framework':<{col['fw']}} "
        f"{'shape':<{col['shape']}} "
        f"{'FLOPs':>{col['flops']}} "
        f"{'bytes(MB)':>{col['bw']}} "
        f"{'AI':>{col['ai']}} "
        f"{'ms(med)':>{col['med']}} "
        f"{'±ms':>{col['sd']}} "
        f"{'effic.':>{col['eff']}} "
        f"{'roofline':^{col['bar']}} "
        f"{'GFLOP/s':>{col['gflops']}} "
        f"{'GB/s':>{col['gbw']}} "
        f"{'bound':<{col['bot']}}"
    )
    print(_SEP)
    print(header)
    print(_SEP)

    last_fw = None
    for r in results:
        if last_fw and r.framework != last_fw:
            print()
        last_fw = r.framework

        eff_clamped = min(r.efficiency, 1.0)
        bar = _bar(eff_clamped)
        eff_str = f"{r.efficiency:.1%}" if r.efficiency < 1.0 else f">{100:.0f}%*"

        print(
            f"{r.framework:<{col['fw']}} "
            f"{r.shape_label:<{col['shape']}} "
            f"{r.flops:>{col['flops']},d} "
            f"{r.bytes_moved / 1e6:>{col['bw']}.1f} "
            f"{r.arith_intensity:>{col['ai']}.1f} "
            f"{r.median_ms:>{col['med']}.3f} "
            f"{r.stddev_ms:>{col['sd']}.3f} "
            f"{eff_str:>{col['eff']}} "
            f"[{bar}] "
            f"{r.achieved_gflops:>{col['gflops']}.1f} "
            f"{r.achieved_gbw:>{col['gbw']}.1f} "
            f"{r.bottleneck:<{col['bot']}}"
        )

    print(_SEP)
    print(
        "  AI = arithmetic intensity (FLOP/byte).  "
        f"Ridge point = {hardware.ridge_point:.1f} FLOP/byte  "
        "(above → compute-bound, below → memory-bound)"
    )
    print(
        "  *Efficiency >100% means the hardware spec is slower than your actual "
        "chip; use --peak-flops / --memory-bandwidth to calibrate."
    )
    print()


def print_summary(results: list[Result]) -> None:
    """Print per-framework aggregate summary."""
    frameworks: dict[str, list[Result]] = {}
    for r in results:
        frameworks.setdefault(r.framework, []).append(r)

    print("┌─ Per-framework summary ────────────────────────────────────────────────────")
    for fw, rs in frameworks.items():
        effs = [r.efficiency for r in rs]
        mean_eff = mean(effs)
        best = max(rs, key=lambda r: r.efficiency)
        worst = min(rs, key=lambda r: r.efficiency)
        print(
            f"│  {fw:<12}  mean efficiency {mean_eff:.1%}  "
            f"best {best.efficiency:.1%} @ {best.shape_label}  "
            f"worst {worst.efficiency:.1%} @ {worst.shape_label}"
        )
    print("└────────────────────────────────────────────────────────────────────────────")
    print()


def register_compare_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "compare",
        help="Compare PyTorch, JAX, and TensorFlow performance against roofline bounds.",
        description="Execute head-to-head empirical benchmarks across available deep learning frameworks.",
    )
    _add_compare_arguments(parser)
    parser.set_defaults(func=run_compare)


def _add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--peak-flops", type=float, default=None, help="Override peak FP32 FLOP/s (e.g. 3.6e12)"
    )
    parser.add_argument(
        "--memory-bandwidth",
        type=float,
        default=None,
        help="Override memory bandwidth in bytes/s (e.g. 100e9)",
    )
    parser.add_argument(
        "--bw-bench-mb",
        type=int,
        default=256,
        help="Working-set size in MiB for the bandwidth benchmark (default 256)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Framework warm-up iterations per shape (default 10)"
    )
    parser.add_argument(
        "--repeats", type=int, default=30, help="Timed iterations per shape (default 30)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output benchmark comparison results in JSON format"
    )


def run_compare(args: argparse.Namespace) -> int:
    if not args.json:
        print("Detecting hardware and measuring memory bandwidth…", flush=True)
    hardware, detection = detect_hardware(bandwidth_benchmark_mb=args.bw_bench_mb)

    if args.peak_flops is not None:
        hardware = HardwareSpec(hardware.name, args.peak_flops, hardware.memory_bandwidth)
    if args.memory_bandwidth is not None:
        hardware = HardwareSpec(hardware.name, hardware.peak_flops, args.memory_bandwidth)

    if not args.json:
        print_hardware_header(hardware, detection.source, detection.measured_bandwidth_gb_s)

    runners = {
        "torch": run_torch,
        "jax": run_jax,
        "tensorflow": run_tensorflow,
    }
    all_results: list[Result] = []

    adapters = available_adapters()
    for package in ["torch", "jax", "tensorflow"]:
        if package in adapters and package in runners:
            if not args.json:
                print(f"Benchmarking {package}…", flush=True)
            all_results.extend(runners[package](hardware, args.warmup, args.repeats))

    if not all_results:
        sys.stderr.write("Install at least one framework extra: torch, jax, or tensorflow.\n")
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "hardware": {
                        "name": hardware.name,
                        "peak_flops": hardware.peak_flops,
                        "memory_bandwidth": hardware.memory_bandwidth,
                        "ridge_point": hardware.ridge_point,
                    },
                    "results": [asdict(r) for r in all_results],
                },
                indent=2,
            )
        )
        return 0

    print()
    print_results(all_results, hardware)
    print_summary(all_results)
    return 0


def compare_main(argv: list[str] | None = None) -> None:
    """Standalone entry point for neural-cost-compare."""
    parser = argparse.ArgumentParser(
        prog="neural-cost-compare",
        description="Compare framework performance against hardware roofline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_compare_arguments(parser)
    args = parser.parse_args(argv)
    code = run_compare(args)
    if code != 0:
        sys.exit(code)
