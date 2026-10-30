"""Pipeline profiling subcommand: time stages with call counts against an end-to-end run."""

from __future__ import annotations

import argparse
import json

from neural_cost.adapters import get_adapter
from neural_cost.cli.helpers import load_target
from neural_cost.pipeline import Pipeline


def register_pipeline_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "pipeline",
        help="Profile a multi-stage pipeline and attribute wall time to each stage.",
        description=(
            "Load TARGET ('script.py:name' or 'module:name'), a neural_cost.Pipeline or a "
            "zero-argument function returning one, benchmark each stage, scale by its call "
            "count, and compare the sum with the end-to-end measurement."
        ),
    )
    parser.add_argument("target", help="script.py:name or module:name of a Pipeline")
    parser.add_argument(
        "--framework",
        default=None,
        help="Adapter used for benchmarking and tracing (torch, jax, tensorflow).",
    )
    parser.add_argument(
        "--roofline",
        action="store_true",
        help="Detect hardware and add a per-stage roofline report (needs --framework).",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup calls per stage.")
    parser.add_argument("--repeats", type=int, default=10, help="Timed calls per stage.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.set_defaults(func=run_pipeline)


def run_pipeline(args: argparse.Namespace) -> int:
    pipeline = load_target(args.target)
    if not isinstance(pipeline, Pipeline) and callable(pipeline):
        pipeline = pipeline()
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"{args.target} is not a Pipeline or a function returning one")
    adapter = get_adapter(args.framework) if args.framework else None
    hardware = None
    if args.roofline:
        if adapter is None:
            raise ValueError("--roofline needs --framework to trace stage operations")
        hardware = (
            adapter.hardware(benchmark_memory=False)[0] if hasattr(adapter, "hardware") else None
        )
        if hardware is None:
            from neural_cost.hardware_detect import detect_hardware

            hardware, _ = detect_hardware(benchmark_memory=False)
    report = pipeline.run(
        adapter=adapter, hardware=hardware, warmup=args.warmup, repeats=args.repeats
    )
    if args.json:
        payload = {
            "measured_total_seconds": (
                report.measured_total.median_seconds if report.measured_total else None
            ),
            "stage_sum_seconds": report.stage_sum_seconds,
            "unexplained_seconds": report.unexplained_seconds,
            "unexplained_ratio": report.unexplained_ratio,
            "stages": [
                {
                    "name": stage.name,
                    "calls": stage.calls,
                    "per_call_seconds": stage.per_call_seconds,
                    "total_seconds": stage.total_seconds,
                    "share": stage.share,
                    "device": stage.measurement.device,
                    "roofline_efficiency": stage.gap.efficiency if stage.gap else None,
                    "bottleneck": stage.gap.bottleneck if stage.gap else None,
                    "coverage": (
                        stage.gap.coverage.fraction if stage.gap and stage.gap.coverage else None
                    ),
                    "note": stage.note,
                }
                for stage in report.stages
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(report.render())
    return 0
