"""Hardware detection and roofline specification subcommand."""

from __future__ import annotations

import argparse
import json
import sys

from neural_cost.cli.helpers import format_bytes, format_flops
from neural_cost.hardware_detect import detect_hardware


def register_hardware_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "hardware",
        help="Detect system accelerator/CPU specs and measure memory bandwidth.",
        description="Inspect detected hardware capabilities, theoretical peak compute, and live STREAM triad bandwidth.",
    )
    parser.add_argument(
        "--bw-bench-mb",
        type=int,
        default=256,
        help="Working-set size in MiB for STREAM benchmark (default: 256).",
    )
    parser.add_argument(
        "--no-bench",
        action="store_true",
        help="Skip the live STREAM memory bandwidth benchmark.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output detected hardware metrics as JSON.",
    )
    parser.set_defaults(func=run_hardware)


def run_hardware(args: argparse.Namespace) -> int:
    benchmark_memory = not args.no_bench
    if not args.json:
        bench_msg = " (running STREAM benchmark)" if benchmark_memory else " (benchmark skipped)"
        print(f"Detecting hardware capabilities{bench_msg}...", file=sys.stderr)

    hw, det = detect_hardware(
        bandwidth_benchmark_mb=args.bw_bench_mb,
        benchmark_memory=benchmark_memory,
    )

    data = {
        "device_name": hw.device_name,
        "detection_source": det.source,
        "peak_flops": hw.peak_flops,
        "memory_bandwidth_bytes_per_sec": hw.memory_bandwidth,
        "measured_bandwidth_gb_s": det.measured_bandwidth_gb_s,
        "ridge_point_flop_per_byte": hw.ridge_point,
    }

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print("┌─ Detected Hardware Specification ──────────────────────────────────────────")
    print(f"│  Device:             {hw.device_name}")
    print(f"│  Detection Source:   {det.source}")
    print(
        f"│  Peak FP32 Compute:  {format_flops(hw.peak_flops)}/s ({hw.peak_flops / 1e12:.2f} TFLOP/s)"
    )
    print(
        f"│  Memory Bandwidth:   {format_bytes(hw.memory_bandwidth)}/s ({hw.memory_bandwidth / 1e9:.2f} GB/s)"
    )
    if det.measured_bandwidth_gb_s is not None:
        print(f"│  STREAM Triad Live:  {det.measured_bandwidth_gb_s:.2f} GB/s")
    print(f"│  Roofline Ridge:     {hw.ridge_point:.2f} FLOP/byte")
    print("└────────────────────────────────────────────────────────────────────────────")
    return 0
