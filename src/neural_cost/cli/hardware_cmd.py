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
        "--device",
        default=None,
        help=(
            "Device the workload runs on: 'cpu' or 'gpu' (also accepts 'cuda', 'mps', "
            "'metal'). Default: the fastest detected accelerator."
        ),
    )
    parser.add_argument(
        "--calibrate",
        choices=["jax"],
        default=None,
        help=(
            "Also measure achievable peak FLOP/s and bandwidth on the active device through "
            "the given framework and report them against the spec-sheet values."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the short calibration sweep with --calibrate.",
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

    calibration = None
    device = args.device
    if args.calibrate == "jax":
        import jax

        from neural_cost.adapters import JaxAdapter

        active = jax.devices()[0]
        device = device or active
        calibration = JaxAdapter().calibrate(device=active, quick=args.quick)
    try:
        hw, det = detect_hardware(
            bandwidth_benchmark_mb=args.bw_bench_mb,
            benchmark_memory=benchmark_memory,
            device=device,
        )
    except ValueError:
        if calibration is None:
            raise
        hw, det = calibration.to_hardware_spec(f"{device} (measured)"), None

    data = {
        "device_name": hw.device_name,
        "device_type": hw.device_type,
        "detection_source": det.source if det else "on-device calibration",
        "peak_flops": hw.peak_flops,
        "memory_bandwidth_bytes_per_sec": hw.memory_bandwidth,
        "measured_bandwidth_gb_s": det.measured_bandwidth_gb_s if det else None,
        "ridge_point_flop_per_byte": hw.ridge_point,
    }
    if calibration is not None:
        flops_fraction, bandwidth_fraction = calibration.fraction_of(hw)
        data["calibration"] = {
            "source": calibration.source,
            "achievable_flops": calibration.achievable_flops,
            "achievable_bandwidth_bytes_per_sec": calibration.achievable_bandwidth,
            "flops_fraction_of_spec": flops_fraction,
            "bandwidth_fraction_of_spec": bandwidth_fraction,
            "dispatch_seconds": calibration.dispatch_seconds,
        }

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print("┌─ Detected Hardware Specification ──────────────────────────────────────────")
    print(f"│  Device:             {hw.device_name} [{hw.device_type or 'unknown'}]")
    print(f"│  Detection Source:   {data['detection_source']}")
    print(
        f"│  Peak FP32 Compute:  {format_flops(hw.peak_flops)}/s ({hw.peak_flops / 1e12:.2f} TFLOP/s)"
    )
    print(
        f"│  Memory Bandwidth:   {format_bytes(hw.memory_bandwidth)}/s ({hw.memory_bandwidth / 1e9:.2f} GB/s)"
    )
    if det is not None and det.measured_bandwidth_gb_s is not None:
        print(f"│  STREAM Triad Live:  {det.measured_bandwidth_gb_s:.2f} GB/s")
    print(f"│  Roofline Ridge:     {hw.ridge_point:.2f} FLOP/byte")
    if calibration is not None:
        flops_fraction, bandwidth_fraction = calibration.fraction_of(hw)
        print(f"│  Calibration:        {calibration.source}")
        print(
            f"│  Achievable Compute: {calibration.achievable_flops / 1e12:.3f} TFLOP/s "
            f"({flops_fraction:.0%} of spec)"
        )
        print(
            f"│  Achievable BW:      {calibration.achievable_bandwidth / 1e9:.2f} GB/s "
            f"({bandwidth_fraction:.0%} of spec)"
        )
        print(f"│  Dispatch / call:    {calibration.dispatch_seconds * 1e6:.1f} µs")
    print("└────────────────────────────────────────────────────────────────────────────")
    return 0
