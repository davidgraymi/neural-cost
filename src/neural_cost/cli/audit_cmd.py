"""CI audit and performance budget assertion subcommand."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from neural_cost.cli.helpers import (
    format_bytes,
    parse_bytes,
    parse_dtype_bytes,
    resolve_hardware,
)
from neural_cost.cli.profile_cmd import (
    profile_convnet,
    profile_mlp,
    profile_moe,
    profile_paged_attention,
    profile_ssm,
    profile_transformer,
)


@dataclass
class AuditCheck:
    name: str
    target_threshold: str
    actual_value: str
    passed: bool
    details: str = ""


def register_audit_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "audit",
        help="Audit architecture performance against strict budget constraints for CI/CD.",
        description="Verify VRAM limits, latency bounds, and arithmetic intensity gates. Exits 0 on success, 1 on violation.",
    )
    _add_audit_arguments(parser)
    parser.set_defaults(func=run_audit)


def _add_audit_arguments(parser: argparse.ArgumentParser) -> None:
    # Budget threshold gates
    parser.add_argument(
        "--max-vram",
        type=str,
        default=None,
        help="Maximum allowable VRAM/memory footprint (e.g. '16GB', '8192MB').",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
        help="Maximum allowable theoretical latency in milliseconds.",
    )
    parser.add_argument(
        "--min-arithmetic-intensity",
        type=float,
        default=None,
        help="Minimum arithmetic intensity in FLOP/byte.",
    )
    parser.add_argument(
        "--max-fragmentation-ratio",
        type=float,
        default=None,
        help="Maximum allowable KV cache fragmentation ratio (0.0 to 1.0).",
    )

    # Architecture parameters
    parser.add_argument(
        "--arch",
        choices=["transformer", "moe", "paged-attention", "mlp", "convnet", "ssm", "mamba"],
        default="transformer",
        help="Architecture family to audit (default: transformer).",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument(
        "--seq-len", "-s", type=int, default=2048, help="Sequence length (default: 2048)."
    )
    parser.add_argument(
        "--embed-dim", "-d", type=int, default=4096, help="Embedding dimension (default: 4096)."
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of query attention heads (default: 32)."
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=None, help="Number of KV heads (for GQA/MQA)."
    )
    parser.add_argument(
        "--num-layers",
        "-l",
        type=int,
        default=32,
        help="Number of transformer layers (default: 32).",
    )
    parser.add_argument(
        "--intermediate-dim", type=int, default=None, help="FFN/MLP intermediate dimension."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "int8", "fp8", "int4"],
        help="Data precision (default: fp16).",
    )

    # MoE specific
    parser.add_argument("--num-experts", type=int, default=8, help="MoE total experts.")
    parser.add_argument("--top-k", type=int, default=2, help="MoE active experts per token.")
    parser.add_argument(
        "--expert-hidden-dim", type=int, default=None, help="MoE expert hidden dimension."
    )
    parser.add_argument(
        "--expert-type", choices=["swiglu", "mlp"], default="swiglu", help="MoE expert type."
    )
    parser.add_argument("--decode", action="store_true", help="Audit autoregressive decode step.")

    # PagedAttention specific
    parser.add_argument("--block-size", type=int, default=16, help="PagedAttention block size.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max sequence length.")
    parser.add_argument(
        "--shared-prefix-tokens", type=int, default=0, help="Shared prompt prefix tokens."
    )

    # State Space Model (SSM / Mamba) specific
    parser.add_argument(
        "--state-dim", type=int, default=16, help="SSM recurrent state dimension N."
    )
    parser.add_argument(
        "--expand-factor", type=int, default=2, help="SSM hidden expansion factor E."
    )
    parser.add_argument(
        "--conv-kernel-size", type=int, default=4, help="SSM 1D depthwise convolution filter width."
    )

    # MLP / Convnet
    parser.add_argument("--in-features", type=int, default=1024, help="MLP input features.")
    parser.add_argument("--out-features", type=int, default=1024, help="MLP output features.")
    parser.add_argument("--in-channels", type=int, default=3, help="ConvNet input channels.")
    parser.add_argument("--out-channels", type=int, default=64, help="ConvNet output channels.")
    parser.add_argument("--height", type=int, default=224, help="ConvNet image height.")
    parser.add_argument("--width", type=int, default=224, help="ConvNet image width.")
    parser.add_argument("--kernel-size", type=int, default=3, help="ConvNet kernel size.")

    # Hardware & output
    parser.add_argument("--peak-flops", type=float, default=None, help="Peak compute FLOP/s.")
    parser.add_argument(
        "--memory-bandwidth", type=float, default=None, help="Memory bandwidth bytes/s."
    )
    parser.add_argument("--device-name", type=str, default=None, help="Target device name.")
    parser.add_argument("--no-bench", action="store_true", help="Skip STREAM memory benchmark.")
    parser.add_argument("--json", action="store_true", help="Output audit results in JSON format.")


def run_audit(args: argparse.Namespace) -> int:
    dtype_bytes = parse_dtype_bytes(args.dtype)

    # 1. Profile architecture
    if args.arch == "transformer":
        res = profile_transformer(args, dtype_bytes)
    elif args.arch == "moe":
        res = profile_moe(args, dtype_bytes)
    elif args.arch == "paged-attention":
        res = profile_paged_attention(args, dtype_bytes)
    elif args.arch == "mlp":
        res = profile_mlp(args, dtype_bytes)
    elif args.arch == "convnet":
        res = profile_convnet(args, dtype_bytes)
    elif args.arch in ("ssm", "mamba"):
        res = profile_ssm(args, dtype_bytes)
    else:
        sys.stderr.write(f"Unknown architecture: {args.arch}\n")
        return 1

    # 2. Hardware roofline
    hw, _ = resolve_hardware(
        peak_flops=args.peak_flops,
        memory_bandwidth=args.memory_bandwidth,
        device_name=args.device_name,
        benchmark_memory=not args.no_bench,
    )

    flops = res["flops"]
    total_bytes = res["total_bytes"]
    ai = res["arithmetic_intensity"]

    compute_sec = flops / hw.peak_flops if hw.peak_flops > 0 else 0.0
    bandwidth_sec = total_bytes / hw.memory_bandwidth if hw.memory_bandwidth > 0 else 0.0
    lower_bound_sec = max(compute_sec, bandwidth_sec)
    lower_bound_ms = lower_bound_sec * 1e3
    bottleneck = "compute" if ai >= hw.ridge_point else "memory"

    # Memory footprint: weights + kv_cache + recurrent state or total_bytes
    vram_bytes = (
        res.get("weight_bytes", 0) + res.get("kv_cache_bytes", 0) + res.get("state_bytes", 0)
    )
    if vram_bytes == 0:
        vram_bytes = total_bytes

    # 3. Evaluate audit assertions
    checks: list[AuditCheck] = []
    has_violations = False

    # Check: Max VRAM
    if args.max_vram is not None:
        max_bytes = parse_bytes(args.max_vram)
        passed = vram_bytes <= max_bytes
        if not passed:
            has_violations = True
        checks.append(
            AuditCheck(
                name="VRAM Footprint",
                target_threshold=f"<= {format_bytes(max_bytes)}",
                actual_value=format_bytes(vram_bytes),
                passed=passed,
                details=f"Over by {format_bytes(vram_bytes - max_bytes)}" if not passed else "OK",
            )
        )

    # Check: Max Latency
    if args.max_latency_ms is not None:
        passed = lower_bound_ms <= args.max_latency_ms
        if not passed:
            has_violations = True
        checks.append(
            AuditCheck(
                name="Latency Floor",
                target_threshold=f"<= {args.max_latency_ms:.2f} ms",
                actual_value=f"{lower_bound_ms:.2f} ms",
                passed=passed,
                details=f"Exceeded by {lower_bound_ms - args.max_latency_ms:.2f} ms"
                if not passed
                else "OK",
            )
        )

    # Check: Min Arithmetic Intensity
    if args.min_arithmetic_intensity is not None:
        passed = ai >= args.min_arithmetic_intensity
        if not passed:
            has_violations = True
        checks.append(
            AuditCheck(
                name="Arithmetic Intensity",
                target_threshold=f">= {args.min_arithmetic_intensity:.1f} FLOP/B",
                actual_value=f"{ai:.1f} FLOP/B",
                passed=passed,
                details="Below target compute intensity" if not passed else "OK",
            )
        )

    # Check: Max Fragmentation Ratio
    if args.max_fragmentation_ratio is not None:
        actual_frag = res.get("fragmentation_ratio", 0.0)
        passed = actual_frag <= args.max_fragmentation_ratio
        if not passed:
            has_violations = True
        checks.append(
            AuditCheck(
                name="KV Fragmentation",
                target_threshold=f"<= {args.max_fragmentation_ratio:.1%}",
                actual_value=f"{actual_frag:.1%}",
                passed=passed,
                details=f"Internal fragmentation {actual_frag:.1%}" if not passed else "OK",
            )
        )

    # Prepare findings and recommendations
    recommendations: list[str] = []
    if has_violations:
        if args.max_vram and vram_bytes > parse_bytes(args.max_vram):
            if res.get("num_heads", 0) and res.get("num_kv_heads") == res.get("num_heads"):
                recommendations.append(
                    "Consider Grouped-Query Attention (GQA) with fewer KV heads (e.g. --num-kv-heads 8)."
                )
            recommendations.append(
                "Consider lower precision weights (e.g. FP8 or INT4) to reduce memory footprint."
            )
        if args.max_latency_ms and lower_bound_ms > args.max_latency_ms:
            if bottleneck == "memory":
                recommendations.append(
                    "Workload is memory-bandwidth bound; increase batch size or use faster HBM hardware."
                )
            else:
                recommendations.append(
                    "Workload is compute-bound; consider pruning, quantization, or higher TFLOP/s hardware."
                )

    report = {
        "status": "FAIL" if has_violations else "PASS",
        "workload": res["architecture"],
        "hardware": hw.device_name,
        "bottleneck": bottleneck,
        "metrics": {
            "vram_bytes": vram_bytes,
            "vram_human": format_bytes(vram_bytes),
            "latency_floor_ms": lower_bound_ms,
            "arithmetic_intensity": ai,
            "fragmentation_ratio": res.get("fragmentation_ratio"),
        },
        "checks": [
            {
                "name": c.name,
                "target": c.target_threshold,
                "actual": c.actual_value,
                "passed": c.passed,
                "details": c.details,
            }
            for c in checks
        ],
        "recommendations": recommendations,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 1 if has_violations else 0

    print("┌─ Neural Cost Performance Audit Gate ───────────────────────────────────────")
    print(f"│  Architecture:   {res['architecture']} ({args.dtype})")
    print(f"│  Target Device:  {hw.device_name} (ridge: {hw.ridge_point:.1f} FLOP/B)")
    print(f"│  Regime:         {bottleneck.upper()}-BOUND")
    print("├─ Budget Assertions ────────────────────────────────────────────────────────")
    if not checks:
        print("│  (No assertions specified; use --max-vram, --max-latency-ms, etc.)")
    for c in checks:
        icon = "✓ PASS" if c.passed else "✗ FAIL"
        print(
            f"│  [{icon}] {c.name:<22} target: {c.target_threshold:<15} actual: {c.actual_value:<12} ({c.details})"
        )

    if recommendations:
        print("├─ Recommendations ──────────────────────────────────────────────────────────")
        for rec in recommendations:
            print(f"│  • {rec}")

    print("└────────────────────────────────────────────────────────────────────────────")

    if has_violations:
        print(
            f"AUDIT FAILED: {sum(1 for c in checks if not c.passed)} constraint(s) violated.",
            file=sys.stderr,
        )
        return 1

    print("AUDIT PASSED: All budget constraints satisfied.")
    return 0
