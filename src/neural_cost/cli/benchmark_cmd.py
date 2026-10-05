"""Benchmark subcommand for empirical latency, bandwidth, and roofline analysis."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from neural_cost.adapters import available_adapters, get_adapter
from neural_cost.analysis import analyze_gap
from neural_cost.api import estimate_model
from neural_cost.benchmarks import (
    benchmark_moe_decode,
    benchmark_paged_attention,
    benchmark_speculative_decoding,
)
from neural_cost.cli.helpers import (
    format_bytes,
    resolve_hardware,
)
from neural_cost.hardware_detect import detect_hardware


def register_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Run dynamic empirical benchmarks and measure real efficiency against roofline bounds.",
        description="Execute wall-clock latency, throughput, and memory bandwidth benchmarks across architectures.",
    )
    _add_benchmark_arguments(parser)
    parser.set_defaults(func=run_benchmark)


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arch",
        choices=["linear", "moe", "speculative", "paged-attention", "stream"],
        default="linear",
        help="Benchmark workload architecture (default: linear).",
    )
    parser.add_argument(
        "--framework",
        choices=["torch", "jax", "tensorflow", "auto"],
        default="auto",
        help="Framework adapter to benchmark with (default: auto).",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size (default: 64)."
    )
    parser.add_argument(
        "--in-features",
        "-k",
        type=int,
        default=1024,
        help="Input dimension for linear (default: 1024).",
    )
    parser.add_argument(
        "--out-features",
        "-n",
        type=int,
        default=1024,
        help="Output dimension for linear (default: 1024).",
    )
    parser.add_argument(
        "--seq-len", "-s", type=int, default=512, help="Context/sequence length (default: 512)."
    )
    parser.add_argument(
        "--embed-dim", "-d", type=int, default=1024, help="Embedding dimension (default: 1024)."
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Query attention heads (default: 8)."
    )
    parser.add_argument("--num-kv-heads", type=int, default=None, help="KV heads for GQA.")
    parser.add_argument(
        "--num-experts", type=int, default=8, help="MoE total experts (default: 8)."
    )
    parser.add_argument(
        "--top-k", type=int, default=2, help="MoE active experts per token (default: 2)."
    )
    parser.add_argument(
        "--expert-hidden-dim", type=int, default=2048, help="MoE expert hidden dimension."
    )
    parser.add_argument(
        "--gamma", type=int, default=4, help="Draft tokens per step for speculative decoding."
    )
    parser.add_argument(
        "--acceptance-rate",
        type=float,
        default=0.75,
        help="Speculative acceptance rate (default: 0.75).",
    )
    parser.add_argument(
        "--block-size", type=int, default=16, help="PagedAttention block size (default: 16)."
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up iterations (default: 5).")
    parser.add_argument(
        "--repeats", type=int, default=15, help="Timed benchmark repetitions (default: 15)."
    )

    # Hardware & output
    parser.add_argument(
        "--peak-flops", type=float, default=None, help="Target peak FLOP/s override."
    )
    parser.add_argument(
        "--memory-bandwidth", type=float, default=None, help="Target memory bandwidth override."
    )
    parser.add_argument(
        "--json", action="store_true", help="Output benchmark results in JSON format."
    )


def run_benchmark(args: argparse.Namespace) -> int:
    hw, _ = resolve_hardware(
        peak_flops=args.peak_flops,
        memory_bandwidth=args.memory_bandwidth,
        benchmark_memory=False,
    )

    if args.arch == "stream":
        _, triad_det = detect_hardware(bandwidth_benchmark_mb=128, benchmark_memory=True)
        res = {
            "workload": "STREAM Triad Memory Bandwidth",
            "measured_bandwidth_gb_s": triad_det.measured_bandwidth_gb_s,
            "device": hw.device_name,
        }
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(
                f"STREAM Triad Live Bandwidth: {triad_det.measured_bandwidth_gb_s:.2f} GB/s ({hw.device_name})"
            )
        return 0

    if args.arch == "moe":
        metrics = benchmark_moe_decode(
            batch_size=args.batch_size,
            seq_len=1,
            embed_dim=args.embed_dim,
            expert_hidden_dim=args.expert_hidden_dim,
            num_experts=args.num_experts,
            top_k=args.top_k,
            warmup=args.warmup,
            repeats=args.repeats,
            hardware=hw,
        )
        if args.json:
            print(json.dumps(asdict(metrics), indent=2))
            return 0
        print("┌─ MoE Empirical Benchmark ──────────────────────────────────────────────────")
        print(f"│  Batch: {args.batch_size} | Experts: {args.num_experts} (top-{args.top_k})")
        print(f"│  Latency:            {metrics.latency_ms:.3f} ms")
        print(f"│  Throughput:         {metrics.achieved_tflops:.2f} TFLOP/s")
        print(f"│  Achieved Bandwidth: {metrics.achieved_gbw:.2f} GB/s")
        print(f"│  DRAM Bandwidth Util:{metrics.memory_bw_util:.1%}")
        print(f"│  Compute Util:       {metrics.compute_util:.1%}")
        print("└────────────────────────────────────────────────────────────────────────────")
        return 0

    if args.arch == "speculative":
        spec_m = benchmark_speculative_decoding(
            gamma=args.gamma,
            acceptance_rate=args.acceptance_rate,
            prompt_len=args.seq_len,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        if args.json:
            print(json.dumps(asdict(spec_m), indent=2))
            return 0
        print("┌─ Speculative Decoding Empirical Benchmark ─────────────────────────────────")
        print(f"│  Gamma: {args.gamma} | Acceptance rate: {args.acceptance_rate:.1%}")
        print(f"│  Expected Tokens/Step:     {spec_m.expected_tokens_per_step:.2f}")
        print(f"│  Draft Latency:            {spec_m.draft_latency_ms:.3f} ms")
        print(f"│  Verify Latency:           {spec_m.verify_latency_ms:.3f} ms")
        print(f"│  Speculative Step Latency: {spec_m.spec_step_latency_ms:.3f} ms")
        print(f"│  Effective ms/token:       {spec_m.effective_ms_per_token:.3f} ms")
        print(f"│  Baseline ms/token:        {spec_m.baseline_ms_per_token:.3f} ms")
        print(f"│  Achieved Wall Speedup:    {spec_m.speedup:.2f}x")
        print(f"│  Breakeven Acceptance Rate:{spec_m.breakeven_acceptance_rate:.1%}")
        print("└────────────────────────────────────────────────────────────────────────────")
        return 0

    if args.arch == "paged-attention":
        paged_m = benchmark_paged_attention(
            batch_size=args.batch_size,
            context_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            block_size=args.block_size,
            warmup=args.warmup,
            repeats=args.repeats,
            hardware=hw,
        )
        if args.json:
            print(json.dumps(asdict(paged_m), indent=2))
            return 0
        print("┌─ PagedAttention Empirical Benchmark ───────────────────────────────────────")
        print(f"│  Batch: {args.batch_size} | Context: {args.seq_len} | Block: {args.block_size}")
        print(f"│  Latency:            {paged_m.latency_ms:.3f} ms")
        print(f"│  KV Allocated:       {format_bytes(paged_m.allocated_kv_bytes)}")
        print(f"│  KV Compulsory:      {format_bytes(paged_m.compulsory_kv_bytes)}")
        print(
            f"│  KV Fragmentation:   {paged_m.fragmentation_ratio:.1%} ({format_bytes(paged_m.fragmentation_bytes)})"
        )
        print(f"│  Achieved Bandwidth: {paged_m.achieved_gbw:.2f} GB/s")
        print(f"│  Bandwidth Util:     {paged_m.memory_bw_util:.1%}")
        print("└────────────────────────────────────────────────────────────────────────────")
        return 0

    # Default: linear matmul benchmark
    fw = args.framework
    avail = available_adapters()
    if fw == "auto":
        for cand in ["torch", "jax", "tensorflow"]:
            if cand in avail:
                fw = cand
                break
        if fw == "auto":
            sys.stderr.write(
                "No deep learning framework available (install torch, jax, or tensorflow).\n"
            )
            return 1

    if fw not in avail:
        sys.stderr.write(f"Framework '{fw}' is not installed.\n")
        return 1

    adapter = get_adapter(fw)
    b = args.batch_size
    k = args.in_features
    n = args.out_features

    if fw == "torch":
        import torch

        model = torch.nn.Linear(k, n, bias=False).eval()
        inputs = (torch.randn(b, k),)
    elif fw == "jax":
        import jax.numpy as jnp

        weight = jnp.ones((k, n))

        def model(x: Any, w: Any = weight) -> Any:
            return jnp.matmul(x, w)

        inputs = (jnp.ones((b, k)), weight)
    else:
        import tensorflow as tf

        model = tf.keras.Sequential([tf.keras.layers.Dense(n, use_bias=False)])
        inputs = (tf.ones((b, k)),)

    est = estimate_model(model, inputs, adapter)
    measurement = adapter.benchmark(model, *inputs, warmup=args.warmup, repeats=args.repeats)
    gap = analyze_gap(est, measurement, hw)

    res = {
        "workload": "linear",
        "framework": fw,
        "batch_size": b,
        "in_features": k,
        "out_features": n,
        "median_ms": measurement.median_seconds * 1e3,
        "achieved_gflops": gap.achieved_flops / 1e9,
        "achieved_gbw": gap.achieved_bandwidth / 1e9,
        "roofline_efficiency": gap.efficiency,
        "bottleneck": gap.bottleneck,
        "hardware": {
            "name": hw.device_name,
            "peak_flops": hw.peak_flops,
            "memory_bandwidth": hw.memory_bandwidth,
        },
    }

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print("┌─ Neural Cost Empirical Benchmark ──────────────────────────────────────────")
    print(f"│  Workload:           {b}×{k}→{n} Linear ({fw.upper()})")
    print(f"│  Median Latency:     {measurement.median_seconds * 1e3:.3f} ms")
    print(f"│  Achieved Compute:   {gap.achieved_flops / 1e9:.2f} GFLOP/s")
    print(f"│  Achieved Bandwidth: {gap.achieved_bandwidth / 1e9:.2f} GB/s")
    print(f"│  Roofline Efficiency:{gap.efficiency:.1%}")
    print(f"│  Regime:             {gap.bottleneck.upper()}-BOUND")
    print("└────────────────────────────────────────────────────────────────────────────")
    return 0
