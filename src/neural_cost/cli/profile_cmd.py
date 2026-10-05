"""Model and architecture static profiling subcommand."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from neural_cost.analysis import (
    analyze_paged_attention_gap,
)
from neural_cost.cli.helpers import (
    format_bytes,
    format_flops,
    parse_dtype_bytes,
    resolve_hardware,
)
from neural_cost.estimate import (
    estimate_moe,
    estimate_operation,
    estimate_paged_attention,
)
from neural_cost.operations import Operation


def register_profile_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "profile",
        help="Profile theoretical compute, memory footprint, and roofline bounds for an architecture.",
        description="Compute forward FLOPs, parameter count, KV cache/activations, arithmetic intensity, and latency floors.",
    )
    _add_profile_arguments(parser)
    parser.set_defaults(func=run_profile)


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arch",
        choices=["transformer", "moe", "paged-attention", "mlp", "convnet"],
        default="transformer",
        help="Architecture family to profile (default: transformer).",
    )
    # General dimensions
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument(
        "--seq-len", "-s", type=int, default=2048, help="Sequence length (default: 2048)."
    )
    parser.add_argument(
        "--embed-dim",
        "-d",
        type=int,
        default=4096,
        help="Hidden/embedding dimension (default: 4096).",
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of query attention heads (default: 32)."
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads for GQA/MQA (default: same as num-heads).",
    )
    parser.add_argument(
        "--num-layers",
        "-l",
        type=int,
        default=32,
        help="Number of transformer layers (default: 32).",
    )
    parser.add_argument(
        "--intermediate-dim",
        type=int,
        default=None,
        help="FFN/MLP intermediate dimension (default: ~8/3 * embed_dim for SwiGLU or 4 * embed_dim).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "int8", "fp8", "int4"],
        help="Data precision (default: fp16).",
    )

    # MoE specific
    parser.add_argument(
        "--num-experts", type=int, default=8, help="Total number of experts in MoE (default: 8)."
    )
    parser.add_argument(
        "--top-k", type=int, default=2, help="Number of active experts per token (default: 2)."
    )
    parser.add_argument(
        "--expert-hidden-dim", type=int, default=None, help="MoE expert hidden dimension."
    )
    parser.add_argument(
        "--expert-type",
        choices=["swiglu", "mlp"],
        default="swiglu",
        help="MoE expert FFN structure.",
    )
    parser.add_argument(
        "--decode", action="store_true", help="Profile token-by-token autoregressive decode step."
    )

    # PagedAttention specific
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="PagedAttention block size in tokens (default: 16).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max sequence length for contiguous reservation comparison.",
    )
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="Number of shared prompt tokens for prefix caching.",
    )

    # MLP / Convnet specific
    parser.add_argument(
        "--in-features", type=int, default=1024, help="Input feature dimension for MLP."
    )
    parser.add_argument(
        "--out-features", type=int, default=1024, help="Output feature dimension for MLP."
    )
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels for ConvNet.")
    parser.add_argument("--out-channels", type=int, default=64, help="Output channels for ConvNet.")
    parser.add_argument("--height", type=int, default=224, help="Image height for ConvNet.")
    parser.add_argument("--width", type=int, default=224, help="Image width for ConvNet.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Kernel size for ConvNet.")

    # Hardware & output
    parser.add_argument(
        "--peak-flops", type=float, default=None, help="Target peak FLOP/s override."
    )
    parser.add_argument(
        "--memory-bandwidth",
        type=float,
        default=None,
        help="Target memory bandwidth in bytes/s override.",
    )
    parser.add_argument("--device-name", type=str, default=None, help="Target device name label.")
    parser.add_argument(
        "--no-bench",
        action="store_true",
        help="Skip live STREAM memory benchmark if detecting hardware.",
    )
    parser.add_argument("--json", action="store_true", help="Output profile report as JSON.")


def profile_transformer(args: argparse.Namespace, dtype_bytes: int) -> dict[str, Any]:
    b = args.batch_size
    s = args.seq_len
    d = args.embed_dim
    h = args.num_heads
    kv_h = args.num_kv_heads or h
    layers = args.num_layers
    inter_d = args.intermediate_dim or int(d * 8 / 3)

    head_dim = d // h

    # Per layer components:
    # 1. Attention: Q, K, V projections + attention score matmul + output projection
    # Q: (b, s, d) x (d, h * head_dim) -> FLOPs = 2 * b * s * d * d
    # K: (b, s, d) x (d, kv_h * head_dim) -> FLOPs = 2 * b * s * d * (kv_h * head_dim)
    # V: (b, s, d) x (d, kv_h * head_dim) -> FLOPs = 2 * b * s * d * (kv_h * head_dim)
    # OutProj: (b, s, d) x (d, d) -> FLOPs = 2 * b * s * d * d
    q_params = d * d
    k_params = d * (kv_h * head_dim)
    v_params = d * (kv_h * head_dim)
    out_params = d * d
    attn_proj_params = q_params + k_params + v_params + out_params

    # Attn core FLOPs: QK^T is 2 * b * h * s * s * head_dim, Score x V is 2 * b * h * s * s * head_dim
    attn_core_flops = 4 * b * h * s * s * head_dim
    attn_proj_flops = 2 * b * s * attn_proj_params
    attn_flops_per_layer = attn_proj_flops + attn_core_flops

    # 2. MLP / SwiGLU:
    # Gate & Up projections: 2 * (d x inter_d)
    # Down projection: (inter_d x d)
    mlp_params = 3 * d * inter_d
    mlp_flops_per_layer = 2 * b * s * mlp_params

    # 3. LayerNorms (2 per layer):
    norm_params = 2 * d
    norm_flops_per_layer = 4 * b * s * d

    # Total per layer:
    layer_params = attn_proj_params + mlp_params + norm_params
    layer_flops = attn_flops_per_layer + mlp_flops_per_layer + norm_flops_per_layer

    total_params = layer_params * layers
    total_flops = layer_flops * layers

    # Memory:
    weight_bytes = total_params * dtype_bytes
    # KV Cache: 2 (K and V) * layers * batch_size * kv_h * head_dim * seq_len * dtype_bytes
    kv_cache_bytes = 2 * layers * b * kv_h * head_dim * s * dtype_bytes
    # Activations per token: input + post-attn + post-mlp
    activation_bytes = layers * (b * s * d * 3) * dtype_bytes

    total_bytes = weight_bytes + kv_cache_bytes + activation_bytes
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0

    return {
        "architecture": "transformer",
        "batch_size": b,
        "seq_len": s,
        "embed_dim": d,
        "num_heads": h,
        "num_kv_heads": kv_h,
        "num_layers": layers,
        "dtype": args.dtype,
        "total_parameters": total_params,
        "weight_bytes": weight_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "activation_bytes": activation_bytes,
        "total_bytes": total_bytes,
        "flops": total_flops,
        "arithmetic_intensity": arithmetic_intensity,
    }


def profile_mlp(args: argparse.Namespace, dtype_bytes: int) -> dict[str, Any]:
    b = args.batch_size
    k = args.in_features
    n = args.out_features
    op = Operation(
        name="linear",
        kind="linear",
        inputs=((b, k), (k, n)),
        output=(b, n),
        dtype_bytes=dtype_bytes,
    )
    est = estimate_operation(op)
    return {
        "architecture": "mlp",
        "batch_size": b,
        "in_features": k,
        "out_features": n,
        "dtype": args.dtype,
        "total_parameters": k * n,
        "weight_bytes": k * n * dtype_bytes,
        "total_bytes": est.total_bytes,
        "flops": est.flops,
        "arithmetic_intensity": est.arithmetic_intensity,
    }


def profile_convnet(args: argparse.Namespace, dtype_bytes: int) -> dict[str, Any]:
    b = args.batch_size
    cin = args.in_channels
    cout = args.out_channels
    h = args.height
    w = args.width
    k = args.kernel_size

    op = Operation(
        name="conv2d",
        kind="conv2d",
        inputs=((b, cin, h, w), (cout, cin, k, k)),
        output=(b, cout, h, w),
        dtype_bytes=dtype_bytes,
    )
    est = estimate_operation(op)
    param_count = cin * cout * k * k
    return {
        "architecture": "convnet",
        "batch_size": b,
        "in_channels": cin,
        "out_channels": cout,
        "height": h,
        "width": w,
        "kernel_size": k,
        "dtype": args.dtype,
        "total_parameters": param_count,
        "weight_bytes": param_count * dtype_bytes,
        "total_bytes": est.total_bytes,
        "flops": est.flops,
        "arithmetic_intensity": est.arithmetic_intensity,
    }


def profile_moe(args: argparse.Namespace, dtype_bytes: int) -> dict[str, Any]:
    b = args.batch_size
    s = args.seq_len if not args.decode else 1
    d = args.embed_dim
    experts = args.num_experts
    k = args.top_k
    hidden = args.expert_hidden_dim or int(d * 8 / 3)
    layers = args.num_layers

    single_layer_est = estimate_moe(
        batch_size=b,
        seq_len=s,
        embed_dim=d,
        expert_hidden_dim=hidden,
        num_experts=experts,
        top_k=k,
        expert_type=args.expert_type,
        is_decode=args.decode,
        dtype_bytes=dtype_bytes,
    )

    total_flops = single_layer_est.total_flops * layers
    total_bytes = single_layer_est.total_bytes * layers
    ai = total_flops / total_bytes if total_bytes > 0 else 0.0

    return {
        "architecture": "moe",
        "batch_size": b,
        "seq_len": s,
        "embed_dim": d,
        "expert_hidden_dim": hidden,
        "num_experts": experts,
        "top_k": k,
        "expert_type": args.expert_type,
        "is_decode": args.decode,
        "num_layers": layers,
        "dtype": args.dtype,
        "total_parameters": single_layer_est.total_parameters * layers,
        "active_parameters": single_layer_est.active_parameters * layers,
        "expected_loaded_experts_per_layer": single_layer_est.expected_loaded_experts,
        "total_bytes": total_bytes,
        "flops": total_flops,
        "arithmetic_intensity": ai,
    }


def profile_paged_attention(args: argparse.Namespace, dtype_bytes: int) -> dict[str, Any]:
    b = args.batch_size
    s = args.seq_len
    d = args.embed_dim
    h = args.num_heads
    kv_h = args.num_kv_heads or h
    block = args.block_size
    layers = args.num_layers
    max_len = args.max_model_len or s
    shared_prefix = args.shared_prefix_tokens

    paged = estimate_paged_attention(
        batch_size=b,
        context_lens=s,
        embed_dim=d,
        num_heads=h,
        num_kv_heads=kv_h,
        block_size=block,
        num_layers=layers,
        dtype_bytes=dtype_bytes,
        shared_prefix_len=shared_prefix,
    )
    gap = analyze_paged_attention_gap(
        paged,
        max_context_len=max_len,
        batch_size=b,
        embed_dim=d,
        num_heads=h,
        num_kv_heads=kv_h,
        num_layers=layers,
        dtype_bytes=dtype_bytes,
    )

    return {
        "architecture": "paged-attention",
        "batch_size": b,
        "context_len": s,
        "embed_dim": d,
        "num_heads": h,
        "num_kv_heads": kv_h,
        "block_size": block,
        "num_layers": layers,
        "dtype": args.dtype,
        "total_blocks": paged.total_blocks,
        "allocated_kv_bytes": paged.allocated_kv_bytes,
        "compulsory_kv_bytes": paged.compulsory_kv_bytes,
        "fragmentation_bytes": paged.fragmentation_bytes,
        "fragmentation_ratio": paged.fragmentation_ratio,
        "contiguous_reserved_bytes": gap.unpaged_contiguous_bytes,
        "memory_saved_bytes": gap.memory_saved_bytes,
        "memory_savings_ratio": gap.memory_savings_ratio,
        "concurrency_multiplier": gap.concurrency_multiplier,
        "flops": paged.total_flops,
        "total_bytes": paged.total_bytes,
        "arithmetic_intensity": paged.arithmetic_intensity,
    }


def run_profile(args: argparse.Namespace) -> int:
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
    else:
        sys.stderr.write(f"Unknown architecture: {args.arch}\n")
        return 1

    # 2. Hardware roofline analysis
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
    bottleneck = "compute" if ai >= hw.ridge_point else "memory"

    res["hardware"] = {
        "name": hw.device_name,
        "peak_flops": hw.peak_flops,
        "memory_bandwidth": hw.memory_bandwidth,
        "ridge_point": hw.ridge_point,
    }
    res["roofline"] = {
        "lower_bound_seconds": lower_bound_sec,
        "lower_bound_ms": lower_bound_sec * 1e3,
        "compute_bound_seconds": compute_sec,
        "bandwidth_bound_seconds": bandwidth_sec,
        "bottleneck": bottleneck,
    }

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print("┌─ Neural Cost Static Profile ───────────────────────────────────────────────")
    print(f"│  Architecture:        {res['architecture']} (dtype: {res['dtype']})")
    if "total_parameters" in res:
        print(f"│  Total Parameters:    {res['total_parameters']:,}")
    if "active_parameters" in res:
        print(f"│  Active Parameters:   {res['active_parameters']:,}")
    print(f"│  Total Compute:       {format_flops(flops)} ({flops / 1e12:.3f} TFLOP)")
    print(f"│  DRAM Traffic:        {format_bytes(total_bytes)} ({total_bytes / 1e9:.3f} GB)")
    if "kv_cache_bytes" in res:
        print(f"│  KV Cache Footprint:  {format_bytes(res['kv_cache_bytes'])}")
    if "activation_bytes" in res:
        print(f"│  Activation Memory:   {format_bytes(res['activation_bytes'])}")
    if "fragmentation_ratio" in res:
        print(
            f"│  KV Fragmentation:    {res['fragmentation_ratio']:.1%} ({format_bytes(res['fragmentation_bytes'])})"
        )
        print(
            f"│  Paged VRAM Saved:    {res['memory_savings_ratio']:.1%} ({res['concurrency_multiplier']:.2f}x concurrency)"
        )
    print(f"│  Arithmetic Intensity:{ai:.2f} FLOP/byte")
    print("├─ Roofline Projection ──────────────────────────────────────────────────────")
    print(f"│  Target Device:       {hw.device_name}")
    print(f"│  Hardware Ridge:      {hw.ridge_point:.1f} FLOP/byte")
    print(f"│  Bottleneck Regime:   {bottleneck.upper()}-BOUND")
    print(f"│  Lower-Bound Latency: {lower_bound_sec * 1e3:.3f} ms")
    print(f"│    Compute bound:     {compute_sec * 1e3:.3f} ms")
    print(f"│    Bandwidth bound:   {bandwidth_sec * 1e3:.3f} ms")
    print("└────────────────────────────────────────────────────────────────────────────")
    return 0
