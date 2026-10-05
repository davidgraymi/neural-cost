"""Distributed 3D parallelism and interconnect roofline subcommand."""

from __future__ import annotations

import argparse
import json

from neural_cost.analysis import analyze_distributed_gap
from neural_cost.cli.helpers import (
    format_bytes,
    parse_dtype_bytes,
)
from neural_cost.estimate import estimate_parallelism
from neural_cost.hardware import (
    INTERCONNECT_PRESETS,
    ClusterTopology,
    HardwareSpec,
    get_interconnect_preset,
)


def register_distributed_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "distributed",
        help="Model 3D parallelism (TP, PP, DP/FSDP), communication volumes, and interconnect rooflines.",
        description="Calculate TP/PP/DP network volume, pipeline bubble overhead, memory sharding, and cluster MFU.",
    )
    _add_distributed_arguments(parser)
    parser.set_defaults(func=run_distributed)


def _add_distributed_arguments(parser: argparse.ArgumentParser) -> None:
    # Model parameters
    parser.add_argument(
        "--parameters",
        "-p",
        type=float,
        default=7e9,
        help="Total model parameter count (e.g. 7e9 for 7B, 70e9 for 70B, default: 7B).",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Global batch size (default: 32)."
    )
    parser.add_argument(
        "--seq-len", "-s", type=int, default=4096, help="Sequence length in tokens (default: 4096)."
    )
    parser.add_argument(
        "--embed-dim", "-d", type=int, default=4096, help="Hidden dimension (default: 4096)."
    )
    parser.add_argument(
        "--num-layers", "-l", type=int, default=32, help="Total transformer layers (default: 32)."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16", "int8", "fp8"],
        help="Data precision (default: bf16).",
    )

    # 3D Parallelism configuration
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallelism degree (default: 1).")
    parser.add_argument(
        "--pp", type=int, default=1, help="Pipeline Parallelism degree (default: 1)."
    )
    parser.add_argument(
        "--dp", type=int, default=1, help="Data Parallelism / ZeRO degree (default: 1)."
    )
    parser.add_argument(
        "--microbatches",
        "-m",
        type=int,
        default=4,
        help="Microbatches per pipeline iteration for 1F1B schedule (default: 4).",
    )
    parser.add_argument(
        "--dp-mode",
        choices=["ddp", "zero1", "zero2", "zero3_fsdp"],
        default="zero3_fsdp",
        help="Data parallelism / ZeRO strategy (default: zero3_fsdp).",
    )
    parser.add_argument(
        "--no-sp",
        action="store_true",
        help="Disable Sequence Parallelism (Megatron-LM SP).",
    )
    parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="Disable full activation checkpointing/recomputation.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Evaluate inference forward step rather than training iteration.",
    )

    # Cluster topology and interconnects
    parser.add_argument(
        "--num-nodes", type=int, default=1, help="Number of physical server nodes (default: 1)."
    )
    parser.add_argument(
        "--devices-per-node",
        type=int,
        default=8,
        help="Accelerators per server node (default: 8).",
    )
    parser.add_argument(
        "--intra-node",
        type=str,
        default="nvlink4",
        help=f"Intra-node interconnect preset (default: nvlink4). Choices: {', '.join(INTERCONNECT_PRESETS.keys())}",
    )
    parser.add_argument(
        "--inter-node",
        type=str,
        default="infiniband_ndr",
        help=f"Inter-node network preset (default: infiniband_ndr). Choices: {', '.join(INTERCONNECT_PRESETS.keys())}",
    )
    parser.add_argument(
        "--device-peak-flops",
        type=float,
        default=989e12,
        help="Per-device peak compute FLOP/s (default: 989 TFLOP/s for H100 SXM5).",
    )
    parser.add_argument(
        "--device-memory-gb",
        type=float,
        default=80.0,
        help="Per-device HBM/VRAM capacity in GB (default: 80 GB).",
    )
    parser.add_argument(
        "--overlap-efficiency",
        type=float,
        default=0.85,
        help="Communication-computation overlap efficiency (0.0 to 1.0, default: 0.85).",
    )
    parser.add_argument("--json", action="store_true", help="Output distributed analysis as JSON.")


def run_distributed(args: argparse.Namespace) -> int:
    dtype_bytes = parse_dtype_bytes(args.dtype)
    is_training = not args.inference
    seq_parallel = not args.no_sp
    act_recompute = not args.no_recompute

    # 1. Resolve interconnects and cluster topology
    try:
        intra_spec = get_interconnect_preset(args.intra_node)
    except ValueError:
        intra_spec = None

    try:
        inter_spec = get_interconnect_preset(args.inter_node)
    except ValueError:
        inter_spec = None

    device = HardwareSpec(
        name="Accelerator",
        peak_flops=args.device_peak_flops,
        memory_bandwidth=3.35e12,  # H100 HBM3 default
        memory_capacity=int(args.device_memory_gb * 1024**3),
    )

    topology = ClusterTopology(
        device=device,
        num_nodes=args.num_nodes,
        devices_per_node=args.devices_per_node,
        intra_node=intra_spec,
        inter_node=inter_spec,
    )

    # If user specified tp/pp/dp, ensure degrees match cluster or adjust DP
    tp = args.tp
    pp = args.pp
    dp = args.dp
    if dp == 1 and (tp * pp) < topology.total_devices:
        dp = max(1, topology.total_devices // (tp * pp))

    # 2. Estimate 3D parallelism cost
    cost = estimate_parallelism(
        total_parameters=int(args.parameters),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        tp_degree=tp,
        pp_degree=pp,
        dp_degree=dp,
        num_microbatches=args.microbatches,
        devices_per_node=args.devices_per_node,
        dp_mode=args.dp_mode,
        sequence_parallel=seq_parallel,
        activation_checkpointing=act_recompute,
        is_training=is_training,
        dtype_bytes=dtype_bytes,
    )

    # 3. Analyze distributed roofline and MFU
    analysis = analyze_distributed_gap(
        cost=cost,
        topology=topology,
        overlap_efficiency=args.overlap_efficiency,
    )

    if args.json:
        data = {
            "cluster": {
                "num_nodes": topology.num_nodes,
                "devices_per_node": topology.devices_per_node,
                "total_devices": topology.total_devices,
                "total_peak_tflops": topology.total_peak_flops / 1e12,
                "intra_node_interconnect": intra_spec.name if intra_spec else None,
                "inter_node_interconnect": inter_spec.name if inter_spec else None,
            },
            "parallelism": {
                "tp": cost.tp_degree,
                "pp": cost.pp_degree,
                "dp": cost.dp_degree,
                "dp_mode": cost.dp_mode,
                "num_microbatches": cost.num_microbatches,
                "pipeline_bubble_fraction": cost.pp_bubble_fraction,
            },
            "communication_bytes": {
                "tp_bytes": cost.tp_bytes_per_step,
                "pp_bytes": cost.pp_bytes_per_step,
                "dp_bytes": cost.dp_bytes_per_step,
                "intra_node_bytes": cost.intra_node_comm_bytes,
                "inter_node_bytes": cost.inter_node_comm_bytes,
                "total_comm_bytes": cost.total_comm_bytes_per_step,
            },
            "timing_ms": {
                "compute_ms": analysis.compute_time_seconds * 1e3,
                "bubble_ms": analysis.bubble_time_seconds * 1e3,
                "intra_comm_ms": analysis.intra_node_comm_time_seconds * 1e3,
                "inter_comm_ms": analysis.inter_node_comm_time_seconds * 1e3,
                "step_ms": analysis.step_time_seconds * 1e3,
            },
            "performance": {
                "model_flops_utilization": analysis.model_flops_utilization,
                "hardware_flops_utilization": analysis.hardware_flops_utilization,
                "samples_per_second": analysis.samples_per_second,
                "tokens_per_second": analysis.tokens_per_second,
                "bottleneck": analysis.bottleneck,
                "memory_fit": analysis.memory_fit,
                "per_device_memory_gb": cost.per_device_total_memory_bytes / 1e9,
            },
            "findings": list(analysis.findings),
        }
        print(json.dumps(data, indent=2))
        return 0

    print("┌─ Distributed 3D Parallelism Roofline ───────────────────────────────────────")
    print(
        f"│  Cluster Topology:    {topology.num_nodes} node(s) x {topology.devices_per_node} devices = {topology.total_devices} GPUs"
    )
    if intra_spec:
        print(f"│  Intra-Node Fabric:   {intra_spec.name} ({intra_spec.bandwidth / 1e9:.0f} GB/s)")
    if inter_spec:
        print(f"│  Inter-Node Fabric:   {inter_spec.name} ({inter_spec.bandwidth / 1e9:.0f} GB/s)")
    print(
        f"│  Parallelism Scheme:  TP={cost.tp_degree} x PP={cost.pp_degree} x DP={cost.dp_degree} ({cost.dp_mode.upper()})"
    )
    print("├─ Performance & Utilization ────────────────────────────────────────────────")
    print(
        f"│  Model FLOPs Util:    {analysis.model_flops_utilization:.1%} MFU (Hardware: {analysis.hardware_flops_utilization:.1%} HFU)"
    )
    print(f"│  Step Time:           {analysis.step_time_seconds * 1e3:.2f} ms")
    print(
        f"│  Training Throughput: {analysis.tokens_per_second:,.0f} tokens/s ({analysis.samples_per_second:.2f} seq/s)"
    )
    print(f"│  Pipeline Bubble:     {cost.pp_bubble_fraction:.1%} idle time")
    print(f"│  Bottleneck Regime:   {analysis.bottleneck.upper()}")
    print("├─ Network Traffic Breakdown ────────────────────────────────────────────────")
    print(
        f"│  Intra-Node Comm:     {format_bytes(cost.intra_node_comm_bytes)} ({analysis.intra_node_comm_time_seconds * 1e3:.2f} ms)"
    )
    print(
        f"│  Inter-Node Comm:     {format_bytes(cost.inter_node_comm_bytes)} ({analysis.inter_node_comm_time_seconds * 1e3:.2f} ms)"
    )
    print(f"│  Comm/Compute Overlap:{analysis.overlap_efficiency:.0%} hidden behind compute")
    print("├─ Memory Allocation per GPU ────────────────────────────────────────────────")
    print(f"│  Parameters:          {format_bytes(cost.per_device_param_bytes)}")
    print(f"│  Optimizer State:     {format_bytes(cost.per_device_optimizer_bytes)}")
    print(f"│  Gradients:           {format_bytes(cost.per_device_grad_bytes)}")
    print(f"│  Activations:         {format_bytes(cost.per_device_activation_bytes)}")
    print(
        f"│  Total GPU Memory:    {format_bytes(cost.per_device_total_memory_bytes)} / {args.device_memory_gb:.1f} GB ({'✓ FIT' if analysis.memory_fit else '✗ OOM'})"
    )
    print("└────────────────────────────────────────────────────────────────────────────")
    for f in analysis.findings:
        print(f"• {f}")
    return 0
