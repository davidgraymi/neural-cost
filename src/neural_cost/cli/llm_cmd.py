"""Dedicated Generative AI and LLM serving cost estimation subcommands."""

from __future__ import annotations

import argparse
import json

from neural_cost.analysis import (
    analyze_continuous_batch_iteration,
    analyze_kernel_launch_floor,
    analyze_moe_gap,
    analyze_paged_attention_gap,
    analyze_speculative_decoding,
    analyze_ssm_gap,
)
from neural_cost.checkpointing import (
    analyze_activation_checkpointing,
    estimate_activation_checkpointing,
)
from neural_cost.cli.helpers import (
    parse_dtype_bytes,
    resolve_hardware,
)
from neural_cost.estimate import (
    CostEstimate,
    estimate_continuous_batch_iteration,
    estimate_moe,
    estimate_paged_attention,
    estimate_ssm,
)
from neural_cost.hardware import HardwareSpec
from neural_cost.quantization import (
    analyze_quantization_gap,
    estimate_quantized_linear,
)


def register_llm_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    llm_parser = subparsers.add_parser(
        "llm",
        help="Modern LLM serving primitives (MoE, Speculative Decoding, PagedAttention, Continuous Batching, SSM).",
        description="Dedicated analytical models and roofline calculators for LLM inference architectures.",
    )
    llm_sub = llm_parser.add_subparsers(dest="llm_subcommand", required=True)

    # Subcommand: moe
    moe_p = llm_sub.add_parser(
        "moe", help="Mixture-of-Experts routing and memory traffic analyzer."
    )
    moe_p.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size.")
    moe_p.add_argument(
        "--seq-len", "-s", type=int, default=1, help="Sequence length (1 for decode)."
    )
    moe_p.add_argument("--embed-dim", "-d", type=int, default=4096, help="Embedding dimension.")
    moe_p.add_argument(
        "--expert-hidden-dim", type=int, default=14336, help="Expert hidden dimension."
    )
    moe_p.add_argument("--num-experts", "-e", type=int, default=8, help="Total experts.")
    moe_p.add_argument("--top-k", "-k", type=int, default=2, help="Active experts per token.")
    moe_p.add_argument("--expert-type", choices=["swiglu", "mlp"], default="swiglu")
    moe_p.add_argument(
        "--decode", action="store_true", default=True, help="Autoregressive decode step."
    )
    moe_p.add_argument("--peak-flops", type=float, default=None)
    moe_p.add_argument("--memory-bandwidth", type=float, default=None)
    moe_p.add_argument("--json", action="store_true")
    moe_p.set_defaults(func=run_moe)

    # Subcommand: speculative
    spec_p = llm_sub.add_parser(
        "speculative", help="Speculative decoding breakeven and speedup analyzer."
    )
    spec_p.add_argument(
        "--gamma", "-g", type=int, default=4, help="Draft lookahead tokens per step."
    )
    spec_p.add_argument(
        "--acceptance-rate", "-a", type=float, default=0.75, help="Draft acceptance probability."
    )
    spec_p.add_argument(
        "--draft-flops", type=float, default=2e9, help="Draft model single-token FLOPs."
    )
    spec_p.add_argument(
        "--draft-bytes", type=float, default=2e9, help="Draft model DRAM bytes per token."
    )
    spec_p.add_argument(
        "--target-verify-flops",
        type=float,
        default=560e9,
        help="Target model parallel verification FLOPs.",
    )
    spec_p.add_argument(
        "--target-verify-bytes",
        type=float,
        default=140e9,
        help="Target model verification DRAM bytes.",
    )
    spec_p.add_argument(
        "--target-decode-flops",
        type=float,
        default=140e9,
        help="Target model autoregressive single-token FLOPs.",
    )
    spec_p.add_argument(
        "--target-decode-bytes",
        type=float,
        default=140e9,
        help="Target model single-token DRAM bytes.",
    )
    spec_p.add_argument("--peak-flops", type=float, default=None)
    spec_p.add_argument("--memory-bandwidth", type=float, default=None)
    spec_p.add_argument("--json", action="store_true")
    spec_p.set_defaults(func=run_speculative)

    # Subcommand: paged
    paged_p = llm_sub.add_parser(
        "paged", help="PagedAttention KV cache fragmentation and capacity analyzer."
    )
    paged_p.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Number of concurrent requests."
    )
    paged_p.add_argument(
        "--context-len", "-s", type=int, default=2048, help="Mean context length in tokens."
    )
    paged_p.add_argument("--embed-dim", "-d", type=int, default=4096, help="Embedding dimension.")
    paged_p.add_argument("--num-heads", type=int, default=32, help="Query attention heads.")
    paged_p.add_argument("--num-kv-heads", type=int, default=8, help="KV heads for GQA/MQA.")
    paged_p.add_argument("--block-size", type=int, default=16, help="Block size in tokens.")
    paged_p.add_argument("--num-layers", "-l", type=int, default=32, help="Number of layers.")
    paged_p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum context length for naive contiguous allocation.",
    )
    paged_p.add_argument(
        "--shared-prefix-tokens", type=int, default=128, help="Shared prompt prefix tokens."
    )
    paged_p.add_argument("--json", action="store_true")
    paged_p.set_defaults(func=run_paged)

    # Subcommand: continuous
    cb_p = llm_sub.add_parser(
        "continuous", help="Continuous batching mixed iteration cost simulator."
    )
    cb_p.add_argument(
        "--decode-streams", type=int, default=64, help="Number of concurrent decode streams."
    )
    cb_p.add_argument(
        "--decode-context-len", type=int, default=1024, help="Context length per decode stream."
    )
    cb_p.add_argument(
        "--prefill-tokens",
        type=int,
        default=512,
        help="Chunked prefill tokens scheduled in this iteration.",
    )
    cb_p.add_argument("--embed-dim", "-d", type=int, default=4096, help="Embedding dimension.")
    cb_p.add_argument("--num-heads", type=int, default=32, help="Query heads.")
    cb_p.add_argument("--num-kv-heads", type=int, default=8, help="KV heads.")
    cb_p.add_argument("--num-layers", "-l", type=int, default=32, help="Number of layers.")
    cb_p.add_argument("--peak-flops", type=float, default=None)
    cb_p.add_argument("--memory-bandwidth", type=float, default=None)
    cb_p.add_argument("--json", action="store_true")
    cb_p.set_defaults(func=run_continuous)

    # Subcommand: ssm
    ssm_p = llm_sub.add_parser(
        "ssm", help="State Space Model (Mamba/S6/SSD) recurrent cost and KV elimination analyzer."
    )
    ssm_p.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size.")
    ssm_p.add_argument(
        "--seq-len",
        "-s",
        type=int,
        default=4096,
        help="Sequence length (or context history for decode).",
    )
    ssm_p.add_argument("--embed-dim", "-d", type=int, default=4096, help="Embedding dimension.")
    ssm_p.add_argument(
        "--state-dim", "-n", type=int, default=16, help="SSM recurrent state dimension."
    )
    ssm_p.add_argument(
        "--expand-factor", "-e", type=int, default=2, help="Hidden expansion factor."
    )
    ssm_p.add_argument(
        "--conv-kernel-size", type=int, default=4, help="1D depthwise convolution kernel size."
    )
    ssm_p.add_argument("--num-layers", "-l", type=int, default=32, help="Number of layers.")
    ssm_p.add_argument(
        "--num-heads", type=int, default=32, help="Attention heads for Transformer comparison."
    )
    ssm_p.add_argument(
        "--num-kv-heads", type=int, default=8, help="KV heads for Transformer comparison."
    )
    ssm_p.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "int8", "fp8", "int4"],
        help="Data precision (default: fp16).",
    )
    ssm_p.add_argument(
        "--decode",
        action="store_true",
        help="Profile token-by-token autoregressive recurrent decode.",
    )
    ssm_p.add_argument("--peak-flops", type=float, default=None)
    ssm_p.add_argument("--memory-bandwidth", type=float, default=None)
    ssm_p.add_argument("--json", action="store_true")
    ssm_p.set_defaults(func=run_ssm)

    # Subcommand: quant
    quant_p = llm_sub.add_parser(
        "quant", help="Sub-byte quantization and dequantization ALU tax analyzer."
    )
    quant_p.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Token batch size (tokens = batch * seq_len).",
    )
    quant_p.add_argument("--in-features", "-k", type=int, default=4096, help="Input dimension K.")
    quant_p.add_argument("--out-features", "-n", type=int, default=4096, help="Output dimension N.")
    quant_p.add_argument(
        "--quantization",
        "-q",
        type=str,
        default="w4a16_awq",
        choices=["w4a16_awq", "w4a16_gptq", "w8a16", "w8a8_fp8", "w8a8_int8", "w4a4_fp4"],
        help="Quantization scheme / preset (default: w4a16_awq).",
    )
    quant_p.add_argument("--peak-flops", type=float, default=None)
    quant_p.add_argument("--memory-bandwidth", type=float, default=None)
    quant_p.add_argument("--json", action="store_true")
    quant_p.set_defaults(func=run_quant)

    # Subcommand: launch-floor
    launch_p = llm_sub.add_parser(
        "launch-floor",
        help="Host CPU kernel launch overhead and CUDA graph speedup analyzer.",
    )
    launch_p.add_argument(
        "--num-kernels",
        "-k",
        type=int,
        default=640,
        help="Number of accelerator kernels dispatched per step (default: 640).",
    )
    launch_p.add_argument(
        "--arithmetic-floor-ms",
        "-a",
        type=float,
        default=0.5,
        help="Theoretical arithmetic roofline latency floor in ms (default: 0.5 ms).",
    )
    launch_p.add_argument(
        "--launch-overhead-us",
        type=float,
        default=5.0,
        help="Host CPU launch overhead per kernel in microseconds (default: 5.0 us).",
    )
    launch_p.add_argument(
        "--measured-latency-ms",
        type=float,
        default=None,
        help="Optional measured wall-clock step latency in ms.",
    )
    launch_p.add_argument("--json", action="store_true")
    launch_p.set_defaults(func=run_launch_floor)

    # Subcommand: checkpointing
    ckpt_p = llm_sub.add_parser(
        "checkpointing",
        help="Activation checkpointing (rematerialization) memory savings and recompute FLOP tax analyzer.",
    )
    ckpt_p.add_argument(
        "--strategy",
        "-m",
        type=str,
        default="selective",
        choices=["none", "full", "selective"],
        help="Checkpointing strategy (default: selective).",
    )
    ckpt_p.add_argument("--batch-size", "-b", type=int, default=4, help="Micro-batch size.")
    ckpt_p.add_argument("--seq-len", "-s", type=int, default=4096, help="Sequence length.")
    ckpt_p.add_argument("--embed-dim", "-d", type=int, default=4096, help="Hidden dimension.")
    ckpt_p.add_argument("--num-layers", "-l", type=int, default=32, help="Number of layers.")
    ckpt_p.add_argument("--num-heads", type=int, default=32, help="Attention heads.")
    ckpt_p.add_argument("--num-kv-heads", type=int, default=8, help="KV heads for GQA.")
    ckpt_p.add_argument("--intermediate-dim", type=int, default=None, help="FFN dimension.")
    ckpt_p.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Model un-fused quadratic attention matrix memory.",
    )
    ckpt_p.add_argument(
        "--vram-gb", type=float, default=80.0, help="Device VRAM capacity in GB (default: 80 GB)."
    )
    ckpt_p.add_argument("--peak-flops", type=float, default=None)
    ckpt_p.add_argument("--memory-bandwidth", type=float, default=None)
    ckpt_p.add_argument("--json", action="store_true")
    ckpt_p.set_defaults(func=run_checkpointing)


def run_moe(args: argparse.Namespace) -> int:
    hw, _ = resolve_hardware(peak_flops=args.peak_flops, memory_bandwidth=args.memory_bandwidth)
    est = estimate_moe(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_type=args.expert_type,
        is_decode=args.decode,
    )
    gap = analyze_moe_gap(est, hw)

    if args.json:
        data = {
            "total_parameters": est.total_parameters,
            "active_parameters": est.active_parameters,
            "expected_loaded_experts": est.expected_loaded_experts,
            "flops": est.total_flops,
            "total_bytes": est.total_bytes,
            "arithmetic_intensity": est.arithmetic_intensity,
            "bottleneck": gap.bottleneck,
            "lower_bound_ms": gap.lower_bound_seconds * 1e3,
            "findings": list(gap.findings),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(gap.render())
    return 0


def run_speculative(args: argparse.Namespace) -> int:
    hw, _ = resolve_hardware(peak_flops=args.peak_flops, memory_bandwidth=args.memory_bandwidth)
    draft_c = CostEstimate(
        flops=args.draft_flops, read_bytes=args.draft_bytes, write_bytes=2048, operations=1
    )
    verify_c = CostEstimate(
        flops=args.target_verify_flops,
        read_bytes=args.target_verify_bytes,
        write_bytes=16384,
        operations=1,
    )
    target_c = CostEstimate(
        flops=args.target_decode_flops,
        read_bytes=args.target_decode_bytes,
        write_bytes=4096,
        operations=1,
    )

    analysis = analyze_speculative_decoding(
        draft_decode_cost=draft_c,
        target_verify_cost=verify_c,
        target_decode_cost=target_c,
        hardware=hw,
        gamma=args.gamma,
        acceptance_rate=args.acceptance_rate,
    )

    if args.json:
        data = {
            "gamma": analysis.gamma,
            "acceptance_rate": analysis.acceptance_rate,
            "expected_tokens_per_step": analysis.expected_tokens_per_step,
            "effective_latency_ms": analysis.latency_per_token_seconds * 1e3,
            "baseline_latency_ms": analysis.baseline_decode_lower_bound_seconds * 1e3,
            "speedup": analysis.speedup,
            "breakeven_acceptance_rate": analysis.breakeven_acceptance_rate,
            "is_favorable": analysis.is_favorable,
        }
        print(json.dumps(data, indent=2))
        return 0

    print(analysis.render())
    return 0


def run_paged(args: argparse.Namespace) -> int:
    max_len = args.max_model_len or args.context_len
    paged = estimate_paged_attention(
        batch_size=args.batch_size,
        context_lens=args.context_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        num_layers=args.num_layers,
        shared_prefix_len=args.shared_prefix_tokens,
    )
    gap = analyze_paged_attention_gap(
        paged,
        max_context_len=max_len,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_layers=args.num_layers,
    )

    if args.json:
        data = {
            "allocated_kv_bytes": paged.allocated_kv_bytes,
            "compulsory_kv_bytes": paged.compulsory_kv_bytes,
            "fragmentation_bytes": paged.fragmentation_bytes,
            "fragmentation_ratio": paged.fragmentation_ratio,
            "memory_saved_bytes": gap.memory_saved_bytes,
            "memory_savings_ratio": gap.memory_savings_ratio,
            "concurrency_multiplier": gap.concurrency_multiplier,
            "findings": list(gap.findings),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(gap.render())
    return 0


def run_continuous(args: argparse.Namespace) -> int:
    hw, _ = resolve_hardware(peak_flops=args.peak_flops, memory_bandwidth=args.memory_bandwidth)
    dec_lens = [args.decode_context_len] * args.decode_streams
    prefill_chunks = [args.prefill_tokens] if args.prefill_tokens > 0 else []

    iteration = estimate_continuous_batch_iteration(
        decode_context_lens=dec_lens,
        prefill_chunk_lens=prefill_chunks,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_layers=args.num_layers,
        hardware=hw,
    )
    gap = analyze_continuous_batch_iteration(iteration, hw)

    if args.json:
        data = {
            "total_tokens": iteration.total_tokens,
            "decode_tokens": iteration.decode_tokens,
            "prefill_tokens": iteration.prefill_tokens,
            "arithmetic_intensity": iteration.arithmetic_intensity,
            "lower_bound_ms": gap.lower_bound_seconds * 1e3,
            "bottleneck": gap.bottleneck,
            "optimal_prefill_tokens_to_saturate": gap.optimal_prefill_tokens_to_saturate,
            "findings": list(gap.findings),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(gap.render())
    return 0


def run_ssm(args: argparse.Namespace) -> int:
    hw, _ = resolve_hardware(peak_flops=args.peak_flops, memory_bandwidth=args.memory_bandwidth)
    dtype_bytes = parse_dtype_bytes(args.dtype)
    est = estimate_ssm(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        state_dim=args.state_dim,
        expand_factor=args.expand_factor,
        conv_kernel_size=args.conv_kernel_size,
        num_layers=args.num_layers,
        dtype_bytes=dtype_bytes,
        is_decode=args.decode,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
    )
    gap = analyze_ssm_gap(est, hw)

    if args.json:
        data = {
            "total_parameters": est.total_parameters,
            "parameter_bytes": est.parameter_bytes,
            "state_bytes": est.state_bytes,
            "flops": est.total_flops,
            "total_bytes": est.total_bytes,
            "arithmetic_intensity": est.arithmetic_intensity,
            "is_decode": est.is_decode,
            "equivalent_transformer_kv_bytes": est.equivalent_transformer_kv_bytes,
            "memory_savings_ratio_vs_transformer": est.memory_savings_ratio_vs_transformer,
            "bottleneck": gap.bottleneck,
            "lower_bound_ms": gap.lower_bound_seconds * 1e3,
            "findings": list(gap.findings),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(gap.render())
    return 0


def run_quant(args: argparse.Namespace) -> int:
    hw, _ = resolve_hardware(peak_flops=args.peak_flops, memory_bandwidth=args.memory_bandwidth)
    est = estimate_quantized_linear(
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features,
        quantization=args.quantization,
    )
    gap = analyze_quantization_gap(est, hw)

    if args.json:
        data = {
            "quantization": est.quantization.name,
            "batch_size": est.batch_size,
            "in_features": est.in_features,
            "out_features": est.out_features,
            "unquantized_weight_bytes": est.unquantized_weight_bytes,
            "quantized_weight_bytes": est.quantized_weight_bytes,
            "scale_zero_bytes": est.scale_zero_bytes,
            "total_weight_bytes": est.total_weight_bytes,
            "weight_compression_ratio": est.weight_compression_ratio,
            "input_activation_bytes": est.input_activation_bytes,
            "output_activation_bytes": est.output_activation_bytes,
            "total_memory_bytes": est.total_memory_bytes,
            "gemm_flops": est.gemm_flops,
            "dequant_unpack_flops": est.dequant_unpack_flops,
            "total_flops": est.total_flops,
            "arithmetic_intensity": est.arithmetic_intensity,
            "unquantized_latency_ms": gap.unquantized_latency_seconds * 1e3,
            "quantized_latency_ms": gap.quantized_latency_seconds * 1e3,
            "speedup": gap.speedup,
            "unquantized_bottleneck": gap.unquantized_bottleneck,
            "quantized_bottleneck": gap.quantized_bottleneck,
            "dequant_overhead_ratio": gap.dequant_overhead_ratio,
            "findings": list(gap.findings),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(gap.render())
    return 0


def run_launch_floor(args: argparse.Namespace) -> int:
    launch_overhead_s = args.launch_overhead_us * 1e-6
    arith_floor_s = args.arithmetic_floor_ms * 1e-3
    measured_s = args.measured_latency_ms * 1e-3 if args.measured_latency_ms is not None else None

    analysis = analyze_kernel_launch_floor(
        num_kernels=args.num_kernels,
        model_arithmetic_floor_seconds=arith_floor_s,
        launch_overhead_seconds=launch_overhead_s,
        measured_latency_seconds=measured_s,
    )

    if args.json:
        data = {
            "num_kernels": analysis.num_kernels,
            "launch_overhead_us": analysis.launch_overhead_seconds * 1e6,
            "total_launch_floor_ms": analysis.total_launch_floor_seconds * 1e3,
            "model_arithmetic_floor_ms": analysis.model_arithmetic_floor_seconds * 1e3,
            "effective_lower_bound_ms": analysis.effective_lower_bound_seconds * 1e3,
            "is_launch_bound": analysis.is_launch_bound,
            "launch_to_arithmetic_ratio": analysis.launch_to_arithmetic_ratio,
            "cuda_graph_potential_speedup": analysis.cuda_graph_potential_speedup,
            "measured_latency_ms": (
                analysis.measured_latency_seconds * 1e3
                if analysis.measured_latency_seconds is not None
                else None
            ),
            "findings": list(analysis.findings),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(analysis.render())
    return 0


def run_checkpointing(args: argparse.Namespace) -> int:
    hw, _ = resolve_hardware(peak_flops=args.peak_flops, memory_bandwidth=args.memory_bandwidth)
    if args.vram_gb is not None:
        hw = HardwareSpec(
            name=hw.name,
            peak_flops=hw.peak_flops,
            memory_bandwidth=hw.memory_bandwidth,
            memory_capacity=int(args.vram_gb * 1e9),
            caches=hw.caches,
        )

    est = estimate_activation_checkpointing(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_dim=args.intermediate_dim,
        num_kv_heads=args.num_kv_heads,
        strategy=args.strategy,
        is_flash_attention=not args.no_flash_attention,
    )
    analysis = analyze_activation_checkpointing(est, hw)

    if args.json:
        data = {
            "strategy": est.strategy.value,
            "batch_size": est.batch_size,
            "seq_len": est.seq_len,
            "embed_dim": est.embed_dim,
            "num_layers": est.num_layers,
            "uncheckpointed_activation_bytes": est.uncheckpointed_activation_bytes,
            "checkpointed_activation_bytes": est.checkpointed_activation_bytes,
            "activation_memory_saved_bytes": est.activation_memory_saved_bytes,
            "activation_savings_ratio": est.activation_savings_ratio,
            "static_model_bytes": analysis.static_model_bytes,
            "total_training_memory_bytes": analysis.total_training_memory_bytes,
            "uncheckpointed_training_memory_bytes": analysis.uncheckpointed_training_memory_bytes,
            "is_oom": analysis.is_oom,
            "uncheckpointed_is_oom": analysis.uncheckpointed_is_oom,
            "memory_capacity_utilization": analysis.memory_capacity_utilization,
            "forward_flops": est.forward_flops,
            "backward_flops": est.backward_flops,
            "recompute_flops": est.recompute_flops,
            "compute_overhead_ratio": est.compute_overhead_ratio,
            "total_step_ms": analysis.total_step_seconds * 1e3,
            "throughput_tokens_per_second": analysis.throughput_tokens_per_second,
            "max_trainable_seq_len": analysis.max_trainable_seq_len,
            "findings": list(analysis.findings),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(analysis.render())
    return 0
