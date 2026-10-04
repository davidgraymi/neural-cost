"""LLM Prefill vs. Decode Phase Discrepancy & KV Cache Benchmarking.

Evaluates operational regimes of Large Language Models:
- Prefill phase: compute-bound parallel processing of prompt context (reporting TTFT and GFLOP/s).
- Decode phase: memory-bandwidth bound autoregressive token generation with KV-cache retrieval
  (reporting tokens/second, memory bandwidth utilization, and gap analysis).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure repo root and src/ are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_cost import HardwareSpec
from neural_cost.hardware_detect import detect_hardware


@dataclass
class PrefillMetrics:
    """Performance metrics for prompt prefill phase."""

    prompt_len: int
    batch_size: int
    latency_ms: float
    ttft_ms: float
    gflops: float
    achieved_gflops: float
    throughput_tokens_s: float
    memory_traffic_bytes: int


@dataclass
class DecodeMetrics:
    """Performance metrics for autoregressive decoding phase."""

    prompt_len: int
    gen_tokens: int
    batch_size: int
    latency_ms: float
    tokens_per_sec: float
    throughput_tokens_s: float
    achieved_gbw: float
    memory_bw_util: float
    kv_cache_bytes: int


def estimate_llm_decode_step_bytes(
    batch: int,
    seq_len: int,
    embed_dim: int,
    num_layers: int = 12,
    dtype_bytes: int = 2,
) -> int:
    """Model memory traffic per autoregressive decode token step.

    T_step = Weights read (12 * D^2 * num_layers * dtype_bytes)
           + KV cache history read (2 * B * L * D * num_layers * dtype_bytes)
           + New KV token write (2 * B * 1 * D * num_layers * dtype_bytes).
    """
    weight_bytes = 12 * (embed_dim**2) * num_layers * dtype_bytes
    kv_read_bytes = 2 * batch * seq_len * embed_dim * num_layers * dtype_bytes
    kv_write_bytes = 2 * batch * 1 * embed_dim * num_layers * dtype_bytes
    return weight_bytes + kv_read_bytes + kv_write_bytes


def benchmark_llm_prefill(
    model: Any = None,
    prompt_len: int = 128,
    batch_size: int = 1,
    embed_dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 4,
    warmup: int = 5,
    repeats: int = 15,
) -> PrefillMetrics:
    """Benchmark LLM prefill phase across prompt context."""
    import torch

    head_dim = embed_dim // num_heads

    # Synthetic or provided attention / transformer block
    q = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    k = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    v = torch.randn(batch_size, num_heads, prompt_len, head_dim)

    # Warmup
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Timing
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    elapsed_ns = time.perf_counter_ns() - t0
    latency_ms = (elapsed_ns / repeats) / 1e6

    # FLOPs: 4 * B * L^2 * D + projections for each layer
    attn_flops = 4 * batch_size * (prompt_len**2) * embed_dim * num_layers
    proj_flops = 8 * batch_size * prompt_len * (embed_dim**2) * num_layers
    total_flops = attn_flops + proj_flops
    achieved_gflops = (total_flops / (latency_ms / 1e3)) / 1e9 if latency_ms > 0 else 0.0

    throughput_tokens_s = (batch_size * prompt_len) / (latency_ms / 1e3) if latency_ms > 0 else 0.0
    mem_traffic = 2 * batch_size * prompt_len * embed_dim * 2 * num_layers

    return PrefillMetrics(
        prompt_len=prompt_len,
        batch_size=batch_size,
        latency_ms=latency_ms,
        ttft_ms=latency_ms,
        gflops=achieved_gflops,
        achieved_gflops=achieved_gflops,
        throughput_tokens_s=throughput_tokens_s,
        memory_traffic_bytes=mem_traffic,
    )


def benchmark_llm_decode(
    model: Any = None,
    prompt_len: int = 128,
    gen_tokens: int = 16,
    batch_size: int = 1,
    embed_dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 4,
    warmup: int = 3,
    repeats: int = 5,
    hardware: HardwareSpec | None = None,
) -> DecodeMetrics:
    """Benchmark token-by-token autoregressive decoding with KV cache."""
    import torch

    head_dim = embed_dim // num_heads

    # Warmup KV cache and single token step
    q = torch.randn(batch_size, num_heads, 1, head_dim)
    k_cache = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    v_cache = torch.randn(batch_size, num_heads, prompt_len, head_dim)

    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k_cache, v_cache)

    total_tokens_timed = 0
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        curr_len = prompt_len
        for _ in range(gen_tokens):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k_cache, v_cache)
            curr_len += 1
            total_tokens_timed += 1
    elapsed_ns = time.perf_counter_ns() - t0
    total_time_s = elapsed_ns / 1e9
    latency_ms = (total_time_s / (repeats * gen_tokens)) * 1e3

    tokens_per_sec = total_tokens_timed / total_time_s if total_time_s > 0 else 0.0

    # KV cache bytes at starting prompt length
    kv_cache_bytes = 2 * batch_size * prompt_len * embed_dim * 2 * num_layers
    step_traffic = estimate_llm_decode_step_bytes(
        batch_size, prompt_len, embed_dim, num_layers=num_layers, dtype_bytes=2
    )
    achieved_gbw = (step_traffic / (latency_ms / 1e3)) / 1e9 if latency_ms > 0 else 0.0

    if hardware is None:
        hardware, _ = detect_hardware()

    bw_util = (achieved_gbw * 1e9) / hardware.memory_bandwidth if hardware.memory_bandwidth > 0 else 0.0

    return DecodeMetrics(
        prompt_len=prompt_len,
        gen_tokens=gen_tokens,
        batch_size=batch_size,
        latency_ms=latency_ms,
        tokens_per_sec=tokens_per_sec,
        throughput_tokens_s=tokens_per_sec,
        achieved_gbw=achieved_gbw,
        memory_bw_util=bw_util,
        kv_cache_bytes=kv_cache_bytes,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for LLM benchmarking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=[128, 512, 2048],
        help="Prompt context lengths to benchmark",
    )
    parser.add_argument("--gen-tokens", type=int, default=16, help="Tokens to generate in decode")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--embed-dim", type=int, default=1024, help="Embedding hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--quick", action="store_true", help="Quick run")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "llm_benchmark_data.json",
        help="Path to save JSON results (default: benchmarks/results/llm_benchmark_data.json)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    hardware, _ = detect_hardware()
    lengths = [128, 512] if args.quick else args.prompt_lengths

    print(f"Hardware: {hardware.name} ({hardware.peak_flops/1e12:.2f} TFLOP/s, {hardware.memory_bandwidth/1e9:.1f} GB/s)")
    print("=" * 80)
    print(f"{'Phase':<10} {'Prompt':<8} {'Batch':<6} {'Latency':<12} {'Throughput':<18} {'Traffic / Util':<20}")
    print("-" * 80)

    prefills = []
    decodes = []

    for p_len in lengths:
        p_met = benchmark_llm_prefill(prompt_len=p_len, batch_size=args.batch_size, embed_dim=args.embed_dim, num_heads=args.num_heads)
        prefills.append(p_met)
        print(f"{'Prefill':<10} {p_met.prompt_len:<8} {p_met.batch_size:<6} {p_met.ttft_ms:8.2f} ms  {p_met.achieved_gflops:8.1f} GFLOP/s   {p_met.throughput_tokens_s:8.0f} tok/s")

        d_met = benchmark_llm_decode(prompt_len=p_len, gen_tokens=args.gen_tokens, batch_size=args.batch_size, embed_dim=args.embed_dim, num_heads=args.num_heads, hardware=hardware)
        decodes.append(d_met)
        print(f"{'Decode':<10} {d_met.prompt_len:<8} {d_met.batch_size:<6} {d_met.latency_ms:8.2f} ms  {d_met.tokens_per_sec:8.1f} tok/s      {d_met.achieved_gbw:6.1f} GB/s ({d_met.memory_bw_util:.1%})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hardware": {
                "name": hardware.name,
                "peak_flops": hardware.peak_flops,
                "memory_bandwidth": hardware.memory_bandwidth,
            },
            "prefill": [asdict(p) for p in prefills],
            "decode": [asdict(d) for d in decodes],
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved {len(prefills) + len(decodes)} LLM records → {args.output}")


if __name__ == "__main__":
    main()
