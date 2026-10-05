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

from neural_cost import (
    HardwareSpec,
    detect_hardware,
    estimate_moe,
    estimate_speculative_decoding,
)


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


@dataclass
class MoEDecodeMetrics:
    """Performance metrics for Mixture-of-Experts decode step."""

    batch_size: int
    num_experts: int
    top_k: int
    embed_dim: int
    expert_hidden_dim: int
    latency_ms: float
    tokens_per_sec: float
    expected_loaded_experts: float
    total_params: int
    active_params: int
    achieved_gbw: float
    memory_bw_util: float
    arithmetic_intensity: float


def benchmark_moe_decode(
    batch_size: int = 1,
    embed_dim: int = 1024,
    expert_hidden_dim: int = 2048,
    num_experts: int = 8,
    top_k: int = 2,
    warmup: int = 3,
    repeats: int = 10,
    hardware: HardwareSpec | None = None,
) -> MoEDecodeMetrics:
    """Benchmark a single token decode step across an MoE layer."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SyntheticMoE(nn.Module):
        def __init__(self, d: int, h: int, e: int, k: int):
            super().__init__()
            self.router = nn.Linear(d, e, bias=False)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, h, bias=False),
                    nn.SiLU(),
                    nn.Linear(h, d, bias=False),
                )
                for _ in range(e)
            ])
            self.k = k

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, s, d = x.shape
            x_flat = x.view(-1, d)
            logits = self.router(x_flat)
            weights, indices = torch.topk(F.softmax(logits, dim=-1), self.k, dim=-1)
            out = torch.zeros_like(x_flat)
            for i in range(self.k):
                expert_idx = indices[:, i]
                for exp_id in torch.unique(expert_idx):
                    mask = expert_idx == exp_id
                    if mask.any():
                        out[mask] += weights[mask, i : i + 1] * self.experts[exp_id](x_flat[mask])
            return out.view(b, s, d)

    model = SyntheticMoE(embed_dim, expert_hidden_dim, num_experts, top_k)
    x = torch.randn(batch_size, 1, embed_dim)

    for _ in range(warmup):
        _ = model(x)

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        _ = model(x)
    elapsed_ns = time.perf_counter_ns() - t0
    latency_ms = (elapsed_ns / repeats) / 1e6

    tokens_per_sec = (batch_size / (latency_ms / 1e3)) if latency_ms > 0 else 0.0

    moe_est = estimate_moe(
        batch_size=batch_size,
        seq_len=1,
        embed_dim=embed_dim,
        expert_hidden_dim=expert_hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        dtype_bytes=4,
        is_decode=True,
    )

    achieved_gbw = (moe_est.total_bytes / (latency_ms / 1e3)) / 1e9 if latency_ms > 0 else 0.0
    if hardware is None:
        hardware, _ = detect_hardware()
    bw_util = (
        (achieved_gbw * 1e9) / hardware.memory_bandwidth
        if hardware.memory_bandwidth > 0
        else 0.0
    )

    return MoEDecodeMetrics(
        batch_size=batch_size,
        num_experts=num_experts,
        top_k=top_k,
        embed_dim=embed_dim,
        expert_hidden_dim=expert_hidden_dim,
        latency_ms=latency_ms,
        tokens_per_sec=tokens_per_sec,
        expected_loaded_experts=moe_est.expected_loaded_experts,
        total_params=moe_est.total_parameters,
        active_params=moe_est.active_parameters,
        achieved_gbw=achieved_gbw,
        memory_bw_util=bw_util,
        arithmetic_intensity=moe_est.arithmetic_intensity,
    )


@dataclass
class SpeculativeDecodeMetrics:
    """Performance metrics for speculative decoding execution."""

    gamma: int
    acceptance_rate: float
    expected_tokens_per_step: float
    draft_latency_ms: float
    verify_latency_ms: float
    spec_step_latency_ms: float
    effective_ms_per_token: float
    baseline_ms_per_token: float
    speedup: float
    breakeven_acceptance_rate: float


def benchmark_speculative_decoding(
    gamma: int = 4,
    acceptance_rate: float = 0.7,
    batch_size: int = 1,
    target_embed_dim: int = 1024,
    draft_embed_dim: int = 512,
    prompt_len: int = 128,
    warmup: int = 3,
    repeats: int = 10,
) -> SpeculativeDecodeMetrics:
    """Benchmark empirical latency of speculative decode cycle vs baseline autoregressive decode."""
    import torch

    draft_head_dim = draft_embed_dim // 8
    q_draft = torch.randn(batch_size, 8, 1, draft_head_dim)
    k_draft = torch.randn(batch_size, 8, prompt_len, draft_head_dim)
    v_draft = torch.randn(batch_size, 8, prompt_len, draft_head_dim)

    target_head_dim = target_embed_dim // 8
    q_verify = torch.randn(batch_size, 8, gamma + 1, target_head_dim)
    k_verify = torch.randn(batch_size, 8, prompt_len + gamma + 1, target_head_dim)
    v_verify = torch.randn(batch_size, 8, prompt_len + gamma + 1, target_head_dim)

    q_target = torch.randn(batch_size, 8, 1, target_head_dim)
    k_target = torch.randn(batch_size, 8, prompt_len, target_head_dim)
    v_target = torch.randn(batch_size, 8, prompt_len, target_head_dim)

    for _ in range(warmup):
        for _ in range(gamma):
            _ = torch.nn.functional.scaled_dot_product_attention(q_draft, k_draft, v_draft)
        _ = torch.nn.functional.scaled_dot_product_attention(q_verify, k_verify, v_verify)
        _ = torch.nn.functional.scaled_dot_product_attention(q_target, k_target, v_target)

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        for _ in range(gamma):
            _ = torch.nn.functional.scaled_dot_product_attention(q_draft, k_draft, v_draft)
    draft_ms = ((time.perf_counter_ns() - t0) / repeats) / 1e6

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        _ = torch.nn.functional.scaled_dot_product_attention(q_verify, k_verify, v_verify)
    verify_ms = ((time.perf_counter_ns() - t0) / repeats) / 1e6

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        _ = torch.nn.functional.scaled_dot_product_attention(q_target, k_target, v_target)
    baseline_ms = ((time.perf_counter_ns() - t0) / repeats) / 1e6

    spec_step_ms = draft_ms + verify_ms

    from neural_cost.estimate import _expected_speculative_tokens, _find_breakeven_alpha

    e_tokens = _expected_speculative_tokens(gamma, acceptance_rate)
    eff_ms_per_token = spec_step_ms / e_tokens if e_tokens > 0 else float("inf")
    speedup = baseline_ms / eff_ms_per_token if eff_ms_per_token > 0 else 0.0

    ratio = spec_step_ms / baseline_ms if baseline_ms > 0 else float("inf")
    breakeven_alpha = _find_breakeven_alpha(gamma, ratio)

    return SpeculativeDecodeMetrics(
        gamma=gamma,
        acceptance_rate=acceptance_rate,
        expected_tokens_per_step=e_tokens,
        draft_latency_ms=draft_ms,
        verify_latency_ms=verify_ms,
        spec_step_latency_ms=spec_step_ms,
        effective_ms_per_token=eff_ms_per_token,
        baseline_ms_per_token=baseline_ms,
        speedup=speedup,
        breakeven_acceptance_rate=breakeven_alpha,
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
