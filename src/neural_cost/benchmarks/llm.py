"""LLM benchmark suite for prefill, decode, MoE, speculative decoding, and PagedAttention."""

from __future__ import annotations

import time
from dataclasses import dataclass

from neural_cost.hardware import HardwareSpec
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


@dataclass
class MoEDecodeMetrics:
    """Performance metrics for Mixture-of-Experts decode step."""

    batch_size: int
    num_experts: int
    top_k: int
    latency_ms: float
    achieved_tflops: float
    achieved_gbw: float
    memory_bw_util: float
    compute_util: float
    expected_loaded_experts: float


@dataclass
class SpeculativeDecodeMetrics:
    """Performance metrics for Speculative Decoding execution."""

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


@dataclass
class PagedAttentionMetrics:
    """Benchmark metrics for PagedAttention execution."""

    batch_size: int
    context_len: int
    block_size: int
    allocated_kv_bytes: int
    compulsory_kv_bytes: int
    fragmentation_bytes: int
    fragmentation_ratio: float
    total_blocks: int
    latency_ms: float
    achieved_gbw: float
    memory_bw_util: float


def benchmark_prefill(
    prompt_len: int = 1024,
    batch_size: int = 1,
    embed_dim: int = 4096,
    num_heads: int = 32,
    num_layers: int = 32,
    warmup: int = 3,
    repeats: int = 10,
    hardware: HardwareSpec | None = None,
) -> PrefillMetrics:
    import torch

    head_dim = embed_dim // num_heads
    q = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    k = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    v = torch.randn(batch_size, num_heads, prompt_len, head_dim)

    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    t1 = time.perf_counter_ns()

    elapsed_ms = ((t1 - t0) / repeats) / 1e6
    ttft_ms = elapsed_ms * num_layers

    # Prefill FLOPs: QK^T + AV = 4 * B * H * S^2 * D
    core_flops = 4 * batch_size * num_heads * (prompt_len**2) * head_dim
    # Projections per layer: Q, K, V, O = 8 * B * S * embed_dim^2
    proj_flops = 8 * batch_size * prompt_len * (embed_dim**2)
    layer_flops = core_flops + proj_flops
    total_flops = layer_flops * num_layers

    achieved_gflops = (total_flops / (ttft_ms / 1e3)) / 1e9 if ttft_ms > 0 else 0.0
    throughput = (batch_size * prompt_len) / (ttft_ms / 1e3) if ttft_ms > 0 else 0.0

    # Memory traffic: model weights loaded once + activation reads/writes
    weight_bytes = 4 * (embed_dim**2) * 2 * num_layers  # Q, K, V, O in FP16
    kv_cache_bytes = 2 * batch_size * num_heads * head_dim * prompt_len * 2 * num_layers
    mem_traffic = weight_bytes + kv_cache_bytes

    return PrefillMetrics(
        prompt_len=prompt_len,
        batch_size=batch_size,
        latency_ms=elapsed_ms,
        ttft_ms=ttft_ms,
        gflops=total_flops / 1e9,
        achieved_gflops=achieved_gflops,
        throughput_tokens_s=throughput,
        memory_traffic_bytes=mem_traffic,
    )


def benchmark_decode(
    prompt_len: int = 1024,
    gen_tokens: int = 16,
    batch_size: int = 1,
    embed_dim: int = 4096,
    num_heads: int = 32,
    num_layers: int = 32,
    warmup: int = 3,
    repeats: int = 5,
    hardware: HardwareSpec | None = None,
) -> DecodeMetrics:
    import torch

    head_dim = embed_dim // num_heads

    if hardware is None:
        hardware, _ = detect_hardware()

    q = torch.randn(batch_size, num_heads, 1, head_dim)
    k = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    v = torch.randn(batch_size, num_heads, prompt_len, head_dim)

    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(gen_tokens):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        t1 = time.perf_counter_ns()
        latencies.append(((t1 - t0) / gen_tokens) / 1e6)

    mean_step_ms = sum(latencies) / len(latencies)
    total_decode_ms = mean_step_ms * num_layers

    tokens_per_sec = (1.0 / (total_decode_ms / 1e3)) if total_decode_ms > 0 else 0.0
    throughput = tokens_per_sec * batch_size

    # Per-token decode DRAM traffic: Full model weights + active KV cache retrieval
    weight_bytes = 4 * (embed_dim**2) * 2 * num_layers
    kv_read_bytes = 2 * batch_size * num_heads * head_dim * prompt_len * 2 * num_layers
    traffic_per_token = weight_bytes + kv_read_bytes

    achieved_gbw = (
        (traffic_per_token / (total_decode_ms / 1e3)) / 1e9 if total_decode_ms > 0 else 0.0
    )
    bw_util = (
        (achieved_gbw * 1e9 / hardware.memory_bandwidth) if hardware.memory_bandwidth > 0 else 0.0
    )

    return DecodeMetrics(
        prompt_len=prompt_len,
        gen_tokens=gen_tokens,
        batch_size=batch_size,
        latency_ms=total_decode_ms,
        tokens_per_sec=tokens_per_sec,
        throughput_tokens_s=throughput,
        achieved_gbw=achieved_gbw,
        memory_bw_util=bw_util,
    )


def benchmark_moe_decode(
    batch_size: int = 1,
    seq_len: int = 1,
    embed_dim: int = 4096,
    expert_hidden_dim: int = 14336,
    num_experts: int = 8,
    top_k: int = 2,
    warmup: int = 3,
    repeats: int = 10,
    hardware: HardwareSpec | None = None,
) -> MoEDecodeMetrics:
    import torch

    from neural_cost.estimate import estimate_moe

    if hardware is None:
        hardware, _ = detect_hardware()

    est = estimate_moe(
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        expert_hidden_dim=expert_hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        expert_type="swiglu",
        is_decode=True,
    )

    w_gate = torch.randn(top_k, embed_dim, expert_hidden_dim)
    w_up = torch.randn(top_k, embed_dim, expert_hidden_dim)
    w_down = torch.randn(top_k, expert_hidden_dim, embed_dim)
    x = torch.randn(batch_size, embed_dim)

    for _ in range(warmup):
        for k_idx in range(top_k):
            h = torch.nn.functional.silu(x @ w_gate[k_idx]) * (x @ w_up[k_idx])
            _ = h @ w_down[k_idx]

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        for k_idx in range(top_k):
            h = torch.nn.functional.silu(x @ w_gate[k_idx]) * (x @ w_up[k_idx])
            _ = h @ w_down[k_idx]
    t1 = time.perf_counter_ns()

    elapsed_ms = ((t1 - t0) / repeats) / 1e6
    achieved_tflops = (est.total_flops / (elapsed_ms / 1e3)) / 1e12 if elapsed_ms > 0 else 0.0
    achieved_gbw = (est.total_bytes / (elapsed_ms / 1e3)) / 1e9 if elapsed_ms > 0 else 0.0

    bw_util = (
        (achieved_gbw * 1e9 / hardware.memory_bandwidth) if hardware.memory_bandwidth > 0 else 0.0
    )
    compute_util = (
        (achieved_tflops * 1e12 / hardware.peak_flops) if hardware.peak_flops > 0 else 0.0
    )

    return MoEDecodeMetrics(
        batch_size=batch_size,
        num_experts=num_experts,
        top_k=top_k,
        latency_ms=elapsed_ms,
        achieved_tflops=achieved_tflops,
        achieved_gbw=achieved_gbw,
        memory_bw_util=bw_util,
        compute_util=compute_util,
        expected_loaded_experts=est.expected_loaded_experts,
    )


def benchmark_speculative_decoding(
    gamma: int = 4,
    acceptance_rate: float = 0.75,
    prompt_len: int = 512,
    batch_size: int = 1,
    draft_head_dim: int = 64,
    target_head_dim: int = 128,
    warmup: int = 3,
    repeats: int = 10,
) -> SpeculativeDecodeMetrics:
    import torch

    from neural_cost.estimate import _expected_speculative_tokens, _find_breakeven_alpha

    q_draft = torch.randn(batch_size, 4, 1, draft_head_dim)
    k_draft = torch.randn(batch_size, 4, prompt_len, draft_head_dim)
    v_draft = torch.randn(batch_size, 4, prompt_len, draft_head_dim)

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


def benchmark_paged_attention(
    batch_size: int = 1,
    context_len: int = 512,
    embed_dim: int = 1024,
    num_heads: int = 8,
    num_kv_heads: int | None = None,
    block_size: int = 16,
    num_layers: int = 4,
    warmup: int = 3,
    repeats: int = 10,
    hardware: HardwareSpec | None = None,
) -> PagedAttentionMetrics:
    import torch

    from neural_cost.estimate import estimate_paged_attention

    if hardware is None:
        hardware, _ = detect_hardware()

    kv_heads = num_kv_heads or num_heads
    head_dim = embed_dim // num_heads

    paged_est = estimate_paged_attention(
        batch_size=batch_size,
        context_lens=context_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=kv_heads,
        block_size=block_size,
        num_layers=num_layers,
    )

    num_blocks = (context_len + block_size - 1) // block_size
    block_k = torch.randn(batch_size, num_blocks, block_size, kv_heads, head_dim)
    block_v = torch.randn(batch_size, num_blocks, block_size, kv_heads, head_dim)
    q = torch.randn(batch_size, num_heads, 1, head_dim)

    # Gather block memory into contiguous sequence for attention simulation
    k_contig = block_k.view(batch_size, num_blocks * block_size, kv_heads, head_dim)[
        :, :context_len, :, :
    ].permute(0, 2, 1, 3)
    v_contig = block_v.view(batch_size, num_blocks * block_size, kv_heads, head_dim)[
        :, :context_len, :, :
    ].permute(0, 2, 1, 3)

    if kv_heads != num_heads:
        k_contig = k_contig.repeat_interleave(num_heads // kv_heads, dim=1)
        v_contig = v_contig.repeat_interleave(num_heads // kv_heads, dim=1)

    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k_contig, v_contig)

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k_contig, v_contig)
    t1 = time.perf_counter_ns()

    step_ms = (((t1 - t0) / repeats) / 1e6) * num_layers

    # Achieved DRAM bandwidth: reading allocated KV cache blocks per decode token
    achieved_gbw = (paged_est.allocated_kv_bytes / (step_ms / 1e3)) / 1e9 if step_ms > 0 else 0.0
    bw_util = (
        (achieved_gbw * 1e9 / hardware.memory_bandwidth) if hardware.memory_bandwidth > 0 else 0.0
    )

    return PagedAttentionMetrics(
        batch_size=batch_size,
        context_len=context_len,
        block_size=block_size,
        allocated_kv_bytes=paged_est.allocated_kv_bytes,
        compulsory_kv_bytes=paged_est.compulsory_kv_bytes,
        fragmentation_bytes=paged_est.fragmentation_bytes,
        fragmentation_ratio=paged_est.fragmentation_ratio,
        total_blocks=paged_est.total_blocks,
        latency_ms=step_ms,
        achieved_gbw=achieved_gbw,
        memory_bw_util=bw_util,
    )
