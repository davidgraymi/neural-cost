---
title: "LLM serving models"
description: "Analytical cost models for MoE, speculative decoding, PagedAttention, continuous batching and SSMs."
section: "Models"
order: 1
---

`neural-cost` provides specialized theoretical models for modern LLM serving paradigms:

## MoE Sparse Routing & Memory Thrashing

Sparse MoE architectures (e.g. Mixtral, DeepSeek) activate only *k* of *E* experts per token. While compute scales with *k*, token-by-token autoregressive decoding requires loading unique experts from DRAM into registers/SRAM, producing severe memory-bandwidth pressure at small batch sizes:

```python
from neural_cost import estimate_moe, analyze_moe_gap, detect_hardware

# Mixtral 8x7B layer: 8 experts, top-2 routing, SwiGLU FFN
est = estimate_moe(
    batch_size=1,
    seq_len=1,
    embed_dim=4096,
    expert_hidden_dim=14336,
    num_experts=8,
    top_k=2,
    expert_type="swiglu",
    is_decode=True,
)

print(f"Total params: {est.total_parameters:,} | Active params: {est.active_parameters:,}")
print(f"Expected loaded experts at B=1: {est.expected_loaded_experts:.1f}")

hardware, _ = detect_hardware()
gap = analyze_moe_gap(est, hardware)
print(gap.render())
```

## Speculative Decoding Breakeven Analysis

Speculative decoding pairs a fast draft model with a parallel target model verification step. `neural-cost` computes the analytical breakeven acceptance rate (α*) required to achieve wall-clock speedup on target hardware:

```python
from neural_cost import (
    CostEstimate,
    detect_hardware,
    estimate_speculative_decoding,
    analyze_speculative_decoding,
)

hardware, _ = detect_hardware()

# Target (70B) vs Draft (1B) step costs
draft_decode = CostEstimate(flops=2e9, read_bytes=2e9, write_bytes=2048, operations=1)
target_verify = CostEstimate(flops=560e9, read_bytes=140e9, write_bytes=16384, operations=1)
target_decode = CostEstimate(flops=140e9, read_bytes=140e9, write_bytes=4096, operations=1)

analysis = analyze_speculative_decoding(
    draft_decode_cost=draft_decode,
    target_verify_cost=target_verify,
    target_decode_cost=target_decode,
    hardware=hardware,
    gamma=4,
    acceptance_rate=0.75,
)

print(analysis.render())
```

Acceptance is rarely uniform: later draft positions are accepted less often, and acceptances are correlated. Pass a per-position profile instead of a scalar, where entry *k* is the probability that draft *k* is accepted given that drafts 1..*k*−1 were. Expected tokens per cycle depend only on these survival probabilities, so this is exact under any correlation. `acceptance_profile_from_lengths` builds the profile from observed accepted-run lengths (e.g. vLLM spec-decode metrics):

```python
from neural_cost import acceptance_profile_from_lengths

profile = acceptance_profile_from_lengths(observed_accepted_lengths, gamma=4)
analysis = analyze_speculative_decoding(
    draft_decode, target_verify, target_decode, hardware, gamma=4, acceptance_rate=profile
)
# analysis.acceptance_rate is the i.i.d. alpha with the same expected tokens/cycle
```

## PagedAttention & KV Cache Memory Modeling

PagedAttention (vLLM) allocates non-contiguous fixed-size blocks (e.g. 16 tokens) to eliminate external fragmentation and over-reservation. `neural-cost` models exact block allocation, compulsory vs wasted fragmentation bytes, prefix cache sharing, and concurrency capacity multipliers:

```python
from neural_cost import estimate_paged_attention, analyze_paged_attention_gap

# 32 concurrent requests with heterogeneous context lengths
context_lens = [256 + (i * 32) for i in range(32)]
paged_cost = estimate_paged_attention(
    batch_size=32,
    context_lens=context_lens,
    embed_dim=4096,
    num_heads=32,
    num_kv_heads=8,  # Grouped-Query Attention (GQA)
    block_size=16,
    num_layers=32,
    max_model_len=4096,
    shared_prefix_tokens=128,  # 128-token shared system prompt
)

gap = analyze_paged_attention_gap(paged_cost)
print(gap.render())
# Reports VRAM saved vs contiguous max reservation, fragmentation ratio, and concurrency gain
```

## Continuous Batching Iteration Cost Simulator

Modern serving engines (vLLM, TGI, TensorRT-LLM) use iteration-level scheduling (continuous/in-flight batching) mixing memory-bound autoregressive decodes and compute-heavy chunked prefills in the same step. `neural-cost` simulates the unified arithmetic intensity and lower-bound execution time across model weights and KV caches:

```python
from neural_cost import (
    estimate_continuous_batch_iteration,
    analyze_continuous_batch_iteration,
    detect_hardware,
)

hardware, _ = detect_hardware()

# Mixed iteration: 64 decode streams + one 512-token chunked prefill
iteration = estimate_continuous_batch_iteration(
    decode_context_lens=[1024] * 64,
    prefill_chunk_lens=[512],
    embed_dim=4096,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    hardware=hardware,
)

gap = analyze_continuous_batch_iteration(iteration, hardware)
print(gap.render())
# Identifies compute vs memory bottleneck and optimal prefill chunk sizing to reach the roofline ridge
```

## State Space Models (Mamba / S6 / SSD) & Linear Attention

State Space Models replace the quadratic O(L²) attention mechanism and linear-growth O(L) KV cache with a fixed-size O(1) hidden recurrent state (h<sub>t</sub> ∈ ℝ<sup>B×D<sub>in</sub>×N</sup>).

`neural-cost` models the associative parallel scan in prefill and constant-memory recurrent step in decode:

```python
from neural_cost import analyze_ssm_gap, detect_hardware, estimate_ssm

# Mamba-3B layer: hidden dim 4096, state dim 16, expand factor 2
est = estimate_ssm(
    batch_size=1,
    seq_len=8192,
    embed_dim=4096,
    state_dim=16,
    expand_factor=2,
    conv_kernel_size=4,
    num_layers=32,
    is_decode=True,
)

print(f"Recurrent state footprint: {est.state_bytes / 1e6:.2f} MB (constant O(1) in seq_len)")
print(
    f"KV cache eliminated: {est.memory_savings_ratio_vs_transformer:.1%} savings vs Transformer KV"
)

hardware, _ = detect_hardware()
gap = analyze_ssm_gap(est, hardware)
print(gap.render())
# Reports constant-memory decode lower bound and throughput advantage over Transformer KV cache retrieval
```
