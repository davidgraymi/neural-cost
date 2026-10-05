"""End-to-end tests for PagedAttention and Continuous Batching serving configurations."""

import pytest

from neural_cost import (
    HardwareSpec,
    analyze_continuous_batch_iteration,
    analyze_paged_attention_gap,
    estimate_continuous_batch_iteration,
    estimate_paged_attention,
)


class TestLlama3ServingE2E:
    """Validate Llama-3 8B serving configurations with PagedAttention and continuous batching."""

    def test_llama3_8b_paged_attention_footprint(self):
        # Llama-3 8B: D=4096, 32 layers, 32 Q heads, 8 KV heads (GQA), head_dim=128
        batch_size = 16
        seq_len = 2048
        est = estimate_paged_attention(
            batch_size=batch_size,
            context_lens=seq_len,
            embed_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            block_size=16,
            num_layers=32,
            dtype_bytes=2,  # BF16
        )

        # 1 token KV footprint per layer = 2 * 8 * 128 * 2 = 4,096 bytes
        # Across 32 layers = 131,072 bytes (~128 KB per token across all layers)
        # Block of 16 tokens = 16 * 128 KB = 2 MB per block
        # For 2048 tokens = 128 blocks = 256 MB per sequence
        # For 16 sequences = 16 * 256 MB = 4,096 MB = 4 GB KV cache
        assert est.total_blocks == 16 * 128  # 2048 blocks
        assert est.allocated_kv_bytes == pytest.approx(4 * 1024**3, rel=0.01)
        # With 2048 exactly divisible by 16, internal fragmentation is 0%
        assert est.fragmentation_bytes == 0
        assert est.fragmentation_ratio == 0.0

    def test_llama3_8b_prefix_caching_concurrency_boost(self):
        # 32 chat sessions sharing a 1024-token system prompt (RAG / Few-shot prompt)
        batch_size = 32
        est = estimate_paged_attention(
            batch_size=batch_size,
            context_lens=1500,
            embed_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            block_size=16,
            num_layers=32,
            dtype_bytes=2,
            shared_prefix_len=1024,
        )

        # 1024 tokens = 64 blocks
        # Saved across 32 sequences = 31 * 64 = 1,984 blocks saved!
        assert est.shared_prefix_blocks == 64
        assert est.shared_saved_bytes > 3.5e9  # >3.5 GB VRAM saved by deduplication

        gap = analyze_paged_attention_gap(
            paged_cost=est,
            max_context_len=4096,  # Naive reservation would allocate 4096 tokens per slot
            batch_size=batch_size,
            embed_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            num_layers=32,
            dtype_bytes=2,
        )
        assert gap.memory_savings_ratio > 0.6  # >60% memory savings vs naive max allocation
        assert gap.concurrency_multiplier > 2.5

    def test_llama3_continuous_batching_h100_saturation(self):
        # NVIDIA H100 SXM5: 989 TFLOP/s FP16/BF16, 3.35 TB/s HBM3
        h100 = HardwareSpec("H100_SXM5", peak_flops=989e12, memory_bandwidth=3.35e12)
        # Ridge point = 989e12 / 3.35e12 = ~295 FLOP/byte

        # Decode-only iteration with 64 streams:
        decode_only = estimate_continuous_batch_iteration(
            decode_context_lens=[1024] * 64,
            prefill_chunk_lens=(),
            embed_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            num_layers=32,
            hardware=h100,
        )
        gap_dec = analyze_continuous_batch_iteration(decode_only, h100)
        assert gap_dec.bottleneck == "memory"
        assert gap_dec.optimal_prefill_tokens_to_saturate > 0

        # Now inject a 1024-token chunked prefill into the same iteration:
        mixed = estimate_continuous_batch_iteration(
            decode_context_lens=[1024] * 64,
            prefill_chunk_lens=(1024,),
            embed_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            num_layers=32,
            hardware=h100,
        )
        gap_mixed = analyze_continuous_batch_iteration(mixed, h100)
        assert mixed.total_tokens == 64 + 1024
        assert mixed.arithmetic_intensity > decode_only.arithmetic_intensity
        assert gap_mixed.bottleneck in ("compute", "memory")
        assert gap_mixed.lower_bound_seconds > 0.0
        assert gap_mixed.render() is not None
