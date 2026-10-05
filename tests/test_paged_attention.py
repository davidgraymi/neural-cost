"""Unit tests for PagedAttention memory allocation, fragmentation, and gap analysis."""

import pytest

from neural_cost import (
    Operation,
    analyze_paged_attention_gap,
    estimate_operation,
    estimate_paged_attention,
)


class TestPagedAttentionEstimation:
    """Validate mathematical correctness of PagedAttention allocation and fragmentation."""

    def test_block_allocation_and_fragmentation(self):
        # 1 sequence with 100 tokens, block_size=16
        # Expected blocks = ceil(100/16) = 7 blocks (capacity 112 tokens)
        # Wasted / internal fragmentation = 12 tokens
        d = 1024
        heads = 8
        layers = 4
        dtype = 2
        est = estimate_paged_attention(
            batch_size=1,
            context_lens=100,
            embed_dim=d,
            num_heads=heads,
            block_size=16,
            num_layers=layers,
            dtype_bytes=dtype,
        )

        bytes_per_tok = 2 * d * layers * dtype  # 2 * 1024 * 4 * 2 = 16,384 bytes/tok
        expected_compulsory = 100 * bytes_per_tok
        expected_allocated = 112 * bytes_per_tok
        expected_fragmentation = 12 * bytes_per_tok

        assert est.total_blocks == 7
        assert est.compulsory_kv_bytes == expected_compulsory
        assert est.allocated_kv_bytes == expected_allocated
        assert est.fragmentation_bytes == expected_fragmentation
        assert est.fragmentation_ratio == pytest.approx(12 / 112, rel=1e-5)

    def test_prefix_caching_deduplication(self):
        # 4 sequences sharing a 64-token system prompt prefix
        # With block_size=16, 64 tokens = exactly 4 blocks
        # Across 4 sequences, 4 - 1 = 3 instances of the 4 blocks are saved (12 blocks saved)
        d = 512
        heads = 4
        layers = 2
        dtype = 2
        est_no_sharing = estimate_paged_attention(
            batch_size=4,
            context_lens=(128, 128, 128, 128),
            embed_dim=d,
            num_heads=heads,
            block_size=16,
            num_layers=layers,
            dtype_bytes=dtype,
            shared_prefix_len=0,
        )
        est_with_sharing = estimate_paged_attention(
            batch_size=4,
            context_lens=(128, 128, 128, 128),
            embed_dim=d,
            num_heads=heads,
            block_size=16,
            num_layers=layers,
            dtype_bytes=dtype,
            shared_prefix_len=64,
        )

        assert est_no_sharing.total_blocks == 4 * 8  # 32 blocks
        assert est_with_sharing.shared_prefix_blocks == 4
        assert est_with_sharing.total_blocks == 32 - 12  # 20 blocks
        assert est_with_sharing.allocated_kv_bytes < est_no_sharing.allocated_kv_bytes
        assert est_with_sharing.shared_saved_bytes > 0

    def test_grouped_query_attention_scaling(self):
        # Standard MHA: 32 heads vs GQA: 8 KV heads
        d = 4096
        heads = 32
        mha = estimate_paged_attention(
            batch_size=2,
            context_lens=256,
            embed_dim=d,
            num_heads=heads,
            num_kv_heads=32,
            block_size=16,
            num_layers=4,
        )
        gqa = estimate_paged_attention(
            batch_size=2,
            context_lens=256,
            embed_dim=d,
            num_heads=heads,
            num_kv_heads=8,
            block_size=16,
            num_layers=4,
        )

        # GQA with 8 KV heads should have exactly 1/4 the KV cache footprint of MHA
        assert gqa.allocated_kv_bytes == mha.allocated_kv_bytes // 4
        assert gqa.compulsory_kv_bytes == mha.compulsory_kv_bytes // 4

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="must be positive"):
            estimate_paged_attention(0, 100, 512, 8)
        with pytest.raises(ValueError, match="expected 2 context lengths"):
            estimate_paged_attention(2, (100,), 512, 8)
        with pytest.raises(ValueError, match="invalid num_kv_heads"):
            estimate_paged_attention(1, 100, 512, 8, num_kv_heads=16)


class TestPagedAttentionOperation:
    """Validate Operation(kind='paged_attention') integration."""

    def test_operation_estimate(self):
        op = Operation(
            name="paged_attn_step",
            kind="paged_attention",
            inputs=((2, 1, 1024),),
            output=(2, 1, 1024),
            dtype_bytes=2,
            attrs={
                "block_size": 16,
                "context_len": 512,
                "num_heads": 8,
                "num_kv_heads": 2,
                "num_layers": 4,
            },
        )
        cost = estimate_operation(op)
        assert cost.flops > 0
        assert cost.read_bytes > 0
        assert cost.write_bytes > 0
        assert cost.arithmetic_intensity > 0


class TestPagedAttentionGapAnalysis:
    """Validate analyze_paged_attention_gap comparison vs contiguous unpaged allocation."""

    def test_analyze_gap_unpaged_savings(self):
        est = estimate_paged_attention(
            batch_size=4,
            context_lens=(100, 200, 300, 400),
            embed_dim=1024,
            num_heads=8,
            num_layers=4,
            block_size=16,
        )
        # Unpaged traditional baseline assumes max_context_len = 2048 for all 4 sequences
        gap = analyze_paged_attention_gap(
            paged_cost=est,
            max_context_len=2048,
            batch_size=4,
            embed_dim=1024,
            num_heads=8,
            num_layers=4,
        )

        assert gap.memory_saved_bytes > 0
        assert gap.memory_savings_ratio > 0.5  # >50% VRAM saved
        assert gap.concurrency_multiplier > 2.0
        assert any("higher request concurrency" in f for f in gap.findings)

        rendered = gap.render()
        assert "PagedAttention Gap Analysis" in rendered
        assert "concurrency boost" in rendered
