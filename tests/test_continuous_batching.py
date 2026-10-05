"""Unit tests for continuous batching iteration cost modeling and gap analysis."""

import pytest

from neural_cost import (
    HardwareSpec,
    analyze_continuous_batch_iteration,
    estimate_continuous_batch_iteration,
)


class TestContinuousBatchingEstimation:
    """Validate continuous batching operational regimes and memory accounting."""

    def test_decode_only_iteration(self):
        # 4 active decode streams, no incoming prefill chunks
        est = estimate_continuous_batch_iteration(
            decode_context_lens=(128, 256, 512, 1024),
            prefill_chunk_lens=(),
            embed_dim=2048,
            num_heads=16,
            num_layers=12,
        )

        assert est.num_decode_requests == 4
        assert est.num_prefill_requests == 0
        assert est.decode_tokens == 4
        assert est.prefill_tokens == 0
        assert est.total_tokens == 4

        # Model weights must be read from DRAM
        assert est.model_weight_bytes > 0
        assert est.kv_cache_read_bytes > 0
        assert est.kv_cache_write_bytes > 0
        # Decode only with small batch is memory bandwidth bound
        assert est.arithmetic_intensity < 20.0
        assert est.is_compute_bound is False

    def test_chunked_prefill_transition_to_compute_bound(self):
        # Co-locating a 512-token prefill chunk with 4 decode requests
        est_mixed = estimate_continuous_batch_iteration(
            decode_context_lens=(128, 256, 512, 1024),
            prefill_chunk_lens=(512,),
            embed_dim=2048,
            num_heads=16,
            num_layers=12,
        )

        assert est_mixed.decode_tokens == 4
        assert est_mixed.prefill_tokens == 512
        assert est_mixed.total_tokens == 516

        # Arithmetic intensity should increase dramatically due to prompt chunk GEMM
        assert est_mixed.arithmetic_intensity > 100.0
        assert est_mixed.is_compute_bound is True

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="at least one decode or prefill request"):
            estimate_continuous_batch_iteration(
                decode_context_lens=(),
                prefill_chunk_lens=(),
            )
        with pytest.raises(ValueError, match="model parameters must be positive"):
            estimate_continuous_batch_iteration(
                decode_context_lens=(128,),
                embed_dim=0,
            )


class TestContinuousBatchingGapAnalysis:
    """Validate analyze_continuous_batch_iteration and saturation recommendations."""

    def test_analyze_decode_memory_bound(self):
        hardware = HardwareSpec("TestGPU", peak_flops=300e12, memory_bandwidth=2e12)
        est = estimate_continuous_batch_iteration(
            decode_context_lens=(128, 256),
            prefill_chunk_lens=(),
            embed_dim=4096,
            num_heads=32,
            num_layers=32,
            hardware=hardware,
        )

        gap = analyze_continuous_batch_iteration(est, hardware)
        assert gap.bottleneck == "memory"
        assert gap.optimal_prefill_tokens_to_saturate > 0
        assert any("memory-bandwidth bound" in f for f in gap.findings)

        rendered = gap.render()
        assert "Continuous Batch Iteration Analysis" in rendered
        assert "target saturation" in rendered

    def test_analyze_mixed_compute_bound(self):
        hardware = HardwareSpec("TestGPU", peak_flops=300e12, memory_bandwidth=2e12)
        est = estimate_continuous_batch_iteration(
            decode_context_lens=(128, 256),
            prefill_chunk_lens=(1024,),
            embed_dim=4096,
            num_heads=32,
            num_layers=32,
            hardware=hardware,
        )

        gap = analyze_continuous_batch_iteration(est, hardware)
        assert gap.bottleneck == "compute"
        assert any("compute-bound" in f for f in gap.findings)
