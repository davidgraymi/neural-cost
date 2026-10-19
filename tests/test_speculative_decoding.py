"""Unit tests for speculative decoding cost modeling and breakeven analysis."""

import pytest

from neural_cost import (
    CostEstimate,
    HardwareSpec,
    analyze_speculative_decoding,
    estimate_speculative_decoding,
)
from neural_cost.estimate import (
    _expected_speculative_tokens,
    _find_breakeven_alpha,
    acceptance_profile_from_lengths,
)


class TestSpeculativeFormulas:
    """Validate mathematical correctness of expected speculative token formulas."""

    def test_expected_tokens_boundary_conditions(self):
        # alpha = 0.0 -> exactly 1 token (only target sample)
        assert _expected_speculative_tokens(4, 0.0) == pytest.approx(1.0)
        # alpha = 1.0 -> 1 + gamma tokens (all draft accepted + target sample)
        assert _expected_speculative_tokens(4, 1.0) == pytest.approx(5.0)
        assert _expected_speculative_tokens(6, 1.0) == pytest.approx(7.0)

    def test_expected_tokens_geometric_sum(self):
        # gamma=4, alpha=0.5 -> 1 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.9375
        assert _expected_speculative_tokens(4, 0.5) == pytest.approx(1.9375)
        # gamma=4, alpha=0.7 -> (1 - 0.7^5) / (1 - 0.7) = 2.7731
        expected = (1.0 - 0.7**5) / (1.0 - 0.7)
        assert _expected_speculative_tokens(4, 0.7) == pytest.approx(expected)

    def test_find_breakeven_alpha_inversion(self):
        gamma = 4
        # Target ratio 2.5 tokens needed to break even
        ratio = 2.5
        alpha_star = _find_breakeven_alpha(gamma, ratio)
        assert 0.0 < alpha_star < 1.0
        # Check that E[N(alpha_star)] == ratio
        assert _expected_speculative_tokens(gamma, alpha_star) == pytest.approx(ratio, rel=1e-4)

    def test_find_breakeven_unattainable(self):
        # Ratio exceeds 1 + gamma (e.g. 10.0 for gamma=4)
        assert _find_breakeven_alpha(4, 10.0) == float("inf")

    def test_find_breakeven_free_lunch(self):
        # Ratio <= 1.0 means speculative cycle is cheaper than 1 baseline token
        assert _find_breakeven_alpha(4, 0.9) == 0.0


class TestSpeculativeCostEstimate:
    """Validate estimate_speculative_decoding behavior."""

    def test_speculative_favorable_regime(self):
        # Target decode: 70B model, 1 token: 140e9 FLOPs, 140e9 bytes
        target_decode = CostEstimate(
            flops=140_000_000_000, read_bytes=140_000_000_000, write_bytes=4096, operations=1
        )
        # Draft decode: 1B model, 1 token: 2e9 FLOPs, 2e9 bytes
        draft_decode = CostEstimate(
            flops=2_000_000_000, read_bytes=2_000_000_000, write_bytes=2048, operations=1
        )
        # Target verify: 70B model verifying gamma=4 tokens in parallel (prefill-like): 140e9 bytes, 560e9 FLOPs
        target_verify = CostEstimate(
            flops=560_000_000_000, read_bytes=140_000_000_000, write_bytes=16384, operations=1
        )

        est = estimate_speculative_decoding(
            draft_decode_cost=draft_decode,
            target_verify_cost=target_verify,
            target_decode_cost=target_decode,
            gamma=4,
            acceptance_rate=0.8,
        )

        assert est.expected_tokens_per_step > 3.0
        # In memory traffic, draft (4 * 2B) + verify (140B) = 148B vs baseline (3.3 * 140B) = 462B
        assert est.speedup_bytes > 2.0
        assert 0.0 < est.breakeven_acceptance_rate_bytes < 0.8

    def test_speculative_unfavorable_low_acceptance(self):
        target_decode = CostEstimate(flops=100e9, read_bytes=100e9, write_bytes=4096, operations=1)
        draft_decode = CostEstimate(flops=10e9, read_bytes=10e9, write_bytes=2048, operations=1)
        target_verify = CostEstimate(flops=400e9, read_bytes=100e9, write_bytes=16384, operations=1)

        est = estimate_speculative_decoding(
            draft_decode_cost=draft_decode,
            target_verify_cost=target_verify,
            target_decode_cost=target_decode,
            gamma=4,
            acceptance_rate=0.05,  # Very low acceptance rate
        )

        assert est.expected_tokens_per_step < 1.1
        # Slowdown because drafting cost is wasted
        assert est.speedup_bytes < 1.0


class TestSpeculativeAnalysis:
    """Validate analyze_speculative_decoding on hardware."""

    def test_analyze_speculative_hardware_speedup(self):
        hardware = HardwareSpec("A100", peak_flops=312e12, memory_bandwidth=2e12)
        target_decode = CostEstimate(
            flops=140_000_000_000, read_bytes=140_000_000_000, write_bytes=4096, operations=1
        )
        draft_decode = CostEstimate(
            flops=2_000_000_000, read_bytes=2_000_000_000, write_bytes=2048, operations=1
        )
        target_verify = CostEstimate(
            flops=560_000_000_000, read_bytes=140_000_000_000, write_bytes=16384, operations=1
        )

        analysis = analyze_speculative_decoding(
            draft_decode_cost=draft_decode,
            target_verify_cost=target_verify,
            target_decode_cost=target_decode,
            hardware=hardware,
            gamma=4,
            acceptance_rate=0.75,
        )

        assert analysis.is_favorable is True
        assert analysis.speedup > 1.5
        assert analysis.breakeven_acceptance_rate < 0.75
        rendered = analysis.render()
        assert "Speculative Decoding Analysis" in rendered
        assert "favorable" in rendered

    def test_analyze_speculative_hardware_slowdown(self):
        hardware = HardwareSpec("A100", peak_flops=312e12, memory_bandwidth=2e12)
        # Draft model is too heavy (half of target)
        target_decode = CostEstimate(flops=100e9, read_bytes=100e9, write_bytes=4096, operations=1)
        draft_decode = CostEstimate(flops=50e9, read_bytes=50e9, write_bytes=4096, operations=1)
        target_verify = CostEstimate(flops=400e9, read_bytes=100e9, write_bytes=16384, operations=1)

        analysis = analyze_speculative_decoding(
            draft_decode_cost=draft_decode,
            target_verify_cost=target_verify,
            target_decode_cost=target_decode,
            hardware=hardware,
            gamma=4,
            acceptance_rate=0.2,
        )

        assert analysis.is_favorable is False
        assert analysis.speedup < 1.0
        assert any("regresses latency" in f for f in analysis.findings)


class TestPositionDependentAcceptance:
    """Per-position conditional acceptance instead of a single i.i.d. alpha."""

    def test_uniform_profile_matches_scalar(self):
        assert _expected_speculative_tokens(4, [0.7] * 4) == pytest.approx(
            _expected_speculative_tokens(4, 0.7)
        )

    def test_profile_expected_tokens_is_survival_sum(self):
        # 1 + 0.9 + 0.9*0.6 + 0.9*0.6*0.5 = 2.71
        assert _expected_speculative_tokens(3, [0.9, 0.6, 0.5]) == pytest.approx(2.71)

    def test_decaying_profile_yields_fewer_tokens_than_its_first_position(self):
        decaying = [0.9, 0.8, 0.7, 0.6]
        assert _expected_speculative_tokens(4, decaying) < _expected_speculative_tokens(4, 0.9)

    def test_profile_length_and_range_validated(self):
        with pytest.raises(ValueError):
            _expected_speculative_tokens(4, [0.5, 0.5])
        with pytest.raises(ValueError):
            _expected_speculative_tokens(2, [0.5, 1.5])

    def test_profile_from_observed_lengths(self):
        # Cycles accepting 0, 1, 2, 2, 4 of gamma=4 drafts.
        profile = acceptance_profile_from_lengths([0, 1, 2, 2, 4], gamma=4)
        assert profile == pytest.approx((4 / 5, 3 / 4, 1 / 3, 1.0))
        # The profile reproduces the empirical mean tokens per cycle (+1 bonus).
        empirical = 1 + sum([0, 1, 2, 2, 4]) / 5
        assert _expected_speculative_tokens(4, profile) == pytest.approx(empirical)

    def test_profile_from_lengths_validates(self):
        with pytest.raises(ValueError):
            acceptance_profile_from_lengths([], gamma=4)
        with pytest.raises(ValueError):
            acceptance_profile_from_lengths([5], gamma=4)
        assert acceptance_profile_from_lengths([0, 0], gamma=2) == (0.0, 0.0)

    def test_estimate_and_analysis_accept_profile(self):
        draft = CostEstimate(flops=1_000, read_bytes=1_000_000, write_bytes=0, operations=1)
        verify = CostEstimate(flops=50_000, read_bytes=10_000_000, write_bytes=0, operations=1)
        target = CostEstimate(flops=10_000, read_bytes=10_000_000, write_bytes=0, operations=1)
        hw = HardwareSpec(name="t", peak_flops=1e12, memory_bandwidth=1e12)
        profile = [0.9, 0.8, 0.6, 0.4]
        est = estimate_speculative_decoding(draft, verify, target, gamma=4, acceptance_rate=profile)
        gap = analyze_speculative_decoding(
            draft, verify, target, hw, gamma=4, acceptance_rate=profile
        )
        e_tokens = _expected_speculative_tokens(4, profile)
        assert est.expected_tokens_per_step == pytest.approx(e_tokens)
        assert gap.expected_tokens_per_step == pytest.approx(e_tokens)
        assert est.acceptance_profile == tuple(profile)
        # Reported scalar alpha is the i.i.d. equivalent with the same E[N].
        assert _expected_speculative_tokens(4, gap.acceptance_rate) == pytest.approx(
            e_tokens, rel=1e-6
        )
        assert "acceptance by position" in gap.render()
