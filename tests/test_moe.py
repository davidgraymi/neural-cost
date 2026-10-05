"""Unit tests for Mixture-of-Experts (MoE) cost modeling and gap analysis."""

import pytest

from neural_cost import HardwareSpec, Operation, analyze_moe_gap, estimate_moe, estimate_operation


class TestMoEEstimation:
    """Validate mathematical correctness of MoE parameter, FLOP, and traffic models."""

    def test_moe_parameter_accounting_swiglu(self):
        """Verify SwiGLU (3 linear projections) parameter calculation."""
        d, h, e, k = 1024, 2048, 8, 2
        est = estimate_moe(
            batch_size=1,
            seq_len=1,
            embed_dim=d,
            expert_hidden_dim=h,
            num_experts=e,
            top_k=k,
            expert_type="swiglu",
            dtype_bytes=2,
        )

        single_expert_params = 3 * d * h  # 6,291,456
        router_params = d * e  # 8,192
        expected_total = router_params + e * single_expert_params  # 50,339,840
        expected_active = router_params + k * single_expert_params  # 12,591,104

        assert est.total_parameters == expected_total
        assert est.active_parameters == expected_active
        assert est.parameter_bytes == expected_total * 2
        assert est.active_parameter_bytes == expected_active * 2

    def test_moe_parameter_accounting_mlp(self):
        """Verify standard MLP (2 linear projections) parameter calculation."""
        d, h, e, k = 1024, 2048, 8, 2
        est = estimate_moe(
            batch_size=1,
            seq_len=1,
            embed_dim=d,
            expert_hidden_dim=h,
            num_experts=e,
            top_k=k,
            expert_type="mlp",
            dtype_bytes=2,
        )

        single_expert_params = 2 * d * h  # 4,194,304
        router_params = d * e  # 8,192
        expected_total = router_params + e * single_expert_params
        expected_active = router_params + k * single_expert_params

        assert est.total_parameters == expected_total
        assert est.active_parameters == expected_active

    def test_moe_shared_experts(self):
        """DeepSeek-style shared experts always run for every token."""
        d, h, e, k = 512, 1024, 8, 2
        shared = 2
        est = estimate_moe(
            batch_size=1,
            seq_len=1,
            embed_dim=d,
            expert_hidden_dim=h,
            num_experts=e,
            top_k=k,
            shared_experts=shared,
            expert_type="swiglu",
            dtype_bytes=2,
        )

        single_expert = 3 * d * h
        router = d * e
        assert est.total_parameters == router + (e + shared) * single_expert
        assert est.active_parameters == router + (k + shared) * single_expert

    def test_moe_decode_expected_loaded_experts_scaling(self):
        """Verify probabilistic expert loading in decode across batch sizes."""
        d, h, e, k = 512, 1024, 8, 2

        # Batch = 1 token: exactly top_k experts loaded
        est_b1 = estimate_moe(
            batch_size=1,
            seq_len=1,
            embed_dim=d,
            expert_hidden_dim=h,
            num_experts=e,
            top_k=k,
            is_decode=True,
        )
        assert est_b1.expected_loaded_experts == pytest.approx(2.0, rel=1e-5)

        # Batch = 4 tokens: expected loaded experts = 8 * (1 - (6/8)^4) = 5.46875
        est_b4 = estimate_moe(
            batch_size=4,
            seq_len=1,
            embed_dim=d,
            expert_hidden_dim=h,
            num_experts=e,
            top_k=k,
            is_decode=True,
        )
        expected_b4 = 8 * (1.0 - (6.0 / 8.0) ** 4)
        assert est_b4.expected_loaded_experts == pytest.approx(expected_b4, rel=1e-4)

        # Large batch (e.g. 64): all experts almost certainly accessed
        est_b64 = estimate_moe(
            batch_size=64,
            seq_len=1,
            embed_dim=d,
            expert_hidden_dim=h,
            num_experts=e,
            top_k=k,
            is_decode=True,
        )
        assert est_b64.expected_loaded_experts == pytest.approx(8.0, rel=1e-3)

        # Traffic should increase with expected loaded experts
        assert est_b1.compulsory_param_read_bytes < est_b4.compulsory_param_read_bytes
        assert est_b4.compulsory_param_read_bytes < est_b64.compulsory_param_read_bytes

    def test_moe_prefill_vs_decode_arithmetic_intensity(self):
        """Prefill should have significantly higher arithmetic intensity than decode."""
        d, h, e, k = 1024, 4096, 8, 2
        prefill = estimate_moe(
            batch_size=1,
            seq_len=1024,
            embed_dim=d,
            expert_hidden_dim=h,
            num_experts=e,
            top_k=k,
            is_decode=False,
        )
        decode = estimate_moe(
            batch_size=1,
            seq_len=1,
            embed_dim=d,
            expert_hidden_dim=h,
            num_experts=e,
            top_k=k,
            is_decode=True,
        )
        assert prefill.arithmetic_intensity > decode.arithmetic_intensity
        assert decode.arithmetic_intensity < 2.0  # Decode single token is bandwidth throttled

    def test_moe_validation_errors(self):
        """Invalid configurations raise ValueError."""
        with pytest.raises(ValueError, match="invalid experts configuration"):
            estimate_moe(1, 1, 512, 1024, num_experts=4, top_k=8)

        with pytest.raises(ValueError, match="dimensions must be positive"):
            estimate_moe(0, 1, 512, 1024)

        with pytest.raises(ValueError, match="shared_experts cannot be negative"):
            estimate_moe(1, 1, 512, 1024, shared_experts=-1)


class TestMoEOperationIntegration:
    """Validate Operation(kind='moe') in neural-cost pipeline."""

    def test_moe_operation_estimate(self):
        op = Operation(
            name="moe_layer_1",
            kind="moe",
            inputs=((2, 128, 512),),
            output=(2, 128, 512),
            dtype_bytes=2,
            attrs={"num_experts": 8, "top_k": 2, "expert_hidden_dim": 2048},
        )
        cost = estimate_operation(op)
        assert cost.flops > 0
        assert cost.read_bytes > 0
        assert cost.write_bytes == 2 * 128 * 512 * 2
        assert cost.arithmetic_intensity > 0

    def test_moe_operation_decode_attr(self):
        op_decode = Operation(
            name="moe_decode",
            kind="moe",
            inputs=((1, 1, 512),),
            output=(1, 1, 512),
            dtype_bytes=2,
            attrs={"num_experts": 8, "top_k": 2, "expert_hidden_dim": 2048, "is_decode": True},
        )
        op_prefill = Operation(
            name="moe_prefill",
            kind="moe",
            inputs=((1, 1, 512),),
            output=(1, 1, 512),
            dtype_bytes=2,
            attrs={"num_experts": 8, "top_k": 2, "expert_hidden_dim": 2048, "is_decode": False},
        )
        cost_decode = estimate_operation(op_decode)
        cost_prefill = estimate_operation(op_prefill)
        assert cost_decode.read_bytes < cost_prefill.read_bytes


class TestMoEGapAnalysis:
    """Validate analyze_moe_gap bottleneck identification and report rendering."""

    def test_analyze_moe_gap_decode(self):
        hardware = HardwareSpec("TestGPU", peak_flops=300e12, memory_bandwidth=2e12)
        est = estimate_moe(
            batch_size=1,
            seq_len=1,
            embed_dim=2048,
            expert_hidden_dim=8192,
            num_experts=8,
            top_k=2,
            is_decode=True,
        )
        gap = analyze_moe_gap(est, hardware, observed_seconds=0.005)
        assert gap.bottleneck == "memory"
        assert gap.efficiency is not None
        assert any("memory-bandwidth bound" in f for f in gap.findings)

        rendered = gap.render()
        assert "MoE Gap Analysis (decode)" in rendered
        assert "expected loaded experts: 2.00" in rendered

    def test_analyze_moe_gap_prefill(self):
        hardware = HardwareSpec("TestGPU", peak_flops=300e12, memory_bandwidth=2e12)
        est = estimate_moe(
            batch_size=16,
            seq_len=1024,
            embed_dim=2048,
            expert_hidden_dim=8192,
            num_experts=8,
            top_k=2,
            is_decode=False,
        )
        gap = analyze_moe_gap(est, hardware)
        assert gap.bottleneck == "compute"
        assert any("compute-bound" in f for f in gap.findings)
        rendered = gap.render()
        assert "MoE Gap Analysis (prefill)" in rendered
