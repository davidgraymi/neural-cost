"""Tests for State Space Model (SSM / Mamba / S6 / SSD) cost estimation, gap analysis, and CLI commands."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from neural_cost.analysis import SSMGapAnalysis, analyze_ssm_gap
from neural_cost.cli.root import main
from neural_cost.estimate import (
    SSMCostEstimate,
    estimate_operation,
    estimate_ssm,
)
from neural_cost.hardware import HardwareSpec
from neural_cost.operations import Operation


def test_estimate_ssm_basic() -> None:
    est = estimate_ssm(
        batch_size=1,
        seq_len=2048,
        embed_dim=4096,
        state_dim=16,
        expand_factor=2,
        conv_kernel_size=4,
        num_layers=32,
        dtype_bytes=2,
        is_decode=False,
    )
    assert isinstance(est, SSMCostEstimate)
    assert est.batch_size == 1
    assert est.seq_len == 2048
    assert est.embed_dim == 4096
    assert est.state_dim == 16
    assert est.num_layers == 32
    assert est.total_parameters > 0
    assert est.parameter_bytes == est.total_parameters * 2
    assert est.total_flops > 0
    assert est.arithmetic_intensity > 0
    assert est.is_decode is False


def test_ssm_state_memory_invariance_with_seq_len() -> None:
    """SSM recurrent state footprint must be strictly invariant O(1) across context lengths."""
    est_1 = estimate_ssm(batch_size=2, seq_len=1, embed_dim=2048, state_dim=16, num_layers=16)
    est_1k = estimate_ssm(batch_size=2, seq_len=1024, embed_dim=2048, state_dim=16, num_layers=16)
    est_32k = estimate_ssm(batch_size=2, seq_len=32768, embed_dim=2048, state_dim=16, num_layers=16)

    # State bytes should be identical regardless of sequence length
    assert est_1.state_bytes == est_1k.state_bytes == est_32k.state_bytes
    # Formula: B * D_in * N * dtype_bytes * num_layers = 2 * (2 * 2048) * 16 * 2 * 16
    expected_state = 2 * (2 * 2048) * 16 * 2 * 16
    assert est_1.state_bytes == expected_state

    # But equivalent Transformer KV cache must scale linearly with seq_len
    assert est_32k.equivalent_transformer_kv_bytes == 32 * est_1k.equivalent_transformer_kv_bytes
    # At 32k context, memory savings ratio must be > 95%
    assert est_32k.memory_savings_ratio_vs_transformer > 0.95


def test_estimate_ssm_decode_mode() -> None:
    est = estimate_ssm(
        batch_size=4,
        seq_len=4096,
        embed_dim=4096,
        state_dim=16,
        num_layers=32,
        is_decode=True,
    )
    assert est.is_decode is True
    # In decode mode, single token step FLOPs are computed
    assert (
        est.total_flops
        < estimate_ssm(4, 4096, 4096, 16, num_layers=32, is_decode=False).total_flops
    )
    # Recurrent state is both read and written in decode step
    assert est.state_read_bytes == est.state_bytes
    assert est.state_write_bytes == est.state_bytes
    assert est.memory_savings_ratio_vs_transformer > 0.9


def test_estimate_ssm_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_ssm(batch_size=-1, seq_len=100, embed_dim=512)
    with pytest.raises(ValueError, match="positive"):
        estimate_ssm(batch_size=1, seq_len=0, embed_dim=512)
    with pytest.raises(ValueError, match="positive"):
        estimate_ssm(batch_size=1, seq_len=10, embed_dim=-4)
    with pytest.raises(ValueError, match="positive"):
        estimate_ssm(batch_size=1, seq_len=10, embed_dim=512, state_dim=-1)
    with pytest.raises(ValueError, match="positive"):
        estimate_ssm(batch_size=1, seq_len=10, embed_dim=512, dtype_bytes=0)


def test_operation_state_space_model() -> None:
    op = Operation(
        name="mamba_block",
        kind="state_space_model",
        inputs=((1, 1024, 2048),),
        output=(1, 1024, 2048),
        dtype_bytes=2,
        attrs={"state_dim": 16, "expand_factor": 2, "conv_kernel_size": 4, "num_layers": 1},
    )
    cost = estimate_operation(op)
    assert cost.flops > 0
    assert cost.read_bytes > 0
    assert cost.write_bytes > 0
    assert cost.arithmetic_intensity > 0


def test_analyze_ssm_gap_compute_bound() -> None:
    est = estimate_ssm(
        batch_size=8,
        seq_len=2048,
        embed_dim=4096,
        state_dim=16,
        num_layers=32,
        is_decode=False,
    )
    # Hardware with high compute and modest bandwidth -> compute bound
    hw = HardwareSpec(name="TestComp", peak_flops=1e14, memory_bandwidth=1e11)
    gap = analyze_ssm_gap(est, hw)
    assert isinstance(gap, SSMGapAnalysis)
    assert gap.bottleneck == "compute"
    assert "Prefill parallel scan is compute-bound" in gap.findings[0]
    render_text = gap.render()
    assert "State Space Model Analysis (prefill" in render_text
    assert "compute-bound" in render_text


def test_analyze_ssm_gap_decode_memory_bound() -> None:
    est = estimate_ssm(
        batch_size=1,
        seq_len=8192,
        embed_dim=4096,
        state_dim=16,
        num_layers=32,
        is_decode=True,
    )
    # High FLOPs, standard memory bandwidth -> memory bound decode
    hw = HardwareSpec(name="TestMem", peak_flops=1e14, memory_bandwidth=2e12)
    gap = analyze_ssm_gap(est, hw)
    assert gap.bottleneck == "memory"
    assert any("memory-bandwidth bound" in f for f in gap.findings)
    assert any("eliminates" in f for f in gap.findings)
    assert gap.speedup_vs_transformer is not None
    assert gap.speedup_vs_transformer > 1.0
    render_text = gap.render()
    assert "decode throughput advantage" in render_text
    assert "KV elimination:" in render_text


def test_cli_profile_ssm(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "neural_cost.cli.profile_cmd.resolve_hardware",
        return_value=(
            HardwareSpec(name="MockGPU", peak_flops=1e14, memory_bandwidth=2e12),
            None,
        ),
    ):
        main(
            [
                "profile",
                "--arch",
                "ssm",
                "--batch-size",
                "1",
                "--seq-len",
                "4096",
                "--embed-dim",
                "4096",
                "--num-layers",
                "32",
                "--no-bench",
            ]
        )
    captured = capsys.readouterr().out
    assert "Architecture:        ssm" in captured
    assert "Recurrent State:" in captured
    assert "KV Elimination:" in captured


def test_cli_profile_mamba_json(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "neural_cost.cli.profile_cmd.resolve_hardware",
        return_value=(
            HardwareSpec(name="MockGPU", peak_flops=1e14, memory_bandwidth=2e12),
            None,
        ),
    ):
        main(
            [
                "profile",
                "--arch",
                "mamba",
                "--batch-size",
                "1",
                "--seq-len",
                "4096",
                "--embed-dim",
                "4096",
                "--num-layers",
                "32",
                "--no-bench",
                "--json",
            ]
        )
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["architecture"] == "mamba"
    assert data["total_parameters"] > 0
    assert data["state_bytes"] > 0
    assert data["memory_savings_ratio_vs_transformer"] > 0.9


def test_cli_llm_ssm(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "llm",
            "ssm",
            "--batch-size",
            "1",
            "--seq-len",
            "4096",
            "--embed-dim",
            "4096",
            "--num-layers",
            "32",
            "--decode",
            "--peak-flops",
            "1e14",
            "--memory-bandwidth",
            "2e12",
        ]
    )
    captured = capsys.readouterr().out
    assert "State Space Model Analysis (decode" in captured
    assert "KV elimination:" in captured
    assert "decode throughput advantage:" in captured


def test_cli_llm_ssm_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "llm",
            "ssm",
            "--batch-size",
            "1",
            "--seq-len",
            "4096",
            "--embed-dim",
            "4096",
            "--num-layers",
            "32",
            "--decode",
            "--peak-flops",
            "1e14",
            "--memory-bandwidth",
            "2e12",
            "--json",
        ]
    )
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["is_decode"] is True
    assert data["state_bytes"] > 0
    assert data["memory_savings_ratio_vs_transformer"] > 0.9
    assert data["bottleneck"] == "memory"


def test_cli_audit_ssm(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "neural_cost.cli.audit_cmd.resolve_hardware",
        return_value=(
            HardwareSpec(name="MockGPU", peak_flops=1e14, memory_bandwidth=2e12),
            None,
        ),
    ):
        main(
            [
                "audit",
                "--arch",
                "ssm",
                "--batch-size",
                "1",
                "--seq-len",
                "4096",
                "--embed-dim",
                "4096",
                "--num-layers",
                "32",
                "--max-vram",
                "16GB",
                "--max-latency-ms",
                "1000",
                "--no-bench",
            ]
        )
    captured = capsys.readouterr().out
    assert "AUDIT PASSED" in captured
