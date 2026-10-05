"""Tests for activation checkpointing, selective recomputation, and training memory modeling."""

from __future__ import annotations

import json

import pytest

from neural_cost import (
    ActivationCheckpointingEstimate,
    ActivationCheckpointingGapAnalysis,
    CheckpointingStrategy,
    HardwareSpec,
    analyze_activation_checkpointing,
    estimate_activation_checkpointing,
)
from neural_cost.cli.root import build_parser

# ---------------------------------------------------------------------------
# Analytical Checkpointing Estimation Tests
# ---------------------------------------------------------------------------


def test_checkpointing_strategy_none() -> None:
    est = estimate_activation_checkpointing(
        batch_size=4,
        seq_len=2048,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        strategy=CheckpointingStrategy.NONE,
    )

    assert isinstance(est, ActivationCheckpointingEstimate)
    assert est.strategy == CheckpointingStrategy.NONE
    assert est.checkpointed_activation_bytes == est.uncheckpointed_activation_bytes
    assert est.activation_memory_saved_bytes == 0
    assert est.activation_savings_ratio == 0.0
    assert est.recompute_flops == 0
    assert est.compute_overhead_ratio == 0.0
    assert est.backward_flops == 2 * est.forward_flops
    assert est.total_training_flops == 3 * est.forward_flops


def test_checkpointing_strategy_full() -> None:
    est = estimate_activation_checkpointing(
        batch_size=4,
        seq_len=2048,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        strategy="full",
    )

    assert est.strategy == CheckpointingStrategy.FULL
    # Full checkpointing saves > 85% of intermediate activation memory
    assert est.activation_savings_ratio > 0.85
    # Full checkpointing recomputes the entire forward pass
    assert est.recompute_flops == est.forward_flops
    # Total training FLOPs increases from 3x to 4x forward (+33.3% overhead)
    assert est.compute_overhead_ratio == pytest.approx(1.0 / 3.0, rel=1e-2)
    assert est.total_training_flops == 4 * est.forward_flops


def test_checkpointing_strategy_selective() -> None:
    est = estimate_activation_checkpointing(
        batch_size=4,
        seq_len=2048,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        strategy=CheckpointingStrategy.SELECTIVE,
    )

    assert est.strategy == CheckpointingStrategy.SELECTIVE
    # Selective checkpointing saves 50-70% activation memory
    assert 0.50 < est.activation_savings_ratio < 0.80
    # But recompute overhead is tiny (< 2% of training compute)
    assert 0.0 < est.compute_overhead_ratio < 0.02
    assert est.recompute_flops > 0


def test_flash_attention_vs_quadratic() -> None:
    # Standard quadratic attention vs FlashAttention
    flash_est = estimate_activation_checkpointing(
        batch_size=2,
        seq_len=8192,
        embed_dim=2048,
        num_layers=8,
        num_heads=16,
        strategy="none",
        is_flash_attention=True,
    )
    unfused_est = estimate_activation_checkpointing(
        batch_size=2,
        seq_len=8192,
        embed_dim=2048,
        num_layers=8,
        num_heads=16,
        strategy="none",
        is_flash_attention=False,
    )

    # Un-fused materializes full S x S matrix, vastly exceeding FlashAttention
    assert (
        unfused_est.uncheckpointed_activation_bytes > flash_est.uncheckpointed_activation_bytes * 2
    )


def test_estimate_activation_checkpointing_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        estimate_activation_checkpointing(0, 2048, 4096, 32, 32)
    with pytest.raises(ValueError, match="must be positive"):
        estimate_activation_checkpointing(4, 2048, 4096, 32, 32, dtype_bytes=0)
    with pytest.raises(ValueError, match="Unknown checkpointing strategy"):
        estimate_activation_checkpointing(4, 2048, 4096, 32, 32, strategy="invalid_strat")


# ---------------------------------------------------------------------------
# Activation Checkpointing Gap Analysis Tests
# ---------------------------------------------------------------------------


def test_analyze_activation_checkpointing_prevents_oom() -> None:
    hw_80gb = HardwareSpec(
        name="H100-80GB",
        peak_flops=989e12,
        memory_bandwidth=3.35e12,
        memory_capacity=80_000_000_000,
    )
    # Long context training on 7B model:
    # B=4, S=4096, D=4096, L=32
    est_none = estimate_activation_checkpointing(
        batch_size=4,
        seq_len=4096,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        strategy=CheckpointingStrategy.NONE,
    )
    gap_none = analyze_activation_checkpointing(est_none, hw_80gb)
    # Uncheckpointed requires > 100 GB -> OOM!
    assert gap_none.is_oom

    est_selective = estimate_activation_checkpointing(
        batch_size=4,
        seq_len=4096,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        strategy=CheckpointingStrategy.SELECTIVE,
    )
    gap_selective = analyze_activation_checkpointing(est_selective, hw_80gb)
    # Selective fits inside 80 GB!
    assert not gap_selective.is_oom
    assert gap_selective.uncheckpointed_is_oom
    assert gap_selective.max_trainable_seq_len > 4000

    rendered = gap_selective.render()
    assert "Activation Checkpointing Analysis (Strategy: SELECTIVE" in rendered
    assert "Checkpointing prevents Out-Of-Memory (OOM)" in rendered
    assert "max trainable seq_len" in rendered


def test_analyze_activation_checkpointing_throughput() -> None:
    hw = HardwareSpec(
        name="A100-40GB",
        peak_flops=312e12,
        memory_bandwidth=1.55e12,
        memory_capacity=40_000_000_000,
    )
    est = estimate_activation_checkpointing(
        batch_size=2,
        seq_len=2048,
        embed_dim=2048,
        num_layers=16,
        num_heads=16,
        strategy=CheckpointingStrategy.FULL,
    )
    analysis = analyze_activation_checkpointing(est, hw)

    assert isinstance(analysis, ActivationCheckpointingGapAnalysis)
    assert analysis.total_step_seconds > 0
    assert analysis.throughput_tokens_per_second > 0
    assert analysis.recompute_seconds == pytest.approx(analysis.forward_seconds)


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------


def test_cli_llm_checkpointing(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "llm",
            "checkpointing",
            "--strategy",
            "selective",
            "--batch-size",
            "2",
            "--seq-len",
            "2048",
            "--embed-dim",
            "2048",
            "--num-layers",
            "16",
            "--vram-gb",
            "40.0",
            "--peak-flops",
            "312e12",
            "--memory-bandwidth",
            "1.55e12",
            "--json",
        ]
    )
    ret = args.func(args)
    assert ret == 0

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["strategy"] == "selective"
    assert data["activation_savings_ratio"] > 0.5
    assert not data["is_oom"]
    assert data["throughput_tokens_per_second"] > 0


def test_cli_profile_with_checkpointing(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "profile",
            "--arch",
            "transformer",
            "--batch-size",
            "2",
            "--seq-len",
            "1024",
            "--embed-dim",
            "1024",
            "--num-layers",
            "8",
            "--num-heads",
            "8",
            "--checkpointing",
            "selective",
            "--training",
            "--peak-flops",
            "100e12",
            "--memory-bandwidth",
            "1e12",
            "--json",
        ]
    )
    ret = args.func(args)
    assert ret == 0

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["is_training"] is True
    assert "checkpointing" in data
    assert data["checkpointing"]["strategy"] == "selective"
    assert data["checkpointing"]["activation_savings_ratio"] > 0.5
    assert data["gradient_bytes"] > 0
    assert data["optimizer_bytes"] > 0
