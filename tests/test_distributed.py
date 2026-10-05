"""Tests for distributed 3D parallelism cost estimation, interconnect rooflines, and CLI command."""

from __future__ import annotations

import json

import pytest

from neural_cost.analysis import DistributedGapAnalysis, analyze_distributed_gap
from neural_cost.cli.root import main
from neural_cost.estimate import (
    estimate_parallelism,
)
from neural_cost.hardware import (
    ClusterTopology,
    HardwareSpec,
    InterconnectSpec,
    get_interconnect_preset,
)


def test_interconnect_spec_transfer_time() -> None:
    nvlink = InterconnectSpec(name="TestLink", bandwidth=100e9, latency_seconds=1.0e-6)
    assert nvlink.transfer_time_seconds(0) == 0.0
    # 100 GB in 100 GB/s = 1.0s + 1.0 us
    time_s = nvlink.transfer_time_seconds(100 * 10**9)
    assert pytest.approx(time_s, rel=1e-4) == 1.000001

    with pytest.raises(ValueError, match="bandwidth must be positive"):
        InterconnectSpec(name="Bad", bandwidth=-10)
    with pytest.raises(ValueError, match="latency_seconds cannot be negative"):
        InterconnectSpec(name="Bad", bandwidth=1e9, latency_seconds=-0.5)


def test_interconnect_presets() -> None:
    nvlink4 = get_interconnect_preset("nvlink4")
    assert nvlink4.bandwidth == 900e9
    ndr = get_interconnect_preset("infiniband_ndr")
    assert ndr.bandwidth == 50e9

    with pytest.raises(ValueError, match="Unknown interconnect preset"):
        get_interconnect_preset("quantum_entanglement_bus")


def test_cluster_topology_properties() -> None:
    gpu = HardwareSpec(
        name="H100", peak_flops=989e12, memory_bandwidth=3.35e12, memory_capacity=80 * 1024**3
    )
    topo = ClusterTopology(device=gpu, num_nodes=4, devices_per_node=8)
    assert topo.total_devices == 32
    assert topo.total_peak_flops == 32 * 989e12
    assert topo.total_memory_capacity == 32 * 80 * 1024**3

    with pytest.raises(ValueError, match="must be positive"):
        ClusterTopology(device=gpu, num_nodes=0)


def test_estimate_parallelism_tp_scaling() -> None:
    # 7B model
    est_tp1 = estimate_parallelism(
        total_parameters=7_000_000_000,
        batch_size=8,
        seq_len=2048,
        embed_dim=4096,
        num_layers=32,
        tp_degree=1,
    )
    assert est_tp1.tp_bytes_per_step == 0

    est_tp2 = estimate_parallelism(
        total_parameters=7_000_000_000,
        batch_size=8,
        seq_len=2048,
        embed_dim=4096,
        num_layers=32,
        tp_degree=2,
    )
    assert est_tp2.tp_bytes_per_step > 0

    est_tp4 = estimate_parallelism(
        total_parameters=7_000_000_000,
        batch_size=8,
        seq_len=2048,
        embed_dim=4096,
        num_layers=32,
        tp_degree=4,
    )
    # TP=4 communication factor is 2*(3/4)=1.5 vs TP=2 factor 2*(1/2)=1.0 (1.5x)
    assert est_tp4.tp_bytes_per_step > est_tp2.tp_bytes_per_step


def test_estimate_parallelism_pp_bubble() -> None:
    # PP=4, Microbatches=4: bubble = (4-1)/(4+4-1) = 3/7 ~ 42.8%
    est = estimate_parallelism(
        total_parameters=7_000_000_000,
        batch_size=16,
        seq_len=2048,
        embed_dim=4096,
        num_layers=32,
        pp_degree=4,
        num_microbatches=4,
    )
    assert pytest.approx(est.pp_bubble_fraction, rel=1e-3) == 3 / 7
    assert est.pp_bytes_per_step > 0


def test_estimate_parallelism_dp_sharding_modes() -> None:
    params = 10_000_000_000  # 10B
    # DDP
    ddp = estimate_parallelism(
        total_parameters=params,
        batch_size=16,
        seq_len=1024,
        embed_dim=4096,
        num_layers=32,
        dp_degree=4,
        dp_mode="ddp",
    )
    # ZeRO-3 / FSDP
    fsdp = estimate_parallelism(
        total_parameters=params,
        batch_size=16,
        seq_len=1024,
        embed_dim=4096,
        num_layers=32,
        dp_degree=4,
        dp_mode="zero3_fsdp",
    )
    # FSDP parameter and optimizer memory per device should be sharded by 4x vs DDP
    assert fsdp.per_device_param_bytes == ddp.per_device_param_bytes // 4
    assert fsdp.per_device_optimizer_bytes == ddp.per_device_optimizer_bytes // 4
    # But FSDP has higher communication volume (AllGather fwd + AllGather bwd + ReduceScatter grads = 3x vs 2x)
    assert fsdp.dp_bytes_per_step > ddp.dp_bytes_per_step


def test_analyze_distributed_gap_compute_bound() -> None:
    gpu = HardwareSpec(
        name="MockH100", peak_flops=1000e12, memory_bandwidth=3e12, memory_capacity=80 * 1024**3
    )
    intra = InterconnectSpec(name="FastNVLink", bandwidth=1800e9)
    inter = InterconnectSpec(name="FastInfiniBand", bandwidth=100e9)
    topo = ClusterTopology(
        device=gpu, num_nodes=2, devices_per_node=8, intra_node=intra, inter_node=inter
    )

    cost = estimate_parallelism(
        total_parameters=7_000_000_000,
        batch_size=64,
        seq_len=2048,
        embed_dim=4096,
        num_layers=32,
        tp_degree=8,
        dp_degree=2,
    )
    gap = analyze_distributed_gap(cost, topo)
    assert isinstance(gap, DistributedGapAnalysis)
    assert gap.memory_fit is True
    assert gap.model_flops_utilization > 0.0
    assert gap.tokens_per_second > 0.0
    render_text = gap.render()
    assert "Distributed 3D Parallelism Roofline" in render_text
    assert "MFU=" in render_text


def test_analyze_distributed_gap_oom() -> None:
    # 70B model with DP=1 on small 16GB GPUs
    small_gpu = HardwareSpec(
        name="SmallGPU", peak_flops=100e12, memory_bandwidth=500e9, memory_capacity=16 * 1024**3
    )
    topo = ClusterTopology(device=small_gpu, num_nodes=1, devices_per_node=1)
    cost = estimate_parallelism(
        total_parameters=70_000_000_000,
        batch_size=8,
        seq_len=4096,
        embed_dim=8192,
        num_layers=80,
        dp_mode="ddp",
    )
    gap = analyze_distributed_gap(cost, topo)
    assert gap.memory_fit is False
    assert gap.bottleneck == "out_of_memory"
    assert "VRAM capacity exceeded" in gap.findings[0]


def test_cli_distributed_command(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "distributed",
            "--parameters",
            "7e9",
            "--batch-size",
            "16",
            "--seq-len",
            "2048",
            "--embed-dim",
            "4096",
            "--num-layers",
            "32",
            "--num-nodes",
            "2",
            "--devices-per-node",
            "8",
            "--tp",
            "8",
            "--dp",
            "2",
        ]
    )
    captured = capsys.readouterr().out
    assert "Distributed 3D Parallelism Roofline" in captured
    assert "Cluster Topology:" in captured
    assert "Parallelism Scheme:  TP=8 x PP=1 x DP=2" in captured
    assert "Model FLOPs Util:" in captured


def test_cli_distributed_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "distributed",
            "--parameters",
            "7e9",
            "--batch-size",
            "16",
            "--seq-len",
            "2048",
            "--embed-dim",
            "4096",
            "--num-layers",
            "32",
            "--num-nodes",
            "2",
            "--devices-per-node",
            "8",
            "--tp",
            "8",
            "--dp",
            "2",
            "--json",
        ]
    )
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["cluster"]["total_devices"] == 16
    assert data["parallelism"]["tp"] == 8
    assert data["parallelism"]["dp"] == 2
    assert "model_flops_utilization" in data["performance"]
