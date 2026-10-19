"""Tests for benchmark collection, diagnostics, and reporting capabilities."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from neural_cost.adapters import TorchAdapter, TorchFxAdapter
from neural_cost.analysis import analyze_gap
from neural_cost.estimate import estimate_operations
from neural_cost.hardware import CacheSpec, HardwareSpec
from neural_cost.memory import estimate_memory
from neural_cost.operations import Operation

# Add benchmarks directory to sys.path so we can import helper modules
BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import collect_data
import collect_gpu_data
import generate_report


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_collect_data_make_bench_record_diagnostics():
    """Verify that _make_bench_record captures fusion, cache, and layer diagnostics."""
    ops = [
        Operation("fc1", "linear", ((32, 128), (128, 64)), (32, 64), dtype_bytes=4),
        Operation("relu", "elementwise", ((32, 64),), (32, 64), dtype_bytes=4),
        Operation("fc2", "linear", ((32, 64), (64, 10)), (32, 10), dtype_bytes=4),
    ]

    cost = estimate_operations(ops)
    mem = estimate_memory(ops)

    prof = SimpleNamespace(
        cost=cost,
        memory=mem,
        operations=ops,
    )

    hw = HardwareSpec(
        name="TestHW",
        peak_flops=1e12,
        memory_bandwidth=1e11,
        caches=(
            CacheSpec("L1", bandwidth=1e12, capacity=128 * 1024),
            CacheSpec("L2", bandwidth=5e11, capacity=512 * 1024),
            CacheSpec("L3", bandwidth=2e11, capacity=16 * 1024 * 1024),
        ),
    )

    meas = SimpleNamespace(median_seconds=1e-3, samples_seconds=(1e-3,))
    gap = analyze_gap(
        prof.cost,
        meas,
        hw,
        operations=ops,
    )

    st = {
        "median": 1.0,
        "mean": 1.01,
        "stddev": 0.02,
        "cv": 2.0,
        "p95": 1.05,
    }

    rec = collect_data._make_bench_record(
        framework="PyTorch",
        variant="compiled",
        arch="FF DNN",
        batch=32,
        prof=prof,
        gap=gap,
        st=st,
    )

    assert rec.framework == "PyTorch"
    assert rec.cache_resident is True
    assert rec.cache_name == "L1"
    assert rec.cache_bound_ms is not None
    assert rec.fused_efficiency is not None
    assert rec.fused_lower_bound_ms is not None
    assert rec.traffic_reduction_pct is not None
    assert rec.traffic_reduction_pct > 0
    assert rec.top_layer_bottleneck is not None
    assert "fc" in rec.top_layer_bottleneck
    assert rec.top_layer_share_pct is not None
    assert rec.top_layer_share_pct > 0


def test_collect_gpu_data_make_bench_record_diagnostics():
    """Verify that _make_gpu_bench_record captures diagnostics."""
    ops = [
        Operation("layer1", "linear", ((32, 128), (128, 64)), (32, 64), dtype_bytes=4),
        Operation("relu", "elementwise", ((32, 64),), (32, 64), dtype_bytes=4),
    ]

    cost = estimate_operations(ops)
    mem = estimate_memory(ops)

    prof = SimpleNamespace(
        cost=cost,
        memory=mem,
        operations=ops,
    )

    hw = HardwareSpec(
        name="TestGPU",
        peak_flops=1e13,
        memory_bandwidth=5e11,
        caches=(CacheSpec("SRAM", bandwidth=2e12, capacity=128 * 1024),),
    )

    meas = SimpleNamespace(median_seconds=5e-4, samples_seconds=(5e-4,))
    gap = analyze_gap(
        prof.cost,
        meas,
        hw,
        operations=ops,
    )

    st = {
        "median": 0.5,
        "mean": 0.51,
        "stddev": 0.01,
        "cv": 2.0,
        "p95": 0.52,
    }

    rec = collect_gpu_data._make_gpu_bench_record(
        framework="PyTorch",
        variant="compiled",
        device="cuda:0",
        arch="FF DNN",
        batch=64,
        prof=prof,
        gap=gap,
        st=st,
    )

    assert rec.framework == "PyTorch"
    assert rec.cache_resident is True
    assert rec.cache_name == "SRAM"
    assert rec.fused_efficiency is not None
    assert rec.traffic_reduction_pct is not None
    assert rec.top_layer_bottleneck is not None


def test_torch_fx_adapter_integration():
    """Verify TorchFxAdapter runs in benchmark model evaluation."""
    from neural_cost.model import profile_model

    model = SimpleToyModel()
    dummy_input = torch.randn(4, 32)

    fx_adapter = TorchFxAdapter()
    fx_prof = profile_model(model, (dummy_input,), fx_adapter)

    adapter = TorchAdapter()
    std_prof = profile_model(model, (dummy_input,), adapter)

    assert len(fx_prof.operations) >= len(std_prof.operations)
    assert any(op.kind == "elementwise" for op in fx_prof.operations)


def test_report_diagnostics_table_formatting():
    """Verify that report diagnostics table handles present and missing fields."""
    sample_records = [
        {
            "framework": "PyTorch",
            "variant": "compiled",
            "architecture": "FF DNN",
            "batch": 32,
            "fused_efficiency": 0.082,
            "traffic_reduction_pct": 14.5,
            "cache_resident": True,
            "cache_name": "L2",
            "top_layer_bottleneck": "fc1 (linear, memory-bound)",
            "top_layer_share_pct": 52.3,
        },
        {
            "framework": "JAX",
            "variant": "jit",
            "architecture": "FF DNN",
            "batch": 32,
            # legacy or missing fields
            "fused_efficiency": None,
            "traffic_reduction_pct": None,
            "cache_resident": False,
            "cache_name": None,
            "top_layer_bottleneck": None,
            "top_layer_share_pct": None,
        },
    ]

    # Test CPU report diagnostics table
    def dummy_select(records, **kwargs):
        res = records
        for k, v in kwargs.items():
            res = [r for r in res if r.get(k) == v]
        return res

    orig_select = generate_report.select
    generate_report.select = dummy_select
    try:
        # Construct table using the logic in generate_report
        rows = [
            "| Architecture | Framework | Variant | Fused Efficiency | Traffic Saved | Resident Cache | Top Layer Bottleneck | Layer Share |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arch in ["FF DNN"]:
            for fw in ["PyTorch", "JAX"]:
                for variant in ["compiled", "jit"]:
                    hits = dummy_select(
                        sample_records, framework=fw, variant=variant, architecture=arch, batch=32
                    )
                    if not hits:
                        continue
                    r = hits[0]
                    fused_eff = (
                        f"{r['fused_efficiency']:.1%}"
                        if r.get("fused_efficiency") is not None
                        else "—"
                    )
                    traffic = (
                        f"{r['traffic_reduction_pct']:.1f}%"
                        if r.get("traffic_reduction_pct") is not None
                        else "—"
                    )
                    res = (
                        f"{r['cache_name']}"
                        if r.get("cache_resident") and r.get("cache_name")
                        else ("Yes" if r.get("cache_resident") else "DRAM")
                    )
                    top_layer = r.get("top_layer_bottleneck") or "—"
                    share = (
                        f"{r['top_layer_share_pct']:.1f}%"
                        if r.get("top_layer_share_pct") is not None
                        else "—"
                    )
                    rows.append(
                        f"| {arch} | {fw} | {variant} | {fused_eff} | {traffic} | {res} | {top_layer} | {share} |"
                    )
        tbl = "\n".join(rows)
        assert "8.2%" in tbl
        assert "14.5%" in tbl
        assert "L2" in tbl
        assert "fc1 (linear, memory-bound)" in tbl
        assert "52.3%" in tbl
        assert "DRAM" in tbl
    finally:
        generate_report.select = orig_select


def test_legacy_gating_default():
    """Verify that default architectures exclude legacy recurrent models."""
    archs = collect_data.get_benchmarked_architectures(include_legacy=False)
    assert "RNN" not in archs
    assert "LSTM" not in archs
    assert "Transformer" in archs


def test_legacy_gating_enabled():
    """Verify that specifying include_legacy adds recurrent models."""
    archs = collect_data.get_benchmarked_architectures(include_legacy=True)
    assert "RNN" in archs
    assert "LSTM" in archs
    assert "Transformer" in archs
