"""generate_gpu_report: methodology wording (issue #91) and partial-run warning (#94)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))

import generate_gpu_report as report

M3_DATA = BENCH / "results" / "apple-m3" / "benchmark_gpu_data.json"


@pytest.mark.parametrize(
    ("hw", "expected"),
    [
        ({"torch_device": "cuda:0 (Tesla T4)", "name": "Tesla T4"}, "cuda"),
        ({"torch_device": "mps (Apple Silicon GPU)", "name": "Apple M3"}, "mps"),
        ({"torch_device": "cpu (no GPU found)", "name": "CPU"}, "cpu"),
        ({"name": "Apple M3 (10-core GPU)"}, "mps"),
        ({"name": "Tesla T4"}, "cuda"),
    ],
)
def test_gpu_backend(hw, expected):
    assert report.gpu_backend(hw) == expected


def _methodology(hw: dict, records: list[dict]) -> str:
    report.ARCHS = report.get_active_architectures(records)
    figures = {
        name: report.FIG_DIR / f"gpu_{name}.png"
        for name in (
            "roofline",
            "latency_bars",
            "heatmap",
            "batch_scaling",
            "speedup",
            "framework_comparison",
            "throughput",
            "throughput_scaling",
            "per_sample_latency",
            "crossover",
            "memory",
        )
    }
    md = report.build_report(hw, records, figures)
    return md.split("## Methodology", 1)[1].split("## Figure", 1)[0]


def test_cuda_report_does_not_describe_metal():
    hw, records = report.load(M3_DATA)
    cuda_hw = {**hw, "name": "Tesla T4", "torch_device": "cuda:0 (Tesla T4)"}
    text = _methodology(cuda_hw, records)
    assert "Eager GPU (`cuda`)" in text
    assert "(`mps`)" not in text
    assert "Metal" not in text
    assert "jax-mps" not in text
    mps_text = _methodology(hw, records)
    assert "Eager GPU (`mps`)" in mps_text and "Metal" in mps_text


def test_partial_run_warning():
    assert report.partial_run_warning({"records": []}) == ""
    assert report.partial_run_warning({"complete": True, "records": []}) == ""
    text = report.partial_run_warning(
        {"complete": False, "records": [{"framework": "PyTorch"}, {"framework": "PyTorch"}]}
    )
    assert "Partial run" in text and "2 configurations" in text and "PyTorch" in text
