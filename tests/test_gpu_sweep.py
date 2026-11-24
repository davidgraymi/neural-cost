"""collect_gpu_data.py sweep behaviour that needs no GPU."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))

import collect_gpu_data


@pytest.mark.parametrize("missing", [0, 2])
def test_missing_device_time_is_reported(capsys, missing):
    recs = [
        SimpleNamespace(device="gpu:0", trace_device_ms=None if i < missing else 1.0)
        for i in range(3)
    ] + [SimpleNamespace(device="cpu:0", trace_device_ms=None)]
    collect_gpu_data._report_missing_device_time("JAX", recs)
    out = capsys.readouterr().out
    assert ("JAX: no device busy time in 2/3 records" in out) == bool(missing)


def _fake_jax(available: dict[str, list]):
    def devices(backend):
        if backend not in available:
            raise RuntimeError(f"Unknown backend {backend}\nmore detail")
        return available[backend]

    return SimpleNamespace(devices=devices)


CPU0 = SimpleNamespace(id=0, device_kind="cpu", platform="cpu")
GPU0 = SimpleNamespace(id=0, device_kind="Tesla T4", platform="gpu")


def test_cuda_sweep_never_falls_back_to_jax_cpu(monkeypatch):
    # Issue #93: on a CUDA host where JAX's CUDA plugin fails, the JAX section
    # must be skipped rather than timed on the CPU and saved as GPU results.
    monkeypatch.setitem(sys.modules, "jax", _fake_jax({"cpu": [CPU0]}))
    dev, label = collect_gpu_data._detect_jax_gpu(cuda=True)
    assert dev is None
    assert label == "Unknown backend gpu"


def test_cuda_sweep_uses_jax_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", _fake_jax({"gpu": [GPU0], "cpu": [CPU0]}))
    dev, label = collect_gpu_data._detect_jax_gpu(cuda=True)
    assert dev is GPU0 and label == "gpu:0 (Tesla T4)"


def test_apple_sweep_keeps_cpu_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", _fake_jax({"cpu": [CPU0]}))
    dev, label = collect_gpu_data._detect_jax_gpu()
    assert dev is CPU0 and label == "cpu:0 (cpu)"


def test_metal_advice_is_not_printed_off_macos(monkeypatch, capsys):
    monkeypatch.setattr(collect_gpu_data.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "jax", _fake_jax({"cpu": [CPU0]}))
    collect_gpu_data._check_jax_mps_driver()
    assert "MPS/Metal" not in capsys.readouterr().out


# Issue #94: results are saved as the sweep goes, so a crash keeps them.


def test_checkpointed_reports_after_each_item():
    records: list[str] = []
    seen: list[list[str]] = []
    for item in collect_gpu_data._checkpointed(["a", "b"], records, lambda r: seen.append(list(r))):
        records.append(item)  # noqa: PERF402 - the generator reads records as it grows
    assert seen == [["a"], ["a", "b"]]


def test_checkpoint_survives_a_crash_mid_section():
    records: list[str] = []
    seen: list[list[str]] = []
    with pytest.raises(MemoryError):
        for item in collect_gpu_data._checkpointed(
            ["a", "b"], records, lambda r: seen.append(list(r))
        ):
            if item == "b":
                raise MemoryError
            records.append(item)
    assert seen == [["a"]]


def test_merge_records_replaces_matching_configurations():
    old = [
        {"framework": "JAX", "variant": "jit", "architecture": "CNN", "batch": 8, "latency": 1},
        {"framework": "JAX", "variant": "jit", "architecture": "CNN", "batch": 32, "latency": 2},
    ]
    new = [{"framework": "JAX", "variant": "jit", "architecture": "CNN", "batch": 8, "latency": 3}]
    merged = collect_gpu_data._merge_records(old, new)
    assert sorted(r["latency"] for r in merged) == [2, 3]


def test_write_json_atomic(tmp_path):
    import json

    path = tmp_path / "out" / "benchmark_gpu_data.json"
    collect_gpu_data._write_json_atomic(path, {"complete": False, "records": [1]})
    collect_gpu_data._write_json_atomic(path, {"complete": True, "records": [1, 2]})
    assert json.loads(path.read_text()) == {"complete": True, "records": [1, 2]}
    assert [p.name for p in path.parent.iterdir()] == [path.name]
    assert collect_gpu_data._previous_records(path) == [1, 2]
    assert collect_gpu_data._previous_records(tmp_path / "missing.json") == []
