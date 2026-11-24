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
