"""validate.py GPU backends for JAX and TensorFlow (issue #96)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import validate


def test_tf_workloads_are_pinned_to_the_requested_device():
    tf = pytest.importorskip("tensorflow")
    workloads = validate.tf_workloads(quick=True, device="/CPU:0")
    assert workloads
    for workload in workloads[:3]:
        (x,) = workload.inputs
        assert "CPU:0" in x.device
        assert all("CPU:0" in w.value.device for w in workload.model.weights)
        out = workload.runner(x)
        assert "CPU:0" in out.device
        assert isinstance(out, tf.Tensor)


def test_jax_gpu_backend_is_listed_when_jax_has_a_gpu(monkeypatch):
    pytest.importorskip("jax")
    gpu = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(
        validate, "_jax_device", lambda platform: gpu if platform == "gpu" else None
    )
    assert "jax-gpu" in validate.available_backends()


def test_jax_workloads_run_on_the_requested_device():
    jax = pytest.importorskip("jax")
    cpu = jax.devices("cpu")[0]
    workload = validate.jax_workloads(True, cpu)[-1]
    assert all(x.committed and x.devices() == {cpu} for x in workload.inputs)
    assert workload.runner(*workload.inputs).devices() == {cpu}
