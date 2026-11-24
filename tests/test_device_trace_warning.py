"""A failed device trace warns instead of leaving trace_device_ms silently empty (#92)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from neural_cost.adapters import jax as jax_adapter
from neural_cost.adapters import tensorflow as tf_adapter


def _fake_jax(trace):
    return SimpleNamespace(profiler=SimpleNamespace(trace=trace))


def test_jax_trace_failure_warns():
    def broken_trace(log_dir):
        raise RuntimeError("CUPTI call error")

    with pytest.warns(RuntimeWarning, match="JAX: no device busy time.*CUPTI call error"):
        assert (
            jax_adapter._profile_device_time(_fake_jax(broken_trace), lambda: 1, (), {}, 2) is None
        )


def test_jax_trace_without_device_timeline_warns(monkeypatch):
    monkeypatch.setattr(jax_adapter, "_block", lambda jax, value: value)
    trace = _fake_jax(lambda log_dir: contextlib.nullcontext())
    with pytest.warns(RuntimeWarning, match="JAX: no device busy time.*no device timeline"):
        assert jax_adapter._profile_device_time(trace, lambda: 1, (), {}, 2) is None


def test_tensorflow_profiler_failure_warns():
    class OpError(Exception):
        pass

    def start(log_dir):
        raise OpError("CUPTI call error")

    tf = SimpleNamespace(
        errors=SimpleNamespace(OpError=OpError),
        profiler=SimpleNamespace(experimental=SimpleNamespace(start=start, stop=lambda: None)),
    )
    with pytest.warns(RuntimeWarning, match="TensorFlow: no device busy time"):
        assert tf_adapter._profile_device_time(tf, lambda: 1, (), {}, 2, lambda out: None) is None
