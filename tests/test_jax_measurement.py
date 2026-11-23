"""Precision of JaxAdapter.benchmark memory and timing telemetry."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from neural_cost.adapters import jax as jax_adapter
from neural_cost.adapters.jax import JaxAdapter, compiled_peak_bytes


def _workload():
    fn = jax.jit(lambda w, x: jnp.tanh(x @ w) @ w.T)
    return fn, jnp.ones((256, 256)), jnp.ones((64, 256))


def test_jitted_peak_comes_from_compiled_plan_without_allocator_stats(monkeypatch):
    monkeypatch.setattr(jax_adapter, "_allocator_stats", lambda device: None)
    fn, w, x = _workload()
    meas = JaxAdapter().benchmark(fn, w, x, warmup=1, repeats=2)
    stats = fn.lower(w, x).compile().memory_analysis()
    expected = stats.argument_size_in_bytes + stats.output_size_in_bytes + stats.temp_size_in_bytes
    assert meas.peak_memory_bytes == expected
    # Arguments alone are 320 KiB; the plan must include them.
    assert meas.peak_memory_bytes >= (256 * 256 + 64 * 256) * 4


def test_unjitted_function_has_no_compiled_peak():
    assert compiled_peak_bytes(lambda x: x + 1, jnp.ones(4)) is None


def test_allocator_peak_used_only_when_the_run_raised_it(monkeypatch):
    fn, w, x = _workload()
    calls = iter(
        [
            {"peak_bytes_in_use": 9_000_000, "bytes_in_use": 1, "bytes_limit": 10**12},
            {
                "peak_bytes_in_use": 9_000_000,
                "bytes_in_use": 2,
                "bytes_limit": 10**12,
                "pool_bytes": 4_000_000,
            },
        ]
    )
    monkeypatch.setattr(jax_adapter, "_allocator_stats", lambda device: next(calls))
    meas = JaxAdapter().benchmark(fn, w, x, warmup=1, repeats=2)
    # A stale lifetime peak from earlier work is not this run's peak.
    assert meas.peak_memory_bytes != 9_000_000
    assert meas.peak_memory_bytes == compiled_peak_bytes(fn, w, x)
    assert meas.allocated_memory_bytes == 2
    # The capacity limit is not memory the allocator reserved.
    assert meas.reserved_memory_bytes == 4_000_000


def test_allocator_peak_reported_when_run_raised_it(monkeypatch):
    fn, w, x = _workload()
    calls = iter([{"peak_bytes_in_use": 100}, {"peak_bytes_in_use": 5_000, "bytes_in_use": 7}])
    monkeypatch.setattr(jax_adapter, "_allocator_stats", lambda device: next(calls))
    meas = JaxAdapter().benchmark(fn, w, x, warmup=1, repeats=2)
    assert meas.peak_memory_bytes == 5_000
    assert meas.reserved_memory_bytes is None


def test_device_found_from_outputs_when_inputs_are_closed_over():
    w = jnp.ones((32, 32))
    meas = JaxAdapter().benchmark(lambda: w @ w, warmup=1, repeats=2)
    assert meas.device is not None
