"""Precision of TensorFlowAdapter.benchmark synchronisation and telemetry."""

import types

import pytest

from neural_cost.adapters import tensorflow as tf_adapter
from neural_cost.adapters.tensorflow import TensorFlowAdapter


@pytest.fixture
def tf():
    # Import inside the test, never at collection: loading TensorFlow before
    # torch._dynamo imports Triton aborts the interpreter on Linux CI.
    return pytest.importorskip("tensorflow")


def test_every_sample_waits_for_the_device(monkeypatch, tf):
    calls = []
    monkeypatch.setattr(tf.test.experimental, "sync_devices", lambda: calls.append(1))
    x = tf.ones((8, 8))
    meas = TensorFlowAdapter().benchmark(lambda a: tf.matmul(a, a), x, warmup=2, repeats=3)
    assert len(calls) == 5  # each warmup and timed call is synchronised
    assert len(meas.samples_seconds) == 3


def test_falls_back_to_waiting_on_outputs_without_sync_devices(monkeypatch, tf):
    monkeypatch.setattr(tf.test, "experimental", types.SimpleNamespace())
    waited = []
    real_reshape = tf.reshape

    def spy(value, shape):
        waited.append(value.shape)
        return real_reshape(value, shape)

    monkeypatch.setattr(tf, "reshape", spy)
    sync = tf_adapter._device_sync(tf)
    sync({"a": tf.ones((2, 3)), "b": tf.zeros((0,)), "c": 4})
    assert waited == [tf.TensorShape((2, 3))]  # empty tensors and non-tensors skipped


def test_memory_device_follows_the_inputs(monkeypatch, tf):
    gpu = types.SimpleNamespace(name="/device:GPU:0")
    monkeypatch.setattr(tf.config, "list_logical_devices", lambda kind: [gpu, gpu])
    on_gpu1 = types.SimpleNamespace(device="/job:localhost/replica:0/task:0/device:GPU:1")
    on_cpu = types.SimpleNamespace(device="/job:localhost/replica:0/task:0/device:CPU:0")
    assert tf_adapter._input_gpu(tf, (on_gpu1,), {}) == "GPU:1"
    assert tf_adapter._input_gpu(tf, (on_cpu,), {}) is None
    assert tf_adapter._input_gpu(tf, (), {}) == "GPU:0"
    monkeypatch.setattr(tf.config, "list_logical_devices", lambda kind: [])
    assert tf_adapter._input_gpu(tf, (on_gpu1,), {}) is None


def test_peak_reset_after_warmup(monkeypatch, tf):
    events = []
    monkeypatch.setattr(tf_adapter, "_input_gpu", lambda tf, a, k: "GPU:0")
    monkeypatch.setattr(
        tf.config.experimental, "reset_memory_stats", lambda d: events.append("reset")
    )
    monkeypatch.setattr(
        tf.config.experimental,
        "get_memory_info",
        lambda d: events.append("read") or {"peak": 10, "current": 4},
    )

    def run(a):
        events.append("call")
        return a

    meas = TensorFlowAdapter().benchmark(run, tf.ones(2), warmup=2, repeats=2)
    assert events == ["call", "call", "reset", "call", "call", "read"]
    assert (meas.peak_memory_bytes, meas.allocated_memory_bytes, meas.device) == (10, 4, "GPU:0")
