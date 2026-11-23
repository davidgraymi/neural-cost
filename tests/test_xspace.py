"""Device busy time from profiler XSpace traces."""

import pytest

from neural_cost.profiler import Measurement
from neural_cost.xspace import DeviceTime, device_time, read_events

GPU = "/device:GPU:0"


def test_overlapping_streams_count_once():
    events = [
        (GPU, "Stream #13(Compute)", 0, 100),
        (GPU, "Stream #14(Compute)", 50, 100),  # overlaps the first by 50 ns
        (GPU, "Stream #13(Compute)", 300, 50),
    ]
    result = device_time(events)
    assert result.busy_seconds == {GPU: pytest.approx(200e-9)}
    assert result.event_count == 3


def test_derived_lines_and_host_planes_are_ignored():
    events = [
        (GPU, "Stream #13(Compute)", 0, 10),
        (GPU, "Stream #13(Compute)", 90, 10),
        (GPU, "XLA Modules", 0, 100),  # spans the idle gap between kernels
        ("/host:CPU", "tf_Compute", 0, 1_000),
    ]
    assert device_time(events).total_seconds == pytest.approx(20e-9)


def test_tpu_uses_xla_ops_line_and_sums_devices():
    events = [
        ("/device:TPU:0", "XLA Ops", 0, 40),
        ("/device:TPU:0", "XLA Modules", 0, 100),
        ("/device:TPU:1", "XLA Ops", 0, 60),
    ]
    result = device_time(events)
    assert result.total_seconds == pytest.approx(100e-9)
    assert set(result.busy_seconds) == {"/device:TPU:0", "/device:TPU:1"}


def test_cpu_only_trace_has_no_device_time():
    assert device_time([("/host:CPU", "tf_Compute", 0, 1_000)]) is None


def _serialized_trace() -> bytes:
    xplane = pytest.importorskip("tensorflow.tsl.profiler.protobuf.xplane_pb2")
    space = xplane.XSpace()
    plane = space.planes.add(id=1, name=GPU)
    plane.event_metadata[1].id = 1
    plane.event_metadata[1].name = "gemm"
    for line_id, name, offsets in ((1, "Stream #7(Compute)", (0, 2_000_000)), (2, "XLA Ops", (0,))):
        line = plane.lines.add(id=line_id, display_id=line_id, name=name, timestamp_ns=1_000)
        for offset_ps in offsets:
            line.events.add(metadata_id=1, offset_ps=offset_ps, duration_ps=1_000_000)
    host = space.planes.add(id=2, name="/host:CPU")
    host.lines.add(id=1, name="main", timestamp_ns=0).events.add(metadata_id=0, duration_ps=10**9)
    return space.SerializeToString()


@pytest.mark.parametrize("parser", ["tensorflow", "jax"])
def test_both_parsers_read_the_same_device_time(parser):
    serialized = _serialized_trace()
    if parser == "jax":
        pytest.importorskip("jax")
    result = device_time(read_events(serialized, parser))
    # Two 1 µs kernels on the stream; the XLA Ops line is ignored when streams exist.
    assert result.total_seconds == pytest.approx(2e-6)
    assert result.event_count == 2


def test_jax_benchmark_fills_device_time_from_profiler(monkeypatch):
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from neural_cost.adapters import jax as jax_adapter

    class FakeAccelerator:
        platform = "gpu"

    monkeypatch.setattr(jax_adapter, "_first_device", lambda jax, tree: FakeAccelerator())
    monkeypatch.setattr(
        jax_adapter,
        "_profile_device_time",
        lambda *a: DeviceTime({GPU: 0.004}, 12),
    )
    meas = jax_adapter.JaxAdapter().benchmark(lambda x: x + 1, jnp.ones(4), warmup=1, repeats=2)
    assert (meas.trace_device_time_seconds, meas.trace_event_count) == (0.004, 12)
    off = jax_adapter.JaxAdapter(trace_device_time=False).benchmark(
        lambda x: x + 1, jnp.ones(4), warmup=1, repeats=2
    )
    assert off.trace_device_time_seconds is None


def test_jax_profiler_pass_runs_and_reports_nothing_on_cpu():
    jax = pytest.importorskip("jax")
    from neural_cost.adapters.jax import _profile_device_time

    fn = jax.jit(lambda x: x @ x)
    assert _profile_device_time(jax, fn, (jax.numpy.ones((64, 64)),), {}, 2) is None


def test_tf_profiler_pass_runs_and_reports_nothing_on_cpu():
    tf = pytest.importorskip("tensorflow")
    from neural_cost.adapters.tensorflow import _device_sync, _profile_device_time

    x = tf.ones((64, 64))
    result = _profile_device_time(tf, lambda a: tf.matmul(a, a), (x,), {}, 2, _device_sync(tf))
    assert result is None


def test_tf_benchmark_fills_device_time_on_gpu(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from neural_cost.adapters import tensorflow as tf_adapter

    monkeypatch.setattr(tf_adapter, "_input_gpu", lambda tf, a, k: "GPU:0")
    monkeypatch.setattr(tf.config.experimental, "reset_memory_stats", lambda d: None)
    monkeypatch.setattr(tf.config.experimental, "get_memory_info", lambda d: {"peak": 1})
    monkeypatch.setattr(tf_adapter, "_profile_device_time", lambda *a: DeviceTime({GPU: 0.5}, 3))
    meas = tf_adapter.TensorFlowAdapter().benchmark(lambda a: a, tf.ones(2), warmup=0, repeats=2)
    assert isinstance(meas, Measurement)
    assert (meas.trace_device_time_seconds, meas.trace_event_count) == (0.5, 3)
