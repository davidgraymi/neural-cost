"""Torch-only pieces of the runtime model: kernel tracing, replay, calibration."""

import pytest

torch = pytest.importorskip("torch")

from neural_cost.adapters import TorchAdapter


def _mlp():
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))


def test_trace_kernels_marks_uncaptured_and_dispatch_level():
    adapter = TorchAdapter()
    records = adapter.trace_kernels(_mlp(), (torch.ones(2, 16),))
    names = [r.name for r in records]
    assert sum(r.heavy for r in records) == 2  # two addmm
    assert any("gelu" in n for n in names)
    assert all(r.python_dispatch for r in records)
    assert adapter.count_kernels(_mlp(), (torch.ones(2, 16),)) == len(records)

    lstm = torch.nn.LSTM(8, 8, batch_first=True)
    lstm_records = adapter.trace_kernels(lstm, (torch.ones(1, 6, 8),))
    # Internal per-timestep kernels of the C++ LSTM do not pay Python dispatch.
    assert sum(not r.python_dispatch for r in lstm_records) > 0


def test_rnn_recurrent_projection_is_sequential():
    lstm = torch.nn.LSTM(8, 8, batch_first=True)
    ops = TorchAdapter().operations(lstm, (torch.ones(3, 5, 8),))
    hh = next(op for op in ops if op.name.endswith(".hh"))
    assert hh.inputs[0] == (3, 8)
    assert hh.attrs["repeat"] == 5


def test_replay_returns_aligned_positive_times():
    adapter = TorchAdapter()
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, stride=2, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 4, 2),
    )
    ops = adapter.operations(model, (torch.ones(1, 3, 8, 8),))
    times = adapter.replay(ops)
    assert len(times) == len(ops)
    assert all(t is not None and t > 0 for t in times)


def test_calibrate_quick_produces_curves():
    cal = TorchAdapter().calibrate("cpu", quick=True)
    assert cal.flops_curve and cal.bandwidth_curve
    assert cal.achievable_flops > 0 and cal.dispatch_seconds > 0
    assert not cal.async_dispatch


def test_benchmark_without_profiler():
    measurement = TorchAdapter(use_profiler=False).benchmark(_mlp(), torch.ones(2, 16), repeats=3)
    assert measurement.median_seconds > 0
    assert len(measurement.samples_seconds) == 3
