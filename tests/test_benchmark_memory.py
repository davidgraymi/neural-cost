import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from neural_cost import (
    Measurement,
    analyze_memory_gap,
    profile_model,
)
from neural_cost.adapters import TorchAdapter


class BenchmarkMemoryTelemetryTests(unittest.TestCase):
    def test_bench_record_schema_includes_memory_telemetry(self) -> None:
        from benchmarks.collect_data import BenchRecord

        record = BenchRecord(
            framework="PyTorch",
            variant="baseline",
            architecture="FF DNN",
            batch=8,
            flops=100_000,
            param_bytes=50_000,
            total_bytes=150_000,
            arith_intensity=0.67,
            latency_median_ms=0.5,
            latency_mean_ms=0.51,
            latency_stddev_ms=0.02,
            latency_cv_pct=3.9,
            latency_p95_ms=0.55,
            roofline_efficiency=0.15,
            achieved_gflops=0.2,
            achieved_gbw=0.3,
            bottleneck="memory",
            peak_allocated_bytes=60_000,
            peak_reserved_bytes=80_000,
            memory_overhead_ratio=1.2,
            theoretical_min_bytes=50_000,
            theoretical_conservative_bytes=70_000,
        )

        d = asdict(record)
        self.assertIn("peak_allocated_bytes", d)
        self.assertIn("peak_reserved_bytes", d)
        self.assertIn("memory_overhead_ratio", d)
        self.assertIn("theoretical_min_bytes", d)
        self.assertIn("theoretical_conservative_bytes", d)
        self.assertEqual(d["peak_allocated_bytes"], 60_000)
        self.assertEqual(d["peak_reserved_bytes"], 80_000)
        self.assertEqual(d["memory_overhead_ratio"], 1.2)

        # Ensure json serializable
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["peak_allocated_bytes"], 60_000)

    def test_torch_block_collects_memory_telemetry(self) -> None:
        pytest.importorskip("torch")
        import torch

        from benchmarks.collect_data import _torch_block

        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        ).eval()
        x = torch.randn(8, 64)

        samples, peak_alloc, peak_res = _torch_block(model, (x,), warmup=2, repeats=5)
        self.assertEqual(len(samples), 5)
        self.assertIsNotNone(peak_alloc)
        self.assertIsNotNone(peak_res)
        self.assertGreater(peak_alloc, 0)
        self.assertGreaterEqual(peak_res, peak_alloc)

    def test_torch_adapter_trace_collects_allocator_telemetry(self) -> None:
        pytest.importorskip("torch")
        import torch

        adapter = TorchAdapter()
        model = torch.nn.Linear(128, 64).eval()
        x = torch.randn(16, 128)

        meas = adapter.trace(model, x, warmup=2, repeats=3)
        self.assertIsInstance(meas, Measurement)
        self.assertEqual(len(meas.samples_seconds), 3)
        self.assertIsNotNone(meas.peak_memory_bytes)
        self.assertGreater(meas.peak_memory_bytes, 0)

    def test_torch_trace_time_counts_nested_ops_once(self) -> None:
        pytest.importorskip("torch")
        import torch

        adapter = TorchAdapter()
        model = torch.nn.Sequential(*[torch.nn.Linear(256, 256) for _ in range(4)]).eval()
        x = torch.randn(64, 256)

        with torch.no_grad():
            meas = adapter.trace(model, x, warmup=2, repeats=5)
        self.assertIsNotNone(meas.trace_device_time_seconds)
        self.assertGreater(meas.trace_device_time_seconds, 0)
        # Profiled op time sits inside the timed loop, so it cannot exceed the
        # wall clock. Summing inclusive times over nested ops roughly doubled it.
        self.assertLessEqual(meas.trace_device_time_seconds, sum(meas.samples_seconds))

    def test_memory_gap_correlation_with_profile(self) -> None:
        pytest.importorskip("torch")
        import torch

        from benchmarks.collect_data import _torch_block

        adapter = TorchAdapter()
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        ).eval()
        x = torch.randn(4, 32)

        prof = profile_model(model, (x,), adapter)
        mem = prof.memory
        self.assertGreater(mem.parameter_bytes, 0)
        self.assertGreater(mem.inference_minimum_bytes, 0)

        _samples, peak_alloc, peak_res = _torch_block(model, (x,), warmup=1, repeats=2)
        meas = Measurement(
            median_seconds=0.001,
            samples_seconds=(0.001, 0.001),
            peak_memory_bytes=peak_alloc,
            allocated_memory_bytes=peak_alloc,
            reserved_memory_bytes=peak_res,
        )
        mem_gap = analyze_memory_gap(mem, meas)
        self.assertEqual(mem_gap.observed_peak_bytes, peak_alloc)
        self.assertEqual(mem_gap.observed_reserved_bytes, peak_res)
        self.assertIsNotNone(mem_gap.overhead_ratio)
        self.assertGreater(len(mem_gap.findings), 0)

    def test_generate_report_rep_batch_and_memory_fig(self) -> None:
        pytest.importorskip("matplotlib")
        from benchmarks.generate_report import fig_memory_utilization, rep_batch

        records = [
            {
                "framework": "PyTorch",
                "variant": "baseline",
                "architecture": "FF DNN",
                "batch": 8,
                "theoretical_min_bytes": 50_000,
                "theoretical_conservative_bytes": 70_000,
                "peak_allocated_bytes": 60_000,
                "peak_reserved_bytes": 60_000,
                "memory_overhead_ratio": 1.2,
            },
            {
                "framework": "PyTorch",
                "variant": "compiled",
                "architecture": "FF DNN",
                "batch": 8,
                "theoretical_min_bytes": 50_000,
                "theoretical_conservative_bytes": 70_000,
                "peak_allocated_bytes": 58_000,
                "peak_reserved_bytes": 58_000,
                "memory_overhead_ratio": 1.16,
            },
        ]
        self.assertEqual(rep_batch(records, preferred=32), 8)
        self.assertEqual(rep_batch([{"batch": 32}, {"batch": 64}], preferred=32), 32)

        hw = {
            "name": "Mock Hardware",
            "peak_flops": 1e12,
            "memory_bandwidth": 50e9,
            "ridge_point": 20.0,
            "measured_bw_gb_s": 45.0,
            "source": "mock",
        }
        fig_path = fig_memory_utilization(hw, records)
        self.assertTrue(Path(fig_path).exists())


def test_torch_cpu_peak_is_high_water_not_sum_of_allocations() -> None:
    torch = pytest.importorskip("torch")

    def churn(x):
        for _ in range(5):
            block = torch.empty(2_500_000)  # 10 MB, freed before the next one
            del block
        return x

    meas = TorchAdapter().trace(churn, torch.ones(1), warmup=1, repeats=3)
    # Only one 10 MB block is ever live; summing allocations reported 150 MB.
    assert meas.peak_memory_bytes is not None
    assert 10_000_000 <= meas.peak_memory_bytes < 11_000_000


def test_torch_benchmark_times_without_profiler_and_keeps_trace_stats(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    profiling = []
    real_profile = torch.profiler.profile

    class Spy(real_profile):
        def __enter__(self):
            profiling.append(True)
            return super().__enter__()

        def __exit__(self, *exc):
            profiling.pop()
            return super().__exit__(*exc)

    monkeypatch.setattr(torch.profiler, "profile", Spy)
    states = []
    model = torch.nn.Linear(64, 64).eval()
    x = torch.randn(4, 64)

    def run(inp):
        states.append(bool(profiling))
        return model(inp)

    meas = TorchAdapter().benchmark(run, x, warmup=1, repeats=4, min_sample_seconds=None)
    # Warmup, overhead probes and 4 timed calls run unprofiled; only the last
    # 4 calls (the separate trace pass) run under the profiler.
    assert states[-4:] == [True] * 4
    assert not any(states[:-4]) and len(states) > 8
    assert meas.trace_event_count and meas.trace_event_count > 0
    assert meas.peak_memory_bytes is not None
    plain = TorchAdapter(use_profiler=False).benchmark(run, x, warmup=0, repeats=2)
    assert plain.trace_event_count is None


def test_torch_mps_reports_gpu_time_and_no_fake_peak() -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.randn(1024, 1024, device="mps")
    meas = TorchAdapter(use_profiler=False).benchmark(lambda a: a @ a, x, warmup=2, repeats=5)
    assert meas.peak_memory_bytes is None  # MPS has no peak counter
    assert meas.allocated_memory_bytes is not None
    assert 0 < meas.trace_device_time_seconds <= sum(meas.samples_seconds)


def test_measurement_spread_and_noise_warning() -> None:
    steady = Measurement(1.0, (0.99, 1.0, 1.0, 1.01, 1.0))
    assert steady.min_seconds == 0.99
    assert steady.relative_spread is not None and steady.relative_spread < 0.05
    assert steady.noise_warning() is None

    noisy = Measurement(1.0, (0.7, 1.0, 1.4, 0.8, 1.3))
    assert noisy.relative_spread > 0.10
    warning = noisy.noise_warning()
    assert warning is not None and "Noisy timing" in warning

    assert Measurement(1.0, (1.0, 2.0)).relative_spread is None  # too few samples


def test_analyze_gap_reports_noisy_timing() -> None:
    from neural_cost import CostEstimate, HardwareSpec, analyze_gap

    hardware = HardwareSpec("test", peak_flops=1e12, memory_bandwidth=1e11)
    estimate = CostEstimate(flops=10**9, read_bytes=10**6, write_bytes=10**6, operations=1)
    noisy = Measurement(1e-3, (0.7e-3, 1e-3, 1.4e-3, 0.8e-3, 1.3e-3))
    assert any("Noisy timing" in f for f in analyze_gap(estimate, noisy, hardware).findings)
    steady = Measurement(1e-3, (1e-3,) * 5)
    assert not any("Noisy timing" in f for f in analyze_gap(estimate, steady, hardware).findings)


def _call_wait_log(**options):
    from neural_cost.profiler import timed_samples

    log = []

    def call():
        log.append("c")
        return len(log)

    result = timed_samples(call, lambda out: log.append("w"), **options)
    return result, "".join(log)


def test_timed_samples_batches_short_calls_with_one_wait_per_sample() -> None:
    (samples, per_sample, overhead), log = _call_wait_log(
        warmup=2, repeats=3, min_sample_seconds=1e-3
    )
    assert per_sample > 1  # appending to a list takes far less than 1 ms
    assert log.endswith(("c" * per_sample + "w") * 3)
    assert log.startswith("cw" * 2 + "cw" * 5)  # warmup, then single-call probes
    assert len(samples) == 3 and overhead >= 0


def test_timed_samples_default_is_one_call_per_sample() -> None:
    (samples, per_sample, _), log = _call_wait_log(warmup=0, repeats=4)
    assert per_sample == 1 and len(samples) == 4
    assert log.endswith("cw" * 4)


def test_sync_overhead_is_single_call_minus_batched_cost() -> None:
    import time

    from neural_cost.profiler import timed_samples

    # A wait that costs 2 ms whenever it is called, like a GPU completion round trip.
    _, _, overhead = timed_samples(lambda: None, lambda out: time.sleep(2e-3), warmup=0, repeats=1)
    assert 1.8e-3 < overhead < 3e-3


def test_short_sample_warning_and_amortized_overhead() -> None:
    short = Measurement(10e-6, (10e-6,) * 5, sync_overhead_seconds=2e-6)
    assert short.sync_overhead_ratio == pytest.approx(0.2)
    assert "min_sample_seconds" in short.sync_overhead_warning()
    batched = Measurement(10e-6, (10e-6,) * 5, calls_per_sample=100, sync_overhead_seconds=2e-6)
    assert batched.sync_overhead_ratio == pytest.approx(0.002)
    assert batched.sync_overhead_warning() is None
    with pytest.raises(ValueError):
        Measurement(1.0, (1.0,), calls_per_sample=0)


def test_attribution_amortizes_call_overhead_for_batched_samples() -> None:
    from neural_cost import HardwareCalibration, HardwareSpec, Operation, attribute_gap

    ops = [Operation("fc", "linear", ((8, 64), (64, 64)), (8, 64), 4)]
    hardware = HardwareSpec("test", peak_flops=1e12, memory_bandwidth=1e11)
    calibration = HardwareCalibration(1e11, 1e10, dispatch_seconds=0, call_overhead_seconds=5e-6)
    single = attribute_gap(ops, Measurement(20e-6, (20e-6,) * 3), hardware, calibration)
    batched = attribute_gap(
        ops, Measurement(20e-6, (20e-6,) * 3, calls_per_sample=10), hardware, calibration
    )

    def dispatch(result):
        return next(c.seconds for c in result.components if c.name == "dispatch")

    assert dispatch(single) == pytest.approx(5e-6)
    assert dispatch(batched) == pytest.approx(0.5e-6)
