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
