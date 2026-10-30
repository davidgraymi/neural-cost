"""Device-aware hardware selection (issue #51)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neural_cost import (
    CostEstimate,
    HardwareSpec,
    Measurement,
    analyze_gap,
    detect_hardware,
    device_mismatch,
    normalize_device_type,
)

APPLE_M3 = ("Apple M3", 3.6e12, 100e9, ())


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", "cpu"),
        ("cpu:0", "cpu"),
        ("TFRT_CPU_0", "cpu"),
        ("cuda:0", "gpu"),
        ("mps", "gpu"),
        ("METAL", "gpu"),
        ("MpsDevice(id=0)", "gpu"),
        ("TPU_0", "tpu"),
        (SimpleNamespace(platform="cpu"), "cpu"),
        (SimpleNamespace(type="mps"), "gpu"),
        (SimpleNamespace(device=SimpleNamespace(platform="gpu")), "gpu"),
        ("banana", None),
        (None, None),
    ],
)
def test_normalize_device_type(device, expected):
    assert normalize_device_type(device) == expected


@patch(
    "neural_cost.hardware_detect._sysctl_int",
    side_effect={"hw.perflevel0.physicalcpu": 4, "hw.perflevel1.physicalcpu": 4}.get,
)
@patch("neural_cost.hardware_detect._probe_cpu_flops", return_value=(8, None))
@patch("neural_cost.hardware_detect._probe_nvidia", return_value=(None, None, None, ()))
@patch("neural_cost.hardware_detect._probe_apple_silicon", return_value=APPLE_M3)
def test_cpu_device_skips_apple_gpu_table(_apple, _nvidia, _cpu, _sysctl):
    gpu, _ = detect_hardware(benchmark_memory=False)
    cpu, detection = detect_hardware(benchmark_memory=False, device="cpu")
    assert gpu.device_type == "gpu" and gpu.peak_flops == 3.6e12
    assert cpu.device_type == "cpu"
    assert cpu.name == "Apple M3 (CPU)"
    # NEON peak: 4 P-cores x 4.05 GHz x 32 + 4 E-cores x 2.75 GHz x 16.
    assert cpu.peak_flops == pytest.approx(4 * 4.05e9 * 32 + 4 * 2.75e9 * 16)
    # Unified memory: without a measurement the chip's bandwidth is used.
    assert cpu.memory_bandwidth == 100e9
    assert "CPU estimate" in detection.source


@patch("neural_cost.hardware_detect._probe_apple_silicon", return_value=APPLE_M3)
def test_framework_device_objects_select_spec(_apple):
    spec, _ = detect_hardware(benchmark_memory=False, device=SimpleNamespace(platform="METAL"))
    assert spec.device_type == "gpu"


@patch("neural_cost.hardware_detect._probe_nvidia", return_value=(None, None, None, ()))
@patch("neural_cost.hardware_detect._probe_apple_silicon", return_value=(None, None, None, ()))
def test_gpu_without_table_and_tpu_raise(_apple, _nvidia):
    with pytest.raises(ValueError, match="no GPU"):
        detect_hardware(benchmark_memory=False, device="cuda")
    with pytest.raises(ValueError, match="TPU"):
        detect_hardware(benchmark_memory=False, device="tpu")
    with pytest.raises(ValueError, match="device type"):
        detect_hardware(benchmark_memory=False, device="banana")


def test_analyze_gap_warns_on_device_mismatch():
    estimate = CostEstimate(10**9, 10**8, 10**8, 1)
    gpu = HardwareSpec("Apple M3", 3.6e12, 100e9, device_type="gpu")
    cpu_run = Measurement(0.02, (0.02,), device="cpu:0")
    gap = analyze_gap(estimate, cpu_run, gpu)
    assert gap.findings[0].startswith("Device mismatch")
    assert device_mismatch(Measurement(0.02, (0.02,), device="mps:0"), gpu) is None
    # Unknown on either side: no warning.
    assert device_mismatch(cpu_run, HardwareSpec("x", 1e12, 1e11)) is None
    assert device_mismatch(Measurement(0.02, (0.02,)), gpu) is None


def test_jax_adapter_hardware_uses_active_device():
    jax = pytest.importorskip("jax")
    from neural_cost.adapters import JaxAdapter

    spec, _ = JaxAdapter().hardware(benchmark_memory=False)
    assert spec.device_type == normalize_device_type(jax.devices()[0])


def test_jax_benchmark_records_device_from_nested_arguments():
    jax = pytest.importorskip("jax")
    from neural_cost.adapters import JaxAdapter

    variables = {"params": {"w": jax.numpy.ones((4, 4))}}
    measurement = JaxAdapter().benchmark(
        lambda v, x: x @ v["params"]["w"], variables, jax.numpy.ones((2, 4)), warmup=0, repeats=1
    )
    assert normalize_device_type(measurement.device) == normalize_device_type(jax.devices()[0])
