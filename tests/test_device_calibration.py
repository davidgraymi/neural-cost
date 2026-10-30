"""Device-aware peak calibration and Apple Silicon variants (issue #55)."""

from unittest.mock import patch

import pytest

from neural_cost import HardwareCalibration, HardwareSpec
from neural_cost.hardware_detect import _probe_apple_silicon


def profiler_output(chip, cores=None):
    text = f"Hardware:\n  Chip: Apple {chip}\nGraphics/Displays:\n"
    if cores is not None:
        text += f"  Total Number of Cores: {cores}\n"
    return text


@pytest.mark.parametrize(
    ("chip", "cores", "tflops", "bandwidth_gb_s"),
    [
        ("M3 Max", 40, 14.2, 400.0),
        ("M3 Max", 30, 14.2 * 30 / 40, 300.0),
        ("M4 Max", 32, 18.4 * 32 / 40, 410.0),
        ("M4 Max", 40, 18.4, 546.0),
        ("M3 Pro", 18, 7.4, 150.0),
        ("M3", 8, 3.6 * 8 / 10, 100.0),
        ("M3 Ultra", 80, 28.4, 819.0),
        ("M3 Max", None, 14.2, 400.0),  # core count unknown: full configuration
    ],
)
@patch("shutil.which", return_value="/usr/sbin/system_profiler")
def test_apple_variants_by_gpu_core_count(_which, chip, cores, tflops, bandwidth_gb_s):
    with patch("subprocess.check_output", return_value=profiler_output(chip, cores)):
        name, flops, bandwidth, _ = _probe_apple_silicon()
    assert flops == pytest.approx(tflops * 1e12)
    assert bandwidth == pytest.approx(bandwidth_gb_s * 1e9)
    assert name.startswith(f"Apple {chip}")
    if cores is not None:
        assert name.endswith(f"({cores}-core GPU)")


def test_calibration_to_spec_and_fraction():
    calibration = HardwareCalibration(2e12, 80e9, source="probe")
    spec = calibration.to_hardware_spec(device_type="tpu")
    assert (spec.peak_flops, spec.memory_bandwidth, spec.device_type) == (2e12, 80e9, "tpu")
    assert "probe" in spec.name
    flops, bandwidth = calibration.fraction_of(HardwareSpec("sheet", 4e12, 100e9))
    assert (flops, bandwidth) == pytest.approx((0.5, 0.8))


def test_jax_calibrate_on_device_is_cached():
    jax = pytest.importorskip("jax")
    from neural_cost.adapters import JaxAdapter

    adapter = JaxAdapter()
    device = jax.devices()[0]
    first = adapter.calibrate(quick=True, device=device)
    assert first is adapter.calibrate(quick=True, device=device)
    assert device.platform in first.source
    assert first.achievable_flops > 0 and first.achievable_bandwidth > 0
    with patch("neural_cost.adapters.jax._measure_calibration", return_value=first) as fresh:
        adapter.calibrate(quick=True, device=device, refresh=True)
    fresh.assert_called_once()


def test_jax_hardware_measure_modes():
    jax = pytest.importorskip("jax")
    from neural_cost.adapters import JaxAdapter

    adapter = JaxAdapter()
    calibration = HardwareCalibration(1e12, 50e9, source="fake")
    with patch.object(JaxAdapter, "calibrate", return_value=calibration):
        spec, detection = adapter.hardware(measure="always", quick=True)
        assert detection is None and spec.peak_flops == 1e12
        assert spec.device_type == jax.devices()[0].platform.lower().replace("cuda", "gpu")
        # No table entry: "auto" falls back to measurement, "never" raises.
        with patch(
            "neural_cost.hardware_detect.detect_hardware", side_effect=ValueError("no table")
        ):
            spec, detection = adapter.hardware(measure="auto")
            assert detection is None and spec.memory_bandwidth == 50e9
            with pytest.raises(ValueError):
                adapter.hardware(measure="never")
    with pytest.raises(ValueError):
        adapter.hardware(measure="sometimes")
