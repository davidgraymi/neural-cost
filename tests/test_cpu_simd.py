"""CPU peak estimate from the SIMD width the OS reports (issue #90)."""

from unittest.mock import patch

import pytest

from neural_cost.hardware_detect import (
    CpuInfo,
    _parse_proc_cpuinfo,
    _probe_cpu_info,
    detect_hardware,
)

XEON_AVX512 = (
    "processor\t: 0\n"
    "model name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n"
    "flags\t\t: fpu sse sse2 avx fma avx2 avx512f avx512dq\n"
    "processor\t: 1\n"
    "model name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n"
    "flags\t\t: fpu sse sse2 avx fma avx2 avx512f avx512dq\n"
)
EPYC_AVX2 = "model name\t: AMD EPYC 7B12\nflags\t\t: fpu sse2 avx fma avx2\n"
AVX_ONLY = "model name\t: Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz\nflags\t\t: sse2 avx\n"
AVX2_NO_FMA = "model name\t: Odd CPU\nflags\t\t: sse2 avx avx2\n"
SSE2 = "model name\t: Old CPU\nflags\t\t: fpu sse sse2\n"
ARM_NEON = "processor\t: 0\nBogoMIPS\t: 50.00\nFeatures\t: fp asimd evtstrm aes crc32\n"
ARM_SVE = "processor\t: 0\nFeatures\t: fp asimd sve sve2\n"


@pytest.mark.parametrize(
    ("text", "name", "simd", "flop"),
    [
        (XEON_AVX512, "Intel(R) Xeon(R) CPU @ 2.20GHz", "AVX-512", 64.0),
        (EPYC_AVX2, "AMD EPYC 7B12", "AVX2+FMA", 32.0),
        (AVX_ONLY, "Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz", "AVX", 16.0),
        (AVX2_NO_FMA, "Odd CPU", "AVX", 16.0),
        (SSE2, "Old CPU", "SSE2", 8.0),
        (ARM_NEON, None, "NEON", 16.0),
        (ARM_SVE, None, "SVE", 32.0),
        ("", None, "scalar", 2.0),
    ],
)
def test_parse_proc_cpuinfo(text, name, simd, flop):
    assert _parse_proc_cpuinfo(text) == CpuInfo(name, simd, flop)


def test_intel_macos_sysctl_features():
    values = {
        "machdep.cpu.brand_string": "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz",
        "machdep.cpu.features": "FPU SSE SSE2 SSE4.1 AVX1.0 FMA",
        "machdep.cpu.leaf7_features": "BMI1 AVX2 BMI2",
    }
    with (
        patch("builtins.open", side_effect=FileNotFoundError),
        patch("neural_cost.hardware_detect._sysctl_str", side_effect=values.get),
    ):
        info = _probe_cpu_info()
    assert info == CpuInfo("Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz", "AVX2+FMA", 32.0)


@patch("neural_cost.hardware_detect._probe_cpu_info")
@patch("neural_cost.hardware_detect._probe_cpu_flops", return_value=(2, 2.2e9))
@patch("neural_cost.hardware_detect._probe_nvidia", return_value=(None, None, None, ()))
@patch("neural_cost.hardware_detect._probe_apple_silicon", return_value=(None, None, None, ()))
def test_linux_cpu_peak_uses_simd_and_name(_apple, _nvidia, _cpu, probe):
    probe.return_value = _parse_proc_cpuinfo(XEON_AVX512)
    spec, detection = detect_hardware(benchmark_memory=False, device="cpu")
    # 2 vCPUs x 2.2 GHz x 64 FLOP/cycle: above the ~120 GFLOP/s a Colab T4
    # host's sustained FP32 matmul reaches, so the lower bound holds.
    assert spec.peak_flops == pytest.approx(2 * 2.2e9 * 64)
    assert spec.peak_flops > 120e9
    assert spec.name == "Intel(R) Xeon(R) CPU @ 2.20GHz"
    assert detection.chip_name == spec.name
    assert "AVX-512 64 FLOP/cycle/core" in detection.source
    half, _ = detect_hardware(benchmark_memory=False, device="cpu", precision="bf16")
    assert half.peak_flops == pytest.approx(2 * spec.peak_flops)
