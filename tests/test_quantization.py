"""Tests for sub-byte quantization, dequantization tax, and kernel launch floor modeling."""

from __future__ import annotations

import json

import pytest

from neural_cost import (
    PRECISION_FORMATS,
    DequantizationCostEstimate,
    HardwareSpec,
    KernelLaunchAnalysis,
    QuantizationGapAnalysis,
    analyze_kernel_launch_floor,
    analyze_quantization_gap,
    estimate_quantized_linear,
    get_precision_format,
    get_quantization_preset,
)
from neural_cost.cli.root import build_parser

# ---------------------------------------------------------------------------
# PrecisionFormat & QuantizationSpec Tests
# ---------------------------------------------------------------------------


def test_precision_formats() -> None:
    fp16 = get_precision_format("fp16")
    assert fp16.bits == 16
    assert fp16.bytes_per_element == 2.0
    assert fp16.is_floating_point

    int4 = get_precision_format("int4")
    assert int4.bits == 4
    assert int4.bytes_per_element == 0.5
    assert int4.is_integer

    int2 = get_precision_format("int2")
    assert int2.bits == 2
    assert int2.bytes_per_element == 0.25

    fp4 = get_precision_format("fp4_e2m1")
    assert fp4.bits == 4
    assert fp4.bytes_per_element == 0.5
    assert fp4.is_floating_point

    # Aliases
    assert get_precision_format("half") == fp16
    assert get_precision_format("fp8") == PRECISION_FORMATS["fp8_e4m3"]
    assert get_precision_format("float32") == PRECISION_FORMATS["fp32"]

    with pytest.raises(ValueError, match="Unknown precision format"):
        get_precision_format("nonexistent_precision")


def test_quantization_presets() -> None:
    awq = get_quantization_preset("w4a16_awq")
    assert awq.is_weight_only
    assert awq.weight_format.bits == 4
    assert awq.activation_format.bits == 16
    assert awq.has_zero_point
    assert not awq.native_hardware_mma

    gptq = get_quantization_preset("gptq")
    assert gptq.is_weight_only
    assert not gptq.has_zero_point

    fp8 = get_quantization_preset("fp8")
    assert not fp8.is_weight_only
    assert fp8.native_hardware_mma

    nvfp4 = get_quantization_preset("nvfp4")
    assert nvfp4.weight_format.bits == 4
    assert nvfp4.activation_format.bits == 4
    assert nvfp4.native_hardware_mma

    with pytest.raises(ValueError, match="Unknown quantization preset"):
        get_quantization_preset("unknown_scheme")


# ---------------------------------------------------------------------------
# Quantized Linear Estimation Tests
# ---------------------------------------------------------------------------


def test_estimate_quantized_linear_w4a16() -> None:
    k = 4096
    n = 4096
    batch = 1
    est = estimate_quantized_linear(batch, k, n, quantization="w4a16_awq")

    assert isinstance(est, DequantizationCostEstimate)
    assert est.batch_size == 1
    assert est.in_features == k
    assert est.out_features == n

    # Unquantized FP16 weights: 4096 * 4096 * 2 = 33,554,432 bytes (~33.55 MB)
    assert est.unquantized_weight_bytes == 4096 * 4096 * 2

    # INT4 weights: 4096 * 4096 * 0.5 = 8,388,608 bytes (8 MB)
    assert est.quantized_weight_bytes == 4096 * 4096 // 2

    # Scale & zero-point overhead:
    # Num weights = 16,777,216. Group size = 128. Num groups = 131,072.
    # Scale: 131,072 * 2 = 262,144 bytes.
    # Zero: 131,072 * 2 = 262,144 bytes.
    # Total scale_zero_bytes = 524,288 bytes (0.5 MB).
    assert est.scale_zero_bytes == 524288
    assert est.total_weight_bytes == 8388608 + 524288

    # Compression ratio: ~3.76x
    assert 3.75 < est.weight_compression_ratio < 3.78

    # GEMM FLOPs: 2 * 1 * 4096 * 4096 = 33,554,432
    assert est.gemm_flops == 2 * batch * k * n

    # Dequant unpack ALU tax: 4 ALU ops per weight element = 4 * 16,777,216 = 67,108,864
    assert est.dequant_unpack_flops == 4 * (k * n)
    assert est.total_flops == est.gemm_flops + est.dequant_unpack_flops
    assert est.arithmetic_intensity > 0


def test_estimate_quantized_linear_w8a8_fp8() -> None:
    k = 4096
    n = 4096
    batch = 32
    est = estimate_quantized_linear(batch, k, n, quantization="w8a8_fp8")

    # Native MMA -> no ALU unpack tax
    assert est.dequant_unpack_flops == 0
    assert est.total_flops == est.gemm_flops
    # Weight compression ratio is ~2x vs FP16
    assert 1.95 < est.weight_compression_ratio < 2.05


def test_estimate_quantized_linear_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        estimate_quantized_linear(0, 4096, 4096)
    with pytest.raises(ValueError, match="must be positive"):
        estimate_quantized_linear(1, -1, 4096)
    with pytest.raises(ValueError, match="must be positive"):
        estimate_quantized_linear(1, 4096, 0)


# ---------------------------------------------------------------------------
# Quantization Gap Analysis Tests
# ---------------------------------------------------------------------------


def test_analyze_quantization_gap_memory_bound() -> None:
    hw = HardwareSpec(
        name="H100-SXM",
        peak_flops=989e12,
        memory_bandwidth=3.35e12,
    )
    # Single-token decode projection: heavily memory-bound
    est = estimate_quantized_linear(1, 4096, 4096, quantization="w4a16_awq")
    analysis = analyze_quantization_gap(est, hw)

    assert isinstance(analysis, QuantizationGapAnalysis)
    assert analysis.unquantized_bottleneck == "memory"
    assert analysis.quantized_bottleneck == "memory"
    # Speedup tracks weight compression
    assert 3.5 < analysis.speedup < 3.8
    assert analysis.dequant_overhead_ratio > 0.5  # dequant tax is significant fraction of total ops

    rendered = analysis.render()
    assert "Quantization Analysis (W4A16 AWQ" in rendered
    assert "weight footprint" in rendered
    assert "dequantization tax" in rendered


def test_analyze_quantization_gap_native_fp8_compute() -> None:
    hw = HardwareSpec(
        name="H100-SXM",
        peak_flops=989e12,
        memory_bandwidth=3.35e12,
    )
    # Large batch size: compute-bound
    est = estimate_quantized_linear(4096, 4096, 4096, quantization="w8a8_fp8")
    analysis = analyze_quantization_gap(est, hw)

    assert analysis.unquantized_bottleneck == "compute"
    assert analysis.quantized_bottleneck == "compute"
    # Native FP8 has 2x peak compute -> 2x speedup
    assert 1.95 < analysis.speedup <= 2.05
    assert analysis.dequant_overhead_ratio == 0.0


# ---------------------------------------------------------------------------
# Kernel Launch Floor Modeling Tests
# ---------------------------------------------------------------------------


def test_analyze_kernel_launch_floor_launch_bound() -> None:
    # 640 kernels with 5 us overhead = 3.2 ms launch floor
    # Model arithmetic floor = 0.5 ms
    analysis = analyze_kernel_launch_floor(
        num_kernels=640,
        model_arithmetic_floor_seconds=0.5e-3,
        launch_overhead_seconds=5.0e-6,
        measured_latency_seconds=3.4e-3,
    )

    assert isinstance(analysis, KernelLaunchAnalysis)
    assert analysis.is_launch_bound
    assert analysis.total_launch_floor_seconds == pytest.approx(3.2e-3)
    assert analysis.effective_lower_bound_seconds == pytest.approx(3.2e-3)
    assert analysis.launch_to_arithmetic_ratio == pytest.approx(6.4)
    # CUDA Graph collapsing 640 launches to 1: speedup is ~ 3.2 / (0.5 + 0.005) = ~6.3x
    assert 6.0 < analysis.cuda_graph_potential_speedup < 6.5

    rendered = analysis.render()
    assert "HOST-DISPATCH BOUND" in rendered
    assert "CUDA Graph" in rendered
    assert "measured gap" in rendered


def test_analyze_kernel_launch_floor_arithmetic_bound() -> None:
    # 10 kernels with 5 us overhead = 0.05 ms launch floor
    # Model arithmetic floor = 20.0 ms
    analysis = analyze_kernel_launch_floor(
        num_kernels=10,
        model_arithmetic_floor_seconds=20.0e-3,
        launch_overhead_seconds=5.0e-6,
    )

    assert not analysis.is_launch_bound
    assert analysis.effective_lower_bound_seconds == pytest.approx(20.0e-3)
    assert analysis.cuda_graph_potential_speedup == pytest.approx(1.0, rel=1e-2)

    rendered = analysis.render()
    assert "ACCELERATOR-EXECUTION BOUND" in rendered


def test_analyze_kernel_launch_floor_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="num_kernels must be positive"):
        analyze_kernel_launch_floor(0, 0.001)
    with pytest.raises(ValueError, match="model_arithmetic_floor_seconds must be positive"):
        analyze_kernel_launch_floor(100, -0.001)
    with pytest.raises(ValueError, match="launch_overhead_seconds cannot be negative"):
        analyze_kernel_launch_floor(100, 0.001, launch_overhead_seconds=-1.0)


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------


def test_cli_llm_quant(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "llm",
            "quant",
            "--batch-size",
            "1",
            "--in-features",
            "4096",
            "--out-features",
            "4096",
            "--quantization",
            "w4a16_awq",
            "--peak-flops",
            "312e12",
            "--memory-bandwidth",
            "2e12",
            "--json",
        ]
    )
    ret = args.func(args)
    assert ret == 0

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["quantization"] == "W4A16 AWQ"
    assert data["weight_compression_ratio"] > 3.7
    assert data["dequant_unpack_flops"] > 0
    assert data["quantized_bottleneck"] == "memory"


def test_cli_llm_launch_floor(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "llm",
            "launch-floor",
            "--num-kernels",
            "640",
            "--arithmetic-floor-ms",
            "0.5",
            "--launch-overhead-us",
            "5.0",
            "--json",
        ]
    )
    ret = args.func(args)
    assert ret == 0

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["num_kernels"] == 640
    assert data["is_launch_bound"] is True
    assert data["cuda_graph_potential_speedup"] > 6.0


def test_cli_profile_with_quantization_and_launch_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "profile",
            "--arch",
            "transformer",
            "--num-layers",
            "8",
            "--embed-dim",
            "1024",
            "--num-heads",
            "8",
            "-b",
            "1",
            "-s",
            "128",
            "--quantization",
            "w4a16_awq",
            "--num-kernels",
            "160",
            "--peak-flops",
            "100e12",
            "--memory-bandwidth",
            "1e12",
            "--json",
        ]
    )
    ret = args.func(args)
    assert ret == 0

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["quantization"] == "W4A16 AWQ"
    assert data["dequant_flops"] > 0
    assert "kernel_launch" in data
    assert data["kernel_launch"]["num_kernels"] == 160
