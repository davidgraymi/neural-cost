"""Sub-byte numerical precision and dequantization cost modeling."""

from __future__ import annotations

from dataclasses import dataclass

from neural_cost.hardware import HardwareSpec


@dataclass(frozen=True, slots=True)
class PrecisionFormat:
    """Numerical precision format specifications."""

    name: str
    bits: int
    is_integer: bool = False
    is_floating_point: bool = True
    exponent_bits: int | None = None
    mantissa_bits: int | None = None

    @property
    def bytes_per_element(self) -> float:
        """Effective memory footprint in bytes per numerical element."""
        return self.bits / 8.0


PRECISION_FORMATS: dict[str, PrecisionFormat] = {
    "fp32": PrecisionFormat(
        "FP32", bits=32, is_integer=False, is_floating_point=True, exponent_bits=8, mantissa_bits=23
    ),
    "fp16": PrecisionFormat(
        "FP16", bits=16, is_integer=False, is_floating_point=True, exponent_bits=5, mantissa_bits=10
    ),
    "bf16": PrecisionFormat(
        "BF16", bits=16, is_integer=False, is_floating_point=True, exponent_bits=8, mantissa_bits=7
    ),
    "fp8_e4m3": PrecisionFormat(
        "FP8-E4M3",
        bits=8,
        is_integer=False,
        is_floating_point=True,
        exponent_bits=4,
        mantissa_bits=3,
    ),
    "fp8_e5m2": PrecisionFormat(
        "FP8-E5M2",
        bits=8,
        is_integer=False,
        is_floating_point=True,
        exponent_bits=5,
        mantissa_bits=2,
    ),
    "int8": PrecisionFormat("INT8", bits=8, is_integer=True, is_floating_point=False),
    "int4": PrecisionFormat("INT4", bits=4, is_integer=True, is_floating_point=False),
    "fp4_e2m1": PrecisionFormat(
        "FP4-E2M1",
        bits=4,
        is_integer=False,
        is_floating_point=True,
        exponent_bits=2,
        mantissa_bits=1,
    ),
    "int2": PrecisionFormat("INT2", bits=2, is_integer=True, is_floating_point=False),
    "ternary": PrecisionFormat("Ternary-1.58b", bits=2, is_integer=True, is_floating_point=False),
}


def get_precision_format(name: str) -> PrecisionFormat:
    """Retrieve numerical precision format by name."""
    norm = name.strip().lower().replace("-", "_")
    if norm in PRECISION_FORMATS:
        return PRECISION_FORMATS[norm]
    # Aliases
    if norm in ("float32", "float"):
        return PRECISION_FORMATS["fp32"]
    if norm in ("float16", "half"):
        return PRECISION_FORMATS["fp16"]
    if norm in ("bfloat16",):
        return PRECISION_FORMATS["bf16"]
    if norm in ("fp8",):
        return PRECISION_FORMATS["fp8_e4m3"]
    available = ", ".join(sorted(PRECISION_FORMATS.keys()))
    raise ValueError(f"Unknown precision format '{name}'. Available: {available}")


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """Specification of weight and activation quantization scheme."""

    name: str
    weight_format: PrecisionFormat
    activation_format: PrecisionFormat
    group_size: int = 128  # block/group quantization size (e.g. 128 for AWQ/GPTQ)
    has_zero_point: bool = True  # asymmetric quantization
    native_hardware_mma: bool = (
        False  # whether accelerator executes native low-precision MMA instructions
    )

    @property
    def is_weight_only(self) -> bool:
        """True if only weights are quantized while activations remain high-precision (FP16/BF16)."""
        return self.weight_format.bits < self.activation_format.bits


QUANTIZATION_PRESETS: dict[str, QuantizationSpec] = {
    # Weight-only 4-bit (AWQ / GPTQ) with FP16 activations
    "w4a16_awq": QuantizationSpec(
        name="W4A16 AWQ",
        weight_format=PRECISION_FORMATS["int4"],
        activation_format=PRECISION_FORMATS["fp16"],
        group_size=128,
        has_zero_point=True,
        native_hardware_mma=False,
    ),
    "w4a16_gptq": QuantizationSpec(
        name="W4A16 GPTQ",
        weight_format=PRECISION_FORMATS["int4"],
        activation_format=PRECISION_FORMATS["fp16"],
        group_size=128,
        has_zero_point=False,
        native_hardware_mma=False,
    ),
    # Weight-only 8-bit with FP16 activations
    "w8a16": QuantizationSpec(
        name="W8A16",
        weight_format=PRECISION_FORMATS["int8"],
        activation_format=PRECISION_FORMATS["fp16"],
        group_size=128,
        has_zero_point=False,
        native_hardware_mma=False,
    ),
    # Native FP8 (Transformer Engine on Hopper H100 / Ada / Blackwell)
    "w8a8_fp8": QuantizationSpec(
        name="W8A8 FP8",
        weight_format=PRECISION_FORMATS["fp8_e4m3"],
        activation_format=PRECISION_FORMATS["fp8_e4m3"],
        group_size=128,
        has_zero_point=False,
        native_hardware_mma=True,
    ),
    # INT8 SmoothQuant
    "w8a8_int8": QuantizationSpec(
        name="W8A8 INT8 (SmoothQuant)",
        weight_format=PRECISION_FORMATS["int8"],
        activation_format=PRECISION_FORMATS["int8"],
        group_size=128,
        has_zero_point=False,
        native_hardware_mma=True,
    ),
    # Sub-byte NVFP4 on Blackwell
    "w4a4_fp4": QuantizationSpec(
        name="W4A4 NVFP4",
        weight_format=PRECISION_FORMATS["fp4_e2m1"],
        activation_format=PRECISION_FORMATS["fp4_e2m1"],
        group_size=64,
        has_zero_point=False,
        native_hardware_mma=True,
    ),
}


def get_quantization_preset(name: str) -> QuantizationSpec:
    """Retrieve standard quantization preset by name."""
    norm = name.strip().lower().replace("-", "_").replace(" ", "_")
    if norm in QUANTIZATION_PRESETS:
        return QUANTIZATION_PRESETS[norm]
    # Aliases
    if norm in ("awq", "int4", "w4a16"):
        return QUANTIZATION_PRESETS["w4a16_awq"]
    if norm in ("gptq",):
        return QUANTIZATION_PRESETS["w4a16_gptq"]
    if norm in ("fp8", "fp8_e4m3"):
        return QUANTIZATION_PRESETS["w8a8_fp8"]
    if norm in ("int8", "smoothquant"):
        return QUANTIZATION_PRESETS["w8a8_int8"]
    if norm in ("nvfp4", "fp4"):
        return QUANTIZATION_PRESETS["w4a4_fp4"]
    available = ", ".join(sorted(QUANTIZATION_PRESETS.keys()))
    raise ValueError(f"Unknown quantization preset '{name}'. Available: {available}")


@dataclass(frozen=True, slots=True)
class DequantizationCostEstimate:
    """Analytical memory, compute, and dequantization unpack overheads for a quantized layer."""

    quantization: QuantizationSpec
    batch_size: int
    in_features: int
    out_features: int
    unquantized_weight_bytes: int
    quantized_weight_bytes: int
    scale_zero_bytes: int
    total_weight_bytes: int
    weight_compression_ratio: float
    input_activation_bytes: int
    output_activation_bytes: int
    total_memory_bytes: int
    gemm_flops: int
    dequant_unpack_flops: int
    total_flops: int
    arithmetic_intensity: float


def estimate_quantized_linear(
    batch_size: int,
    in_features: int,
    out_features: int,
    quantization: QuantizationSpec | str = "w4a16_awq",
) -> DequantizationCostEstimate:
    """Estimate compute FLOPs, compulsory weight traffic, and dequantization ALU tax for a linear projection.

    Parameters:
        batch_size: Number of tokens (tokens = batch_size * seq_len).
        in_features: Input dimension K.
        out_features: Output dimension N.
        quantization: QuantizationSpec or preset name (e.g. 'w4a16_awq', 'w8a8_fp8').
    """
    if batch_size <= 0 or in_features <= 0 or out_features <= 0:
        raise ValueError("batch_size, in_features, and out_features must be positive")

    spec = get_quantization_preset(quantization) if isinstance(quantization, str) else quantization

    num_weights = in_features * out_features
    # Baseline unquantized model weights are in standard FP16 (2.0 bytes per weight)
    baseline_bytes_per_elem = 2.0
    unquantized_weight_bytes = int(num_weights * baseline_bytes_per_elem)

    # 1. Quantized weight bytes
    quant_weight_bytes = int(num_weights * spec.weight_format.bytes_per_element)

    # 2. Scale & zero-point parameters traffic:
    # Scale: FP16 (2 bytes) per group. Zero-point: FP16 or INT4 (2 bytes) per group if present.
    num_groups = (num_weights + spec.group_size - 1) // spec.group_size
    scale_bytes = num_groups * 2
    zero_bytes = num_groups * 2 if spec.has_zero_point else 0
    scale_zero_bytes = scale_bytes + zero_bytes

    total_weight_bytes = quant_weight_bytes + scale_zero_bytes
    compression_ratio = (
        unquantized_weight_bytes / total_weight_bytes if total_weight_bytes > 0 else 1.0
    )

    # 3. Activation traffic
    in_act_bytes = int(batch_size * in_features * spec.activation_format.bytes_per_element)
    out_act_bytes = int(batch_size * out_features * spec.activation_format.bytes_per_element)
    total_mem_bytes = total_weight_bytes + in_act_bytes + out_act_bytes

    # 4. Arithmetic FLOPs:
    # Standard MMA GEMM: 2 * M * K * N
    gemm_flops = 2 * batch_size * in_features * out_features

    # 5. Dequantization Unpack ALU Tax:
    # If hardware has no native MMA for this sub-byte precision, weights must be unpacked:
    # Bit shift + mask: 2 ALU ops
    # FP convert & scale/zero multiply-add: 2 ALU ops
    # Total dequant tax: ~4 ops per weight element loaded from DRAM per token batch
    if not spec.native_hardware_mma and spec.is_weight_only:
        dequant_flops = 4 * num_weights
    else:
        dequant_flops = 0

    total_flops = gemm_flops + dequant_flops
    arithmetic_intensity = total_flops / total_mem_bytes if total_mem_bytes > 0 else 0.0

    return DequantizationCostEstimate(
        quantization=spec,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        unquantized_weight_bytes=unquantized_weight_bytes,
        quantized_weight_bytes=quant_weight_bytes,
        scale_zero_bytes=scale_zero_bytes,
        total_weight_bytes=total_weight_bytes,
        weight_compression_ratio=compression_ratio,
        input_activation_bytes=in_act_bytes,
        output_activation_bytes=out_act_bytes,
        total_memory_bytes=total_mem_bytes,
        gemm_flops=gemm_flops,
        dequant_unpack_flops=dequant_flops,
        total_flops=total_flops,
        arithmetic_intensity=arithmetic_intensity,
    )


@dataclass(frozen=True, slots=True)
class QuantizationGapAnalysis:
    """Gap analysis comparing unquantized baseline vs quantized execution on target hardware."""

    cost: DequantizationCostEstimate
    hardware: HardwareSpec
    unquantized_latency_seconds: float
    quantized_latency_seconds: float
    speedup: float
    unquantized_bottleneck: str
    quantized_bottleneck: str
    dequant_overhead_ratio: float
    findings: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f"Quantization Analysis ({self.cost.quantization.name}, B={self.cost.batch_size}, K={self.cost.in_features}, N={self.cost.out_features})",
            f"  weight footprint: {self.cost.total_weight_bytes / 1e6:.2f} MB ({self.cost.weight_compression_ratio:.2f}x compression vs {self.cost.unquantized_weight_bytes / 1e6:.2f} MB)",
            f"  dequantization tax: {self.cost.dequant_unpack_flops / 1e6:.2f} MFLOPs ({self.dequant_overhead_ratio:.1%} of compute)",
            f"  arithmetic intensity: {self.cost.arithmetic_intensity:.2f} FLOP/B (ridge: {self.hardware.ridge_point:.1f} FLOP/B)",
            f"  performance: {self.quantized_latency_seconds * 1e3:.3f} ms vs {self.unquantized_latency_seconds * 1e3:.3f} ms ({self.speedup:.2f}x speedup, {self.quantized_bottleneck}-bound)",
        ]
        lines.extend(f"  finding: {f}" for f in self.findings)
        return "\n".join(lines)


def analyze_quantization_gap(
    cost: DequantizationCostEstimate,
    hardware: HardwareSpec,
) -> QuantizationGapAnalysis:
    """Analyze roofline speedup and dequantization tax of a quantized operator against hardware."""
    # 1. Unquantized baseline (FP16 weights and activations)
    unquant_act_bytes = int(cost.batch_size * (cost.in_features + cost.out_features) * 2.0)
    unquant_bytes = cost.unquantized_weight_bytes + unquant_act_bytes
    t_unquant_comp = cost.gemm_flops / hardware.peak_flops if hardware.peak_flops > 0 else 0.0
    t_unquant_mem = (
        unquant_bytes / hardware.memory_bandwidth if hardware.memory_bandwidth > 0 else 0.0
    )
    unquant_bound_s = max(t_unquant_comp, t_unquant_mem)
    unquant_bottleneck = "compute" if t_unquant_comp >= t_unquant_mem else "memory"

    # 2. Quantized execution
    # If native FP8/INT8 MMA is supported, peak compute doubles
    peak_flops = hardware.peak_flops
    if cost.quantization.native_hardware_mma:
        peak_flops = hardware.peak_flops * 2.0

    t_quant_comp = cost.total_flops / peak_flops if peak_flops > 0 else 0.0
    t_quant_mem = (
        cost.total_memory_bytes / hardware.memory_bandwidth
        if hardware.memory_bandwidth > 0
        else 0.0
    )
    quant_bound_s = max(t_quant_comp, t_quant_mem)
    quant_bottleneck = "compute" if t_quant_comp >= t_quant_mem else "memory"

    speedup = unquant_bound_s / quant_bound_s if quant_bound_s > 0 else 1.0
    dequant_ratio = cost.dequant_unpack_flops / cost.total_flops if cost.total_flops > 0 else 0.0

    findings: list[str] = []
    if cost.quantization.is_weight_only:
        if quant_bottleneck == "memory":
            findings.append(
                f"Weight-only quantization successfully accelerates memory-bound execution by {speedup:.2f}x."
            )
        else:
            findings.append(
                "Workload shifted to compute-bound; dequantization ALU unpacking consumes execution cycles."
            )
        if dequant_ratio > 0.1:
            findings.append(
                f"Dequantization unpacking tax represents {dequant_ratio:.1%} of arithmetic operations."
            )
    else:
        findings.append(
            f"Native low-precision Tensor Cores deliver {speedup:.2f}x speedup with doubled peak compute."
        )

    return QuantizationGapAnalysis(
        cost=cost,
        hardware=hardware,
        unquantized_latency_seconds=unquant_bound_s,
        quantized_latency_seconds=quant_bound_s,
        speedup=speedup,
        unquantized_bottleneck=unquant_bottleneck,
        quantized_bottleneck=quant_bottleneck,
        dequant_overhead_ratio=dequant_ratio,
        findings=tuple(findings),
    )
