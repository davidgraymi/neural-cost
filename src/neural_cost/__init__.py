"""Framework-neutral neural-network compute and memory cost analysis."""

from .adapters import available_adapters, get_adapter
from .analysis import (
    GapAnalysis,
    LayerGapAnalysis,
    MemoryGapAnalysis,
    ModelGapAnalysis,
    MoEGapAnalysis,
    SpeculativeDecodingAnalysis,
    analyze_gap,
    analyze_layers_gap,
    analyze_memory_gap,
    analyze_model_gap,
    analyze_moe_gap,
    analyze_speculative_decoding,
)
from .api import estimate_model
from .estimate import (
    CostEstimate,
    FusedCostEstimate,
    MoECostEstimate,
    SpeculativeCostEstimate,
    estimate_adamw_traffic,
    estimate_conv2d,
    estimate_fused_operations,
    estimate_moe,
    estimate_operation,
    estimate_operations,
    estimate_speculative_decoding,
)
from .hardware import CacheSpec, HardwareSpec
from .hardware_detect import DetectionResult, detect_hardware
from .memory import MemoryEstimate, estimate_memory
from .model import ModelProfile, profile_model
from .operations import Operation
from .profiler import Measurement, benchmark

__all__ = [
    "CacheSpec",
    "CostEstimate",
    "DetectionResult",
    "FusedCostEstimate",
    "GapAnalysis",
    "HardwareSpec",
    "LayerGapAnalysis",
    "Measurement",
    "MemoryEstimate",
    "MemoryGapAnalysis",
    "MoECostEstimate",
    "MoEGapAnalysis",
    "ModelGapAnalysis",
    "ModelProfile",
    "Operation",
    "SpeculativeCostEstimate",
    "SpeculativeDecodingAnalysis",
    "analyze_gap",
    "analyze_layers_gap",
    "analyze_memory_gap",
    "analyze_model_gap",
    "analyze_moe_gap",
    "analyze_speculative_decoding",
    "available_adapters",
    "benchmark",
    "detect_hardware",
    "estimate_adamw_traffic",
    "estimate_conv2d",
    "estimate_fused_operations",
    "estimate_memory",
    "estimate_model",
    "estimate_moe",
    "estimate_operation",
    "estimate_operations",
    "estimate_speculative_decoding",
    "get_adapter",
    "profile_model",
]
