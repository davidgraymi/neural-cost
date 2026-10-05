"""Framework-neutral neural-network compute and memory cost analysis."""

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("neural-cost")
    except PackageNotFoundError:
        __version__ = "0.7.0.dev0"
except ImportError:
    __version__ = "0.7.0.dev0"

from .adapters import available_adapters, get_adapter
from .analysis import (
    ContinuousBatchGapAnalysis,
    GapAnalysis,
    LayerGapAnalysis,
    MemoryGapAnalysis,
    ModelGapAnalysis,
    MoEGapAnalysis,
    PagedAttentionGapAnalysis,
    SpeculativeDecodingAnalysis,
    SSMGapAnalysis,
    analyze_continuous_batch_iteration,
    analyze_gap,
    analyze_layers_gap,
    analyze_memory_gap,
    analyze_model_gap,
    analyze_moe_gap,
    analyze_paged_attention_gap,
    analyze_speculative_decoding,
    analyze_ssm_gap,
)
from .api import estimate_model
from .estimate import (
    ContinuousBatchIterationEstimate,
    CostEstimate,
    FusedCostEstimate,
    MoECostEstimate,
    PagedAttentionCostEstimate,
    SpeculativeCostEstimate,
    SSMCostEstimate,
    estimate_adamw_traffic,
    estimate_continuous_batch_iteration,
    estimate_conv2d,
    estimate_fused_operations,
    estimate_moe,
    estimate_operation,
    estimate_operations,
    estimate_paged_attention,
    estimate_speculative_decoding,
    estimate_ssm,
)
from .hardware import CacheSpec, HardwareSpec
from .hardware_detect import DetectionResult, detect_hardware
from .memory import MemoryEstimate, estimate_memory
from .model import ModelProfile, profile_model
from .operations import Operation
from .profiler import Measurement, benchmark

__all__ = [
    "CacheSpec",
    "ContinuousBatchGapAnalysis",
    "ContinuousBatchIterationEstimate",
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
    "PagedAttentionCostEstimate",
    "PagedAttentionGapAnalysis",
    "SSMCostEstimate",
    "SSMGapAnalysis",
    "SpeculativeCostEstimate",
    "SpeculativeDecodingAnalysis",
    "__version__",
    "analyze_continuous_batch_iteration",
    "analyze_gap",
    "analyze_layers_gap",
    "analyze_memory_gap",
    "analyze_model_gap",
    "analyze_moe_gap",
    "analyze_paged_attention_gap",
    "analyze_speculative_decoding",
    "analyze_ssm_gap",
    "available_adapters",
    "benchmark",
    "detect_hardware",
    "estimate_adamw_traffic",
    "estimate_continuous_batch_iteration",
    "estimate_conv2d",
    "estimate_fused_operations",
    "estimate_memory",
    "estimate_model",
    "estimate_moe",
    "estimate_operation",
    "estimate_operations",
    "estimate_paged_attention",
    "estimate_speculative_decoding",
    "estimate_ssm",
    "get_adapter",
    "profile_model",
]
