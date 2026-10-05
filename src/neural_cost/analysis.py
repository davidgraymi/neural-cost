from collections.abc import Iterable
from dataclasses import dataclass

from .estimate import (
    ContinuousBatchIterationEstimate,
    CostEstimate,
    FusedCostEstimate,
    MoECostEstimate,
    PagedAttentionCostEstimate,
    estimate_fused_operations,
    estimate_operation,
)
from .hardware import HardwareSpec
from .memory import MemoryEstimate
from .model import ModelProfile
from .operations import Operation
from .profiler import Measurement


@dataclass(frozen=True, slots=True)
class LayerGapAnalysis:
    """Roofline attribution and bottleneck breakdown for a single operation/layer."""

    name: str
    kind: str
    flops: int
    read_bytes: int
    write_bytes: int
    total_bytes: int
    arithmetic_intensity: float
    compute_bound_seconds: float
    bandwidth_bound_seconds: float
    lower_bound_seconds: float
    bottleneck: str
    time_share_ratio: float


@dataclass(frozen=True, slots=True)
class GapAnalysis:
    lower_bound_seconds: float
    compute_bound_seconds: float
    bandwidth_bound_seconds: float
    observed_seconds: float
    efficiency: float
    achieved_flops: float
    achieved_bandwidth: float
    bottleneck: str
    findings: tuple[str, ...]
    cache_bound_seconds: float | None = None
    resident_cache_level: str | None = None
    cache_efficiency: float | None = None
    fused_lower_bound_seconds: float | None = None
    fused_efficiency: float | None = None
    layer_analyses: tuple[LayerGapAnalysis, ...] = ()

    def render(self) -> str:
        """Return a compact, terminal-friendly performance-gap report."""
        lines = [
            "Neural cost gap analysis",
            (
                f"  bound: {self.lower_bound_seconds * 1e3:.3f} ms "
                f"(compute {self.compute_bound_seconds * 1e3:.3f} ms, "
                f"memory {self.bandwidth_bound_seconds * 1e3:.3f} ms)"
            ),
        ]
        if self.fused_lower_bound_seconds is not None:
            lines.append(
                f"  fused bound: {self.fused_lower_bound_seconds * 1e3:.3f} ms"
                + (
                    f" (efficiency {self.fused_efficiency:.1%})"
                    if self.fused_efficiency is not None
                    else ""
                )
            )
        if self.resident_cache_level and self.cache_bound_seconds is not None:
            lines.append(
                f"  cache residency: {self.resident_cache_level} "
                f"(bound {self.cache_bound_seconds * 1e3:.3f} ms"
                + (
                    f", efficiency {self.cache_efficiency:.1%}"
                    if self.cache_efficiency is not None
                    else ""
                )
                + ")"
            )
        lines.extend(
            [
                f"  observed: {self.observed_seconds * 1e3:.3f} ms",
                f"  roofline efficiency: {self.efficiency:.1%} ({self.bottleneck}-bound)",
                (
                    f"  achieved: {self.achieved_flops / 1e9:.3f} GFLOP/s, "
                    f"{self.achieved_bandwidth / 1e9:.3f} GB/s"
                ),
            ]
        )
        if self.layer_analyses:
            lines.append("  top layer bottlenecks:")
            sorted_layers = sorted(
                self.layer_analyses, key=lambda layer: layer.time_share_ratio, reverse=True
            )[:3]
            for idx, layer in enumerate(sorted_layers, 1):
                lines.append(
                    f"    {idx}. {layer.name} ({layer.kind}): {layer.lower_bound_seconds * 1e3:.3f} ms "
                    f"({layer.time_share_ratio:.1%} share, {layer.bottleneck}-bound, AI: {layer.arithmetic_intensity:.1f} FLOP/B)"
                )
        lines.extend(f"  next: {finding}" for finding in self.findings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MemoryGapAnalysis:
    """Comparison between idealized tensor storage and allocator telemetry."""

    theoretical_minimum_bytes: int
    theoretical_conservative_bytes: int
    observed_peak_bytes: int | None
    observed_reserved_bytes: int | None
    overhead_ratio: float | None
    findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelGapAnalysis:
    """Combined compute and memory gap analysis for one workload."""

    performance: GapAnalysis
    memory: MemoryGapAnalysis


def analyze_layers_gap(
    operations: Iterable[Operation], hardware: HardwareSpec
) -> tuple[LayerGapAnalysis, ...]:
    """Compute per-layer roofline bounds, arithmetic intensity, and bottleneck classification."""
    op_list = list(operations)
    if not op_list:
        return ()

    raw_layers: list[tuple[Operation, CostEstimate, float, float, float, str]] = []
    total_lower_bound = 0.0

    for op in op_list:
        est = estimate_operation(op)
        compute = est.flops / hardware.peak_flops
        bandwidth = est.total_bytes / hardware.memory_bandwidth
        bound = max(compute, bandwidth)
        bottleneck = "compute" if compute >= bandwidth else "memory"
        total_lower_bound += bound
        raw_layers.append((op, est, compute, bandwidth, bound, bottleneck))

    results: list[LayerGapAnalysis] = []
    for op, est, compute, bandwidth, bound, bottleneck in raw_layers:
        share = bound / total_lower_bound if total_lower_bound > 0 else 0.0
        results.append(
            LayerGapAnalysis(
                name=op.name,
                kind=op.kind,
                flops=est.flops,
                read_bytes=est.read_bytes,
                write_bytes=est.write_bytes,
                total_bytes=est.total_bytes,
                arithmetic_intensity=est.arithmetic_intensity,
                compute_bound_seconds=compute,
                bandwidth_bound_seconds=bandwidth,
                lower_bound_seconds=bound,
                bottleneck=bottleneck,
                time_share_ratio=share,
            )
        )
    return tuple(results)


def analyze_gap(
    estimate: CostEstimate | FusedCostEstimate,
    measurement: Measurement,
    hardware: HardwareSpec,
    operations: Iterable[Operation] | None = None,
) -> GapAnalysis:
    """Analyze observed runtime against a roofline lower bound.

    Findings are hypotheses to investigate, not assertions about kernel-level
    behavior.  The compulsory-traffic model makes the achieved bandwidth an
    effective value, especially where intermediates are materialized.
    """
    compute = estimate.flops / hardware.peak_flops
    bandwidth = estimate.total_bytes / hardware.memory_bandwidth
    lower_bound = max(compute, bandwidth)
    observed = measurement.median_seconds
    efficiency = min(1.0, lower_bound / observed)
    bottleneck = "compute" if compute >= bandwidth else "memory"
    findings: list[str] = []

    layer_analyses = ()
    if operations is not None:
        layer_analyses = analyze_layers_gap(operations, hardware)
        if layer_analyses:
            top_layer = max(layer_analyses, key=lambda layer: layer.time_share_ratio)
            findings.append(
                f"Top bottleneck layer: '{top_layer.name}' ({top_layer.kind}) accounts for "
                f"{top_layer.time_share_ratio:.1%} of theoretical execution time ({top_layer.bottleneck}-bound)."
            )

    fused_lower_bound_seconds = None
    fused_efficiency = None
    fused_est: FusedCostEstimate | None = None
    if isinstance(estimate, FusedCostEstimate) and estimate.eliminated_bytes > 0:
        fused_est = estimate
    elif operations is not None:
        candidate_fused = estimate_fused_operations(operations)
        if candidate_fused.eliminated_bytes > 0:
            fused_est = candidate_fused

    if fused_est is not None:
        fused_bandwidth = fused_est.total_bytes / hardware.memory_bandwidth
        fused_lower_bound_seconds = max(compute, fused_bandwidth)
        fused_efficiency = min(1.0, fused_lower_bound_seconds / observed)
        findings.append(
            f"Fusion optimization: kernel fusion eliminates {fused_est.eliminated_bytes / 1024:.1f} KB of traffic "
            f"({fused_est.traffic_reduction_ratio:.1%} reduction), raising arithmetic intensity to "
            f"{fused_est.arithmetic_intensity:.1f} FLOP/byte and fused lower bound to {fused_lower_bound_seconds * 1e3:.3f} ms."
        )

    resident_cache = hardware.find_resident_cache(estimate.total_bytes)
    cache_bound_seconds = None
    resident_cache_level = None
    cache_efficiency = None

    if resident_cache is not None:
        resident_cache_level = resident_cache.name
        cache_mem_bound = estimate.total_bytes / resident_cache.bandwidth
        cache_bound_seconds = max(compute, cache_mem_bound)
        cache_efficiency = min(1.0, cache_bound_seconds / observed)
        findings.append(
            f"Cache-resident: total tensor traffic ({estimate.total_bytes / (1024 * 1024):.2f} MB) fits in {resident_cache.name} "
            f"({resident_cache.capacity / (1024 * 1024):.0f} MB, {resident_cache.bandwidth / 1e9:.0f} GB/s). "
            f"Hierarchical roofline bound is {cache_bound_seconds * 1e3:.3f} ms."
        )

    if bottleneck == "memory":
        findings.append(
            "Memory-bound: consider fusion, reduced precision, or fewer materialized tensors."
        )
    else:
        findings.append(
            "Compute-bound: consider faster kernels, tensor cores, or greater parallelism."
        )

    achieved_bw = estimate.total_bytes / observed
    bw_utilization = (
        achieved_bw / hardware.memory_bandwidth if hardware.memory_bandwidth > 0 else 0.0
    )
    if bw_utilization >= 0.60:
        findings.append(
            f"Memory-bandwidth saturation: achieved bandwidth ({achieved_bw / 1e9:.1f} GB/s) reaches "
            f"{bw_utilization:.1%} of hardware peak ({hardware.memory_bandwidth / 1e9:.1f} GB/s)."
        )

    if efficiency < 0.5:
        findings.append(
            "Large roofline gap: inspect launch overhead, synchronization, shape padding, and data movement."
        )
    if hardware.memory_capacity is not None and estimate.total_bytes > hardware.memory_capacity:
        findings.append(
            "Compulsory traffic exceeds device memory capacity; partitioning or offload may be required."
        )
    return GapAnalysis(
        lower_bound,
        compute,
        bandwidth,
        observed,
        efficiency,
        estimate.flops / observed,
        estimate.total_bytes / observed,
        bottleneck,
        tuple(findings),
        cache_bound_seconds=cache_bound_seconds,
        resident_cache_level=resident_cache_level,
        cache_efficiency=cache_efficiency,
        fused_lower_bound_seconds=fused_lower_bound_seconds,
        fused_efficiency=fused_efficiency,
        layer_analyses=layer_analyses,
    )


def analyze_memory_gap(estimate: MemoryEstimate, measurement: Measurement) -> MemoryGapAnalysis:
    """Explain allocator memory above a static tensor-storage estimate.

    If an adapter cannot obtain allocator telemetry, the result reports the
    static bounds and explicitly leaves the observed fields unset.
    """
    observed = measurement.peak_memory_bytes or measurement.allocated_memory_bytes
    reserved = measurement.reserved_memory_bytes
    findings: list[str] = []
    ratio = None
    if observed is None:
        findings.append("No allocator telemetry was collected for this device.")
    elif estimate.inference_conservative_bytes == 0:
        findings.append("No tensor storage was attributed to the static operation trace.")
    else:
        ratio = observed / estimate.inference_conservative_bytes
        if ratio > 1.5:
            findings.append(
                "Observed peak exceeds conservative tensor storage; inspect allocator pools, "
                "workspaces, retained tensors, and fragmentation."
            )
        else:
            findings.append("Observed peak is near the conservative static tensor-storage bound.")
    if reserved is not None and observed is not None and reserved > observed:
        findings.append(
            "Reserved memory exceeds allocated memory; the caching allocator retains a pool."
        )
    return MemoryGapAnalysis(
        estimate.inference_minimum_bytes,
        estimate.inference_conservative_bytes,
        observed,
        reserved,
        ratio,
        tuple(findings),
    )


def analyze_model_gap(
    profile: ModelProfile, measurement: Measurement, hardware: HardwareSpec
) -> ModelGapAnalysis:
    """Analyze compute roofline efficiency and memory allocation in one call."""
    cost_to_analyze = profile.fused_cost if profile.fused_cost is not None else profile.cost
    return ModelGapAnalysis(
        analyze_gap(cost_to_analyze, measurement, hardware, operations=profile.operations),
        analyze_memory_gap(profile.memory, measurement),
    )


@dataclass(frozen=True, slots=True)
class SpeculativeDecodingAnalysis:
    """Hardware roofline speedup and breakeven analysis for speculative decoding."""

    gamma: int
    acceptance_rate: float
    expected_tokens_per_step: float
    draft_step_lower_bound_seconds: float
    target_verify_lower_bound_seconds: float
    spec_step_lower_bound_seconds: float
    latency_per_token_seconds: float
    baseline_decode_lower_bound_seconds: float
    speedup: float
    breakeven_acceptance_rate: float
    is_favorable: bool
    findings: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f"Speculative Decoding Analysis (gamma={self.gamma}, alpha={self.acceptance_rate:.1%})",
            f"  expected tokens/cycle: {self.expected_tokens_per_step:.2f}",
            f"  draft cycle bound: {self.draft_step_lower_bound_seconds * 1e3:.3f} ms",
            f"  target verify bound: {self.target_verify_lower_bound_seconds * 1e3:.3f} ms",
            f"  speculative step bound: {self.spec_step_lower_bound_seconds * 1e3:.3f} ms",
            f"  effective latency/token: {self.latency_per_token_seconds * 1e3:.3f} ms",
            f"  baseline decode bound: {self.baseline_decode_lower_bound_seconds * 1e3:.3f} ms",
            f"  speedup: {self.speedup:.2f}x ({'favorable' if self.is_favorable else 'slowdown'})",
            f"  breakeven alpha: {self.breakeven_acceptance_rate:.1%}"
            if self.breakeven_acceptance_rate <= 1.0
            else "  breakeven alpha: unattainable (overhead exceeds baseline)",
        ]
        lines.extend(f"  finding: {f}" for f in self.findings)
        return "\n".join(lines)


def analyze_speculative_decoding(
    draft_decode_cost: CostEstimate,
    target_verify_cost: CostEstimate,
    target_decode_cost: CostEstimate,
    hardware: HardwareSpec,
    gamma: int = 4,
    acceptance_rate: float = 0.7,
) -> SpeculativeDecodingAnalysis:
    """Analyze speculative decoding wall-clock speedup on target hardware.

    Evaluates whether the parallel verification in the target model plus the
    draft autoregressive steps achieves lower latency per token than standard
    autoregressive decoding in the target model alone.
    """
    if gamma <= 0:
        raise ValueError("gamma (speculation depth) must be positive")
    if not (0.0 <= acceptance_rate <= 1.0):
        raise ValueError("acceptance_rate must be between 0.0 and 1.0")

    draft_t_comp = draft_decode_cost.flops / hardware.peak_flops
    draft_t_mem = draft_decode_cost.total_bytes / hardware.memory_bandwidth
    draft_single_bound = max(draft_t_comp, draft_t_mem)
    draft_step_bound = gamma * draft_single_bound

    verify_t_comp = target_verify_cost.flops / hardware.peak_flops
    verify_t_mem = target_verify_cost.total_bytes / hardware.memory_bandwidth
    verify_bound = max(verify_t_comp, verify_t_mem)

    spec_step_bound = draft_step_bound + verify_bound

    base_t_comp = target_decode_cost.flops / hardware.peak_flops
    base_t_mem = target_decode_cost.total_bytes / hardware.memory_bandwidth
    baseline_bound = max(base_t_comp, base_t_mem)

    from .estimate import _expected_speculative_tokens, _find_breakeven_alpha

    e_tokens = _expected_speculative_tokens(gamma, acceptance_rate)
    eff_latency_per_token = spec_step_bound / e_tokens if e_tokens > 0 else float("inf")
    speedup = baseline_bound / eff_latency_per_token if eff_latency_per_token > 0 else 0.0

    ratio = spec_step_bound / baseline_bound if baseline_bound > 0 else float("inf")
    breakeven_alpha = _find_breakeven_alpha(gamma, ratio)

    findings: list[str] = []
    is_favorable = speedup > 1.0
    if is_favorable:
        findings.append(
            f"Speculative decoding achieves {speedup:.2f}x speedup; acceptance rate ({acceptance_rate:.1%}) exceeds breakeven ({breakeven_alpha:.1%})."
        )
    else:
        if breakeven_alpha > 1.0:
            findings.append(
                f"Speculative decoding regresses latency ({speedup:.2f}x); draft + verify latency ratio ({ratio:.2f}) exceeds max possible speculative yield ({1 + gamma}). Consider a lighter draft model."
            )
        else:
            findings.append(
                f"Speculative decoding regresses latency ({speedup:.2f}x); acceptance rate ({acceptance_rate:.1%}) is below the {breakeven_alpha:.1%} breakeven threshold."
            )

    return SpeculativeDecodingAnalysis(
        gamma=gamma,
        acceptance_rate=acceptance_rate,
        expected_tokens_per_step=e_tokens,
        draft_step_lower_bound_seconds=draft_step_bound,
        target_verify_lower_bound_seconds=verify_bound,
        spec_step_lower_bound_seconds=spec_step_bound,
        latency_per_token_seconds=eff_latency_per_token,
        baseline_decode_lower_bound_seconds=baseline_bound,
        speedup=speedup,
        breakeven_acceptance_rate=breakeven_alpha,
        is_favorable=is_favorable,
        findings=tuple(findings),
    )


@dataclass(frozen=True, slots=True)
class MoEGapAnalysis:
    """Gap analysis and memory thrashing diagnosis for an MoE layer."""

    cost: MoECostEstimate
    lower_bound_seconds: float
    compute_bound_seconds: float
    bandwidth_bound_seconds: float
    bottleneck: str
    observed_seconds: float | None
    efficiency: float | None
    active_to_total_ratio: float
    loaded_to_total_ratio: float
    findings: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f"MoE Gap Analysis ({'decode' if self.cost.is_decode else 'prefill'})",
            f"  total params: {self.cost.total_parameters:,} ({self.cost.parameter_bytes / 1e6:.1f} MB)",
            f"  active params/token: {self.cost.active_parameters:,} ({self.active_to_total_ratio:.1%} sparsity)",
            f"  expected loaded experts: {self.cost.expected_loaded_experts:.2f} ({self.loaded_to_total_ratio:.1%} of pool)",
            f"  arithmetic intensity: {self.cost.arithmetic_intensity:.2f} FLOP/B",
            f"  lower bound: {self.lower_bound_seconds * 1e3:.3f} ms ({self.bottleneck}-bound)",
        ]
        if self.observed_seconds is not None:
            lines.append(f"  observed: {self.observed_seconds * 1e3:.3f} ms")
            if self.efficiency is not None:
                lines.append(f"  roofline efficiency: {self.efficiency:.1%}")
        lines.extend(f"  finding: {f}" for f in self.findings)
        return "\n".join(lines)


def analyze_moe_gap(
    moe_cost: MoECostEstimate,
    hardware: HardwareSpec,
    observed_seconds: float | None = None,
) -> MoEGapAnalysis:
    """Analyze compute roofline efficiency and memory thrashing for an MoE layer."""
    compute_s = moe_cost.total_flops / hardware.peak_flops
    bandwidth_s = moe_cost.total_bytes / hardware.memory_bandwidth
    bound_s = max(compute_s, bandwidth_s)
    bottleneck = "compute" if compute_s >= bandwidth_s else "memory"

    efficiency = (bound_s / observed_seconds) if observed_seconds and observed_seconds > 0 else None
    if efficiency is not None:
        efficiency = min(1.0, efficiency)

    active_ratio = (
        moe_cost.active_parameters / moe_cost.total_parameters
        if moe_cost.total_parameters > 0
        else 1.0
    )
    loaded_ratio = (
        moe_cost.compulsory_param_read_bytes / moe_cost.parameter_bytes
        if moe_cost.parameter_bytes > 0
        else 1.0
    )

    findings: list[str] = []
    if moe_cost.is_decode:
        if bottleneck == "memory":
            findings.append(
                f"Decode is severely memory-bandwidth bound (AI: {moe_cost.arithmetic_intensity:.2f} FLOP/B < ridge point {hardware.ridge_point:.1f} FLOP/B)."
            )
            if loaded_ratio > 0.5:
                findings.append(
                    "High expert dispersion across tokens forces loading >50% of the expert pool; batching amortizes DRAM traffic."
                )
            else:
                findings.append(
                    f"Sparse activation loads only {loaded_ratio:.1%} of parameters from DRAM per step, saving {(1 - loaded_ratio):.1%} bandwidth vs dense."
                )
    else:
        if bottleneck == "compute":
            findings.append(
                "Prefill phase achieves high arithmetic intensity and is compute-bound."
            )
        else:
            findings.append(
                "Prefill phase remains memory-bound; consider increasing sequence length or batch size."
            )

    return MoEGapAnalysis(
        cost=moe_cost,
        lower_bound_seconds=bound_s,
        compute_bound_seconds=compute_s,
        bandwidth_bound_seconds=bandwidth_s,
        bottleneck=bottleneck,
        observed_seconds=observed_seconds,
        efficiency=efficiency,
        active_to_total_ratio=active_ratio,
        loaded_to_total_ratio=loaded_ratio,
        findings=tuple(findings),
    )


@dataclass(frozen=True, slots=True)
class PagedAttentionGapAnalysis:
    """Memory overhead and concurrency gain analysis for PagedAttention vs contiguous allocation."""

    cost: PagedAttentionCostEstimate
    unpaged_contiguous_bytes: int
    paged_allocated_bytes: int
    memory_saved_bytes: int
    memory_savings_ratio: float
    concurrency_multiplier: float
    fragmentation_ratio: float
    findings: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f"PagedAttention Gap Analysis (block_size={self.cost.block_size})",
            f"  unpaged contiguous memory: {self.unpaged_contiguous_bytes / 1e6:.2f} MB",
            f"  paged allocated memory: {self.paged_allocated_bytes / 1e6:.2f} MB",
            f"  memory saved: {self.memory_saved_bytes / 1e6:.2f} MB ({self.memory_savings_ratio:.1%} reduction)",
            f"  concurrency boost: {self.concurrency_multiplier:.2f}x capacity",
            f"  internal fragmentation: {self.cost.fragmentation_bytes / 1e3:.1f} KB ({self.fragmentation_ratio:.1%})",
        ]
        if self.cost.shared_prefix_blocks > 0:
            lines.append(
                f"  prefix sharing: {self.cost.shared_prefix_blocks} blocks shared ({self.cost.shared_saved_bytes / 1e6:.2f} MB deduplicated)"
            )
        lines.extend(f"  finding: {f}" for f in self.findings)
        return "\n".join(lines)


def analyze_paged_attention_gap(
    paged_cost: PagedAttentionCostEstimate,
    max_context_len: int,
    batch_size: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int | None = None,
    num_layers: int = 1,
    dtype_bytes: int = 2,
) -> PagedAttentionGapAnalysis:
    """Analyze memory savings and concurrency gains of PagedAttention vs traditional reservation."""
    kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
    head_dim = embed_dim // num_heads
    bytes_per_token = 2 * (kv_heads * head_dim) * num_layers * dtype_bytes

    unpaged_contiguous = batch_size * max_context_len * bytes_per_token
    paged_allocated = paged_cost.allocated_kv_bytes

    saved_bytes = max(0, unpaged_contiguous - paged_allocated)
    savings_ratio = saved_bytes / unpaged_contiguous if unpaged_contiguous > 0 else 0.0
    concurrency_mult = unpaged_contiguous / paged_allocated if paged_allocated > 0 else 1.0

    findings: list[str] = []
    if savings_ratio > 0.4:
        findings.append(
            f"PagedAttention saves {savings_ratio:.1%} VRAM vs contiguous reservation; supports {concurrency_mult:.2f}x higher request concurrency."
        )
    if paged_cost.fragmentation_ratio > 0.2:
        findings.append(
            f"Internal fragmentation is {paged_cost.fragmentation_ratio:.1%}; consider smaller block_size for short sequences."
        )
    if paged_cost.shared_prefix_blocks > 0:
        findings.append(
            f"Prefix caching deduplicates {paged_cost.shared_saved_bytes / 1e6:.2f} MB across concurrent sessions."
        )

    return PagedAttentionGapAnalysis(
        cost=paged_cost,
        unpaged_contiguous_bytes=unpaged_contiguous,
        paged_allocated_bytes=paged_allocated,
        memory_saved_bytes=saved_bytes,
        memory_savings_ratio=savings_ratio,
        concurrency_multiplier=concurrency_mult,
        fragmentation_ratio=paged_cost.fragmentation_ratio,
        findings=tuple(findings),
    )


@dataclass(frozen=True, slots=True)
class ContinuousBatchGapAnalysis:
    """Operational regime and throughput gap analysis for a continuous batching iteration."""

    iteration: ContinuousBatchIterationEstimate
    hardware: HardwareSpec
    lower_bound_seconds: float
    compute_bound_seconds: float
    bandwidth_bound_seconds: float
    bottleneck: str
    optimal_prefill_tokens_to_saturate: int
    findings: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f"Continuous Batch Iteration Analysis ({self.iteration.total_tokens} total tokens: {self.iteration.decode_tokens} decode, {self.iteration.prefill_tokens} prefill)",
            f"  arithmetic intensity: {self.iteration.arithmetic_intensity:.2f} FLOP/B (ridge point: {self.hardware.ridge_point:.1f} FLOP/B)",
            f"  iteration bound: {self.lower_bound_seconds * 1e3:.3f} ms ({self.bottleneck}-bound)",
            f"  memory traffic: {self.iteration.total_bytes / 1e6:.2f} MB (weights {self.iteration.model_weight_bytes / 1e6:.2f} MB, KV read {self.iteration.kv_cache_read_bytes / 1e6:.2f} MB)",
        ]
        if (
            self.bottleneck == "memory"
            and self.optimal_prefill_tokens_to_saturate > self.iteration.prefill_tokens
        ):
            lines.append(
                f"  target saturation: inject ~{self.optimal_prefill_tokens_to_saturate} prefill chunk tokens to reach compute roofline"
            )
        lines.extend(f"  finding: {f}" for f in self.findings)
        return "\n".join(lines)


def analyze_continuous_batch_iteration(
    iteration: ContinuousBatchIterationEstimate,
    hardware: HardwareSpec,
) -> ContinuousBatchGapAnalysis:
    """Evaluate continuous batch operational regime and compute prefill chunking targets."""
    t_comp = iteration.total_flops / hardware.peak_flops
    t_mem = iteration.total_bytes / hardware.memory_bandwidth
    lower_bound_s = max(t_comp, t_mem)
    bottleneck = "compute" if t_comp >= t_mem else "memory"

    ridge = hardware.ridge_point
    findings: list[str] = []

    if bottleneck == "compute":
        findings.append(
            f"Iteration is compute-bound (AI: {iteration.arithmetic_intensity:.1f} FLOP/B >= ridge: {ridge:.1f} FLOP/B); Tensor Cores are fully utilized."
        )
        opt_prefill = iteration.prefill_tokens
    else:
        flops_per_tok = iteration.total_flops / max(1, iteration.total_tokens)
        needed_tokens = max(
            1,
            int((iteration.model_weight_bytes * ridge) / flops_per_tok),
        )
        opt_prefill = max(0, needed_tokens - iteration.decode_tokens)
        findings.append(
            f"Iteration is memory-bandwidth bound (AI: {iteration.arithmetic_intensity:.1f} FLOP/B < ridge: {ridge:.1f} FLOP/B)."
        )
        if iteration.prefill_tokens == 0:
            findings.append(
                "Decode-only iteration; co-locating chunked prefill will amortize weight DRAM reads and increase throughput."
            )
        else:
            findings.append(
                f"Increase prefill chunk budget to ~{opt_prefill} tokens to transition iteration to compute-bound."
            )

    return ContinuousBatchGapAnalysis(
        iteration=iteration,
        hardware=hardware,
        lower_bound_seconds=lower_bound_s,
        compute_bound_seconds=t_comp,
        bandwidth_bound_seconds=t_mem,
        bottleneck=bottleneck,
        optimal_prefill_tokens_to_saturate=opt_prefill,
        findings=tuple(findings),
    )
