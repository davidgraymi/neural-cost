"""Activation checkpointing, selective recomputation, and training memory cost modeling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from neural_cost.hardware import HardwareSpec


class CheckpointingStrategy(str, Enum):
    """Activation checkpointing and rematerialization policy."""

    NONE = "none"
    FULL = "full"
    SELECTIVE = "selective"


@dataclass(frozen=True, slots=True)
class ActivationCheckpointingEstimate:
    """Analytical activation memory footprint and recomputation FLOP accounting."""

    strategy: CheckpointingStrategy
    batch_size: int
    seq_len: int
    embed_dim: int
    intermediate_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    dtype_bytes: int
    is_flash_attention: bool
    uncheckpointed_activation_bytes: int
    checkpointed_activation_bytes: int
    activation_memory_saved_bytes: int
    activation_savings_ratio: float
    forward_flops: int
    backward_flops: int
    recompute_flops: int
    total_training_flops: int
    compute_overhead_ratio: float


def estimate_activation_checkpointing(
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    intermediate_dim: int | None = None,
    num_kv_heads: int | None = None,
    dtype_bytes: int = 2,
    strategy: CheckpointingStrategy | str = CheckpointingStrategy.SELECTIVE,
    is_flash_attention: bool = True,
) -> ActivationCheckpointingEstimate:
    """Estimate activation memory retention and recomputation FLOP overhead across checkpointing strategies.

    Parameters:
        batch_size: Micro-batch size B.
        seq_len: Sequence length S.
        embed_dim: Hidden dimension D.
        num_layers: Number of transformer layers L.
        num_heads: Number of attention query heads H.
        intermediate_dim: FFN/MLP intermediate dimension (default: ~8/3 * embed_dim).
        num_kv_heads: KV attention heads for GQA/MQA (default: same as num_heads).
        dtype_bytes: Precision footprint in bytes (default: 2 for FP16/BF16).
        strategy: 'none' (stash all), 'full' (layer boundary inputs only), or 'selective' (recompute low-FLOP ops).
        is_flash_attention: Whether fused FlashAttention is used (avoids materializing full S x S attention matrix).
    """
    if batch_size <= 0 or seq_len <= 0 or embed_dim <= 0 or num_layers <= 0 or num_heads <= 0:
        raise ValueError(
            "batch_size, seq_len, embed_dim, num_layers, and num_heads must be positive"
        )
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")

    if isinstance(strategy, str):
        norm = strategy.strip().lower()
        if norm in ("none", "no", "off"):
            strat = CheckpointingStrategy.NONE
        elif norm in ("full", "layer", "all"):
            strat = CheckpointingStrategy.FULL
        elif norm in ("selective", "megatron", "flash"):
            strat = CheckpointingStrategy.SELECTIVE
        else:
            raise ValueError(
                f"Unknown checkpointing strategy '{strategy}'. Choose from 'none', 'full', 'selective'."
            )
    else:
        strat = strategy

    kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    inter_dim = intermediate_dim if intermediate_dim is not None else int(embed_dim * 8 / 3)
    head_dim = embed_dim // num_heads

    tokens = batch_size * seq_len

    # --- 1. Forward and Backward FLOPs per layer ---
    # Attention GEMMs: Q, K, V, Out projections + Attention Score matmuls
    # Q: 2 * tokens * D * D
    # K, V: 2 * 2 * tokens * D * (kv_heads * head_dim)
    # Out: 2 * tokens * D * D
    # Attention QK^T and Score*V: 4 * tokens * seq_len * D
    attn_gemm_flops = (
        2 * tokens * embed_dim * embed_dim  # Q
        + 4 * tokens * embed_dim * (kv_heads * head_dim)  # K, V
        + 2 * tokens * embed_dim * embed_dim  # Out
        + 4 * tokens * seq_len * embed_dim  # Attention core matmuls
    )

    # Attention lightweight ops (Softmax, Dropout, Norms):
    # LayerNorm: 4 * tokens * D
    # Softmax: 3 * tokens * num_heads * seq_len
    attn_low_flops = (
        4 * tokens * embed_dim  # Pre-attention LayerNorm / RMSNorm
        + 3 * tokens * num_heads * seq_len  # Attention Softmax
    )

    # MLP GEMMs (SwiGLU: Gate, Up, Down projections):
    # 2 * (D x inter_dim) + (inter_dim x D) = 3 projections
    mlp_gemm_flops = 6 * tokens * embed_dim * inter_dim

    # MLP lightweight ops:
    # Post-attention / Pre-MLP LayerNorm: 4 * tokens * D
    # SwiGLU activation (Swish + elementwise multiply): 3 * tokens * inter_dim
    mlp_low_flops = (
        4 * tokens * embed_dim  # Pre-MLP LayerNorm
        + 3 * tokens * inter_dim  # SwiGLU activation
    )

    layer_fwd_flops = attn_gemm_flops + attn_low_flops + mlp_gemm_flops + mlp_low_flops
    fwd_flops = layer_fwd_flops * num_layers

    # Backward pass compute is ~2x forward pass:
    bwd_flops = 2 * fwd_flops

    # --- 2. Activation Memory Accounting per layer ---
    # Boundary activation: Layer input tensor (saved to start backward recomputation)
    boundary_bytes = tokens * embed_dim * dtype_bytes

    # Uncheckpointed detailed intermediate tensors held for backward:
    # 1. Attention inputs & intermediate tensors:
    #    - QKV projection input: tokens * D * dtype_bytes
    #    - Q, K, V activations: tokens * (D + 2 * kv_heads * head_dim) * dtype_bytes
    #    - Attention matrix:
    #      If standard un-fused: B * H * S * S * dtype_bytes (quadratic in S)
    #      If FlashAttention: only softmax logsumexp: B * H * S * 4 bytes (linear in S)
    attn_matrix_bytes = (
        tokens * num_heads * 4
        if is_flash_attention
        else batch_size * num_heads * seq_len * seq_len * dtype_bytes
    )
    #    - Attention output projection input: tokens * D * dtype_bytes
    #    - Pre-attention norm input: tokens * D * dtype_bytes
    attn_internal_bytes = (
        tokens * embed_dim * dtype_bytes  # input to QKV
        + tokens * (embed_dim + 2 * kv_heads * head_dim) * dtype_bytes  # Q, K, V
        + attn_matrix_bytes  # softmax scores / LSE
        + tokens * embed_dim * dtype_bytes  # input to OutProj
        + tokens * embed_dim * dtype_bytes  # pre-attn norm
    )

    # 2. MLP intermediate tensors:
    #    - Pre-MLP norm input: tokens * D * dtype_bytes
    #    - Gate & Up projection activations: 2 * tokens * inter_dim * dtype_bytes
    #    - Down projection input (post-SwiGLU): tokens * inter_dim * dtype_bytes
    mlp_internal_bytes = (
        tokens * embed_dim * dtype_bytes  # pre-mlp norm
        + 3 * tokens * inter_dim * dtype_bytes  # gate, up, down inputs
    )

    total_layer_act_bytes = boundary_bytes + attn_internal_bytes + mlp_internal_bytes
    uncheckpointed_total_act_bytes = total_layer_act_bytes * num_layers

    # --- 3. Checkpointing Strategy Application ---
    if strat == CheckpointingStrategy.NONE:
        checkpointed_act_bytes = uncheckpointed_total_act_bytes
        recompute_flops = 0

    elif strat == CheckpointingStrategy.FULL:
        # Full checkpointing stores only layer boundary inputs: L * boundary_bytes
        # Plus 1 layer of materialized activations during active backward execution:
        checkpointed_act_bytes = (num_layers * boundary_bytes) + total_layer_act_bytes
        # Entire forward pass is recomputed during backward:
        recompute_flops = fwd_flops

    elif strat == CheckpointingStrategy.SELECTIVE:
        # Selective checkpointing (Megatron-LM / FlashAttention):
        # We save GEMM inputs & outputs (QKV, OutProj, MLP Gate/Up/Down):
        # - boundary_bytes (layer input)
        # - out_proj input (tokens * D)
        # - mlp gate/up input (tokens * D)
        # - mlp down input (tokens * inter_dim)
        retained_per_layer = (
            boundary_bytes
            + tokens * embed_dim * dtype_bytes  # out_proj input
            + tokens * embed_dim * dtype_bytes  # mlp input
            + tokens * inter_dim * dtype_bytes  # mlp down input
        )
        checkpointed_act_bytes = retained_per_layer * num_layers
        # Recompute only lightweight ops: LayerNorms, Softmax, SwiGLU activation:
        recompute_per_layer = attn_low_flops + mlp_low_flops
        recompute_flops = recompute_per_layer * num_layers

    saved_bytes = max(0, uncheckpointed_total_act_bytes - checkpointed_act_bytes)
    savings_ratio = (
        saved_bytes / uncheckpointed_total_act_bytes if uncheckpointed_total_act_bytes > 0 else 0.0
    )

    total_training_flops = fwd_flops + bwd_flops + recompute_flops
    baseline_training_flops = fwd_flops + bwd_flops
    compute_overhead_ratio = (
        recompute_flops / baseline_training_flops if baseline_training_flops > 0 else 0.0
    )

    return ActivationCheckpointingEstimate(
        strategy=strat,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        intermediate_dim=inter_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=kv_heads,
        dtype_bytes=dtype_bytes,
        is_flash_attention=is_flash_attention,
        uncheckpointed_activation_bytes=uncheckpointed_total_act_bytes,
        checkpointed_activation_bytes=checkpointed_act_bytes,
        activation_memory_saved_bytes=saved_bytes,
        activation_savings_ratio=savings_ratio,
        forward_flops=fwd_flops,
        backward_flops=bwd_flops,
        recompute_flops=recompute_flops,
        total_training_flops=total_training_flops,
        compute_overhead_ratio=compute_overhead_ratio,
    )


@dataclass(frozen=True, slots=True)
class ActivationCheckpointingGapAnalysis:
    """Gap analysis evaluating training memory pressure, OOM avoidance, and compute throughput trade-offs."""

    estimate: ActivationCheckpointingEstimate
    hardware: HardwareSpec
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    static_model_bytes: int
    total_training_memory_bytes: int
    uncheckpointed_training_memory_bytes: int
    is_oom: bool
    uncheckpointed_is_oom: bool
    memory_capacity_utilization: float
    forward_seconds: float
    backward_seconds: float
    recompute_seconds: float
    total_step_seconds: float
    baseline_step_seconds: float
    throughput_tokens_per_second: float
    max_trainable_seq_len: int
    findings: tuple[str, ...]

    def render(self) -> str:
        strat_name = self.estimate.strategy.value.upper()
        lines = [
            f"Activation Checkpointing Analysis (Strategy: {strat_name}, L={self.estimate.num_layers}, D={self.estimate.embed_dim}, S={self.estimate.seq_len})",
            f"  activation memory: {self.estimate.checkpointed_activation_bytes / 1e9:.2f} GB ({self.estimate.activation_savings_ratio:.1%} saved vs {self.estimate.uncheckpointed_activation_bytes / 1e9:.2f} GB uncheckpointed)",
            f"  total training footprint: {self.total_training_memory_bytes / 1e9:.2f} GB (static: {self.static_model_bytes / 1e9:.2f} GB, activations: {self.estimate.checkpointed_activation_bytes / 1e9:.2f} GB)",
        ]
        if self.hardware.memory_capacity:
            cap_gb = self.hardware.memory_capacity / 1e9
            status = (
                "OOM EXCEEDED"
                if self.is_oom
                else f"{self.memory_capacity_utilization:.1%} capacity"
            )
            lines.append(
                f"  VRAM capacity check ({self.hardware.device_name} {cap_gb:.0f} GB): {status}"
            )
            lines.append(
                f"  max trainable seq_len on device: {self.max_trainable_seq_len:,} tokens"
            )

        lines.extend(
            [
                f"  recomputation tax: +{self.estimate.recompute_flops / 1e12:.2f} TFLOP ({self.estimate.compute_overhead_ratio:.1%} training compute overhead)",
                f"  step time: {self.total_step_seconds * 1e3:.2f} ms (fwd: {self.forward_seconds * 1e3:.2f} ms, bwd: {self.backward_seconds * 1e3:.2f} ms, recompute: {self.recompute_seconds * 1e3:.2f} ms)",
                f"  training throughput: {self.throughput_tokens_per_second:,.1f} tokens/s",
            ]
        )
        lines.extend(f"  finding: {f}" for f in self.findings)
        return "\n".join(lines)


def analyze_activation_checkpointing(
    estimate: ActivationCheckpointingEstimate,
    hardware: HardwareSpec,
    parameter_count: int | None = None,
    optimizer_multiplier: float = 2.0,  # Adam moments (2x parameter bytes)
) -> ActivationCheckpointingGapAnalysis:
    """Analyze training memory footprint, OOM safety margins, and throughput trade-offs on target hardware.

    Parameters:
        estimate: ActivationCheckpointingEstimate.
        hardware: Target HardwareSpec.
        parameter_count: Optional explicit total parameter count. If omitted, estimated from architecture.
        optimizer_multiplier: Optimizer state memory multiplier (default: 2.0 for standard Adam FP32/FP16 states).
    """
    # 1. Parameter count calculation if not given
    if parameter_count is None:
        # Standard transformer parameter estimate:
        # Per layer: 4 * D^2 (attn) + 3 * D * inter_dim (SwiGLU) + 4 * D (norms)
        d = estimate.embed_dim
        inter_d = estimate.intermediate_dim
        per_layer = 4 * d * d + 3 * d * inter_d + 4 * d
        params = per_layer * estimate.num_layers
    else:
        params = parameter_count

    param_bytes = params * estimate.dtype_bytes
    grad_bytes = params * estimate.dtype_bytes
    opt_bytes = int(params * estimate.dtype_bytes * optimizer_multiplier)
    static_bytes = param_bytes + grad_bytes + opt_bytes

    total_mem_bytes = static_bytes + estimate.checkpointed_activation_bytes
    uncheckpointed_mem_bytes = static_bytes + estimate.uncheckpointed_activation_bytes

    vram_cap = hardware.memory_capacity
    is_oom = total_mem_bytes > vram_cap if vram_cap is not None else False
    unquant_is_oom = uncheckpointed_mem_bytes > vram_cap if vram_cap is not None else False

    utilization = total_mem_bytes / vram_cap if vram_cap and vram_cap > 0 else 0.0

    # 2. Maximum trainable sequence length calculation
    # S_max such that static_bytes + activation_bytes(S_max) <= vram_cap
    if vram_cap and vram_cap > static_bytes:
        avail_act_bytes = vram_cap - static_bytes
        # Activation bytes per token under this strategy:
        act_per_token = (
            estimate.checkpointed_activation_bytes / (estimate.batch_size * estimate.seq_len)
            if (estimate.batch_size * estimate.seq_len) > 0
            else 1.0
        )
        max_tokens = int(avail_act_bytes / act_per_token)
        max_seq_len = max(1, max_tokens // estimate.batch_size)
    else:
        max_seq_len = 0

    # 3. Step latency and throughput
    peak_flops = hardware.peak_flops
    fwd_time = estimate.forward_flops / peak_flops if peak_flops > 0 else 0.0
    bwd_time = estimate.backward_flops / peak_flops if peak_flops > 0 else 0.0
    recomp_time = estimate.recompute_flops / peak_flops if peak_flops > 0 else 0.0

    total_step_time = fwd_time + bwd_time + recomp_time
    baseline_step_time = fwd_time + bwd_time

    total_tokens = estimate.batch_size * estimate.seq_len
    throughput = total_tokens / total_step_time if total_step_time > 0 else 0.0

    findings: list[str] = []
    if unquant_is_oom and not is_oom:
        findings.append(
            f"Checkpointing prevents Out-Of-Memory (OOM)! Uncheckpointed training requires {uncheckpointed_mem_bytes / 1e9:.1f} GB, exceeding device capacity."
        )
    elif is_oom:
        findings.append(
            f"Total training footprint ({total_mem_bytes / 1e9:.1f} GB) exceeds VRAM capacity ({vram_cap / 1e9:.1f} GB). Consider reducing batch size or sequence length."
        )

    if estimate.strategy == CheckpointingStrategy.SELECTIVE:
        findings.append(
            f"Selective checkpointing saves {estimate.activation_savings_ratio:.1%} activation memory with negligible {estimate.compute_overhead_ratio:.1%} recomputation compute tax."
        )
    elif estimate.strategy == CheckpointingStrategy.FULL:
        findings.append(
            f"Full checkpointing minimizes activation footprint ({estimate.activation_savings_ratio:.1%} savings) at the cost of +{estimate.compute_overhead_ratio:.1%} extra compute (+1 full forward pass)."
        )

    return ActivationCheckpointingGapAnalysis(
        estimate=estimate,
        hardware=hardware,
        parameter_bytes=param_bytes,
        gradient_bytes=grad_bytes,
        optimizer_state_bytes=opt_bytes,
        static_model_bytes=static_bytes,
        total_training_memory_bytes=total_mem_bytes,
        uncheckpointed_training_memory_bytes=uncheckpointed_mem_bytes,
        is_oom=is_oom,
        uncheckpointed_is_oom=unquant_is_oom,
        memory_capacity_utilization=utilization,
        forward_seconds=fwd_time,
        backward_seconds=bwd_time,
        recompute_seconds=recomp_time,
        total_step_seconds=total_step_time,
        baseline_step_seconds=baseline_step_time,
        throughput_tokens_per_second=throughput,
        max_trainable_seq_len=max_seq_len,
        findings=tuple(findings),
    )
