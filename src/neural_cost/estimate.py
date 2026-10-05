"""Theoretical operation cost estimation."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

from .hardware import HardwareSpec
from .operations import Operation, numel


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Theoretical work and minimum tensor traffic for a workload."""

    flops: int
    read_bytes: int
    write_bytes: int
    operations: int

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_bytes if self.total_bytes else 0.0

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        return CostEstimate(
            self.flops + other.flops,
            self.read_bytes + other.read_bytes,
            self.write_bytes + other.write_bytes,
            self.operations + other.operations,
        )


def estimate_conv2d(
    data: tuple[int, ...] | Operation,
    kernel: tuple[int, ...] | None = None,
    output: tuple[int, ...] | None = None,
    groups: int = 1,
) -> int:
    """Calculate FLOPs for 2D convolution with grouped and depthwise support.

    Accepts either an Operation instance or explicit (data, kernel, output, groups) shapes.
    Enforces the channel invariant: data[1] == kernel[1] * groups.
    """
    if isinstance(data, Operation):
        op = data
        if len(op.inputs) < 2:
            raise ValueError("conv2d requires input and kernel shapes")
        data_shape, kernel_shape = op.inputs[:2]
        output_shape = op.output
        groups_val = int(op.attrs.get("groups", groups))
    else:
        if kernel is None or output is None:
            raise ValueError("explicit shapes require data, kernel, and output")
        data_shape = data
        kernel_shape = kernel
        output_shape = output
        groups_val = groups

    if len(data_shape) != 4 or len(kernel_shape) != 4 or len(output_shape) != 4:
        raise ValueError("conv2d expects NCHW input, OIHW kernel, and NCHW output")
    if groups_val <= 0:
        raise ValueError("groups must be a positive integer")

    if data_shape[1] != kernel_shape[1] * groups_val or output_shape[1] != kernel_shape[0]:
        raise ValueError("conv2d channel dimensions do not match")

    assert data_shape[1] == kernel_shape[1] * groups_val

    return 2 * numel(output_shape) * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]


def estimate_operation(operation: Operation) -> CostEstimate:
    """Estimate FLOPs and compulsory tensor I/O for one operation.

    The estimate intentionally models useful work and compulsory traffic, not
    cache misses, workspace, fusion, or allocator effects.  Those differences
    become visible in :func:`neural_cost.analyze_gap`.
    """
    read_bytes = sum(numel(shape) for shape in operation.inputs) * operation.dtype_bytes + int(
        operation.attrs.get("parameter_bytes", 0)
    )
    write_bytes = numel(operation.output) * operation.dtype_bytes
    kind = operation.kind

    if kind in {"matmul", "linear"}:
        if len(operation.inputs) < 2:
            raise ValueError(f"{kind} requires input and weight shapes")
        left, right = operation.inputs[:2]
        if len(left) < 2 or len(right) != 2:
            raise ValueError(f"{kind} expects [..., M, K] and [K, N] shapes")
        if left[-1] != right[0]:
            raise ValueError("matrix inner dimensions do not match")
        flops = 2 * prod(left[:-1]) * right[0] * right[1]
    elif kind == "conv2d":
        flops = estimate_conv2d(operation)
    elif kind == "elementwise":
        flops = int(operation.attrs.get("flops_per_element", 1) * numel(operation.output))
    elif kind == "custom":
        if "flops" not in operation.attrs:
            raise ValueError("custom operations require attrs['flops']")
        flops = int(operation.attrs["flops"])
    elif kind == "softmax" or kind in {"layernorm", "batchnorm"}:
        flops = 5 * numel(operation.output)
    elif kind == "rmsnorm":
        flops = 3 * numel(operation.output)
    elif kind == "swiglu":
        flops = int(
            operation.attrs.get(
                "flops", operation.attrs.get("flops_per_element", 3) * numel(operation.output)
            )
        )
    elif kind == "embedding":
        # Embedding lookup: no multiply-accumulate FLOPs, just one read per token.
        # output shape is (batch, seq_len, embed_dim) or (N, embed_dim).
        flops = numel(operation.output)  # 1 FLOP/element as a nominal indexing cost
    elif kind == "attention":
        # Multi-head self-attention cost:
        #   4 linear projections  : 4 × 2 × B × T × D² / num_heads × num_heads = 4×2×B×T×D²
        #   QKᵀ per head         : B × H × T × T × head_dim  (2 FLOPs per element)
        #   softmax               : 5 × B × H × T²
        #   weighted sum (AV)     : B × H × T × T × head_dim (2 FLOPs)
        # We approximate using output shape (B, T, D) and attrs.
        if len(operation.output) >= 4:
            b = operation.output[0]
            num_heads = int(operation.attrs.get("num_heads", operation.output[1]))
            seq_len = int(operation.attrs.get("seq_len", operation.output[2]))
            head_dim = int(operation.attrs.get("head_dim", operation.output[3]))
            d = num_heads * head_dim
            b_t = b * seq_len
        elif len(operation.output) >= 2:
            b_t = prod(operation.output[:-1])  # batch * seq combined
            d = operation.output[-1]
            num_heads = int(operation.attrs.get("num_heads", 1))
            head_dim = int(operation.attrs.get("head_dim", d // max(num_heads, 1)))
            seq_len = int(
                operation.attrs.get(
                    "seq_len", operation.output[-2] if len(operation.output) >= 2 else 1
                )
            )
        else:
            b_t, d, num_heads, head_dim, seq_len = (
                1,
                numel(operation.output),
                1,
                numel(operation.output),
                1,
            )

        include_projections = bool(operation.attrs.get("include_projections", True))
        proj_flops = 4 * 2 * b_t * d * d if include_projections else 0
        attn_flops = 4 * b_t * num_heads * seq_len * head_dim
        softmax_flops = 5 * b_t * num_heads * seq_len
        flops = proj_flops + attn_flops + softmax_flops
    elif kind == "pooling":
        kernel_size = operation.attrs.get("kernel_size")
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError("pooling requires attrs['kernel_size'] as (kH, kW)")
        flops = numel(operation.output) * kernel_size[0] * kernel_size[1]
    elif kind == "moe":
        # Mixture-of-Experts layer
        out_shape = operation.output
        embed_dim = out_shape[-1]
        n_tokens = numel(out_shape[:-1]) if len(out_shape) > 1 else 1

        num_experts = int(operation.attrs.get("num_experts", 8))
        top_k = int(operation.attrs.get("top_k", 2))
        expert_hidden_dim = int(operation.attrs.get("expert_hidden_dim", 4 * embed_dim))
        shared_experts = int(operation.attrs.get("shared_experts", 0))
        expert_type = str(operation.attrs.get("expert_type", "swiglu")).lower()
        is_decode = bool(operation.attrs.get("is_decode", False))

        if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
            raise ValueError(f"invalid experts configuration: {num_experts=}, {top_k=}")

        # Router: Linear projection from embed_dim to num_experts
        router_flops = 2 * n_tokens * embed_dim * num_experts
        router_params = embed_dim * num_experts

        # Single expert parameter & FLOP count
        if expert_type == "swiglu":
            # Gate (D -> H), Up (D -> H), Down (H -> D)
            single_expert_params = 3 * embed_dim * expert_hidden_dim
            single_expert_flops = 6 * embed_dim * expert_hidden_dim + 3 * expert_hidden_dim
        else:
            # Standard MLP: Up (D -> H), Down (H -> D)
            single_expert_params = 2 * embed_dim * expert_hidden_dim
            single_expert_flops = 4 * embed_dim * expert_hidden_dim + 1 * expert_hidden_dim

        active_per_token = top_k + shared_experts
        expert_flops = n_tokens * active_per_token * single_expert_flops
        combine_flops = n_tokens * top_k * embed_dim
        flops = router_flops + expert_flops + combine_flops

        total_experts = num_experts + shared_experts
        total_params = router_params + total_experts * single_expert_params

        if is_decode:
            # Under decode (token-by-token generation), model expected unique experts loaded
            p_not_picked = (1.0 - (top_k / num_experts)) ** n_tokens if num_experts > 0 else 0.0
            expected_loaded = num_experts * (1.0 - p_not_picked)
            loaded_params = int(
                router_params + (expected_loaded + shared_experts) * single_expert_params
            )
            compulsory_param_bytes = loaded_params * operation.dtype_bytes
        else:
            compulsory_param_bytes = int(
                operation.attrs.get("parameter_bytes", total_params * operation.dtype_bytes)
            )

        # Override read_bytes to incorporate compulsory parameter traffic
        token_read_bytes = sum(numel(shape) for shape in operation.inputs) * operation.dtype_bytes
        read_bytes = token_read_bytes + compulsory_param_bytes
    elif kind == "paged_attention":
        out_shape = operation.output
        embed_dim = out_shape[-1]
        n_tokens = numel(out_shape[:-1]) if len(out_shape) > 1 else 1

        block_size = int(operation.attrs.get("block_size", 16))
        context_len = int(
            operation.attrs.get("context_len", operation.attrs.get("prompt_len", 128))
        )
        num_heads = int(operation.attrs.get("num_heads", 8))
        num_kv_heads = int(operation.attrs.get("num_kv_heads", num_heads))
        head_dim = int(operation.attrs.get("head_dim", embed_dim // max(num_heads, 1)))
        num_layers = int(operation.attrs.get("num_layers", 1))

        if block_size <= 0 or context_len <= 0:
            raise ValueError("block_size and context_len must be positive")

        attn_flops = 4 * n_tokens * context_len * (num_heads * head_dim) * num_layers
        softmax_flops = 5 * n_tokens * num_heads * context_len * num_layers
        include_projections = bool(operation.attrs.get("include_projections", True))
        proj_flops = (
            4 * 2 * n_tokens * embed_dim * embed_dim * num_layers if include_projections else 0
        )
        flops = attn_flops + softmax_flops + proj_flops

        num_blocks = (context_len + block_size - 1) // block_size
        kv_head_dim = num_kv_heads * head_dim
        allocated_kv_bytes = (
            n_tokens
            * num_blocks
            * block_size
            * 2
            * kv_head_dim
            * num_layers
            * operation.dtype_bytes
        )
        block_table_bytes = n_tokens * num_blocks * 8

        token_read_bytes = sum(numel(shape) for shape in operation.inputs) * operation.dtype_bytes
        read_bytes = token_read_bytes + allocated_kv_bytes + block_table_bytes
        new_kv_write = n_tokens * 2 * kv_head_dim * num_layers * operation.dtype_bytes
        write_bytes = numel(operation.output) * operation.dtype_bytes + new_kv_write
    elif kind == "state_space_model":
        out_shape = operation.output
        embed_dim = out_shape[-1]
        n_tokens = numel(out_shape[:-1]) if len(out_shape) > 1 else 1

        state_dim = int(operation.attrs.get("state_dim", 16))
        expand_factor = int(operation.attrs.get("expand_factor", 2))
        conv_kernel_size = int(operation.attrs.get("conv_kernel_size", 4))
        num_layers = int(operation.attrs.get("num_layers", 1))
        is_decode = bool(operation.attrs.get("is_decode", False))

        d_in = expand_factor * embed_dim

        in_proj_flops = 2 * n_tokens * embed_dim * (2 * d_in) * num_layers
        conv_flops = 2 * n_tokens * d_in * conv_kernel_size * num_layers
        dt_rank = max(1, embed_dim // 16)
        delta_flops = 2 * n_tokens * (d_in * dt_rank + dt_rank * d_in) * num_layers
        bc_flops = 2 * 2 * n_tokens * d_in * state_dim * num_layers
        core_flops = 6 * n_tokens * d_in * state_dim * num_layers
        gate_flops = n_tokens * d_in * num_layers
        out_proj_flops = 2 * n_tokens * d_in * embed_dim * num_layers

        flops = (
            in_proj_flops
            + conv_flops
            + delta_flops
            + bc_flops
            + core_flops
            + gate_flops
            + out_proj_flops
        )

        in_proj_params = embed_dim * (2 * d_in)
        conv_params = d_in * conv_kernel_size
        delta_params = d_in * dt_rank + dt_rank * d_in
        bc_params = 2 * d_in * state_dim
        out_proj_params = d_in * embed_dim
        total_params = (
            in_proj_params + conv_params + delta_params + bc_params + out_proj_params
        ) * num_layers

        batch_size = n_tokens if is_decode else (out_shape[0] if len(out_shape) > 1 else 1)
        state_bytes = batch_size * d_in * state_dim * operation.dtype_bytes * num_layers
        token_bytes = sum(numel(shape) for shape in operation.inputs) * operation.dtype_bytes
        param_bytes = total_params * operation.dtype_bytes

        if is_decode:
            read_bytes = token_bytes + param_bytes + state_bytes
            write_bytes = numel(operation.output) * operation.dtype_bytes + state_bytes
        else:
            read_bytes = token_bytes + param_bytes
            write_bytes = numel(operation.output) * operation.dtype_bytes
    else:  # Defensive in case a caller bypasses static typing.
        raise ValueError(f"unsupported operation kind: {kind}")
    return CostEstimate(flops, read_bytes, write_bytes, 1)


def estimate_operations(operations: Iterable[Operation]) -> CostEstimate:
    """Aggregate theoretical cost for an iterable of operations."""
    total = CostEstimate(0, 0, 0, 0)
    for operation in operations:
        total += estimate_operation(operation)
    return total


@dataclass(frozen=True, slots=True)
class FusedCostEstimate:
    """Cost estimate accounting for compiler operator fusion."""

    unfused: CostEstimate
    fused_read_bytes: int
    fused_write_bytes: int
    eliminated_bytes: int
    fused_groups_count: int

    @property
    def flops(self) -> int:
        return self.unfused.flops

    @property
    def read_bytes(self) -> int:
        return self.fused_read_bytes

    @property
    def write_bytes(self) -> int:
        return self.fused_write_bytes

    @property
    def operations(self) -> int:
        return self.unfused.operations

    @property
    def total_bytes(self) -> int:
        return self.fused_read_bytes + self.fused_write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_bytes if self.total_bytes else 0.0

    @property
    def traffic_reduction_ratio(self) -> float:
        return self.eliminated_bytes / self.unfused.total_bytes if self.unfused.total_bytes else 0.0


_FUSIBLE_CONSUMER_KINDS = {
    "elementwise",
    "softmax",
    "layernorm",
    "rmsnorm",
    "batchnorm",
    "pooling",
    "swiglu",
}


def estimate_fused_operations(operations: Iterable[Operation]) -> FusedCostEstimate:
    """Estimate theoretical work and reduced tensor traffic under operator fusion.

    Identifies producer-consumer patterns (such as linear/conv2d followed by
    elementwise, normalization, or pooling layers) that modern optimizing
    compilers (e.g., PyTorch Inductor, JAX/XLA) fuse into single kernels,
    eliminating intermediate DRAM roundtrips.
    """
    op_list = list(operations)
    unfused = estimate_operations(op_list)
    if not op_list:
        return FusedCostEstimate(unfused, 0, 0, 0, 0)

    eliminated_read_bytes = 0
    eliminated_write_bytes = 0
    fused_groups = 0

    i = 0
    while i < len(op_list):
        current_op = op_list[i]
        fused_with_current = False
        j = i + 1
        last_out_shape = current_op.output
        last_dtype = current_op.dtype_bytes

        while j < len(op_list):
            next_op = op_list[j]
            if (
                next_op.kind in _FUSIBLE_CONSUMER_KINDS
                and next_op.inputs
                and next_op.inputs[0] == last_out_shape
            ):
                intermediate_bytes = numel(last_out_shape) * min(last_dtype, next_op.dtype_bytes)
                eliminated_write_bytes += intermediate_bytes
                eliminated_read_bytes += intermediate_bytes
                fused_with_current = True
                last_out_shape = next_op.output
                last_dtype = next_op.dtype_bytes
                j += 1
            else:
                break

        if fused_with_current:
            fused_groups += 1
            i = j
        else:
            i += 1

    fused_read = max(0, unfused.read_bytes - eliminated_read_bytes)
    fused_write = max(0, unfused.write_bytes - eliminated_write_bytes)
    eliminated_total = eliminated_read_bytes + eliminated_write_bytes

    return FusedCostEstimate(
        unfused=unfused,
        fused_read_bytes=fused_read,
        fused_write_bytes=fused_write,
        eliminated_bytes=eliminated_total,
        fused_groups_count=fused_groups,
    )


def estimate_adamw_traffic(num_parameters: int, dtype_bytes: int = 4) -> int:
    """Model DRAM traffic for an AdamW optimizer step across model parameters.

    For parameter count P:
      Reads:  Parameter (P), Gradient (P), First Moment (P), Second Moment (P) = 4P
      Writes: Updated Parameter (P), First Moment (P), Second Moment (P)       = 3P
      Total DRAM bytes: (4P + 3P) * dtype_bytes = 7 * P * dtype_bytes.
    """
    return 7 * int(num_parameters) * int(dtype_bytes)


@dataclass(frozen=True, slots=True)
class MoECostEstimate:
    """Theoretical cost and memory traffic breakdown for a Mixture-of-Experts layer."""

    total_flops: int
    router_flops: int
    expert_flops: int
    combine_flops: int
    total_parameters: int
    active_parameters: int
    expected_loaded_experts: float
    parameter_bytes: int
    active_parameter_bytes: int
    compulsory_param_read_bytes: int
    token_read_bytes: int
    token_write_bytes: int
    total_bytes: int
    arithmetic_intensity: float
    is_decode: bool


def estimate_moe(
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    expert_hidden_dim: int,
    num_experts: int = 8,
    top_k: int = 2,
    shared_experts: int = 0,
    expert_type: str = "swiglu",
    dtype_bytes: int = 2,
    is_decode: bool = False,
) -> MoECostEstimate:
    """Model compute, parameter footprint, and compulsory traffic for an MoE layer.

    Parameters:
        batch_size: Batch dimension (number of independent sequences).
        seq_len: Sequence length (1 for decode, >1 for prefill).
        embed_dim: Model hidden dimension (e.g., 4096).
        expert_hidden_dim: Intermediate hidden dimension of each expert FFN (e.g., 14336).
        num_experts: Number of routed experts in the pool (e.g., 8, 64).
        top_k: Number of routed experts selected per token (e.g., 2, 8).
        shared_experts: Number of shared experts always executed for all tokens (e.g., DeepSeek).
        expert_type: 'swiglu' (3 linear projections) or 'mlp' (2 linear projections).
        dtype_bytes: Bytes per parameter and activation element (e.g., 2 for FP16/BF16).
        is_decode: If True, models token-by-token generation with probabilistic expert loading.
    """
    if batch_size <= 0 or seq_len <= 0 or embed_dim <= 0 or expert_hidden_dim <= 0:
        raise ValueError("dimensions must be positive")
    if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
        raise ValueError(f"invalid experts configuration: {num_experts=}, {top_k=}")
    if shared_experts < 0:
        raise ValueError("shared_experts cannot be negative")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")

    n_tokens = batch_size * seq_len
    norm_type = expert_type.lower()
    if norm_type == "swiglu":
        single_expert_params = 3 * embed_dim * expert_hidden_dim
        single_expert_flops = 6 * embed_dim * expert_hidden_dim + 3 * expert_hidden_dim
    else:
        single_expert_params = 2 * embed_dim * expert_hidden_dim
        single_expert_flops = 4 * embed_dim * expert_hidden_dim + 1 * expert_hidden_dim

    router_params = embed_dim * num_experts
    total_experts = num_experts + shared_experts
    total_params = router_params + total_experts * single_expert_params
    total_param_bytes = total_params * dtype_bytes

    active_params_per_token = router_params + (top_k + shared_experts) * single_expert_params
    active_param_bytes = active_params_per_token * dtype_bytes

    router_flops = 2 * n_tokens * embed_dim * num_experts
    expert_flops = n_tokens * (top_k + shared_experts) * single_expert_flops
    combine_flops = n_tokens * top_k * embed_dim
    total_flops = router_flops + expert_flops + combine_flops

    if is_decode:
        # Over n_tokens, probability an expert is never selected:
        p_not_picked = (1.0 - (top_k / num_experts)) ** n_tokens if num_experts > 0 else 0.0
        expected_loaded = num_experts * (1.0 - p_not_picked)
        compulsory_param_bytes = int(
            (router_params + (expected_loaded + shared_experts) * single_expert_params)
            * dtype_bytes
        )
    else:
        expected_loaded = float(num_experts)
        compulsory_param_bytes = total_param_bytes

    token_read_bytes = n_tokens * embed_dim * dtype_bytes
    token_write_bytes = n_tokens * embed_dim * dtype_bytes
    total_bytes = token_read_bytes + compulsory_param_bytes + token_write_bytes
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0

    return MoECostEstimate(
        total_flops=total_flops,
        router_flops=router_flops,
        expert_flops=expert_flops,
        combine_flops=combine_flops,
        total_parameters=total_params,
        active_parameters=active_params_per_token,
        expected_loaded_experts=expected_loaded,
        parameter_bytes=total_param_bytes,
        active_parameter_bytes=active_param_bytes,
        compulsory_param_read_bytes=compulsory_param_bytes,
        token_read_bytes=token_read_bytes,
        token_write_bytes=token_write_bytes,
        total_bytes=total_bytes,
        arithmetic_intensity=arithmetic_intensity,
        is_decode=is_decode,
    )


@dataclass(frozen=True, slots=True)
class SpeculativeCostEstimate:
    """Theoretical cost, speedup, and acceptance threshold for speculative decoding."""

    gamma: int
    acceptance_rate: float
    expected_tokens_per_step: float
    draft_decode_flops: int
    target_verify_flops: int
    total_step_flops: int
    draft_decode_bytes: int
    target_verify_bytes: int
    total_step_bytes: int
    baseline_target_flops: int
    baseline_target_bytes: int
    speedup_flops: float
    speedup_bytes: float
    breakeven_acceptance_rate_bytes: float
    breakeven_acceptance_rate_flops: float


def _expected_speculative_tokens(gamma: int, alpha: float) -> float:
    """Compute expected number of accepted + bonus tokens: E[N] = (1 - alpha^(gamma + 1)) / (1 - alpha)."""
    if gamma <= 0:
        return 1.0
    alpha = max(0.0, min(1.0, float(alpha)))
    if abs(alpha - 1.0) < 1e-9:
        return float(1 + gamma)
    return (1.0 - (alpha ** (gamma + 1))) / (1.0 - alpha)


def _find_breakeven_alpha(gamma: int, cost_ratio: float) -> float:
    """Solve for alpha* in [0, 1] where E[N(alpha*)] == cost_ratio.

    If cost_ratio <= 1.0, breakeven is 0.0 (always beneficial).
    If cost_ratio > 1 + gamma, breakeven is unattainable (returns inf).
    """
    if cost_ratio <= 1.0:
        return 0.0
    max_tokens = float(1 + gamma)
    if cost_ratio > max_tokens:
        return float("inf")

    low, high = 0.0, 1.0
    for _ in range(30):
        mid = (low + high) / 2.0
        val = _expected_speculative_tokens(gamma, mid)
        if val < cost_ratio:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def estimate_speculative_decoding(
    draft_decode_cost: CostEstimate,
    target_verify_cost: CostEstimate,
    target_decode_cost: CostEstimate,
    gamma: int = 4,
    acceptance_rate: float = 0.7,
) -> SpeculativeCostEstimate:
    """Estimate speculative decoding compute, memory traffic, and breakeven acceptance rate.

    Args:
        draft_decode_cost: Cost of generating 1 token autoregressively with draft model.
        target_verify_cost: Cost of parallel verification of gamma tokens in target model.
        target_decode_cost: Cost of generating 1 token autoregressively with target model (baseline).
        gamma: Lookahead speculation depth (number of candidate tokens proposed).
        acceptance_rate: Mean probability alpha of accepting a draft token.
    """
    if gamma <= 0:
        raise ValueError("gamma (speculation depth) must be positive")
    if not (0.0 <= acceptance_rate <= 1.0):
        raise ValueError("acceptance_rate must be between 0.0 and 1.0")

    e_tokens = _expected_speculative_tokens(gamma, acceptance_rate)

    draft_total_flops = gamma * draft_decode_cost.flops
    total_step_flops = draft_total_flops + target_verify_cost.flops
    baseline_flops_for_e_tokens = int(e_tokens * target_decode_cost.flops)
    speedup_flops = (
        (e_tokens * target_decode_cost.flops) / total_step_flops if total_step_flops > 0 else 0.0
    )

    draft_total_bytes = gamma * draft_decode_cost.total_bytes
    total_step_bytes = draft_total_bytes + target_verify_cost.total_bytes
    baseline_bytes_for_e_tokens = int(e_tokens * target_decode_cost.total_bytes)
    speedup_bytes = (
        (e_tokens * target_decode_cost.total_bytes) / total_step_bytes
        if total_step_bytes > 0
        else 0.0
    )

    ratio_flops = (
        total_step_flops / target_decode_cost.flops
        if target_decode_cost.flops > 0
        else float("inf")
    )
    ratio_bytes = (
        total_step_bytes / target_decode_cost.total_bytes
        if target_decode_cost.total_bytes > 0
        else float("inf")
    )

    breakeven_flops = _find_breakeven_alpha(gamma, ratio_flops)
    breakeven_bytes = _find_breakeven_alpha(gamma, ratio_bytes)

    return SpeculativeCostEstimate(
        gamma=gamma,
        acceptance_rate=acceptance_rate,
        expected_tokens_per_step=e_tokens,
        draft_decode_flops=draft_total_flops,
        target_verify_flops=target_verify_cost.flops,
        total_step_flops=total_step_flops,
        draft_decode_bytes=draft_total_bytes,
        target_verify_bytes=target_verify_cost.total_bytes,
        total_step_bytes=total_step_bytes,
        baseline_target_flops=baseline_flops_for_e_tokens,
        baseline_target_bytes=baseline_bytes_for_e_tokens,
        speedup_flops=speedup_flops,
        speedup_bytes=speedup_bytes,
        breakeven_acceptance_rate_bytes=breakeven_bytes,
        breakeven_acceptance_rate_flops=breakeven_flops,
    )


@dataclass(frozen=True, slots=True)
class PagedAttentionCostEstimate:
    """Theoretical cost, fragmentation, and memory metrics for PagedAttention."""

    total_flops: int
    compulsory_kv_bytes: int
    allocated_kv_bytes: int
    fragmentation_bytes: int
    fragmentation_ratio: float
    block_size: int
    total_blocks: int
    shared_prefix_blocks: int
    shared_saved_bytes: int
    block_table_bytes: int
    total_read_bytes: int
    total_write_bytes: int
    total_bytes: int
    arithmetic_intensity: float


def estimate_paged_attention(
    batch_size: int,
    context_lens: int | tuple[int, ...] | list[int],
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int | None = None,
    block_size: int = 16,
    num_layers: int = 1,
    dtype_bytes: int = 2,
    shared_prefix_len: int = 0,
    include_projections: bool = True,
) -> PagedAttentionCostEstimate:
    """Calculate PagedAttention memory allocation, fragmentation, and traffic.

    Parameters:
        batch_size: Number of concurrent sequences.
        context_lens: Either a single integer context length (for uniform sequences)
                      or a tuple of context lengths per sequence.
        embed_dim: Total embedding dimension (e.g. 4096).
        num_heads: Number of Query attention heads.
        num_kv_heads: Number of Key/Value attention heads (for GQA/MQA). Defaults to num_heads.
        block_size: Number of tokens per physical block/page (typically 16 or 32).
        num_layers: Number of Transformer layers (default 1).
        dtype_bytes: Bytes per element (e.g., 2 for FP16/BF16, 1 for FP8).
        shared_prefix_len: Length of common system prompt/prefix shared across all sequences.
        include_projections: Whether to include QKV and output projection FLOPs.
    """
    if batch_size <= 0 or embed_dim <= 0 or num_heads <= 0 or block_size <= 0:
        raise ValueError("batch_size, embed_dim, num_heads, and block_size must be positive")
    if num_layers <= 0 or dtype_bytes <= 0:
        raise ValueError("num_layers and dtype_bytes must be positive")
    if shared_prefix_len < 0:
        raise ValueError("shared_prefix_len cannot be negative")

    if isinstance(context_lens, int):
        if context_lens <= 0:
            raise ValueError("context_lens must be positive")
        c_lens = [context_lens] * batch_size
    else:
        c_lens = list(context_lens)
        if len(c_lens) != batch_size:
            raise ValueError(f"expected {batch_size} context lengths, got {len(c_lens)}")
        if any(cl <= 0 for cl in c_lens):
            raise ValueError("all context lengths must be positive")

    kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
    if kv_heads <= 0 or kv_heads > num_heads:
        raise ValueError(f"invalid num_kv_heads={kv_heads}; must be between 1 and num_heads")

    head_dim = embed_dim // num_heads
    kv_head_dim = kv_heads * head_dim
    bytes_per_token_kv = 2 * kv_head_dim * num_layers * dtype_bytes
    block_bytes = block_size * bytes_per_token_kv

    # Calculate allocated blocks per sequence
    blocks_per_seq = [(cl + block_size - 1) // block_size for cl in c_lens]
    raw_total_blocks = sum(blocks_per_seq)

    # Prefix sharing (Prompt / System Prompt caching)
    shared_blocks = shared_prefix_len // block_size if shared_prefix_len > 0 else 0
    saved_blocks = (batch_size - 1) * shared_blocks if (shared_blocks > 0 and batch_size > 1) else 0
    effective_total_blocks = max(1, raw_total_blocks - saved_blocks)

    allocated_kv_bytes = effective_total_blocks * block_bytes
    shared_saved_bytes = saved_blocks * block_bytes

    # Compulsory KV bytes: exactly useful tokens across sequences
    raw_compulsory_kv = sum(cl * bytes_per_token_kv for cl in c_lens)
    saved_compulsory_tokens = (
        (batch_size - 1) * shared_prefix_len if (shared_prefix_len > 0 and batch_size > 1) else 0
    )
    compulsory_kv_bytes = max(0, raw_compulsory_kv - saved_compulsory_tokens * bytes_per_token_kv)

    fragmentation_bytes = max(0, allocated_kv_bytes - compulsory_kv_bytes)
    fragmentation_ratio = (
        fragmentation_bytes / allocated_kv_bytes if allocated_kv_bytes > 0 else 0.0
    )

    block_table_bytes = effective_total_blocks * 8

    total_tokens = batch_size
    attn_flops = 4 * sum(cl for cl in c_lens) * (num_heads * head_dim) * num_layers
    softmax_flops = 5 * sum(cl for cl in c_lens) * num_heads * num_layers
    proj_flops = (
        4 * 2 * total_tokens * embed_dim * embed_dim * num_layers if include_projections else 0
    )
    total_flops = attn_flops + softmax_flops + proj_flops

    q_read_bytes = total_tokens * embed_dim * dtype_bytes
    total_read_bytes = q_read_bytes + allocated_kv_bytes + block_table_bytes

    o_write_bytes = total_tokens * embed_dim * dtype_bytes
    new_kv_write_bytes = total_tokens * bytes_per_token_kv
    total_write_bytes = o_write_bytes + new_kv_write_bytes

    total_bytes = total_read_bytes + total_write_bytes
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0

    return PagedAttentionCostEstimate(
        total_flops=total_flops,
        compulsory_kv_bytes=compulsory_kv_bytes,
        allocated_kv_bytes=allocated_kv_bytes,
        fragmentation_bytes=fragmentation_bytes,
        fragmentation_ratio=fragmentation_ratio,
        block_size=block_size,
        total_blocks=effective_total_blocks,
        shared_prefix_blocks=shared_blocks,
        shared_saved_bytes=shared_saved_bytes,
        block_table_bytes=block_table_bytes,
        total_read_bytes=total_read_bytes,
        total_write_bytes=total_write_bytes,
        total_bytes=total_bytes,
        arithmetic_intensity=arithmetic_intensity,
    )


@dataclass(frozen=True, slots=True)
class ContinuousBatchIterationEstimate:
    """Theoretical compute, memory traffic, and operational regime for a continuous batching step."""

    num_decode_requests: int
    num_prefill_requests: int
    decode_tokens: int
    prefill_tokens: int
    total_tokens: int
    model_weight_bytes: int
    kv_cache_read_bytes: int
    kv_cache_write_bytes: int
    activation_bytes: int
    total_bytes: int
    total_flops: int
    decode_flops: int
    prefill_flops: int
    arithmetic_intensity: float
    is_compute_bound: bool
    hardware_lower_bound_seconds: float | None = None


def estimate_continuous_batch_iteration(
    decode_context_lens: tuple[int, ...] | list[int] = (),
    prefill_chunk_lens: tuple[int, ...] | list[int] = (),
    embed_dim: int = 4096,
    num_heads: int = 32,
    num_kv_heads: int | None = None,
    intermediate_dim: int | None = None,
    num_layers: int = 32,
    dtype_bytes: int = 2,
    hardware: HardwareSpec | None = None,
) -> ContinuousBatchIterationEstimate:
    """Model a single iteration of continuous batching with co-located prefill and decode.

    In continuous batching (e.g., Orca, vLLM, Sarathi-Serve):
    - Decode requests each execute 1 token step reading historical KV cache.
    - Chunked prefill requests process chunks of prompt tokens with high arithmetic intensity.
    - Model weights are read once per iteration from DRAM and amortized across all tokens.
    """
    n_decode = len(decode_context_lens)
    n_prefill = len(prefill_chunk_lens)
    if n_decode == 0 and n_prefill == 0:
        raise ValueError("iteration must contain at least one decode or prefill request")
    if embed_dim <= 0 or num_heads <= 0 or num_layers <= 0 or dtype_bytes <= 0:
        raise ValueError("model parameters must be positive")

    kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
    h_dim = intermediate_dim if intermediate_dim is not None else int(2.7 * embed_dim)
    head_dim = embed_dim // num_heads

    decode_tokens = n_decode
    prefill_tokens = sum(prefill_chunk_lens)
    total_tokens = decode_tokens + prefill_tokens

    # Standard Llama-style SwiGLU block params:
    attn_params_per_layer = embed_dim * embed_dim * 2 + 2 * embed_dim * (kv_heads * head_dim)
    ffn_params_per_layer = 3 * embed_dim * h_dim
    norm_params_per_layer = 2 * embed_dim
    params_per_layer = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    total_model_params = num_layers * params_per_layer
    model_weight_bytes = total_model_params * dtype_bytes

    kv_bytes_per_token = 2 * (kv_heads * head_dim) * num_layers * dtype_bytes
    kv_cache_read_bytes = sum(cl * kv_bytes_per_token for cl in decode_context_lens)
    kv_cache_write_bytes = total_tokens * kv_bytes_per_token

    activation_bytes = 2 * total_tokens * embed_dim * dtype_bytes
    total_bytes = model_weight_bytes + kv_cache_read_bytes + kv_cache_write_bytes + activation_bytes

    linear_flops = 2 * total_tokens * total_model_params
    decode_attn_flops = (
        4 * sum(cl for cl in decode_context_lens) * (num_heads * head_dim) * num_layers
    )
    decode_flops = 2 * decode_tokens * total_model_params + decode_attn_flops

    prefill_attn_flops = (
        4 * sum(cl * cl for cl in prefill_chunk_lens) * (num_heads * head_dim) * num_layers
    )
    prefill_flops = 2 * prefill_tokens * total_model_params + prefill_attn_flops

    total_flops = linear_flops + decode_attn_flops + prefill_attn_flops
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0

    lower_bound_s = None
    if hardware is not None:
        t_comp = total_flops / hardware.peak_flops
        t_mem = total_bytes / hardware.memory_bandwidth
        lower_bound_s = max(t_comp, t_mem)
        is_compute_bound = t_comp >= t_mem
    else:
        is_compute_bound = arithmetic_intensity >= 50.0

    return ContinuousBatchIterationEstimate(
        num_decode_requests=n_decode,
        num_prefill_requests=n_prefill,
        decode_tokens=decode_tokens,
        prefill_tokens=prefill_tokens,
        total_tokens=total_tokens,
        model_weight_bytes=model_weight_bytes,
        kv_cache_read_bytes=kv_cache_read_bytes,
        kv_cache_write_bytes=kv_cache_write_bytes,
        activation_bytes=activation_bytes,
        total_bytes=total_bytes,
        total_flops=total_flops,
        decode_flops=decode_flops,
        prefill_flops=prefill_flops,
        arithmetic_intensity=arithmetic_intensity,
        is_compute_bound=is_compute_bound,
        hardware_lower_bound_seconds=lower_bound_s,
    )


@dataclass(frozen=True, slots=True)
class SSMCostEstimate:
    """Theoretical compute, parameter footprint, and state traffic for a State Space Model (e.g. Mamba/RWKV)."""

    total_flops: int
    in_proj_flops: int
    conv_flops: int
    ssm_core_flops: int
    out_proj_flops: int
    total_parameters: int
    parameter_bytes: int
    state_bytes: int
    input_read_bytes: int
    output_write_bytes: int
    state_read_bytes: int
    state_write_bytes: int
    total_bytes: int
    arithmetic_intensity: float
    is_decode: bool
    batch_size: int
    seq_len: int
    embed_dim: int
    state_dim: int
    expand_factor: int
    num_layers: int
    equivalent_transformer_kv_bytes: int
    memory_savings_ratio_vs_transformer: float


def estimate_ssm(
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    state_dim: int = 16,
    expand_factor: int = 2,
    conv_kernel_size: int = 4,
    num_layers: int = 1,
    dtype_bytes: int = 2,
    is_decode: bool = False,
    num_heads: int = 32,
    num_kv_heads: int | None = None,
) -> SSMCostEstimate:
    """Model compute FLOPs, parameter footprint, and state traffic for a State Space Model (Mamba/S6/SSD).

    Parameters:
        batch_size: Number of concurrent sequences.
        seq_len: Sequence length (use 1 for autoregressive decode token step).
        embed_dim: Hidden dimension D.
        state_dim: Recurrent state dimension N (e.g. 16 for Mamba-1, 64-128 for Mamba-2).
        expand_factor: Hidden dimension expansion E (usually 2, so D_in = E * D).
        conv_kernel_size: 1D depthwise convolution filter width (usually 4).
        num_layers: Number of SSM layers.
        dtype_bytes: Bytes per floating point element (default 2 for FP16/BF16).
        is_decode: If True, evaluates single-token autoregressive recurrent generation.
        num_heads: Attention heads in equivalent Transformer (for KV cache comparison).
        num_kv_heads: KV heads in equivalent Transformer (for GQA/MQA comparison).
    """
    if batch_size <= 0 or seq_len <= 0 or embed_dim <= 0:
        raise ValueError("batch_size, seq_len, and embed_dim must be positive")
    if state_dim <= 0 or expand_factor <= 0 or conv_kernel_size <= 0 or num_layers <= 0:
        raise ValueError(
            "state_dim, expand_factor, conv_kernel_size, and num_layers must be positive"
        )
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")

    d_in = expand_factor * embed_dim
    n_tokens = batch_size if is_decode else batch_size * seq_len

    # 1. Input projection: D -> 2 * D_in (gated branch z and main branch x')
    in_proj_params_per_layer = embed_dim * (2 * d_in)
    in_proj_flops = 2 * n_tokens * in_proj_params_per_layer * num_layers

    # 2. 1D Depthwise Conv: kernel_size * D_in
    conv_params_per_layer = d_in * conv_kernel_size
    conv_flops = 2 * n_tokens * conv_params_per_layer * num_layers

    # 3. SSM parameter projections:
    dt_rank = max(1, embed_dim // 16)
    delta_params_per_layer = d_in * dt_rank + dt_rank * d_in
    delta_flops = 2 * n_tokens * delta_params_per_layer * num_layers

    bc_params_per_layer = 2 * d_in * state_dim
    bc_flops = 2 * n_tokens * bc_params_per_layer * num_layers

    # 4. SSM recurrence core:
    # In prefill (parallel scan): ~6 FLOPs per token per state element
    # In decode (recurrent update): h_t = A * h_{t-1} + B * x_t, y_t = C * h_t (~6 FLOPs)
    core_flops = 6 * n_tokens * d_in * state_dim * num_layers

    # 5. Output projection and gating:
    gate_flops = n_tokens * d_in * num_layers
    out_proj_params_per_layer = d_in * embed_dim
    out_proj_flops = (2 * n_tokens * out_proj_params_per_layer + gate_flops) * num_layers

    total_flops = in_proj_flops + conv_flops + delta_flops + bc_flops + core_flops + out_proj_flops

    params_per_layer = (
        in_proj_params_per_layer
        + conv_params_per_layer
        + delta_params_per_layer
        + bc_params_per_layer
        + out_proj_params_per_layer
    )
    total_params = params_per_layer * num_layers
    param_bytes = total_params * dtype_bytes

    # Recurrent state: B * D_in * N * dtype_bytes per layer
    # Constant O(1) in sequence length!
    state_bytes = batch_size * d_in * state_dim * dtype_bytes * num_layers

    input_read_bytes = n_tokens * embed_dim * dtype_bytes
    output_write_bytes = n_tokens * embed_dim * dtype_bytes

    if is_decode:
        # Decode: reading weights + reading state + writing updated state + token I/O
        state_read_bytes = state_bytes
        state_write_bytes = state_bytes
    else:
        # Prefill: associative parallel scan in SRAM tile; state stays on-chip
        state_read_bytes = 0
        state_write_bytes = state_bytes

    total_bytes = (
        param_bytes + input_read_bytes + output_write_bytes + state_read_bytes + state_write_bytes
    )
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0

    # Equivalent Transformer KV cache comparison:
    tf_kv_heads = num_kv_heads or num_heads
    tf_head_dim = embed_dim // num_heads
    equiv_tf_kv_bytes = (
        2 * batch_size * seq_len * (tf_kv_heads * tf_head_dim) * dtype_bytes * num_layers
    )
    savings_bytes = max(0, equiv_tf_kv_bytes - state_bytes)
    savings_ratio = savings_bytes / equiv_tf_kv_bytes if equiv_tf_kv_bytes > 0 else 0.0

    return SSMCostEstimate(
        total_flops=total_flops,
        in_proj_flops=in_proj_flops,
        conv_flops=conv_flops,
        ssm_core_flops=core_flops,
        out_proj_flops=out_proj_flops,
        total_parameters=total_params,
        parameter_bytes=param_bytes,
        state_bytes=state_bytes,
        input_read_bytes=input_read_bytes,
        output_write_bytes=output_write_bytes,
        state_read_bytes=state_read_bytes,
        state_write_bytes=state_write_bytes,
        total_bytes=total_bytes,
        arithmetic_intensity=arithmetic_intensity,
        is_decode=is_decode,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        state_dim=state_dim,
        expand_factor=expand_factor,
        num_layers=num_layers,
        equivalent_transformer_kv_bytes=equiv_tf_kv_bytes,
        memory_savings_ratio_vs_transformer=savings_ratio,
    )
