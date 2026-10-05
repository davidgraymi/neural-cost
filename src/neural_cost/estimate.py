"""Theoretical operation cost estimation."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

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
