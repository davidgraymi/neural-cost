"""Combined static compute and memory profiles for models."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .adapters.base import FrameworkAdapter
from .coverage import Coverage
from .estimate import (
    CostEstimate,
    FusedCostEstimate,
    estimate_fused_operations,
    estimate_operations,
)
from .memory import MemoryEstimate, estimate_memory
from .operations import Operation


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Theoretical compute and tensor-storage profile for concrete inputs."""

    cost: CostEstimate
    memory: MemoryEstimate
    fused_cost: FusedCostEstimate | None = None
    operations: tuple[Operation, ...] = ()
    coverage: Coverage | None = None


def profile_model(
    model: Any,
    example_inputs: Sequence[Any],
    adapter: FrameworkAdapter,
    *,
    training: bool = False,
    optimizer_state_multiplier: float = 0.0,
) -> ModelProfile:
    """Build a framework-neutral static profile through an adapter."""
    operations = tuple(adapter.operations(model, example_inputs))
    return ModelProfile(
        cost=estimate_operations(operations),
        memory=estimate_memory(
            operations,
            training=training,
            optimizer_state_multiplier=optimizer_state_multiplier,
        ),
        fused_cost=estimate_fused_operations(operations),
        operations=operations,
        coverage=getattr(adapter, "coverage", None),
    )
