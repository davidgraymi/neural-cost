"""Static model-memory accounting built from portable operations."""

from collections.abc import Iterable
from dataclasses import dataclass

from .operations import Operation, numel


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Idealized tensor-storage accounting for an inference or training step.

    Activation liveness is framework and graph dependent.  The minimum peak is
    the largest single output tensor, while the conservative peak assumes all
    forward outputs remain live.  Real allocator telemetry belongs in
    :class:`neural_cost.Measurement` and should be compared against these
    explicitly idealized bounds.
    """

    parameter_bytes: int
    activation_bytes: int
    minimum_peak_activation_bytes: int
    conservative_peak_activation_bytes: int
    gradient_bytes: int = 0
    optimizer_state_bytes: int = 0

    @property
    def inference_minimum_bytes(self) -> int:
        """Parameters plus the lower bound for live forward activations."""
        return self.parameter_bytes + self.minimum_peak_activation_bytes

    @property
    def inference_conservative_bytes(self) -> int:
        """Parameters plus all materialized forward outputs held concurrently."""
        return self.parameter_bytes + self.conservative_peak_activation_bytes

    @property
    def training_minimum_bytes(self) -> int:
        """Idealized training footprint including gradients and optimizer state."""
        return self.inference_minimum_bytes + self.gradient_bytes + self.optimizer_state_bytes


def estimate_memory(
    operations: Iterable[Operation],
    *,
    training: bool = False,
    optimizer_state_multiplier: float = 0.0,
) -> MemoryEstimate:
    """Estimate parameter and activation storage from portable operations.

    Adapters attach a stable ``parameter_id`` so shared modules are counted
    once.  Hand-authored linear and convolution operations fall back to their
    weight tensor as a parameter.  ``optimizer_state_multiplier=2`` models
    Adam's two parameter-sized moment buffers; enable ``training`` to include
    one parameter-sized gradient buffer as well.
    """
    if optimizer_state_multiplier < 0:
        raise ValueError("optimizer_state_multiplier must be non-negative")

    parameter_bytes = 0
    activations: list[int] = []
    seen_parameters: set[int] = set()
    for operation in operations:
        output_bytes = numel(operation.output) * operation.dtype_bytes
        activations.append(output_bytes)

        declared = operation.attrs.get("parameter_bytes")
        parameter_id = operation.attrs.get("parameter_id")
        if (
            declared is None
            and operation.kind in {"linear", "conv2d", "conv"}
            and len(operation.inputs) >= 2
        ):
            declared = numel(operation.inputs[1]) * operation.dtype_bytes
        if declared is None:
            continue
        if not isinstance(declared, (int, float)) or declared < 0:
            raise ValueError("attrs['parameter_bytes'] must be a non-negative number")
        if parameter_id is not None:
            if not isinstance(parameter_id, int):
                raise ValueError("attrs['parameter_id'] must be an integer")
            if parameter_id in seen_parameters:
                continue
            seen_parameters.add(parameter_id)
        parameter_bytes += int(declared)

    activation_bytes = sum(activations)
    minimum_peak = max(activations, default=0)
    gradients = parameter_bytes if training else 0
    optimizer = int(parameter_bytes * optimizer_state_multiplier) if training else 0
    return MemoryEstimate(
        parameter_bytes,
        activation_bytes,
        minimum_peak,
        activation_bytes,
        gradients,
        optimizer,
    )
