"""Framework adapter contract."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from ..operations import Operation
from ..profiler import Measurement


class FrameworkAdapter(ABC):
    """Bridge a framework's model representation to neural-cost.

    Implement ``operations`` for theoretical analysis.  Override ``benchmark``
    when the framework requires synchronization or offers memory telemetry.
    """

    name: str = "custom"
    #: Set by ``operations()`` in adapters that track it: how much of the
    #: traced model was modeled (see :class:`neural_cost.coverage.Coverage`).
    coverage: Any = None

    @abstractmethod
    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        """Return portable operations for ``model`` and concrete example inputs."""

    def benchmark(
        self,
        function: Callable[..., Any],
        *args: Any,
        warmup: int = 3,
        repeats: int = 10,
        **kwargs: Any,
    ) -> Measurement:
        from ..profiler import benchmark

        return benchmark(function, *args, warmup=warmup, repeats=repeats, **kwargs)

    def profile(
        self,
        model: Any,
        example_inputs: Sequence[Any],
        *,
        training: bool = False,
        optimizer_state_multiplier: float = 0.0,
    ) -> Any:
        """Convenience method to profile a model using this adapter."""
        from ..model import profile_model

        return profile_model(
            model,
            example_inputs,
            self,
            training=training,
            optimizer_state_multiplier=optimizer_state_multiplier,
        )
