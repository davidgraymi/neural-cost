"""A small, portable operation representation.

Framework adapters translate their graphs/modules into these records.  Custom
frameworks can construct them directly without depending on any ML runtime.
"""

from dataclasses import dataclass, field
from typing import Literal, get_args

OperationKind = Literal[
    "matmul",
    "linear",
    "conv2d",
    "elementwise",
    "custom",
    "softmax",
    "layernorm",
    "rmsnorm",
    "batchnorm",
    "pooling",
    "embedding",
    "attention",
    "swiglu",
    "moe",
    "paged_attention",
]


@dataclass(frozen=True, slots=True)
class Operation:
    """An operation and the tensor shapes needed to estimate its cost.

    Shapes are tuples of concrete dimensions.  ``attrs`` holds operation
    parameters such as ``kernel_size``, ``stride``, or a custom ``flops`` value.
    """

    name: str
    kind: OperationKind
    inputs: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]
    dtype_bytes: int = 4
    attrs: dict[str, int | float | tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_kinds = get_args(OperationKind)
        if self.kind not in valid_kinds:
            raise ValueError(
                f"unsupported operation kind {self.kind!r}; must be one of {valid_kinds}"
            )
        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")
        for shape in (*self.inputs, self.output):
            if any(d <= 0 for d in shape):
                raise ValueError("shapes must contain only positive dimensions")


def numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result
