"""How much of a traced function an adapter actually modeled.

A roofline efficiency only describes the operations an adapter understood.
If a search loop spends most of its time in primitives the adapter skipped,
a "69% efficient" headline is misleading.  Adapters record a
:class:`Coverage` after :meth:`~neural_cost.adapters.base.FrameworkAdapter.operations`
so reports can say how much of the workload the roofline covers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_COVERAGE_THRESHOLD = 0.9


@dataclass(frozen=True, slots=True)
class Coverage:
    """Counts of modeled, free and unmodeled units (jaxpr equations, leaf modules).

    ``modeled`` units produced an operation.  ``free`` units are intentionally
    zero-cost (reshapes, transposes, dtype casts that only change metadata).
    ``unmodeled`` units were skipped, so their cost is missing from every
    estimate; ``unmodeled_primitives`` lists them by frequency and
    ``unmodeled_output_bytes`` sums their output sizes (scaled by loop trip
    counts) as a rough measure of the traffic left out.  ``unit`` names what
    was counted.
    """

    modeled: int
    free: int
    unmodeled: int
    unmodeled_primitives: tuple[tuple[str, int], ...] = ()
    unmodeled_output_bytes: int = 0
    unit: str = "equations"

    def __post_init__(self) -> None:
        if min(self.modeled, self.free, self.unmodeled, self.unmodeled_output_bytes) < 0:
            raise ValueError("coverage counts must be non-negative")

    @property
    def total(self) -> int:
        return self.modeled + self.free + self.unmodeled

    @property
    def fraction(self) -> float:
        """Share of units that were modeled or are free by construction (1.0 if empty)."""
        return (self.modeled + self.free) / self.total if self.total else 1.0

    @classmethod
    def from_counts(
        cls,
        modeled: int,
        free: int,
        unmodeled: Iterable[str],
        unmodeled_output_bytes: int = 0,
        unit: str = "equations",
    ) -> Coverage:
        counts = Counter(unmodeled)
        ranked = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        return cls(modeled, free, sum(counts.values()), ranked, unmodeled_output_bytes, unit)

    def summary(self, top: int = 5) -> str:
        """One-line summary, e.g. ``coverage: 82.0% of 50 equations modeled (unmodeled: sort x4)``."""
        line = f"coverage: {self.fraction:.1%} of {self.total} {self.unit} modeled"
        if self.unmodeled_primitives:
            names = ", ".join(f"{name} x{count}" for name, count in self.unmodeled_primitives[:top])
            more = len(self.unmodeled_primitives) - top
            line += f" (unmodeled: {names}{f', +{more} more' if more > 0 else ''})"
        return line

    def warning(self, threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> str | None:
        """A warning when coverage is below ``threshold``, otherwise ``None``."""
        if self.fraction >= threshold:
            return None
        return (
            f"Low coverage: only {self.fraction:.1%} of the traced {self.unit} were modeled; "
            "the roofline bound and efficiency describe part of the workload, not all of it."
        )
