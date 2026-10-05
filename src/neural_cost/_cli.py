"""Console entry point and backward compatibility shim for neural-cost-compare."""

from __future__ import annotations

from neural_cost.cli.compare_cmd import (
    SHAPES,
    Result,
    compare_main,
    evaluate,
    print_hardware_header,
    print_results,
    print_summary,
    run_compare,
    run_jax,
    run_tensorflow,
    run_torch,
)


def main() -> None:
    compare_main()


__all__ = [
    "SHAPES",
    "Result",
    "compare_main",
    "evaluate",
    "main",
    "print_hardware_header",
    "print_results",
    "print_summary",
    "run_compare",
    "run_jax",
    "run_tensorflow",
    "run_torch",
]

if __name__ == "__main__":
    main()
