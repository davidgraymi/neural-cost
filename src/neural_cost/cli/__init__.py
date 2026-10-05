"""Unified CLI package for neural-cost."""

from __future__ import annotations

from neural_cost.cli.compare_cmd import compare_main
from neural_cost.cli.root import build_parser, main

__all__ = ["build_parser", "compare_main", "main"]
