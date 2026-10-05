"""Benchmark modules for empirical execution and roofline comparison."""

from __future__ import annotations

from neural_cost.benchmarks.llm import (
    DecodeMetrics,
    MoEDecodeMetrics,
    PagedAttentionMetrics,
    PrefillMetrics,
    SpeculativeDecodeMetrics,
    benchmark_decode,
    benchmark_moe_decode,
    benchmark_paged_attention,
    benchmark_prefill,
    benchmark_speculative_decoding,
)

__all__ = [
    "DecodeMetrics",
    "MoEDecodeMetrics",
    "PagedAttentionMetrics",
    "PrefillMetrics",
    "SpeculativeDecodeMetrics",
    "benchmark_decode",
    "benchmark_moe_decode",
    "benchmark_paged_attention",
    "benchmark_prefill",
    "benchmark_speculative_decoding",
]
