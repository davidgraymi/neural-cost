"""End-to-end tests for MoE and Speculative Decoding in LLM Benchmarks."""

import sys
from pathlib import Path

import pytest

from neural_cost import HardwareSpec, estimate_moe

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))


class TestLLMBenchmarkMoEAndSpeculative:
    """Validate benchmark extensions in benchmarks/llm_bench.py."""

    def test_llm_bench_exports(self):
        import llm_bench

        assert hasattr(llm_bench, "benchmark_moe_decode")
        assert hasattr(llm_bench, "benchmark_speculative_decoding")
        assert hasattr(llm_bench, "MoEDecodeMetrics")
        assert hasattr(llm_bench, "SpeculativeDecodeMetrics")

    def test_e2e_benchmark_moe_decode(self):
        pytest.importorskip("torch")
        import llm_bench

        hardware = HardwareSpec("M1_Mock", peak_flops=2.6e12, memory_bandwidth=68e9)
        metrics = llm_bench.benchmark_moe_decode(
            batch_size=1,
            embed_dim=64,
            expert_hidden_dim=128,
            num_experts=4,
            top_k=2,
            warmup=1,
            repeats=2,
            hardware=hardware,
        )

        assert isinstance(metrics, llm_bench.MoEDecodeMetrics)
        assert metrics.batch_size == 1
        assert metrics.latency_ms > 0
        assert metrics.tokens_per_sec > 0
        assert metrics.expected_loaded_experts == pytest.approx(2.0, rel=1e-4)
        assert metrics.total_params > metrics.active_params
        assert metrics.achieved_gbw > 0
        assert 0.0 <= metrics.memory_bw_util <= 1.0

    def test_e2e_benchmark_speculative_decoding(self):
        pytest.importorskip("torch")
        import llm_bench

        metrics = llm_bench.benchmark_speculative_decoding(
            gamma=3,
            acceptance_rate=0.75,
            batch_size=1,
            target_embed_dim=128,
            draft_embed_dim=64,
            prompt_len=32,
            warmup=1,
            repeats=2,
        )

        assert isinstance(metrics, llm_bench.SpeculativeDecodeMetrics)
        assert metrics.gamma == 3
        assert metrics.expected_tokens_per_step > 2.0
        assert metrics.draft_latency_ms > 0
        assert metrics.verify_latency_ms > 0
        assert metrics.spec_step_latency_ms == pytest.approx(
            metrics.draft_latency_ms + metrics.verify_latency_ms, rel=1e-5
        )
        assert metrics.effective_ms_per_token > 0
        assert metrics.baseline_ms_per_token > 0
        assert metrics.speedup > 0


class TestRealisticArchitecturesMoE:
    """Validate scaling models against known open-weight MoE specifications."""

    def test_mixtral_8x7b_layer_dimensions(self):
        """Mixtral 8x7B: D=4096, H=14336, E=8, k=2, SwiGLU."""
        est_decode = estimate_moe(
            batch_size=1,
            seq_len=1,
            embed_dim=4096,
            expert_hidden_dim=14336,
            num_experts=8,
            top_k=2,
            expert_type="swiglu",
            dtype_bytes=2,
            is_decode=True,
        )

        # Single expert SwiGLU = 3 * 4096 * 14336 = 176,160,768 params
        # 8 experts = 1,409,286,144 params per MoE layer (~1.4B params per layer, ~47B total across 32 layers)
        # Active params per token = 2 * 176M + router = ~352M params
        assert est_decode.total_parameters > 1.4e9
        assert est_decode.active_parameters < 400e6
        # Sparsity ratio should be ~25%
        assert est_decode.active_parameters / est_decode.total_parameters == pytest.approx(
            0.25, abs=0.01
        )
        assert est_decode.expected_loaded_experts == pytest.approx(2.0, rel=1e-4)

    def test_deepseek_v2_style_shared_and_finegrained_experts(self):
        """DeepSeek-V2 style: fine-grained experts (E=64, k=6) + 2 shared experts."""
        est = estimate_moe(
            batch_size=8,
            seq_len=1,
            embed_dim=2048,
            expert_hidden_dim=1536,
            num_experts=64,
            top_k=6,
            shared_experts=2,
            expert_type="swiglu",
            dtype_bytes=2,
            is_decode=True,
        )

        assert est.total_parameters > est.active_parameters
        # In batch=8, not all 64 experts are loaded:
        # P(expert not picked by 1 token) = 1 - 6/64 = 58/64 = 0.90625
        # P(not picked by 8 tokens) = 0.90625^8 = 0.444
        # Expected loaded = 64 * (1 - 0.444) = ~35.5 experts
        assert 30.0 < est.expected_loaded_experts < 40.0
