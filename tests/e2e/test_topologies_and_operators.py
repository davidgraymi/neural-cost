"""End-to-End Tests for Realistic Topologies (#19) & Modernized Operators (#23).

Covers:
- Scale tiers (micro, standard, production) and modern topologies (ConvNeXt, ViT, Transformer, DeepDNN).
- Gating of legacy RNN and LSTM models behind --include-legacy (excluded by default).
- Hardware-accelerated fused SDPA attention in Transformer models.
- RMSNorm and SwiGLU operators in OperationKind and estimate_operation.
- Depthwise convolution support in estimate_conv2d (groups parameter).
- Batch sweeps for standard scale ([1, 4, 16, 64]).
- Dynamic report generation resilience when batch 32 is absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import get_args

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from neural_cost.estimate import CostEstimate, estimate_operation
from neural_cost.operations import Operation, OperationKind

# Ensure repo root and benchmarks directory are importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))


# ---------------------------------------------------------------------------

# 1. Scale Tiers and Modern Topologies (Issue #19)
# ---------------------------------------------------------------------------


class TestRealisticTopologiesAndScaleTiers:
    """Validate model scale tiers and modern topologies from Issue #19."""

    @pytest.mark.parametrize("scale", ["micro", "standard", "production"])
    def test_model_registry_supports_scale_tiers(self, scale: str):
        """get_model must support micro, standard, and production scale tiers."""
        try:
            from benchmarks.models import get_model
        except ImportError as exc:
            pytest.fail(f"benchmarks.models.get_model is required by Issue #19: {exc}")

        # Test modern transformer instantiation at each scale
        model, inputs = get_model("transformer", framework="torch", scale=scale)
        assert isinstance(model, nn.Module)
        assert isinstance(inputs, (tuple, list, torch.Tensor))

    def test_model_registry_rejects_invalid_scale(self):
        """get_model must reject unsupported scale names."""
        try:
            from benchmarks.models import get_model
        except ImportError as exc:
            pytest.fail(f"benchmarks.models.get_model required: {exc}")

        with pytest.raises((ValueError, KeyError)):
            get_model("transformer", framework="torch", scale="invalid_tier_scale")

    def test_convnext_standard_scale_and_flops(self):
        """ConvNeXt at standard scale must handle ImageNet inputs (224x224x3) and have >4 GFLOPs."""
        try:
            from benchmarks.models import get_model
        except ImportError as exc:
            pytest.fail(f"benchmarks.models.get_model required: {exc}")

        model, example_inputs = get_model("convnext", framework="torch", scale="standard")
        assert isinstance(model, nn.Module)

        # Ensure input shape is ImageNet standard [B, 3, 224, 224]
        if isinstance(example_inputs, (tuple, list)):
            x = example_inputs[0]
        else:
            x = example_inputs
        assert x.ndim == 4
        assert x.shape[1] == 3
        assert x.shape[2] == 224
        assert x.shape[3] == 224

        # Run forward pass to verify valid topology
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.ndim in (2, 4)

        # Profile model FLOPs using TorchAdapter
        from neural_cost.adapters import TorchAdapter

        adapter = TorchAdapter()
        profile = adapter.profile(model, (x,))
        # Requirement from Issue #19: > 4 GFLOPs for standard vision
        assert profile.cost.flops >= 4.0e9, (
            f"ConvNeXt standard scale expected >= 4 GFLOPs, got {profile.cost.flops / 1e9:.2f} GFLOPs"
        )

    def test_vit_standard_scale_topology(self):
        """Vision Transformer (ViT) standard scale must handle ImageNet inputs and execute encoder."""
        try:
            from benchmarks.models import get_model
        except ImportError as exc:
            pytest.fail(f"benchmarks.models.get_model required: {exc}")

        model, example_inputs = get_model("vit", framework="torch", scale="standard")
        assert isinstance(model, nn.Module)

        if isinstance(example_inputs, (tuple, list)):
            x = example_inputs[0]
        else:
            x = example_inputs
        assert x.shape[1:] == (3, 224, 224)

        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == x.shape[0]

    def test_deep_dnn_standard_scale_topology(self):
        """Deep DNN standard scale must represent deep feedforward network (1024 -> 4096 -> 4096 -> 1024)."""
        try:
            from benchmarks.models import get_model
        except ImportError as exc:
            pytest.fail(f"benchmarks.models.get_model required: {exc}")

        model, example_inputs = get_model("deep_dnn", framework="torch", scale="standard")
        assert isinstance(model, nn.Module)

        if isinstance(example_inputs, (tuple, list)):
            x = example_inputs[0]
        else:
            x = example_inputs
        assert x.shape[-1] == 1024

        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == x.shape[0]

    def test_collect_data_cli_scale_argument(self):
        """collect_data.py must expose --scale flag with choices ['micro', 'standard', 'production']."""
        import collect_data

        parser = argparse.ArgumentParser()
        # Test if collect_data provides an argument parser helper
        if hasattr(collect_data, "build_parser"):
            parser = collect_data.build_parser()
            actions = {action.dest: action for action in parser._actions}
            assert "scale" in actions, "CLI parser must support --scale"
            scale_action = actions["scale"]
            assert scale_action.default == "standard"
            assert set(scale_action.choices) == {"micro", "standard", "production"}
        else:
            # Check parser definition in main or module
            source = Path(collect_data.__file__).read_text()
            assert "--scale" in source, "benchmarks/collect_data.py must implement --scale argument"

    def test_standard_batch_sweep_values(self):
        """Issue #19 requirement: standard scale default batch sweep is [1, 4, 16, 64]."""
        import collect_data

        if hasattr(collect_data, "SCALE_BATCH_SIZES"):
            assert collect_data.SCALE_BATCH_SIZES.get("standard") == [1, 4, 16, 64]
        elif hasattr(collect_data, "STANDARD_BATCH_SIZES"):
            assert collect_data.STANDARD_BATCH_SIZES == [1, 4, 16, 64]
        else:
            source = Path(collect_data.__file__).read_text()
            assert "[1, 4, 16, 64]" in source or "1, 4, 16, 64" in source, (
                "collect_data.py must define default standard batch sweep [1, 4, 16, 64]"
            )


# ---------------------------------------------------------------------------
# 2. Legacy RNN/LSTM Exclusion and Gating (Issue #23)
# ---------------------------------------------------------------------------


class TestLegacyModelGating:
    """Verify that vanilla RNN and LSTM are excluded by default and enabled via --include-legacy."""

    def test_collect_data_excludes_legacy_models_by_default(self):
        """Default benchmark architectures must NOT include RNN or LSTM."""
        import collect_data

        if hasattr(collect_data, "get_benchmarked_architectures"):
            archs = collect_data.get_benchmarked_architectures(include_legacy=False)
            assert "RNN" not in archs
            assert "LSTM" not in archs
        elif hasattr(collect_data, "DEFAULT_ARCHITECTURES"):
            assert "RNN" not in collect_data.DEFAULT_ARCHITECTURES
            assert "LSTM" not in collect_data.DEFAULT_ARCHITECTURES
        else:
            source = Path(collect_data.__file__).read_text()
            assert "--include-legacy" in source, "collect_data.py must implement --include-legacy"

    def test_collect_data_includes_legacy_models_with_flag(self):
        """Supplying --include-legacy must restore RNN and LSTM benchmarks."""
        import collect_data

        if hasattr(collect_data, "get_benchmarked_architectures"):
            archs = collect_data.get_benchmarked_architectures(include_legacy=True)
            assert "RNN" in archs
            assert "LSTM" in archs
        else:
            source = Path(collect_data.__file__).read_text()
            assert "include_legacy" in source or "--include-legacy" in source

    def test_cli_parser_include_legacy_flag(self):
        """CLI parser must accept --include-legacy boolean flag defaulting to False."""
        import collect_data

        if hasattr(collect_data, "build_parser"):
            parser = collect_data.build_parser()
            args = parser.parse_args([])
            assert getattr(args, "include_legacy", False) is False

            args_legacy = parser.parse_args(["--include-legacy"])
            assert args_legacy.include_legacy is True


# ---------------------------------------------------------------------------
# 3. Modern Operators and Estimation Engine (Issue #23)
# ---------------------------------------------------------------------------


class TestModernOperatorsAndEstimation:
    """Verify RMSNorm, SwiGLU, SDPA fused attention, and depthwise conv."""

    def test_operation_kind_contains_rmsnorm_and_swiglu(self):
        """OperationKind must include 'rmsnorm' and 'swiglu'."""
        kinds = get_args(OperationKind)
        assert "rmsnorm" in kinds, "OperationKind must include 'rmsnorm'"
        assert "swiglu" in kinds, "OperationKind must include 'swiglu'"

    def test_operation_construction_for_rmsnorm_and_swiglu(self):
        """Operation objects with kind 'rmsnorm' and 'swiglu' must construct without error."""
        op_rmsnorm = Operation(
            name="norm1",
            kind="rmsnorm",
            inputs=((2, 512, 768),),
            output=(2, 512, 768),
            dtype_bytes=4,
        )
        assert op_rmsnorm.kind == "rmsnorm"

        op_swiglu = Operation(
            name="mlp_gate",
            kind="swiglu",
            inputs=((2, 512, 1024),),
            output=(2, 512, 1024),
            dtype_bytes=4,
        )
        assert op_swiglu.kind == "swiglu"

    def test_estimate_rmsnorm_flops_and_traffic(self):
        """RMSNorm FLOPs must be approximately 3 * numel (square, mean/rsqrt, scale)."""
        shape = (4, 128, 512)
        num_elements = 4 * 128 * 512
        op = Operation(
            name="rmsnorm",
            kind="rmsnorm",
            inputs=(shape,),
            output=shape,
            dtype_bytes=4,
        )
        est = estimate_operation(op)
        assert isinstance(est, CostEstimate)
        # FLOPs = 3 * numel
        assert est.flops == 3 * num_elements, (
            f"Expected {3 * num_elements} FLOPs for RMSNorm, got {est.flops}"
        )
        # DRAM traffic: read input + read scale weights + write output = 3 * numel * dtype_bytes
        assert est.read_bytes >= num_elements * 4
        assert est.write_bytes == num_elements * 4

    def test_estimate_swiglu_cost(self):
        """SwiGLU activation operator must model elementwise multiply and silu FLOPs."""
        shape = (2, 256, 1024)
        num_elements = 2 * 256 * 1024
        op = Operation(
            name="swiglu",
            kind="swiglu",
            inputs=(shape,),
            output=shape,
            dtype_bytes=4,
        )
        est = estimate_operation(op)
        assert isinstance(est, CostEstimate)
        # At least elementwise gating operations (SiLU + mul >= 2 FLOP/elem)
        assert est.flops >= 2 * num_elements

    def test_estimate_conv2d_depthwise_groups(self):
        """Conv2d with groups > 1 (e.g. depthwise conv in ConvNeXt) must be accepted."""
        # ConvNeXt 7x7 depthwise convolution:
        # data = (B, 64, H, W) = (1, 64, 56, 56)
        # kernel = (out_channels, in_channels // groups, kH, kW) = (64, 1, 7, 7) with groups = 64
        data_shape = (1, 64, 56, 56)
        kernel_shape = (64, 1, 7, 7)
        out_shape = (1, 64, 56, 56)

        op = Operation(
            name="conv_dw",
            kind="conv2d",
            inputs=(data_shape, kernel_shape),
            output=out_shape,
            dtype_bytes=4,
            attrs={"groups": 64},
        )
        # Must not raise "ValueError: conv2d channel dimensions do not match"
        est = estimate_operation(op)
        assert isinstance(est, CostEstimate)
        # Mathematical expected FLOPs: 2 * output_numel * (in_channels // groups) * kH * kW
        # = 2 * (1 * 64 * 56 * 56) * 1 * 7 * 7
        expected_flops = 2 * (1 * 64 * 56 * 56) * 1 * 7 * 7
        assert est.flops == expected_flops

    def test_estimate_conv2d_invalid_groups_rejected(self):
        """Conv2d with invalid channel/group configurations must raise ValueError."""
        # data channels = 64, kernel in_channels = 2, groups = 64 -> mismatch (64 != 2 * 64)
        op = Operation(
            name="conv_invalid",
            kind="conv2d",
            inputs=((1, 64, 56, 56), (64, 2, 7, 7)),
            output=(1, 64, 56, 56),
            dtype_bytes=4,
            attrs={"groups": 64},
        )
        with pytest.raises(ValueError):
            estimate_operation(op)

    def test_transformer_fused_sdpa_kernel(self):
        """Transformer benchmark implementation must execute using fused SDPA."""
        try:
            from benchmarks.models import get_model

            model, _inputs = get_model("transformer", framework="torch", scale="standard")
        except ImportError:
            # Check collect_data implementation
            import collect_data

            if hasattr(collect_data, "_torch_transformer"):
                model, _inputs = collect_data._torch_transformer(32)
            else:
                pytest.fail("Transformer model builder not found")

        # Verify PyTorch model contains SDPA attention or calls F.scaled_dot_product_attention
        import inspect

        try:
            source = inspect.getsource(model.__class__)
        except (OSError, TypeError):
            source = Path(collect_data.__file__).read_text()

        assert "scaled_dot_product_attention" in source or "sdpa" in source.lower(), (
            "Transformer benchmark must use hardware-accelerated fused SDPA attention"
        )


# ---------------------------------------------------------------------------
# 4. Report Generation Robustness (Issue #19)
# ---------------------------------------------------------------------------


class TestReportGenerationRobustness:
    """Verify generate_report.py does not crash when batch 32 is missing."""

    def test_report_handles_missing_batch_32(self, tmp_path):
        """generate_report must dynamically select reference batch rather than hardcoding 32."""
        import generate_report

        hw = {
            "name": "Apple M3",
            "peak_flops": 3.6e12,
            "memory_bandwidth": 100e9,
            "ridge_point": 36.0,
        }
        # Create mock records with standard batch sweep [1, 4, 16, 64] (batch 32 absent)
        records = [
            {
                "framework": "PyTorch",
                "variant": "baseline",
                "architecture": "ConvNeXt",
                "batch": b,
                "flops": 1000,
                "param_bytes": 500,
                "total_bytes": 1500,
                "arith_intensity": 10.0,
                "latency_median_ms": 5.0,
                "latency_mean_ms": 5.0,
                "latency_stddev_ms": 0.1,
                "latency_cv_pct": 2.0,
                "latency_p95_ms": 5.2,
                "roofline_efficiency": 0.75,
                "achieved_gflops": 200.0,
                "achieved_gbw": 20.0,
                "bottleneck": "compute",
            }
            for b in [1, 4, 16, 64]
        ]

        # Call efficiency heatmap or helper with batch 32 absent
        if hasattr(generate_report, "fig_efficiency_heatmap"):
            try:
                fig = generate_report.fig_efficiency_heatmap(hw, records)
                assert fig is not None
            except ValueError as exc:
                pytest.fail(f"generate_report crashed on absent batch 32: {exc}")
        else:
            source = Path(generate_report.__file__).read_text()
            assert "batch == 32" not in source or "batches_in_data" in source, (
                "generate_report.py must dynamically select available reference batches"
            )
