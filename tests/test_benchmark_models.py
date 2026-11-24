"""Tests for realistic model topologies, scale tiers, and report robustness (#19)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from benchmarks.models import (
    SCALE_TIERS,
    get_available_models,
    get_batch_sizes,
    get_model,
)
from neural_cost.adapters import TorchAdapter
from neural_cost.estimate import CostEstimate, estimate_conv2d, estimate_operation
from neural_cost.operations import Operation


def test_scale_tier_registry():
    """Verify scale tier validation and batch size sweeps."""
    assert set(SCALE_TIERS) == {"micro", "standard", "production"}
    assert get_batch_sizes("standard") == [1, 4, 16, 64]
    assert get_batch_sizes("micro") == [1, 8, 32, 128]
    assert get_batch_sizes("production") == [1, 2, 4, 8, 16]

    with pytest.raises(ValueError):
        get_batch_sizes("invalid_tier")

    modern = get_available_models(include_legacy=False)
    assert "ConvNeXt" in modern
    assert "ViT" in modern
    assert "RNN" not in modern
    assert "LSTM" not in modern

    all_archs = get_available_models(include_legacy=True)
    assert "RNN" in all_archs
    assert "LSTM" in all_archs


def test_invalid_scale_rejected():
    with pytest.raises((ValueError, KeyError)):
        get_model("convnext", framework="torch", scale="non_existent")


def test_torch_convnext_standard_scale():
    pytest.importorskip("torch")
    model, inputs = get_model("convnext", framework="torch", scale="standard", batch=1)
    x = inputs[0]
    assert x.shape == (1, 3, 224, 224)

    out = model(x)
    assert out.shape == (1, 10)

    adapter = TorchAdapter()
    profile = adapter.profile(model, inputs)
    assert profile.cost.flops >= 4.0e9, f"ConvNeXt FLOPs: {profile.cost.flops}"


def test_torch_vit_standard_scale():
    pytest.importorskip("torch")
    model, inputs = get_model("vit", framework="torch", scale="standard", batch=1)
    x = inputs[0]
    assert x.shape == (1, 3, 224, 224)

    out = model(x)
    assert out.shape == (1, 10)

    adapter = TorchAdapter()
    profile = adapter.profile(model, inputs)
    assert profile.cost.flops >= 4.0e9, f"ViT FLOPs: {profile.cost.flops}"


def test_torch_deep_dnn_standard_scale():
    pytest.importorskip("torch")
    model, inputs = get_model("deep_dnn", framework="torch", scale="standard", batch=2)
    x = inputs[0]
    assert x.shape == (2, 1024)

    out = model(x)
    assert out.shape == (2, 10)


def test_torch_transformer_standard_scale():
    pytest.importorskip("torch")
    model, inputs = get_model("transformer", framework="torch", scale="standard", batch=1)
    x = inputs[0]
    assert x.shape == (1, 512, 768)

    out = model(x)
    assert out.shape == (1, 10)


def test_estimate_conv2d_depthwise():
    op = Operation(
        name="dw_conv",
        kind="conv2d",
        inputs=((1, 64, 56, 56), (64, 1, 7, 7)),
        output=(1, 64, 56, 56),
        dtype_bytes=4,
        attrs={"groups": 64},
    )
    est = estimate_operation(op)
    assert isinstance(est, CostEstimate)
    assert est.flops == 2 * (1 * 64 * 56 * 56) * 1 * 7 * 7

    # Directly test estimate_conv2d
    flops = estimate_conv2d(
        data=(1, 64, 56, 56),
        kernel=(64, 1, 7, 7),
        output=(1, 64, 56, 56),
        groups=64,
    )
    assert flops == est.flops


def test_estimate_conv2d_channel_mismatch_raises():
    op = Operation(
        name="bad_dw_conv",
        kind="conv2d",
        inputs=((1, 64, 56, 56), (64, 2, 7, 7)),
        output=(1, 64, 56, 56),
        dtype_bytes=4,
        attrs={"groups": 64},
    )
    with pytest.raises(ValueError):
        estimate_operation(op)


def test_generate_report_dynamic_batch(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import generate_report

    monkeypatch.setattr(generate_report, "FIG_DIR", tmp_path)

    hw = {
        "name": "Apple M3",
        "peak_flops": 3.6e12,
        "memory_bandwidth": 100e9,
        "ridge_point": 36.0,
    }
    records = [
        {
            "framework": "PyTorch",
            "variant": "compiled",
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

    ref_b = generate_report.get_reference_batch(records)
    assert ref_b in [1, 4, 16, 64]

    p1 = generate_report.fig_efficiency_heatmap(hw, records, batch=ref_b)
    assert p1.exists()

    p2 = generate_report.fig_cv_heatmap(hw, records, batch=ref_b)
    assert p2.exists()


# ---------------------------------------------------------------------------
# Cross-framework agreement (#86): every framework builds the same network.
# ---------------------------------------------------------------------------

# JAX counts LayerNorm and GELU as their primitive ops, which costs a few
# percent more than the fused counts on micro models; JAX also counts the ViT
# position embedding as a parameter.  Anything larger means the models differ.
FLOP_TOLERANCE = 0.05
PARAM_TOLERANCE = 0.03


def _profile(arch: str, framework: str):
    from neural_cost import profile_model
    from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchFxAdapter

    adapter = {
        "torch": TorchFxAdapter,
        "jax": JaxAdapter,
        "tensorflow": TensorFlowAdapter,
    }[framework]()
    model, inputs = get_model(arch, framework=framework, scale="micro", batch=2)
    prof = profile_model(model, inputs, adapter)
    return prof.cost.flops, prof.memory.parameter_bytes


@pytest.mark.parametrize("arch", ["FF DNN", "ConvNeXt", "ViT", "Transformer"])
@pytest.mark.parametrize("framework", ["jax", "tensorflow"])
def test_flops_and_params_match_pytorch(arch, framework):
    pytest.importorskip("torch")
    pytest.importorskip(framework)
    ref_flops, ref_params = _profile(arch, "torch")
    flops, params = _profile(arch, framework)
    assert flops == pytest.approx(ref_flops, rel=FLOP_TOLERANCE), (arch, framework)
    assert params == pytest.approx(ref_params, rel=PARAM_TOLERANCE), (arch, framework)


@pytest.mark.parametrize(
    ("arch", "shape"),
    [("ConvNeXt", (3, 32, 32, 3)), ("ViT", (3, 32, 32, 3))],
)
def test_tensorflow_vision_models_run(arch, shape):
    pytest.importorskip("tensorflow")
    model, inputs = get_model(arch, framework="tensorflow", scale="micro", batch=3)
    assert tuple(inputs[0].shape) == shape
    assert tuple(model(*inputs, training=False).shape) == (3, 10)


def test_unknown_tensorflow_architecture_raises():
    pytest.importorskip("tensorflow")
    with pytest.raises(KeyError):
        get_model("resnet", framework="tensorflow", scale="micro")
