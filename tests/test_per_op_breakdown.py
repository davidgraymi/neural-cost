"""Per-op breakdown in render(), including JAX normalization layers (issue #56)."""

import re

import pytest

from neural_cost import (
    HardwareSpec,
    Measurement,
    Operation,
    analyze_gap,
    estimate_operation,
    estimate_operations,
)
from neural_cost.analysis import analyze_layers_gap, group_layers_by_scope

HW = HardwareSpec("test", 1e12, 1e11)


def ops():
    return [
        Operation("net/conv/mm", "matmul", ((64, 256), (256, 256)), (64, 256)),
        Operation("net/norm/sub", "elementwise", ((64, 256), (64, 1)), (64, 256)),
        Operation("net/norm/mul", "elementwise", ((64, 256), (64, 256)), (64, 256)),
        Operation("loose", "elementwise", ((8,),), (8,)),
    ]


def gap():
    operations = ops()
    return analyze_gap(
        estimate_operations(operations), Measurement(0.01, (0.01,)), HW, operations=operations
    )


def table_names(text):
    return re.findall(r"^\s+\d+\s+(\S+)", text, flags=re.MULTILINE)


def test_render_top_n_sorted_by_share_bytes_and_flops():
    result = gap()
    by_share = result.render(top=2)
    assert "top 2 of 4 operations by share of lower bound:" in by_share
    assert "FLOP/B" in by_share and "regime" in by_share
    assert len(table_names(by_share)) == 2
    assert table_names(result.render(sort="flops"))[0] == "net/conv/mm"
    by_bytes = table_names(result.render(top=10, sort="bytes"))
    assert by_bytes[-1] == "loose" and len(by_bytes) == 4
    assert "operations by" not in result.render(top=0)
    with pytest.raises(ValueError):
        result.render(sort="latency")


def test_group_by_scope_sums_members():
    layers = analyze_layers_gap(ops(), HW)
    grouped = {layer.name: layer for layer in group_layers_by_scope(layers)}
    assert set(grouped) == {"net/conv", "net/norm", "loose"}
    norm = grouped["net/norm"]
    members = [layer for layer in layers if layer.name.startswith("net/norm/")]
    assert norm.total_bytes == sum(m.total_bytes for m in members)
    assert norm.time_share_ratio == pytest.approx(sum(m.time_share_ratio for m in members))
    assert norm.kind == "elementwise"
    text = gap().render(group_by_scope=True)
    assert "of 3 scopes" in text and "net/norm" in text


def test_jax_group_norm_is_attributed_by_scope():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from neural_cost.adapters import JaxAdapter

    def group_norm(x, scale, groups=4):
        n, h, w, c = x.shape
        g = x.reshape(n, h, w, groups, c // groups)
        mean = g.mean(axis=(1, 2, 4), keepdims=True)
        var = ((g - mean) ** 2).mean(axis=(1, 2, 4), keepdims=True)
        return ((g - mean) * jax.lax.rsqrt(var + 1e-5)).reshape(n, h, w, c) * scale

    def model(x, kernel, scale):
        with jax.named_scope("block_0"):
            with jax.named_scope("Conv_0"):
                y = jax.lax.conv_general_dilated(
                    x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
                )
            with jax.named_scope("GroupNorm_0"):
                return jax.nn.relu(group_norm(y, scale))

    args = (jnp.ones((2, 8, 8, 16)), jnp.ones((3, 3, 16, 16)), jnp.ones(16))
    adapter = JaxAdapter()
    operations = adapter.operations(jax.jit(model), args)
    norm_ops = [op for op in operations if "GroupNorm_0" in op.name]
    assert any("reduce_sum" in op.name for op in norm_ops)
    assert any(op.kind == "conv2d" and "Conv_0" in op.name for op in operations)
    assert adapter.coverage.unmodeled == 0
    # A reduction does one FLOP per input element, not per output element.
    reduce_op = next(op for op in norm_ops if "reduce_sum" in op.name)
    assert estimate_operation(reduce_op).flops == 2 * 8 * 8 * 16

    result = analyze_gap(
        estimate_operations(operations),
        Measurement(0.01, (0.01,)),
        HW,
        operations=operations,
        coverage=adapter.coverage,
    )
    names = table_names(result.render(group_by_scope=True))
    assert any(name.endswith("block_0/GroupNorm_0") for name in names)
    assert any(name.endswith("block_0/Conv_0") for name in names)
