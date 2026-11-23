"""1-D, 3-D, transposed, depthwise and separable convolutions and N-d pooling."""

from __future__ import annotations

import pytest

from neural_cost import Operation, estimate_conv, estimate_operation, flop_breakdown


def test_conv1d_and_conv3d_flops():
    conv1d = Operation("c1", "conv", ((2, 4, 16), (8, 4, 3)), (2, 8, 16))
    assert estimate_conv(conv1d) == 2 * (2 * 8 * 16) * 4 * 3
    conv3d = Operation(
        "c3", "conv", ((1, 4, 8, 8, 8), (6, 2, 3, 3, 3)), (1, 6, 8, 8, 8), 4, {"groups": 2}
    )
    assert estimate_operation(conv3d).flops == 2 * (6 * 512) * 2 * 27


def test_conv_kind_matches_conv2d_for_two_spatial_dims():
    shapes = ((2, 3, 8, 8), (16, 3, 3, 3))
    plain = estimate_operation(Operation("a", "conv2d", shapes, (2, 16, 6, 6)))
    general = estimate_operation(Operation("b", "conv", shapes, (2, 16, 6, 6)))
    assert plain == general


def test_transposed_conv_scatters_each_input():
    # ConvTranspose2d(4 -> 6, k=3, stride=2, groups=2) on a 5x5 map.
    op = Operation(
        "t",
        "conv",
        ((1, 4, 5, 5), (6, 2, 3, 3)),
        (1, 6, 11, 11),
        4,
        {"transposed": True, "groups": 2},
    )
    assert estimate_conv(op) == 2 * (4 * 25) * (6 // 2) * 9
    # The regular-conv formula would count the zero-inserted positions.
    assert estimate_conv(op) < 2 * (6 * 121) * 2 * 9


@pytest.mark.parametrize(
    "inputs, output, attrs",
    [
        (((2, 4, 16), (8, 3, 3)), (2, 8, 16), {}),  # channel mismatch
        (((2, 4, 16), (8, 4, 3)), (2, 7, 16), {}),  # output channels mismatch
        (((2, 4), (8, 4)), (2, 8), {}),  # no spatial dimensions
        (((1, 4, 2, 2, 2, 2), (8, 4, 1, 1, 1, 1)), (1, 8, 2, 2, 2, 2), {}),  # 4-D
        (((2, 4, 16), (6, 1, 3)), (2, 6, 16), {"groups": 4}),  # C_out % groups
        (((2, 4, 16), (8, 4, 3)), (2, 8, 16), {"groups": 0}),
    ],
)
def test_conv_rejects_bad_shapes(inputs, output, attrs):
    with pytest.raises(ValueError):
        estimate_conv(Operation("bad", "conv", inputs, output, 4, attrs))


def test_conv_is_dense_and_counts_parameters():
    op = Operation("c", "conv", ((2, 4, 16), (8, 4, 3)), (2, 8, 16))
    breakdown = flop_breakdown([op])
    assert breakdown.dense_flops == estimate_conv(op)
    assert breakdown.parameter_count == 8 * 4 * 3


def test_pooling_accepts_one_to_three_dims():
    pool1d = Operation("p1", "pooling", ((2, 8, 16),), (2, 8, 8), 4, {"kernel_size": (2,)})
    pool3d = Operation(
        "p3", "pooling", ((1, 4, 8, 8, 8),), (1, 4, 4, 4, 4), 4, {"kernel_size": (2, 2, 2)}
    )
    assert estimate_operation(pool1d).flops == 2 * 8 * 8 * 2
    assert estimate_operation(pool3d).flops == 4 * 64 * 8
    with pytest.raises(ValueError):
        estimate_operation(Operation("p", "pooling", ((1, 1, 2),), (1, 1, 1), 4, {}))


@pytest.fixture(params=["eager", "fx"])
def torch_adapter(request):
    pytest.importorskip("torch")
    from neural_cost.adapters import TorchAdapter, TorchFxAdapter

    return TorchAdapter(use_profiler=False) if request.param == "eager" else TorchFxAdapter()


def test_torch_conv_family_is_captured(torch_adapter):
    torch = pytest.importorskip("torch")
    nn = torch.nn
    model = nn.Sequential(
        nn.Conv3d(2, 4, 3, padding=1),
        nn.BatchNorm3d(4),
        nn.MaxPool3d(2),
        nn.ConvTranspose3d(4, 6, 2, stride=2, groups=2),
        nn.AdaptiveAvgPool3d(1),
    )
    ops = list(torch_adapter.operations(model, (torch.randn(1, 2, 8, 8, 8),)))
    assert [op.kind for op in ops] == ["conv", "batchnorm", "pooling", "conv", "pooling"]
    conv3d, _, pool, deconv, gap = ops
    assert estimate_conv(conv3d) == 2 * (4 * 512) * 2 * 27
    # ConvTranspose3d weight (4, 3, 2, 2, 2) becomes portable (6, 2, 2, 2, 2).
    assert deconv.inputs[1] == (6, 2, 2, 2, 2)
    assert deconv.attrs["transposed"] is True and deconv.attrs["groups"] == 2
    assert estimate_conv(deconv) == 2 * (4 * 64) * 3 * 8
    assert pool.attrs["kernel_size"] == (2, 2, 2)
    assert gap.attrs["kernel_size"] == (8, 8, 8)
    if torch_adapter.coverage is not None:
        assert torch_adapter.coverage.unmodeled == 0


def test_torch_conv1d_and_unbatched(torch_adapter):
    torch = pytest.importorskip("torch")
    nn = torch.nn
    model = nn.Sequential(
        nn.Conv1d(4, 8, 3, padding=1), nn.AvgPool1d(2), nn.ConvTranspose1d(8, 4, 2)
    )
    ops = list(torch_adapter.operations(model, (torch.randn(4, 16),)))  # unbatched (C, L)
    conv, pool, deconv = ops
    assert conv.inputs[0] == (1, 4, 16) and conv.output == (1, 8, 16)
    assert estimate_conv(conv) == 2 * (8 * 16) * 4 * 3
    assert pool.attrs["kernel_size"] == (2,)
    assert estimate_conv(deconv) == 2 * (8 * 8) * 4 * 2


def test_torch_plain_conv2d_keeps_conv2d_kind(torch_adapter):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ConvTranspose2d(4, 2, 2))
    kinds = [op.kind for op in torch_adapter.operations(model, (torch.randn(1, 3, 8, 8),))]
    assert kinds == ["conv2d", "conv"]


def test_torch_fx_functional_convs():
    torch = pytest.importorskip("torch")
    from neural_cost.adapters import TorchFxAdapter

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.randn(8, 4, 3))
            self.wt = torch.nn.Parameter(torch.randn(8, 2, 2))  # (C_in, C_out/groups, k)

        def forward(self, x):
            y = torch.nn.functional.conv1d(x, self.w1, padding=1)
            return torch.nn.functional.conv_transpose1d(y, self.wt, stride=2, groups=2)

    ops = [
        op
        for op in TorchFxAdapter().operations(Block(), (torch.randn(2, 4, 16),))
        if op.kind == "conv"
    ]
    assert len(ops) == 2
    assert ops[1].attrs == {"groups": 2, "transposed": True}
    assert ops[1].inputs[1] == (4, 4, 2)
    assert estimate_conv(ops[1]) == 2 * (2 * 8 * 16) * 2 * 2


def test_torch_replay_runs_nd_convs():
    pytest.importorskip("torch")
    from neural_cost.adapters._torch_replay import replay_operations

    ops = [
        Operation("c1", "conv", ((2, 4, 16), (8, 4, 3)), (2, 8, 16)),
        Operation(
            "c3", "conv", ((1, 2, 8, 8, 8), (4, 1, 3, 3, 3)), (1, 4, 8, 8, 8), 4, {"groups": 2}
        ),
        Operation(
            "t2", "conv", ((1, 6, 6, 6), (4, 6, 3, 3)), (1, 4, 13, 13), 4, {"transposed": True}
        ),
        Operation(
            "t3",
            "conv",
            ((1, 4, 4, 4, 4), (6, 2, 2, 2, 2)),
            (1, 6, 8, 8, 8),
            4,
            {"transposed": True, "groups": 2},
        ),
        Operation("p1", "pooling", ((2, 8, 16),), (2, 8, 8), 4, {"kernel_size": (2,)}),
        Operation(
            "p3", "pooling", ((1, 4, 8, 8, 8),), (1, 4, 4, 4, 4), 4, {"kernel_size": (2, 2, 2)}
        ),
    ]
    times = replay_operations(ops, target_seconds=1e-4)
    assert all(seconds is not None and seconds > 0 for seconds in times)


def test_jax_conv1d_and_conv_transpose():
    jax = pytest.importorskip("jax")
    from neural_cost.adapters import JaxAdapter

    jnp = jax.numpy
    adapter = JaxAdapter()

    def conv1d(x, w):
        return jax.lax.conv_general_dilated(
            x, w, (1,), "SAME", dimension_numbers=("NWC", "WIO", "NWC")
        )

    (op,) = adapter.operations(conv1d, (jnp.ones((2, 16, 4)), jnp.ones((3, 4, 8))))
    assert op.kind == "conv" and op.inputs == ((2, 4, 16), (8, 4, 3))
    assert adapter.coverage.unmodeled == 0

    def deconv3d(x, w):
        return jax.lax.conv_transpose(
            x, w, (2, 2, 2), "VALID", dimension_numbers=("NCDHW", "OIDHW", "NCDHW")
        )

    (t,) = JaxAdapter().operations(deconv3d, (jnp.ones((1, 4, 4, 4, 4)), jnp.ones((6, 4, 2, 2, 2))))
    assert t.attrs["transposed"] is True and t.output == (1, 6, 8, 8, 8)
    # Zero-inserted positions of the dilated input are not counted.
    assert estimate_conv(t) == 2 * (4 * 64) * 6 * 8


def test_keras_conv_family():
    tf = pytest.importorskip("tensorflow")
    from neural_cost.adapters import TensorFlowAdapter

    layers = tf.keras.layers
    x_in = tf.keras.Input((16, 4))
    x = layers.Conv1D(8, 3, padding="same")(x_in)
    x = layers.MaxPooling1D(2)(x)
    x = layers.SeparableConv1D(6, 3, depth_multiplier=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    model = tf.keras.Model(x_in, x)
    ops = TensorFlowAdapter().operations(model, (tf.ones((2, 16, 4)),))
    assert [op.kind for op in ops] == ["conv", "pooling", "conv", "conv", "pooling"]
    conv, pool, depthwise, pointwise, gap = ops
    assert conv.inputs == ((2, 4, 16), (8, 4, 3))
    assert pool.attrs["kernel_size"] == (2,)
    assert depthwise.attrs["groups"] == 8 and depthwise.output == (2, 16, 6)
    assert pointwise.inputs[1] == (6, 16, 1)
    assert gap.attrs["kernel_size"] == (6,)

    x_in = tf.keras.Input((8, 8, 8, 2))
    y = layers.Conv3D(4, 3, padding="same", groups=2)(x_in)
    y = layers.Conv3DTranspose(6, 2, strides=2)(y)
    model = tf.keras.Model(x_in, y)
    conv3d, deconv = TensorFlowAdapter().operations(model, (tf.ones((1, 8, 8, 8, 2)),))
    assert conv3d.attrs["groups"] == 2
    assert estimate_conv(conv3d) == 2 * (4 * 512) * 1 * 27
    assert deconv.attrs["transposed"] is True
    assert estimate_conv(deconv) == 2 * (4 * 512) * 6 * 8

    x_in = tf.keras.Input((8, 8, 3))
    z = layers.DepthwiseConv2D(3, depth_multiplier=2)(x_in)
    (dw,) = TensorFlowAdapter().operations(tf.keras.Model(x_in, z), (tf.ones((1, 8, 8, 3)),))
    assert dw.kind == "conv2d" and dw.inputs[1] == (6, 1, 3, 3) and dw.attrs["groups"] == 3
