"""The same mathematical model must produce the same canonical counts in every adapter.

Each case builds one architecture in PyTorch, JAX and TensorFlow with identical
shapes and parameterization, then asserts dense FLOPs and unique parameter
counts agree exactly (rtol=0).  Frameworks that are not installed are skipped;
a case needs at least two frameworks to be meaningful.
"""

from __future__ import annotations

import importlib.util

import pytest

from neural_cost import cross_check, flop_breakdown

B, T, D, H, OUT = 4, 8, 32, 4, 10

HAS = {name: importlib.util.find_spec(name) is not None for name in ("torch", "jax", "tensorflow")}


def _require_two() -> None:
    if sum(HAS.values()) < 2:
        pytest.skip("needs at least two of torch/jax/tensorflow")


# --------------------------------------------------------------------------- MLP
def _torch_mlp():
    import torch

    from neural_cost.adapters import TorchAdapter

    model = torch.nn.Sequential(torch.nn.Linear(D, 64), torch.nn.ReLU(), torch.nn.Linear(64, OUT))
    return TorchAdapter().operations(model, (torch.ones(B, D),))


def _jax_mlp():
    import jax.numpy as jnp

    from neural_cost.adapters import JaxAdapter

    def model(x, w1, b1, w2, b2):
        return jnp.maximum(x @ w1 + b1, 0) @ w2 + b2

    args = (jnp.ones((B, D)), jnp.ones((D, 64)), jnp.ones(64), jnp.ones((64, OUT)), jnp.ones(OUT))
    return JaxAdapter().operations(model, args)


def _tf_mlp():
    import tensorflow as tf

    from neural_cost.adapters import TensorFlowAdapter

    inp = tf.keras.Input((D,))
    out = tf.keras.layers.Dense(OUT)(tf.keras.layers.Dense(64, activation="relu")(inp))
    return TensorFlowAdapter().operations(tf.keras.Model(inp, out), (tf.ones((B, D)),))


# --------------------------------------------------------------------------- CNN
def _torch_cnn():
    import torch

    from neural_cost.adapters import TorchAdapter

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, OUT),
    )
    return TorchAdapter().operations(model, (torch.ones(B, 3, 16, 16),))


def _jax_cnn():
    import jax.numpy as jnp
    from jax import lax

    from neural_cost.adapters import JaxAdapter

    def model(x, k1, b1, k2, b2, w, b):
        # NHWC / HWIO like Flax, exercising layout-aware conv accounting.
        dn = ("NHWC", "HWIO", "NHWC")
        y = lax.conv_general_dilated(x, k1, (1, 1), "SAME", dimension_numbers=dn) + b1
        y = jnp.maximum(y, 0)
        y = lax.conv_general_dilated(y, k2, (2, 2), "SAME", dimension_numbers=dn) + b2
        y = jnp.maximum(y, 0).mean(axis=(1, 2))
        return y @ w + b

    args = (
        jnp.ones((B, 16, 16, 3)),
        jnp.ones((3, 3, 3, 16)),
        jnp.ones(16),
        jnp.ones((3, 3, 16, 32)),
        jnp.ones(32),
        jnp.ones((32, OUT)),
        jnp.ones(OUT),
    )
    return JaxAdapter().operations(model, args)


def _tf_cnn():
    import tensorflow as tf

    from neural_cost.adapters import TensorFlowAdapter

    inp = tf.keras.Input((16, 16, 3))
    y = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    y = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(y)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    out = tf.keras.layers.Dense(OUT)(y)
    return TensorFlowAdapter().operations(tf.keras.Model(inp, out), (tf.ones((B, 16, 16, 3)),))


# -------------------------------------------------------------------------- LSTM
# Bias-free: PyTorch LSTMs carry two bias vectors (b_ih + b_hh) while Keras
# carries one, a real parameterization difference, not a counting bug.
def _torch_lstm():
    import torch

    from neural_cost.adapters import TorchAdapter

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(D, D, num_layers=2, batch_first=True, bias=False)
            self.head = torch.nn.Linear(D, OUT, bias=False)

        def forward(self, x):
            return self.head(self.lstm(x)[0][:, -1])

    return TorchAdapter().operations(Model(), (torch.ones(B, T, D),))


def _jax_lstm():
    import jax
    import jax.numpy as jnp
    from jax import lax

    from neural_cost.adapters import JaxAdapter

    def layer(xs, wih, whh):
        def step(carry, xt):
            h, c = carry
            i, f, g, o = jnp.split(xt @ wih + h @ whh, 4, axis=-1)
            c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            return (h, c), h

        zeros = jnp.zeros((xs.shape[1], D))
        return lax.scan(step, (zeros, zeros), xs)[1]

    def model(x, wih0, whh0, wih1, whh1, w):
        xs = jnp.swapaxes(x, 0, 1)
        ys = layer(layer(xs, wih0, whh0), wih1, whh1)
        return ys[-1] @ w

    args = (
        (jnp.ones((B, T, D)),)
        + tuple(jnp.ones((D, 4 * D)) for _ in range(4))
        + (jnp.ones((D, OUT)),)
    )
    return JaxAdapter().operations(model, args)


def _tf_lstm():
    import tensorflow as tf

    from neural_cost.adapters import TensorFlowAdapter

    inp = tf.keras.Input((T, D))
    y = tf.keras.layers.LSTM(D, return_sequences=True, use_bias=False)(inp)
    y = tf.keras.layers.LSTM(D, use_bias=False)(y)
    out = tf.keras.layers.Dense(OUT, use_bias=False)(y)
    return TensorFlowAdapter().operations(tf.keras.Model(inp, out), (tf.ones((B, T, D)),))


# --------------------------------------------------------------------- Attention
def _torch_attention():
    import torch

    from neural_cost.adapters import TorchAdapter

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(D, H, batch_first=True)

        def forward(self, x):
            return self.attn(x, x, x, need_weights=False)[0]

    return TorchAdapter().operations(Model(), (torch.ones(B, T, D),))


def _jax_attention():
    import jax
    import jax.numpy as jnp

    from neural_cost.adapters import JaxAdapter

    def model(x, wq, bq, wk, bk, wv, bv, wo, bo):
        def heads(t):
            return t.reshape(B, T, H, D // H)

        q, k, v = heads(x @ wq + bq), heads(x @ wk + bk), heads(x @ wv + bv)
        return jax.nn.dot_product_attention(q, k, v).reshape(B, T, D) @ wo + bo

    weights = []
    for _ in range(4):
        weights += [jnp.ones((D, D)), jnp.ones(D)]
    return JaxAdapter().operations(model, (jnp.ones((B, T, D)), *weights))


def _tf_attention():
    import tensorflow as tf

    from neural_cost.adapters import TensorFlowAdapter

    inp = tf.keras.Input((T, D))
    out = tf.keras.layers.MultiHeadAttention(H, D // H)(inp, inp)
    return TensorFlowAdapter().operations(tf.keras.Model(inp, out), (tf.ones((B, T, D)),))


CASES = {
    "mlp": {"torch": _torch_mlp, "jax": _jax_mlp, "tensorflow": _tf_mlp},
    "cnn": {"torch": _torch_cnn, "jax": _jax_cnn, "tensorflow": _tf_cnn},
    "lstm": {"torch": _torch_lstm, "jax": _jax_lstm, "tensorflow": _tf_lstm},
    "attention": {"torch": _torch_attention, "jax": _jax_attention, "tensorflow": _tf_attention},
}

EXPECTED_DENSE = {
    "mlp": 2 * B * (D * 64 + 64 * OUT),
    "cnn": 2 * B * (16 * 16 * 16 * 3 * 9 + 8 * 8 * 32 * 16 * 9 + 32 * OUT),
    "lstm": 2 * B * T * 2 * (2 * D * 4 * D) + 2 * B * D * OUT,
    "attention": 4 * 2 * B * T * D * D + 4 * B * T * T * D,
}
EXPECTED_PARAMS = {
    "mlp": D * 64 + 64 + 64 * OUT + OUT,
    "cnn": 3 * 9 * 16 + 16 + 16 * 9 * 32 + 32 + 32 * OUT + OUT,
    "lstm": 2 * (2 * D * 4 * D) + D * OUT,
    "attention": 4 * (D * D + D),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_frameworks_agree_on_canonical_counts(case: str) -> None:
    _require_two()
    workloads = {fw: build() for fw, build in CASES[case].items() if HAS[fw]}
    report = cross_check(workloads, rtol=0.0)
    assert report.consistent, report.render()
    for fw, ops in workloads.items():
        breakdown = flop_breakdown(ops)
        assert breakdown.dense_flops == EXPECTED_DENSE[case], (fw, report.render())
        assert breakdown.parameter_count == EXPECTED_PARAMS[case], (fw, report.render())


def test_cross_check_flags_outlier() -> None:
    from neural_cost import Operation

    good = [Operation("fc", "linear", ((4, 8), (8, 2)), (4, 2))]
    bad = [Operation("fc", "linear", ((4, 8), (8, 4)), (4, 4))]
    report = cross_check({"a": good, "b": good, "c": bad})
    assert not report.consistent
    assert report.outliers == ("c",)
    assert "mismatch" in report.render()


def test_fx_adapter_matches_hook_adapter():
    torch = pytest.importorskip("torch")
    from neural_cost.adapters import TorchAdapter, TorchFxAdapter

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(D)
            self.attn = torch.nn.MultiheadAttention(D, H, batch_first=True)
            self.lstm = torch.nn.LSTM(D, D, num_layers=2, batch_first=True)
            self.head = torch.nn.Linear(D, OUT)

        def forward(self, x):
            y = self.norm(x)
            y = y + self.attn(y, y, y)[0]
            return self.head(self.lstm(y)[0][:, -1])

    model, inputs = Model().eval(), (torch.ones(B, T, D),)
    report = cross_check(
        {
            "hooks": TorchAdapter().operations(model, inputs),
            "fx": TorchFxAdapter().operations(model, inputs),
        },
        rtol=0.0,
    )
    assert report.consistent, report.render()
