"""TensorFlow (Keras) benchmark model architectures across scale tiers.

Each builder mirrors the PyTorch model of the same name in torch_models.py layer
for layer, reading its dimensions from configs.py.  Images are NHWC, the layout
TensorFlow convolutions support on CPU; the FLOPs are the same as NCHW.

Every operation that does arithmetic is a Keras layer, because
``TensorFlowAdapter`` counts layers.  Residual additions and the ViT position
embedding are the exceptions: they cost one add per element, which is noise next
to the convolutions and matmuls.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from .configs import (
    CONVNEXT,
    DEEP_DNN,
    IN_CHANNELS,
    NUM_CLASSES,
    TRANSFORMER,
    VIT,
    VIT_MLP_RATIO,
    scale_config,
)

layers = tf.keras.layers


class PositionEmbedding(layers.Layer):
    """Learned position embedding added to a (batch, tokens, dim) sequence."""

    def build(self, input_shape: Any) -> None:
        self.pos = self.add_weight(
            name="pos", shape=(1, int(input_shape[1]), int(input_shape[2])), initializer="zeros"
        )

    def call(self, x: Any) -> Any:
        return x + self.pos


def _gelu() -> Any:
    # nn.GELU() in PyTorch is the exact (erf) form.
    return layers.Activation(lambda v: tf.nn.gelu(v, approximate=False), name=None)


def _layer_norm(eps: float = 1e-5) -> Any:
    return layers.LayerNormalization(epsilon=eps)


def _encoder_block(x: Any, dim: int, num_heads: int, hidden: int) -> Any:
    """Pre-norm encoder block, as ViTBlock / TransformerBlock in PyTorch."""
    h = _layer_norm()(x)
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)(h, h)
    x = layers.Add()([x, h])
    h = _layer_norm()(x)
    h = layers.Dense(hidden)(h)
    h = _gelu()(h)
    h = layers.Dense(dim)(h)
    return layers.Add()([x, h])


def _finish(inputs: Any, outputs: Any, x: Any, name: str) -> tuple[Any, tuple[Any]]:
    model = tf.keras.Model(inputs, outputs, name=name)
    model(x, training=False)
    return model, (x,)


def _random(shape: tuple[int, ...]) -> Any:
    return tf.random.stateless_normal(shape, seed=(0, 0))


# ---------------------------------------------------------------------------
# Modern architectures
# ---------------------------------------------------------------------------


def build_tf_convnext(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any]]:
    cfg = scale_config(CONVNEXT, scale)
    depths, dims, stem, size = cfg["depths"], cfg["dims"], cfg["stem"], cfg["img_size"]

    inputs = tf.keras.Input(shape=(size, size, IN_CHANNELS))
    h = layers.Conv2D(dims[0], stem, strides=stem)(inputs)
    h = _layer_norm(1e-6)(h)
    for i, (depth, dim) in enumerate(zip(depths, dims, strict=True)):
        for _ in range(depth):
            res = h
            h = layers.DepthwiseConv2D(7, padding="same")(h)
            h = _layer_norm(1e-6)(h)
            h = layers.Conv2D(4 * dim, 1)(h)
            h = _gelu()(h)
            h = layers.Conv2D(dim, 1)(h)
            h = layers.Add()([res, h])
        if i < len(dims) - 1:
            h = _layer_norm(1e-6)(h)
            h = layers.Conv2D(dims[i + 1], 2, strides=2)(h)
    h = layers.GlobalAveragePooling2D()(h)
    h = _layer_norm()(h)
    outputs = layers.Dense(NUM_CLASSES)(h)
    return _finish(inputs, outputs, _random((batch, size, size, IN_CHANNELS)), "convnext")


def build_tf_vit(scale: str = "standard", batch: int = 1, **kwargs: Any) -> tuple[Any, tuple[Any]]:
    cfg = scale_config(VIT, scale)
    size, patch, dim, heads = cfg["img_size"], cfg["patch_size"], cfg["embed_dim"], cfg["num_heads"]
    num_patches = (size // patch) ** 2

    inputs = tf.keras.Input(shape=(size, size, IN_CHANNELS))
    h = layers.Conv2D(dim, patch, strides=patch)(inputs)
    h = layers.Reshape((num_patches, dim))(h)
    h = PositionEmbedding()(h)
    for _ in range(cfg["depth"]):
        h = _encoder_block(h, dim, heads, dim * VIT_MLP_RATIO)
    h = _layer_norm()(h)
    h = layers.GlobalAveragePooling1D()(h)
    outputs = layers.Dense(NUM_CLASSES)(h)
    return _finish(inputs, outputs, _random((batch, size, size, IN_CHANNELS)), "vit")


def build_tf_deep_dnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any]]:
    cfg = scale_config(DEEP_DNN, scale)
    d_in, hidden, d_out = cfg["in_features"], cfg["hidden_features"], cfg["out_features"]

    inputs = tf.keras.Input(shape=(d_in,))
    h = inputs
    for width in (hidden, hidden, d_out):
        h = layers.Dense(width)(h)
        h = _gelu()(h)
        h = _layer_norm()(h)
    outputs = layers.Dense(NUM_CLASSES)(h)
    return _finish(inputs, outputs, _random((batch, d_in)), "deep_dnn")


def build_tf_transformer(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any]]:
    cfg = scale_config(TRANSFORMER, scale)
    seq, dim = cfg["seq_len"], cfg["embed_dim"]

    inputs = tf.keras.Input(shape=(seq, dim))
    h = inputs
    for _ in range(cfg["num_layers"]):
        h = _encoder_block(h, dim, cfg["num_heads"], cfg["ffn_dim"])
    h = _layer_norm()(h)
    h = layers.GlobalAveragePooling1D()(h)
    outputs = layers.Dense(NUM_CLASSES)(h)
    return _finish(inputs, outputs, _random((batch, seq, dim)), "transformer")


# ---------------------------------------------------------------------------
# Legacy architectures (mirror LegacyCNN / LegacyRNN / LegacyLSTM)
# ---------------------------------------------------------------------------


def build_tf_cnn(scale: str = "standard", batch: int = 1, **kwargs: Any) -> tuple[Any, tuple[Any]]:
    size = 32 if scale == "micro" else 224
    inputs = tf.keras.Input(shape=(size, size, IN_CHANNELS))
    h = layers.Conv2D(64, 3, padding="same")(inputs)
    h = layers.ReLU()(h)
    h = layers.Conv2D(128, 3, strides=2, padding="same")(h)
    h = layers.ReLU()(h)
    h = layers.GlobalAveragePooling2D()(h)
    outputs = layers.Dense(NUM_CLASSES)(h)
    return _finish(inputs, outputs, _random((batch, size, size, IN_CHANNELS)), "cnn")


def _recurrent(cell: Any, scale: str, batch: int, name: str) -> tuple[Any, tuple[Any]]:
    seq = 32 if scale == "micro" else 128
    dim = 128 if scale == "micro" else 256
    inputs = tf.keras.Input(shape=(seq, dim))
    h = cell(dim, return_sequences=True)(inputs)
    h = layers.GlobalAveragePooling1D()(h)
    outputs = layers.Dense(NUM_CLASSES)(h)
    return _finish(inputs, outputs, _random((batch, seq, dim)), name)


def build_tf_rnn(scale: str = "standard", batch: int = 1, **kwargs: Any) -> tuple[Any, tuple[Any]]:
    return _recurrent(layers.SimpleRNN, scale, batch, "rnn")


def build_tf_lstm(scale: str = "standard", batch: int = 1, **kwargs: Any) -> tuple[Any, tuple[Any]]:
    return _recurrent(layers.LSTM, scale, batch, "lstm")


TF_BUILDERS: dict[str, Any] = {
    "convnext": build_tf_convnext,
    "vit": build_tf_vit,
    "deep_dnn": build_tf_deep_dnn,
    "transformer": build_tf_transformer,
    "cnn": build_tf_cnn,
    "rnn": build_tf_rnn,
    "lstm": build_tf_lstm,
}
