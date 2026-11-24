"""JAX benchmark model architectures across scale tiers."""

from __future__ import annotations

from typing import Any

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp

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


# ---------------------------------------------------------------------------
# Building blocks.  Each mirrors the PyTorch module of the same name in
# torch_models.py: same layers, biases and LayerNorms, NCHW images.
# ---------------------------------------------------------------------------

_DN = ("NCHW", "OIHW", "NCHW")


class _Init:
    """Deterministic small random weights (all-ones weights overflow deep stacks)."""

    def __init__(self, seed: int = 0) -> None:
        self.key = jax.random.PRNGKey(seed)

    def __call__(self, shape: tuple[int, ...], std: float = 0.02) -> Any:
        self.key, sub = jax.random.split(self.key)
        return std * jax.random.normal(sub, shape, dtype=jnp.float32)


def _linear_params(init: _Init, d_in: int, d_out: int) -> dict[str, Any]:
    return {"w": init((d_in, d_out)), "b": jnp.zeros((d_out,))}


def _norm_params(dim: int) -> dict[str, Any]:
    return {"g": jnp.ones((dim,)), "b": jnp.zeros((dim,))}


def _conv_params(init: _Init, c_in: int, c_out: int, k: int, groups: int = 1) -> dict[str, Any]:
    return {"w": init((c_out, c_in // groups, k, k)), "b": jnp.zeros((c_out,))}


def _linear(x: Any, p: dict[str, Any]) -> Any:
    return x @ p["w"] + p["b"]


def _layer_norm(x: Any, p: dict[str, Any], eps: float = 1e-5, axis: int = -1) -> Any:
    mean = x.mean(axis=axis, keepdims=True)
    var = jnp.square(x - mean).mean(axis=axis, keepdims=True)
    shape = [1] * x.ndim
    shape[axis] = -1
    return (x - mean) * lax.rsqrt(var + eps) * p["g"].reshape(shape) + p["b"].reshape(shape)


def _conv(x: Any, p: dict[str, Any], stride: int, padding: Any, groups: int = 1) -> Any:
    y = lax.conv_general_dilated(
        x, p["w"], (stride, stride), padding, feature_group_count=groups, dimension_numbers=_DN
    )
    return y + p["b"].reshape((1, -1, 1, 1))


def _gelu(x: Any) -> Any:
    # nn.GELU() in PyTorch is the exact (erf) form.
    return jnn.gelu(x, approximate=False)


def _attention_params(init: _Init, dim: int) -> dict[str, Any]:
    return {name: _linear_params(init, dim, dim) for name in ("q", "k", "v", "o")}


def _attention(x: Any, p: dict[str, Any], num_heads: int) -> Any:
    b, l, d = x.shape
    head_dim = d // num_heads
    q = _linear(x, p["q"]).reshape((b, l, num_heads, head_dim))
    k = _linear(x, p["k"]).reshape((b, l, num_heads, head_dim))
    v = _linear(x, p["v"]).reshape((b, l, num_heads, head_dim))
    return _linear(jax.nn.dot_product_attention(q, k, v).reshape((b, l, d)), p["o"])


def _mlp_params(init: _Init, dim: int, hidden: int) -> dict[str, Any]:
    return {"fc1": _linear_params(init, dim, hidden), "fc2": _linear_params(init, hidden, dim)}


def _mlp(x: Any, p: dict[str, Any]) -> Any:
    return _linear(_gelu(_linear(x, p["fc1"])), p["fc2"])


def _encoder_block_params(init: _Init, dim: int, hidden: int) -> dict[str, Any]:
    return {
        "norm1": _norm_params(dim),
        "attn": _attention_params(init, dim),
        "norm2": _norm_params(dim),
        "mlp": _mlp_params(init, dim, hidden),
    }


def _encoder_block(x: Any, p: dict[str, Any], num_heads: int) -> Any:
    """Pre-norm encoder block, as ViTBlock / TransformerBlock in PyTorch."""
    x = x + _attention(_layer_norm(x, p["norm1"]), p["attn"], num_heads)
    return x + _mlp(_layer_norm(x, p["norm2"]), p["mlp"])


# ---------------------------------------------------------------------------
# Modern architectures
# ---------------------------------------------------------------------------


def build_jax_convnext(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    cfg = scale_config(CONVNEXT, scale)
    depths, dims, stem = cfg["depths"], cfg["dims"], cfg["stem"]
    init = _Init()
    params: dict[str, Any] = {
        "stem": _conv_params(init, IN_CHANNELS, dims[0], stem),
        "stem_norm": _norm_params(dims[0]),
        "stages": [
            [
                {
                    "dw": _conv_params(init, dim, dim, 7, groups=dim),
                    "norm": _norm_params(dim),
                    "pw1": _conv_params(init, dim, 4 * dim, 1),
                    "pw2": _conv_params(init, 4 * dim, dim, 1),
                }
                for _ in range(depth)
            ]
            for depth, dim in zip(depths, dims, strict=True)
        ],
        "down": [
            {"norm": _norm_params(dims[i]), "conv": _conv_params(init, dims[i], dims[i + 1], 2)}
            for i in range(len(dims) - 1)
        ],
        "head_norm": _norm_params(dims[-1]),
        "head": _linear_params(init, dims[-1], NUM_CLASSES),
    }

    def model(x: Any, p: dict[str, Any]) -> Any:
        h = _conv(x, p["stem"], stem, "VALID")
        h = _layer_norm(h, p["stem_norm"], eps=1e-6, axis=1)
        for i, (blocks, dim) in enumerate(zip(p["stages"], dims, strict=True)):
            for blk in blocks:
                res = h
                h = _conv(h, blk["dw"], 1, ((3, 3), (3, 3)), groups=dim)
                h = _layer_norm(h, blk["norm"], eps=1e-6, axis=1)
                h = _conv(h, blk["pw1"], 1, "VALID")
                h = _gelu(h)
                h = _conv(h, blk["pw2"], 1, "VALID")
                h = res + h
            if i < len(p["down"]):
                h = _layer_norm(h, p["down"][i]["norm"], eps=1e-6, axis=1)
                h = _conv(h, p["down"][i]["conv"], 2, "VALID")
        h = h.mean(axis=(2, 3))
        return _linear(_layer_norm(h, p["head_norm"]), p["head"])

    x = jnp.ones((batch, IN_CHANNELS, cfg["img_size"], cfg["img_size"]))
    return model, (x, params)


def build_jax_vit(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    cfg = scale_config(VIT, scale)
    dim, heads, patch = cfg["embed_dim"], cfg["num_heads"], cfg["patch_size"]
    num_patches = (cfg["img_size"] // patch) ** 2
    init = _Init()
    params: dict[str, Any] = {
        "patch": _conv_params(init, IN_CHANNELS, dim, patch),
        "pos": jnp.zeros((1, num_patches, dim)),
        "blocks": [
            _encoder_block_params(init, dim, dim * VIT_MLP_RATIO) for _ in range(cfg["depth"])
        ],
        "norm": _norm_params(dim),
        "head": _linear_params(init, dim, NUM_CLASSES),
    }

    def model(x: Any, p: dict[str, Any]) -> Any:
        h = _conv(x, p["patch"], patch, "VALID")
        b, d, gh, gw = h.shape
        h = h.reshape((b, d, gh * gw)).transpose((0, 2, 1)) + p["pos"]
        for blk in p["blocks"]:
            h = _encoder_block(h, blk, heads)
        h = _layer_norm(h, p["norm"])
        return _linear(h.mean(axis=1), p["head"])

    x = jnp.ones((batch, IN_CHANNELS, cfg["img_size"], cfg["img_size"]))
    return model, (x, params)


def build_jax_deep_dnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    cfg = scale_config(DEEP_DNN, scale)
    d_in, hidden, d_out = cfg["in_features"], cfg["hidden_features"], cfg["out_features"]
    init = _Init()
    params: dict[str, Any] = {
        "layers": [
            {"fc": _linear_params(init, a, b), "norm": _norm_params(b)}
            for a, b in ((d_in, hidden), (hidden, hidden), (hidden, d_out))
        ],
        "head": _linear_params(init, d_out, NUM_CLASSES),
    }

    def model(x: Any, p: dict[str, Any]) -> Any:
        for layer in p["layers"]:
            x = _layer_norm(_gelu(_linear(x, layer["fc"])), layer["norm"])
        return _linear(x, p["head"])

    x = jnp.ones((batch, d_in))
    return model, (x, params)


def build_jax_transformer(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    cfg = scale_config(TRANSFORMER, scale)
    dim, heads = cfg["embed_dim"], cfg["num_heads"]
    init = _Init()
    params: dict[str, Any] = {
        "blocks": [
            _encoder_block_params(init, dim, cfg["ffn_dim"]) for _ in range(cfg["num_layers"])
        ],
        "norm": _norm_params(dim),
        "head": _linear_params(init, dim, NUM_CLASSES),
    }

    def model(x: Any, p: dict[str, Any]) -> Any:
        for blk in p["blocks"]:
            x = _encoder_block(x, blk, heads)
        x = _layer_norm(x, p["norm"])
        return _linear(x.mean(axis=1), p["head"])

    x = jnp.ones((batch, cfg["seq_len"], dim))
    return model, (x, params)


# ---------------------------------------------------------------------------
# Legacy architectures
# ---------------------------------------------------------------------------


def build_jax_cnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    img_size = 32 if scale == "micro" else 224
    embed_dim = 128
    k1 = jnp.ones((embed_dim // 2, 3, 3, 3))
    k2 = jnp.ones((embed_dim, embed_dim // 2, 3, 3))
    wfc = jnp.ones((embed_dim, 10))

    def model(x, _k1=k1, _k2=k2, _wfc=wfc):
        y = jnp.tanh(
            lax.conv_general_dilated(
                x, _k1, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
            )
        )
        y = jnp.tanh(
            lax.conv_general_dilated(
                y, _k2, (2, 2), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
            )
        )
        return y.mean(axis=(2, 3)) @ _wfc

    x = jnp.ones((batch, 3, img_size, img_size))
    return model, (x, k1, k2, wfc)


def build_jax_rnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    seq_len = 32 if scale == "micro" else 128
    dim = 128 if scale == "micro" else 256
    wih = jnp.ones((dim, dim))
    whh = jnp.ones((dim, dim))
    wfc = jnp.ones((dim, 10))

    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], dim))
        for t in range(seq_len):
            h = jnp.tanh(x[:, t, :] @ _wih + h @ _whh)
        return h @ _wfc

    x = jnp.ones((batch, seq_len, dim))
    return model, (x, wih, whh, wfc)


def build_jax_lstm(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    seq_len = 32 if scale == "micro" else 128
    dim = 128 if scale == "micro" else 256
    wih = jnp.ones((dim, dim * 4))
    whh = jnp.ones((dim, dim * 4))
    wfc = jnp.ones((dim, 10))

    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], dim))
        c = jnp.zeros((x.shape[0], dim))
        for t in range(seq_len):
            gates = x[:, t, :] @ _wih + h @ _whh
            i_g = jnn.sigmoid(gates[:, :dim])
            f_g = jnn.sigmoid(gates[:, dim : 2 * dim])
            g_g = jnp.tanh(gates[:, 2 * dim : 3 * dim])
            o_g = jnn.sigmoid(gates[:, 3 * dim :])
            c = f_g * c + i_g * g_g
            h = o_g * jnp.tanh(c)
        return h @ _wfc

    x = jnp.ones((batch, seq_len, dim))
    return model, (x, wih, whh, wfc)


JAX_BUILDERS: dict[str, Any] = {
    "convnext": build_jax_convnext,
    "vit": build_jax_vit,
    "deep_dnn": build_jax_deep_dnn,
    "transformer": build_jax_transformer,
    "cnn": build_jax_cnn,
    "rnn": build_jax_rnn,
    "lstm": build_jax_lstm,
}
