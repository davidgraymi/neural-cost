"""End-to-end architecture comparison across PyTorch, JAX, and TensorFlow.

Benchmarks five canonical neural-network architecture families on a single
shared workload per family so framework overhead and roofline efficiency are
directly comparable:

  • FF DNN     – multi-layer feedforward classifier
  • CNN        – 3-block ConvNet with pooling
  • RNN        – 2-layer vanilla (tanh) RNN encoder, bias-free
  • LSTM       – 2-layer LSTM encoder, bias-free
  • Transformer – 2-layer encoder (multi-head attention + FFN)

Run after installing one or more framework extras:

    pip install "neural-cost[torch,jax,tensorflow]"
    # Or from local source: pip install -e '.[torch,jax,tensorflow]'
    python examples/architecture_comparison.py

Hardware is auto-detected (Apple Silicon, NVIDIA, CPU fallback). Override:

    python examples/architecture_comparison.py --peak-flops 312e12 --memory-bandwidth 1.6e12

Usage flags:
    --warmup N        warm-up iterations per shape (default: 5)
    --repeats N       timed iterations per shape (default: 20)
    --batch N         batch size (default: 16)
    --seq-len N       sequence length for RNN/LSTM/Transformer (default: 32)
    --bw-bench-mb N   bandwidth benchmark working-set MiB (default: 256)
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

from neural_cost import HardwareSpec, analyze_gap, estimate_model, profile_model
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter
from neural_cost.adapters.base import FrameworkAdapter
from neural_cost.hardware_detect import detect_hardware


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

EMBED_DIM = 128          # hidden/channel/embedding dimension
NUM_HEADS = 4            # attention heads (Transformer)
NUM_CLASSES = 10         # output classes


def installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    framework: str
    architecture: str
    flops: int
    bytes_moved: int
    param_bytes: int
    arith_intensity: float
    median_ms: float
    stddev_ms: float
    efficiency: float
    achieved_gflops: float
    achieved_gbw: float
    bottleneck: str
    fused_efficiency: float | None = None
    fused_lower_bound_ms: float | None = None
    cache_resident: bool = False
    cache_name: str | None = None
    top_layer_bottleneck: str | None = None
    top_layer_share_pct: float | None = None


def evaluate(
    framework: str,
    architecture: str,
    model: Callable[..., Any],
    inputs: tuple[Any, ...],
    adapter: FrameworkAdapter,
    hardware: HardwareSpec,
    warmup: int,
    repeats: int,
) -> Result:
    profile = profile_model(model, inputs, adapter)
    runner = model
    if framework == "JAX":
        import jax

        runner = jax.jit(model)  # eager lax.scan would recompile on every call
    measurement = adapter.benchmark(runner, *inputs, warmup=warmup, repeats=repeats)
    gap = analyze_gap(profile.cost, measurement, hardware, operations=profile.operations)

    samples_ms = [s * 1e3 for s in measurement.samples_seconds]
    sd = stdev(samples_ms) if len(samples_ms) > 1 else 0.0

    fused_lb_ms = (
        round(gap.fused_lower_bound_seconds * 1e3, 3)
        if gap.fused_lower_bound_seconds is not None
        else None
    )
    cache_resident = gap.resident_cache_level is not None
    cache_name = gap.resident_cache_level

    top_layer_bneck = None
    top_layer_share = None
    if gap.layer_analyses:
        top_l = max(gap.layer_analyses, key=lambda l: l.time_share_ratio)
        top_layer_bneck = f"{top_l.name} ({top_l.kind}, {top_l.bottleneck}-bound)"
        top_layer_share = round(top_l.time_share_ratio * 100, 1)

    return Result(
        framework=framework,
        architecture=architecture,
        flops=profile.cost.flops,
        bytes_moved=profile.cost.total_bytes,
        param_bytes=profile.memory.parameter_bytes,
        arith_intensity=profile.cost.arithmetic_intensity,
        median_ms=measurement.median_seconds * 1e3,
        stddev_ms=sd,
        efficiency=gap.efficiency,
        achieved_gflops=gap.achieved_flops / 1e9,
        achieved_gbw=gap.achieved_bandwidth / 1e9,
        bottleneck=gap.bottleneck,
        fused_efficiency=gap.fused_efficiency,
        fused_lower_bound_ms=fused_lb_ms,
        cache_resident=cache_resident,
        cache_name=cache_name,
        top_layer_bottleneck=top_layer_bneck,
        top_layer_share_pct=top_layer_share,
    )


# ---------------------------------------------------------------------------
# PyTorch model builders
# ---------------------------------------------------------------------------

def torch_ff_dnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(784, EMBED_DIM),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(EMBED_DIM),
        torch.nn.Linear(EMBED_DIM, EMBED_DIM),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(EMBED_DIM),
        torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval()
    return model, (torch.randn(batch, 784),)


def torch_cnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, EMBED_DIM // 2, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM // 2),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(EMBED_DIM // 2, EMBED_DIM, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval()
    return model, (torch.randn(batch, 3, 32, 32),)


def torch_rnn(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch

    class RNNModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = torch.nn.RNN(
                EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True, bias=False
            )
            self.fc = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x: Any) -> Any:
            out, _ = self.rnn(x)
            return self.fc(out[:, -1])

    return RNNModel().eval(), (torch.randn(batch, seq, EMBED_DIM),)


def torch_lstm(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch

    class LSTMModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(
                EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True, bias=False
            )
            self.fc = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    return LSTMModel().eval(), (torch.randn(batch, seq, EMBED_DIM),)


def torch_transformer(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch

    class TransformerEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn1 = torch.nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=False)
            self.norm1a = torch.nn.LayerNorm(EMBED_DIM)
            self.ff1 = torch.nn.Linear(EMBED_DIM, EMBED_DIM * 4)
            self.ff1b = torch.nn.Linear(EMBED_DIM * 4, EMBED_DIM)
            self.norm1b = torch.nn.LayerNorm(EMBED_DIM)
            self.attn2 = torch.nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=False)
            self.norm2a = torch.nn.LayerNorm(EMBED_DIM)
            self.ff2 = torch.nn.Linear(EMBED_DIM, EMBED_DIM * 4)
            self.ff2b = torch.nn.Linear(EMBED_DIM * 4, EMBED_DIM)
            self.norm2b = torch.nn.LayerNorm(EMBED_DIM)
            self.head = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x: Any) -> Any:  # x: (B, T, D)
            x_t = x.permute(1, 0, 2)  # (T, B, D) for MHA
            a1, _ = self.attn1(x_t, x_t, x_t)
            x_t = self.norm1a(x_t + a1)
            f1 = torch.relu(self.ff1(x_t))
            x_t = self.norm1b(x_t + self.ff1b(f1))
            a2, _ = self.attn2(x_t, x_t, x_t)
            x_t = self.norm2a(x_t + a2)
            f2 = torch.relu(self.ff2(x_t))
            x_t = self.norm2b(x_t + self.ff2b(f2))
            return self.head(x_t.mean(0))

    return TransformerEncoder().eval(), (torch.randn(batch, seq, EMBED_DIM),)


# ---------------------------------------------------------------------------
# JAX model builders
#
# Each builder is the same mathematical model as its PyTorch and TensorFlow
# counterparts (same layers, biases and normalization parameters), so
# ``neural_cost.cross_check`` can assert identical dense FLOPs and parameters.
# Parameters are passed as one pytree argument after the input.
# ---------------------------------------------------------------------------

def _jax_dense(key: Any, n_in: int, n_out: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    return {"w": jax.random.normal(key, (n_in, n_out)) * n_in**-0.5, "b": jnp.zeros(n_out)}


def _jax_layer_norm(x: Any, p: dict[str, Any]) -> Any:
    import jax.numpy as jnp

    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5) * p["g"] + p["b"]


def _jax_norm_params(dim: int) -> dict[str, Any]:
    import jax.numpy as jnp

    return {"g": jnp.ones(dim), "b": jnp.zeros(dim)}


def jax_ff_dnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import jax
    import jax.numpy as jnp

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    params = {
        "fc1": _jax_dense(k1, 784, EMBED_DIM), "ln1": _jax_norm_params(EMBED_DIM),
        "fc2": _jax_dense(k2, EMBED_DIM, EMBED_DIM), "ln2": _jax_norm_params(EMBED_DIM),
        "fc3": _jax_dense(k3, EMBED_DIM, NUM_CLASSES),
    }

    def model(x: Any, p: Any) -> Any:
        x = _jax_layer_norm(jnp.maximum(x @ p["fc1"]["w"] + p["fc1"]["b"], 0), p["ln1"])
        x = _jax_layer_norm(jnp.maximum(x @ p["fc2"]["w"] + p["fc2"]["b"], 0), p["ln2"])
        return x @ p["fc3"]["w"] + p["fc3"]["b"]

    return model, (jnp.ones((batch, 784)), params)


def jax_cnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import jax
    import jax.lax as lax
    import jax.numpy as jnp

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    c1, c2 = EMBED_DIM // 2, EMBED_DIM
    params = {
        "k1": jax.random.normal(k1, (c1, 3, 3, 3)) * 0.1, "b1": jnp.zeros(c1),
        "bn1": {"scale": jnp.ones(c1), "shift": jnp.zeros(c1)},
        "k2": jax.random.normal(k2, (c2, c1, 3, 3)) * 0.1, "b2": jnp.zeros(c2),
        "bn2": {"scale": jnp.ones(c2), "shift": jnp.zeros(c2)},
        "fc": _jax_dense(k3, c2, NUM_CLASSES),
    }
    dn = ("NCHW", "OIHW", "NCHW")

    def channel(v: Any) -> Any:
        return v.reshape(1, -1, 1, 1)

    def model(x: Any, p: Any) -> Any:  # x: (B, C, H, W), eval-mode BatchNorm
        y = lax.conv_general_dilated(x, p["k1"], (1, 1), "SAME", dimension_numbers=dn)
        y = jnp.maximum(y + channel(p["b1"]), 0)
        y = y * channel(p["bn1"]["scale"]) + channel(p["bn1"]["shift"])
        y = lax.reduce_window(y, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID")
        y = lax.conv_general_dilated(y, p["k2"], (1, 1), "SAME", dimension_numbers=dn)
        y = jnp.maximum(y + channel(p["b2"]), 0)
        y = y * channel(p["bn2"]["scale"]) + channel(p["bn2"]["shift"])
        return y.mean(axis=(2, 3)) @ p["fc"]["w"] + p["fc"]["b"]

    return model, (jnp.ones((batch, 3, 32, 32)), params)


def _jax_recurrent(cell: str, batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    """Two-layer bias-free tanh RNN or LSTM via ``lax.scan`` plus a dense head."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    gates = 4 if cell == "lstm" else 1
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    scale = EMBED_DIM**-0.5
    layers = [
        {
            "ih": jax.random.normal(keys[2 * i], (EMBED_DIM, gates * EMBED_DIM)) * scale,
            "hh": jax.random.normal(keys[2 * i + 1], (EMBED_DIM, gates * EMBED_DIM)) * scale,
        }
        for i in range(2)
    ]
    params = {"layers": layers, "fc": _jax_dense(keys[4], EMBED_DIM, NUM_CLASSES)}

    def run_layer(xs: Any, p: Any) -> Any:  # xs: (T, B, D)
        zeros = jnp.zeros((xs.shape[1], EMBED_DIM))
        if cell == "lstm":
            def step(carry: Any, xt: Any) -> Any:
                h, c = carry
                i, f, g, o = jnp.split(xt @ p["ih"] + h @ p["hh"], 4, axis=-1)
                c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
                h = jax.nn.sigmoid(o) * jnp.tanh(c)
                return (h, c), h

            return lax.scan(step, (zeros, zeros), xs)[1]

        def rnn_step(h: Any, xt: Any) -> Any:
            h = jnp.tanh(xt @ p["ih"] + h @ p["hh"])
            return h, h

        return lax.scan(rnn_step, zeros, xs)[1]

    def model(x: Any, p: Any) -> Any:
        ys = jnp.swapaxes(x, 0, 1)
        for layer in p["layers"]:
            ys = run_layer(ys, layer)
        return ys[-1] @ p["fc"]["w"] + p["fc"]["b"]

    return model, (jnp.ones((batch, seq, EMBED_DIM)), params)


def jax_rnn(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    return _jax_recurrent("rnn", batch, seq)


def jax_lstm(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    return _jax_recurrent("lstm", batch, seq)


def jax_transformer(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    """Two post-norm encoder layers: multi-head attention + ReLU FFN, then a dense head."""
    import jax
    import jax.numpy as jnp

    keys = iter(jax.random.split(jax.random.PRNGKey(0), 16))
    layers = [
        {
            "q": _jax_dense(next(keys), EMBED_DIM, EMBED_DIM),
            "k": _jax_dense(next(keys), EMBED_DIM, EMBED_DIM),
            "v": _jax_dense(next(keys), EMBED_DIM, EMBED_DIM),
            "o": _jax_dense(next(keys), EMBED_DIM, EMBED_DIM),
            "ln_a": _jax_norm_params(EMBED_DIM),
            "ff1": _jax_dense(next(keys), EMBED_DIM, 4 * EMBED_DIM),
            "ff2": _jax_dense(next(keys), 4 * EMBED_DIM, EMBED_DIM),
            "ln_b": _jax_norm_params(EMBED_DIM),
        }
        for _ in range(2)
    ]
    params = {"layers": layers, "head": _jax_dense(next(keys), EMBED_DIM, NUM_CLASSES)}

    def dense(x: Any, p: Any) -> Any:
        return x @ p["w"] + p["b"]

    def model(x: Any, p: Any) -> Any:  # x: (B, T, D)
        b, t, d = x.shape

        def heads(z: Any) -> Any:
            return z.reshape(b, t, NUM_HEADS, d // NUM_HEADS)

        for layer in p["layers"]:
            q, k, v = (heads(dense(x, layer[name])) for name in ("q", "k", "v"))
            attn = jax.nn.dot_product_attention(q, k, v).reshape(b, t, d)
            x = _jax_layer_norm(x + dense(attn, layer["o"]), layer["ln_a"])
            ff = dense(jnp.maximum(dense(x, layer["ff1"]), 0), layer["ff2"])
            x = _jax_layer_norm(x + ff, layer["ln_b"])
        return dense(x.mean(axis=1), p["head"])

    return model, (jnp.ones((batch, seq, EMBED_DIM)), params)


# ---------------------------------------------------------------------------
# TensorFlow model builders
# ---------------------------------------------------------------------------

def tf_ff_dnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(NUM_CLASSES),
    ])
    inp = tf.ones((batch, 784))
    model(inp)  # build
    return model, (inp,)


def tf_cnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(EMBED_DIM // 2, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(EMBED_DIM, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES),
    ])
    inp = tf.ones((batch, 32, 32, 3))
    model(inp)
    return model, (inp,)


def tf_rnn(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf
    # Bias-free: PyTorch RNNs carry two bias vectors, Keras one.
    inp_layer = tf.keras.layers.Input(shape=(seq, EMBED_DIM))
    x = tf.keras.layers.SimpleRNN(EMBED_DIM, return_sequences=True, use_bias=False)(inp_layer)
    x = tf.keras.layers.SimpleRNN(EMBED_DIM, use_bias=False)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    model = tf.keras.Model(inp_layer, out)
    inp = tf.ones((batch, seq, EMBED_DIM))
    model(inp)
    return model, (inp,)


def tf_lstm(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf
    inp_layer = tf.keras.layers.Input(shape=(seq, EMBED_DIM))
    x = tf.keras.layers.LSTM(EMBED_DIM, return_sequences=True, use_bias=False)(inp_layer)
    x = tf.keras.layers.LSTM(EMBED_DIM, use_bias=False)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    model = tf.keras.Model(inp_layer, out)
    inp = tf.ones((batch, seq, EMBED_DIM))
    model(inp)
    return model, (inp,)


def tf_transformer(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf

    inp_layer = tf.keras.layers.Input(shape=(seq, EMBED_DIM))
    # Layer 1
    x = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(inp_layer, inp_layer)
    x = tf.keras.layers.LayerNormalization()(inp_layer + x)
    ff = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x)
    ff = tf.keras.layers.Dense(EMBED_DIM)(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)
    # Layer 2
    x2 = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(x, x)
    x2 = tf.keras.layers.LayerNormalization()(x + x2)
    ff2 = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x2)
    ff2 = tf.keras.layers.Dense(EMBED_DIM)(ff2)
    x2 = tf.keras.layers.LayerNormalization()(x2 + ff2)
    # Head
    pooled = tf.keras.layers.GlobalAveragePooling1D()(x2)
    out = tf.keras.layers.Dense(NUM_CLASSES)(pooled)

    model = tf.keras.Model(inp_layer, out)
    inp = tf.ones((batch, seq, EMBED_DIM))
    model(inp)
    return model, (inp,)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCHITECTURES = ["FF DNN", "CNN", "RNN", "LSTM", "Transformer"]

BUILDERS: dict[str, dict[str, Any]] = {
    "PyTorch":     {"FF DNN": torch_ff_dnn, "CNN": torch_cnn, "RNN": torch_rnn, "LSTM": torch_lstm, "Transformer": torch_transformer},
    "JAX":         {"FF DNN": jax_ff_dnn,   "CNN": jax_cnn,   "RNN": jax_rnn,   "LSTM": jax_lstm,   "Transformer": jax_transformer},
    "TensorFlow":  {"FF DNN": tf_ff_dnn,    "CNN": tf_cnn,    "RNN": tf_rnn,    "LSTM": tf_lstm,    "Transformer": tf_transformer},
}

ADAPTERS: dict[str, Any] = {
    "PyTorch": TorchAdapter,
    "JAX": JaxAdapter,
    "TensorFlow": TensorFlowAdapter,
}

PACKAGES: dict[str, str] = {
    "PyTorch": "torch",
    "JAX": "jax",
    "TensorFlow": "tensorflow",
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SEP = "─" * 135


def _bar(fraction: float, width: int = 18) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.3f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:,}"


def print_hardware(hardware: HardwareSpec, detection: Any) -> None:
    print()
    print("┌─ Hardware ──────────────────────────────────────────────────────────────────")
    print(f"│  Device          : {hardware.name}")
    print(f"│  Peak FP32       : {hardware.peak_flops / 1e12:.2f} TFLOP/s")
    print(f"│  Peak bandwidth  : {hardware.memory_bandwidth / 1e9:.1f} GB/s  "
          f"(STREAM triad: {detection.measured_bandwidth_gb_s:.1f} GB/s)")
    print(f"│  Ridge point     : {hardware.ridge_point:.1f} FLOP/byte")
    print(f"│  Detection source: {detection.source}")
    print("└────────────────────────────────────────────────────────────────────────────")
    print()


def print_architecture_section(arch: str, results: list[Result], hardware: HardwareSpec) -> None:
    fw_results = [r for r in results if r.architecture == arch]
    if not fw_results:
        return

    print(f"\n{'━' * 135}")
    print(f"  {arch}")
    print(f"{'━' * 135}")

    col = dict(fw=12, flops=10, params=10, mbytes=9, ai=7, ms=9, sd=7, eff=8, bar=20, gflops=9, gbw=8, bot=8)
    hdr = (
        f"{'Framework':<{col['fw']}} "
        f"{'FLOPs':>{col['flops']}} "
        f"{'Param B':>{col['params']}} "
        f"{'I/O MB':>{col['mbytes']}} "
        f"{'AI':>{col['ai']}} "
        f"{'ms(med)':>{col['ms']}} "
        f"{'±ms':>{col['sd']}} "
        f"{'effic.':>{col['eff']}} "
        f"{'roofline':^{col['bar']}} "
        f"{'GFLOP/s':>{col['gflops']}} "
        f"{'GB/s':>{col['gbw']}} "
        f"{'bound':<{col['bot']}}"
    )
    print(hdr)
    print("─" * 135)

    for r in fw_results:
        eff_clamped = min(r.efficiency, 1.0)
        bar = _bar(eff_clamped)
        eff_str = f"{r.efficiency:.1%}" if r.efficiency < 1.0 else f">{100:.0f}%*"
        print(
            f"{r.framework:<{col['fw']}} "
            f"{_fmt_flops(r.flops):>{col['flops']}} "
            f"{_fmt_flops(r.param_bytes):>{col['params']}} "
            f"{r.bytes_moved / 1e6:>{col['mbytes']}.1f} "
            f"{r.arith_intensity:>{col['ai']}.2f} "
            f"{r.median_ms:>{col['ms']}.3f} "
            f"{r.stddev_ms:>{col['sd']}.3f} "
            f"{eff_str:>{col['eff']}} "
            f"[{bar}] "
            f"{r.achieved_gflops:>{col['gflops']}.2f} "
            f"{r.achieved_gbw:>{col['gbw']}.2f} "
            f"{r.bottleneck:<{col['bot']}}"
        )

    print("─" * 135)
    print("  Diagnostics (top layer bottleneck & fusion):")
    for r in fw_results:
        fused_info = f", Fused Eff: {r.fused_efficiency:.1%}" if r.fused_efficiency is not None else ""
        cache_info = f", Resident in {r.cache_name}" if r.cache_resident and r.cache_name else ""
        top_info = f"Top Bottleneck: {r.top_layer_bottleneck} ({r.top_layer_share_pct}%)" if r.top_layer_bottleneck else "Top Bottleneck: N/A"
        print(f"    • {r.framework}: {top_info}{fused_info}{cache_info}")
    print("─" * 135)
    print(f"  AI = arithmetic intensity (FLOP/byte).  "
          f"Ridge point = {hardware.ridge_point:.1f} FLOP/byte  "
          "(above → compute-bound, below → memory-bound)")


def print_cross_framework_summary(results: list[Result]) -> None:
    print(f"\n{'━' * 135}")
    print("  Per-Architecture × Per-Framework Summary")
    print(f"{'━' * 135}")

    arch_col = 14
    fw_col = 28  # "framework: eff% (Xms)"

    header = f"  {'Architecture':<{arch_col}}"
    all_fws: list[str] = []
    seen: set[str] = set()
    for r in results:
        if r.framework not in seen:
            all_fws.append(r.framework)
            seen.add(r.framework)
    for fw in all_fws:
        header += f"  {fw:<{fw_col}}"
    print(header)
    print("  " + "─" * (arch_col + len(all_fws) * (fw_col + 2)))

    archs = list(dict.fromkeys(r.architecture for r in results))
    for arch in archs:
        row = f"  {arch:<{arch_col}}"
        for fw in all_fws:
            match = next((r for r in results if r.architecture == arch and r.framework == fw), None)
            if match:
                cell = f"{match.efficiency:.1%} eff  {match.median_ms:.2f}ms  {_fmt_flops(match.flops)} FLOPs"
            else:
                cell = "(skipped)"
            row += f"  {cell:<{fw_col}}"
        print(row)

    print(f"\n  *Efficiency >100% means hardware spec is conservative; pass --peak-flops to calibrate.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--peak-flops", type=float, default=None,
                        help="Override peak FP32 FLOP/s (e.g. 3.6e12)")
    parser.add_argument("--memory-bandwidth", type=float, default=None,
                        help="Override memory bandwidth bytes/s (e.g. 100e9)")
    parser.add_argument("--bw-bench-mb", type=int, default=256,
                        help="Working-set MiB for bandwidth benchmark (default 256)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warm-up iterations per workload (default 5)")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Timed iterations per workload (default 20)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default 16)")
    parser.add_argument("--seq-len", type=int, default=32,
                        help="Sequence length for RNN/LSTM/Transformer (default 32)")
    args = parser.parse_args()

    print("Detecting hardware…", flush=True)
    hardware, detection = detect_hardware(bandwidth_benchmark_mb=args.bw_bench_mb)
    if args.peak_flops is not None:
        hardware = HardwareSpec(
            hardware.name, args.peak_flops, hardware.memory_bandwidth, caches=hardware.caches
        )
    if args.memory_bandwidth is not None:
        hardware = HardwareSpec(
            hardware.name, hardware.peak_flops, args.memory_bandwidth, caches=hardware.caches
        )
    print_hardware(hardware, detection)

    all_results: list[Result] = []

    for fw_name, pkg in PACKAGES.items():
        if not installed(pkg):
            print(f"  [skipping {fw_name} – not installed]")
            continue
        if fw_name == "PyTorch":
            from neural_cost.adapters import TorchFxAdapter
            try:
                adapter: FrameworkAdapter = TorchFxAdapter(use_profiler=False)
            except Exception:  # noqa: BLE001
                adapter = ADAPTERS[fw_name]()
        else:
            adapter = ADAPTERS[fw_name]()
        builders = BUILDERS[fw_name]
        print(f"Benchmarking {fw_name}…", flush=True)
        for arch in ARCHITECTURES:
            builder = builders[arch]
            try:
                model, inputs = builder(args.batch, args.seq_len)
                result = evaluate(
                    fw_name, arch, model, inputs, adapter, hardware,
                    warmup=args.warmup, repeats=args.repeats,
                )
                all_results.append(result)
                status = f"{result.median_ms:.2f} ms  {result.efficiency:.1%} eff"
                print(f"    {arch:<14} {status}")
            except Exception as exc:  # noqa: BLE001
                print(f"    {arch:<14} ERROR: {exc}")

    if not all_results:
        raise SystemExit(
            "Install at least one framework: pip install 'neural-cost[torch,jax,tensorflow]'"
        )

    print()
    for arch in ARCHITECTURES:
        arch_results = [r for r in all_results if r.architecture == arch]
        if arch_results:
            print_architecture_section(arch, all_results, hardware)

    print_cross_framework_summary(all_results)
    print_consistency(args.batch, args.seq_len)
    print()


def print_consistency(batch: int, seq: int) -> None:
    """Verify every framework sees the same dense FLOPs and parameters per architecture."""
    from neural_cost import cross_check

    adapters = {
        fw: ADAPTERS[fw]() for fw, pkg in PACKAGES.items() if installed(pkg)
    }
    if len(adapters) < 2:
        return
    print()
    print("  Canonical cross-framework consistency (dense FLOPs, unique parameters)")
    for arch in ARCHITECTURES:
        workloads = {}
        for fw, adapter in adapters.items():
            model, inputs = BUILDERS[fw][arch](batch, seq)
            workloads[fw] = adapter.operations(model, inputs)
        report = cross_check(workloads, rtol=0.0)
        b = report.counts[0].breakdown
        status = "identical" if report.consistent else "MISMATCH: " + ", ".join(report.outliers)
        print(
            f"  {arch:<12} {b.dense_flops / 1e6:>9.1f}M dense FLOPs  "
            f"{b.parameter_count:>9,} params  {status}"
        )


if __name__ == "__main__":
    main()
