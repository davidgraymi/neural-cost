"""Scientific evaluation: collect benchmark data across architectures × frameworks × batch sizes.

Writes results/benchmark_data.json when complete.

Key improvement over the basic example:
  - JAX models are wrapped with jax.jit() (the dominant performance lever)
  - PyTorch models are wrapped with torch.compile() where supported
  - Baseline (un-compiled) vs optimised variants are both measured
  - Multiple batch sizes sweep from memory-bound to compute-bound regimes
  - 15 warmup + 40 repeat iterations for stable statistics

Run:
    python benchmarks/collect_data.py [--quick]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

# Ensure repo root and src/ are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.models import (
    SCALE_BATCH_SIZES,
    SCALE_TIERS,
    STANDARD_BATCH_SIZES,
    get_available_models,
    get_batch_sizes,
    get_model,
)
from neural_cost import (
    CostEstimate,
    HardwareSpec,
    Measurement,
    analyze_gap,
    analyze_memory_gap,
    estimate_adamw_traffic,
    estimate_fused_operations,
    estimate_memory,
    profile_model,
)
from neural_cost.adapters import (
    JaxAdapter,
    TensorFlowAdapter,
    TorchAdapter,
    TorchFxAdapter,
)
from neural_cost.adapters.base import FrameworkAdapter
from neural_cost.hardware_detect import detect_hardware

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_DIM = 128
NUM_HEADS = 4
NUM_CLASSES = 10
IMG_SIZE = 32

BATCH_SIZES = [1, 4, 16, 64]
QUICK_BATCH_SIZES = [4, 16]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class BenchRecord:
    framework: str
    variant: str  # "baseline" or "optimised"
    architecture: str
    batch: int
    flops: int
    param_bytes: int
    total_bytes: int
    arith_intensity: float
    latency_median_ms: float
    latency_mean_ms: float
    latency_stddev_ms: float
    latency_cv_pct: float  # coefficient of variation
    latency_p95_ms: float
    roofline_efficiency: float
    achieved_gflops: float
    achieved_gbw: float
    bottleneck: str
    scale: str = "standard"
    precision: str = "fp32"
    mode: str = "inference"
    fused_efficiency: float | None = None
    fused_lower_bound_ms: float | None = None
    traffic_reduction_pct: float | None = None
    cache_resident: bool = False
    cache_name: str | None = None
    cache_bound_ms: float | None = None
    top_layer_bottleneck: str | None = None
    top_layer_share_pct: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    memory_overhead_ratio: float | None = None
    theoretical_min_bytes: int | None = None
    theoretical_conservative_bytes: int | None = None
    training_minimum_bytes: int | None = None


def get_mode_flops_multiplier(mode: str) -> float:
    """Return theoretical FLOP multiplier relative to forward pass."""
    mode_clean = mode.lower().strip()
    if mode_clean == "train_step":
        return 3.0
    if mode_clean == "backward_only":
        return 2.0
    return 1.0


def get_precision_bytes(precision: str) -> int:
    """Return byte width for target precision."""
    p = precision.lower().strip()
    if p in ("fp16", "bf16"):
        return 2
    if p in ("int8", "fp8"):
        return 1
    return 4


def _make_bench_record(
    framework: str,
    variant: str,
    arch: str,
    batch: int,
    prof: Any,
    gap: Any,
    st: dict[str, float],
    scale: str = "standard",
    precision: str = "fp32",
    mode: str = "inference",
    peak_alloc: int | None = None,
    peak_res: int | None = None,
) -> BenchRecord:
    fused_lb_ms = (
        round(gap.fused_lower_bound_seconds * 1e3, 3)
        if getattr(gap, "fused_lower_bound_seconds", None) is not None
        else None
    )
    traffic_red_pct = None
    if getattr(gap, "fused_lower_bound_seconds", None) is not None and getattr(
        prof, "operations", None
    ):
        fused_est = estimate_fused_operations(prof.operations)
        traffic_red_pct = round(fused_est.traffic_reduction_ratio * 100, 1)

    cache_resident = getattr(gap, "resident_cache_level", None) is not None
    cache_name = getattr(gap, "resident_cache_level", None)
    cache_bound_ms = (
        round(gap.cache_bound_seconds * 1e3, 3)
        if getattr(gap, "cache_bound_seconds", None) is not None
        else None
    )

    top_layer_bneck = None
    top_layer_share = None
    if getattr(gap, "layer_analyses", None):
        top_l = max(gap.layer_analyses, key=lambda l: l.time_share_ratio)
        top_layer_bneck = f"{top_l.name} ({top_l.kind}, {top_l.bottleneck}-bound)"
        top_layer_share = round(top_l.time_share_ratio * 100, 1)

    theo_min = None
    theo_cons = None
    ratio = None
    if hasattr(prof, "memory") and prof.memory is not None:
        theo_min = getattr(prof.memory, "inference_minimum_bytes", None)
        theo_cons = getattr(prof.memory, "inference_conservative_bytes", None)
        if peak_alloc is not None and theo_min is not None and theo_min > 0:
            ratio = round(peak_alloc / theo_min, 4)

    training_min_b = None
    if mode == "train_step" and getattr(prof, "operations", None):
        try:
            mem_train = estimate_memory(prof.operations, training=True, optimizer_state_multiplier=2.0)
            training_min_b = mem_train.training_minimum_bytes
        except Exception:
            training_min_b = None

    return BenchRecord(
        framework=framework,
        variant=variant,
        architecture=arch,
        batch=batch,
        flops=prof.cost.flops,
        param_bytes=prof.memory.parameter_bytes,
        total_bytes=prof.cost.total_bytes,
        arith_intensity=prof.cost.arithmetic_intensity,
        latency_median_ms=st["median"],
        latency_mean_ms=st["mean"],
        latency_stddev_ms=st["stddev"],
        latency_cv_pct=st["cv"],
        latency_p95_ms=st["p95"],
        roofline_efficiency=gap.efficiency,
        achieved_gflops=gap.achieved_flops / 1e9,
        achieved_gbw=gap.achieved_bandwidth / 1e9,
        bottleneck=gap.bottleneck,
        scale=scale,
        precision=precision,
        mode=mode,
        fused_efficiency=getattr(gap, "fused_efficiency", None),
        fused_lower_bound_ms=fused_lb_ms,
        traffic_reduction_pct=traffic_red_pct,
        cache_resident=cache_resident,
        cache_name=cache_name,
        cache_bound_ms=cache_bound_ms,
        top_layer_bottleneck=top_layer_bneck,
        top_layer_share_pct=top_layer_share,
        peak_allocated_bytes=peak_alloc,
        peak_reserved_bytes=peak_res,
        memory_overhead_ratio=ratio,
        theoretical_min_bytes=theo_min,
        theoretical_conservative_bytes=theo_cons,
        training_minimum_bytes=training_min_b,
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _jax_block(
    fn: Callable, args: tuple, warmup: int, repeats: int
) -> tuple[list[float], int | None, int | None]:
    import jax

    def wait(v: Any) -> None:
        for leaf in jax.tree.leaves(v):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

    for _ in range(warmup):
        wait(fn(*args))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        wait(fn(*args))
        samples.append((time.perf_counter_ns() - t0) / 1e6)

    peak_allocated = None
    peak_reserved = None
    try:
        devices = jax.devices()
        if devices and hasattr(devices[0], "memory_stats"):
            stats = devices[0].memory_stats()
            if stats:
                peak_allocated = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
                peak_reserved = stats.get("bytes_limit") or stats.get("bytes_reserved")
    except Exception:
        pass
    if peak_allocated is None:
        try:
            total_nbytes = sum(getattr(leaf, "nbytes", 0) for leaf in jax.tree.leaves(args))
            if total_nbytes > 0:
                peak_allocated = int(total_nbytes)
                peak_reserved = peak_allocated
        except Exception:
            pass
    return samples, peak_allocated, peak_reserved


def _torch_block(
    fn: Callable, args: tuple, warmup: int, repeats: int
) -> tuple[list[float], int | None, int | None]:
    import torch

    device = next((arg.device for arg in args if isinstance(arg, torch.Tensor)), None)
    is_cuda = device is not None and device.type == "cuda"
    is_mps = device is not None and device.type == "mps"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    elif is_mps:
        torch.mps.synchronize()

    for _ in range(warmup):
        fn(*args)

    if is_cuda:
        torch.cuda.synchronize(device)
    elif is_mps:
        torch.mps.synchronize()

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        if is_cuda:
            torch.cuda.synchronize(device)
        elif is_mps:
            torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e6)

    peak_allocated: int | None = None
    peak_reserved: int | None = None
    if is_cuda:
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    elif is_mps:
        peak_allocated = int(torch.mps.current_allocated_memory())
        peak_reserved = int(torch.mps.driver_allocated_memory())
    else:
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
            ) as prof:
                fn(*args)
            events = prof.events()
            act_allocs = [e.cpu_memory_usage for e in events if e.cpu_memory_usage > 0]
            param_bytes = 0
            if hasattr(fn, "parameters") and callable(fn.parameters):
                param_bytes = sum(p.numel() * p.element_size() for p in fn.parameters())
            peak_allocated = int(param_bytes + sum(act_allocs))
            peak_reserved = peak_allocated
        except Exception:
            peak_allocated = None
            peak_reserved = None

    return samples, peak_allocated, peak_reserved


def _tf_block(
    fn: Callable, args: tuple, warmup: int, repeats: int
) -> tuple[list[float], int | None, int | None]:
    import tensorflow as tf

    async_wait = getattr(tf.experimental, "async_wait", None)

    def wait():
        if async_wait:
            async_wait()

    for _ in range(warmup):
        fn(*args)
        wait()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        wait()
        samples.append((time.perf_counter_ns() - t0) / 1e6)

    peak_allocated = None
    peak_reserved = None
    try:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            info = tf.config.experimental.get_memory_info("GPU:0")
            peak_allocated = info.get("peak")
            peak_reserved = info.get("current") or peak_allocated
    except Exception:
        pass
    if peak_allocated is None:
        try:
            total_bytes = 0
            for a in args:
                if hasattr(a, "numpy"):
                    total_bytes += a.numpy().nbytes
            if total_bytes > 0:
                peak_allocated = int(total_bytes)
                peak_reserved = peak_allocated
        except Exception:
            pass
    return samples, peak_allocated, peak_reserved


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    n = len(s)
    med = median(s)
    mn = mean(s)
    sd = stdev(s) if n > 1 else 0.0
    cv = 100 * sd / mn if mn > 0 else 0.0
    p95 = s[min(int(0.95 * n), n - 1)]
    return dict(median=med, mean=mn, stddev=sd, cv=cv, p95=p95)


# ---------------------------------------------------------------------------
# PyTorch model builders  (return model, call_fn, inputs_for_adapter)
# ---------------------------------------------------------------------------


def _torch_ff_dnn(batch: int):
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
    x = torch.randn(batch, 784)
    return model, (x,)


def _torch_cnn(batch: int):
    import torch

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, EMBED_DIM // 2, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM // 2),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(EMBED_DIM // 2, EMBED_DIM, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval()
    x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    return model, (x,)


def _torch_rnn(batch: int):
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.RNN(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1])

    m = M().eval()
    x = torch.randn(batch, 32, EMBED_DIM)
    return m, (x,)


def _torch_lstm(batch: int):
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    m = M().eval()
    x = torch.randn(batch, 32, EMBED_DIM)
    return m, (x,)


def _torch_transformer(batch: int):
    import torch
    import torch.nn.functional as F

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight

    class SDPASelfAttention(torch.nn.Module):
        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

        def forward(self, x):
            b, l, d = x.shape
            q = self.q_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.out_proj(out.transpose(1, 2).contiguous().view(b, l, d))

    class SwiGLU(torch.nn.Module):
        def __init__(self, in_features: int, hidden_features: int):
            super().__init__()
            self.w_gate = torch.nn.Linear(in_features, hidden_features, bias=False)
            self.w_up = torch.nn.Linear(in_features, hidden_features, bias=False)
            self.w_down = torch.nn.Linear(hidden_features, in_features, bias=False)

        def forward(self, x):
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    class M(torch.nn.Module):
        """Transformer model using scaled_dot_product_attention (SDPA)."""

        def __init__(self):
            super().__init__()
            self.norm1 = RMSNorm(EMBED_DIM)
            self.attn = SDPASelfAttention(EMBED_DIM, NUM_HEADS)
            self.norm2 = RMSNorm(EMBED_DIM)
            self.ffn = SwiGLU(EMBED_DIM, int(EMBED_DIM * 8 / 3))
            self.head = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return self.head(x.mean(1))

    m = M().eval()
    x = torch.randn(batch, 32, EMBED_DIM)
    return m, (x,)


TORCH_BUILDERS = {
    "FF DNN": _torch_ff_dnn,
    "CNN": _torch_cnn,
    "RNN": _torch_rnn,
    "LSTM": _torch_lstm,
    "Transformer": _torch_transformer,
}


# ---------------------------------------------------------------------------
# JAX model builders
# ---------------------------------------------------------------------------


def _jax_ff_dnn(batch: int):
    import jax.numpy as jnp

    w1 = jnp.ones((784, EMBED_DIM))
    w2 = jnp.ones((EMBED_DIM, EMBED_DIM))
    w3 = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def model(x, _w1=w1, _w2=w2, _w3=w3):
        return jnp.tanh(jnp.tanh(x @ _w1) @ _w2) @ _w3

    return model, (jnp.ones((batch, 784)), w1, w2, w3)


def _jax_cnn(batch: int):
    import jax.lax as lax
    import jax.numpy as jnp

    k1 = jnp.ones((EMBED_DIM // 2, 3, 3, 3))
    k2 = jnp.ones((EMBED_DIM, EMBED_DIM // 2, 3, 3))
    wfc = jnp.ones((EMBED_DIM, NUM_CLASSES))

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

    return model, (jnp.ones((batch, 3, IMG_SIZE, IMG_SIZE)), k1, k2, wfc)


def _jax_rnn(batch: int):
    import jax.numpy as jnp

    T = 32
    wih = jnp.ones((EMBED_DIM, EMBED_DIM))
    whh = jnp.ones((EMBED_DIM, EMBED_DIM))
    wfc = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], EMBED_DIM))
        for t in range(T):
            h = jnp.tanh(x[:, t] @ _wih + h @ _whh)
        return h @ _wfc

    return model, (jnp.ones((batch, T, EMBED_DIM)), wih, whh, wfc)


def _jax_lstm(batch: int):
    import jax.numpy as jnp

    T, G = 32, 4
    wih = jnp.ones((EMBED_DIM, G * EMBED_DIM))
    whh = jnp.ones((EMBED_DIM, G * EMBED_DIM))
    wfc = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def sigmoid(x):
        return 1.0 / (1.0 + jnp.exp(-x))

    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], EMBED_DIM))
        c = jnp.zeros((x.shape[0], EMBED_DIM))
        for t in range(T):
            gv = x[:, t] @ _wih + h @ _whh
            i, f, g, o = jnp.split(gv, 4, axis=-1)
            c = sigmoid(f) * c + sigmoid(i) * jnp.tanh(g)
            h = sigmoid(o) * jnp.tanh(c)
        return h @ _wfc

    return model, (jnp.ones((batch, T, EMBED_DIM)), wih, whh, wfc)


def _jax_transformer(batch: int):
    import jax.nn as jnn
    import jax.numpy as jnp

    T = 32
    hd = EMBED_DIM // NUM_HEADS
    d_inter = int(EMBED_DIM * 8 / 3)
    gamma1 = jnp.ones(EMBED_DIM)
    wq = jnp.ones((EMBED_DIM, EMBED_DIM))
    wk = jnp.ones((EMBED_DIM, EMBED_DIM))
    wv = jnp.ones((EMBED_DIM, EMBED_DIM))
    wo = jnp.ones((EMBED_DIM, EMBED_DIM))
    gamma2 = jnp.ones(EMBED_DIM)
    w_gate = jnp.ones((EMBED_DIM, d_inter))
    w_up = jnp.ones((EMBED_DIM, d_inter))
    w_down = jnp.ones((d_inter, EMBED_DIM))
    wfc = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def rmsnorm(u, gamma):
        return (u / jnp.sqrt(jnp.mean(jnp.square(u), -1, keepdims=True) + 1e-6)) * gamma

    def attn(u, _wq, _wk, _wv, _wo):
        b, l, d = u.shape
        q = (u @ _wq).reshape((b, l, NUM_HEADS, hd))
        k = (u @ _wk).reshape((b, l, NUM_HEADS, hd))
        v = (u @ _wv).reshape((b, l, NUM_HEADS, hd))
        return jax.nn.dot_product_attention(q, k, v).reshape((b, l, d)) @ _wo

    def model(
        x,
        _g1=gamma1,
        _wq=wq,
        _wk=wk,
        _wv=wv,
        _wo=wo,
        _g2=gamma2,
        _wg=w_gate,
        _wu=w_up,
        _wd=w_down,
        _wfc=wfc,
    ):
        x = x + attn(rmsnorm(x, _g1), _wq, _wk, _wv, _wo)
        x = x + ((jnn.silu(rmsnorm(x, _g2) @ _wg) * (rmsnorm(x, _g2) @ _wu)) @ _wd)
        return x.mean(1) @ _wfc

    return model, (
        jnp.ones((batch, T, EMBED_DIM)),
        gamma1,
        wq,
        wk,
        wv,
        wo,
        gamma2,
        w_gate,
        w_up,
        w_down,
        wfc,
    )


JAX_BUILDERS = {
    "FF DNN": _jax_ff_dnn,
    "CNN": _jax_cnn,
    "RNN": _jax_rnn,
    "LSTM": _jax_lstm,
    "Transformer": _jax_transformer,
}


# ---------------------------------------------------------------------------
# TF model builders
# ---------------------------------------------------------------------------


def _tf_ff_dnn(batch: int):
    import tensorflow as tf

    m = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(NUM_CLASSES),
        ]
    )
    x = tf.ones((batch, 784))
    m(x)
    return m, (x,)


def _tf_cnn(batch: int):
    import tensorflow as tf

    m = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(EMBED_DIM // 2, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(EMBED_DIM, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(NUM_CLASSES),
        ]
    )
    x = tf.ones((batch, IMG_SIZE, IMG_SIZE, 3))
    m(x)
    return m, (x,)


def _tf_rnn(batch: int):
    import tensorflow as tf

    inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
    x = tf.keras.layers.GRU(EMBED_DIM, return_sequences=True)(inp)
    x = tf.keras.layers.GRU(EMBED_DIM)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    m = tf.keras.Model(inp, out)
    xi = tf.ones((batch, 32, EMBED_DIM))
    m(xi)
    return m, (xi,)


def _tf_lstm(batch: int):
    import tensorflow as tf

    inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
    x = tf.keras.layers.LSTM(EMBED_DIM, return_sequences=True)(inp)
    x = tf.keras.layers.LSTM(EMBED_DIM)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    m = tf.keras.Model(inp, out)
    xi = tf.ones((batch, 32, EMBED_DIM))
    m(xi)
    return m, (xi,)


def _tf_transformer(batch: int):
    import tensorflow as tf

    inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
    x = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(
        inp, inp
    )
    x = tf.keras.layers.LayerNormalization()(inp + x)
    ff = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x)
    ff = tf.keras.layers.Dense(EMBED_DIM)(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)
    x2 = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(
        x, x
    )
    x2 = tf.keras.layers.LayerNormalization()(x + x2)
    ff2 = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x2)
    ff2 = tf.keras.layers.Dense(EMBED_DIM)(ff2)
    x2 = tf.keras.layers.LayerNormalization()(x2 + ff2)
    p = tf.keras.layers.GlobalAveragePooling1D()(x2)
    out = tf.keras.layers.Dense(NUM_CLASSES)(p)
    m = tf.keras.Model(inp, out)
    xi = tf.ones((batch, 32, EMBED_DIM))
    m(xi)
    return m, (xi,)


TF_BUILDERS = {
    "FF DNN": _tf_ff_dnn,
    "CNN": _tf_cnn,
    "RNN": _tf_rnn,
    "LSTM": _tf_lstm,
    "Transformer": _tf_transformer,
}

DEFAULT_ARCHITECTURES: list[str] = ["FF DNN", "ConvNeXt", "ViT", "Transformer"]
LEGACY_ARCHITECTURES: list[str] = ["CNN", "RNN", "LSTM"]


def get_benchmarked_architectures(include_legacy: bool = False) -> list[str]:
    """Return active benchmark architectures, gating legacy models."""
    if include_legacy:
        return list(DEFAULT_ARCHITECTURES + LEGACY_ARCHITECTURES)
    return list(DEFAULT_ARCHITECTURES)


ARCHITECTURES = get_benchmarked_architectures(include_legacy=False)


# ---------------------------------------------------------------------------
# Per-framework evaluation
# ---------------------------------------------------------------------------


def eval_torch(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    use_fx: bool = True,
    scale: str = "standard",
    architectures: list[str] | None = None,
    precision: str = "fp32",
    mode: str = "inference",
) -> list[BenchRecord]:
    import torch

    # Use TorchFxAdapter by default to trace functional calls and activations;
    # automatically falls back to module hooks if graph is untraceable.
    if use_fx:
        try:
            adapter: FrameworkAdapter = TorchFxAdapter()
        except Exception:
            adapter = TorchAdapter()
    else:
        adapter = TorchAdapter()

    active_archs = architectures if architectures is not None else ARCHITECTURES
    records: list[BenchRecord] = []
    for arch in active_archs:
        for batch in batches:
            try:
                model, inputs = get_model(arch, framework="torch", scale=scale, batch=batch)
                if precision == "fp16":
                    model = model.half()
                    inputs = tuple(
                        x.half() if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                        for x in inputs
                    )
                elif precision == "bf16":
                    model = model.to(torch.bfloat16)
                    inputs = tuple(
                        x.to(torch.bfloat16)
                        if isinstance(x, torch.Tensor) and x.is_floating_point()
                        else x
                        for x in inputs
                    )
            except Exception as exc:
                print(f"  PyTorch get_model {arch} scale={scale}: {exc}")
                continue
            # Static profile
            try:
                prof = profile_model(model, inputs, adapter)
            except Exception as exc:
                print(f"  PyTorch profile {arch} B={batch}: {exc}")
                continue

            flop_multiplier = get_mode_flops_multiplier(mode)
            cost = prof.cost
            if flop_multiplier != 1.0:
                cost = CostEstimate(
                    flops=int(prof.cost.flops * flop_multiplier),
                    read_bytes=prof.cost.read_bytes,
                    write_bytes=prof.cost.write_bytes,
                    operations=prof.cost.operations,
                )

            # Baseline
            try:
                if mode == "train_step":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

                    def run_step(*x: Any) -> None:
                        optimizer.zero_grad(set_to_none=True)
                        out = model(*x)
                        loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
                        loss.backward()
                        optimizer.step()

                    samp, peak_alloc, peak_res = _torch_block(run_step, inputs, warmup, repeats)
                elif mode == "backward_only":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

                    def run_bwd(*x: Any) -> None:
                        model.zero_grad(set_to_none=True)
                        out = model(*x)
                        loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
                        loss.backward()

                    samp, peak_alloc, peak_res = _torch_block(run_bwd, inputs, warmup, repeats)
                else:
                    samp, peak_alloc, peak_res = _torch_block(model, inputs, warmup, repeats)

                st = _stats(samp)
                meas = Measurement(
                    median_seconds=st["median"] / 1e3,
                    samples_seconds=tuple(s / 1e3 for s in samp),
                    peak_memory_bytes=peak_alloc,
                    allocated_memory_bytes=peak_alloc,
                    reserved_memory_bytes=peak_res,
                )
                gap = analyze_gap(
                    cost,
                    meas,
                    hardware,
                    operations=prof.operations,
                )
                records.append(
                    _make_bench_record(
                        "PyTorch",
                        "baseline",
                        arch,
                        batch,
                        prof,
                        gap,
                        st,
                        scale=scale,
                        precision=precision,
                        mode=mode,
                        peak_alloc=peak_alloc,
                        peak_res=peak_res,
                    )
                )
            except Exception as exc:
                print(f"  PyTorch baseline {arch} B={batch}: {exc}")
            # Optimised: torch.compile
            try:
                compiled = torch.compile(model)
                # Warmup compile
                for _ in range(max(3, warmup)):
                    compiled(*inputs)
                if mode == "train_step":
                    optimizer = torch.optim.AdamW(compiled.parameters(), lr=1e-3)

                    def run_compiled_step(*x: Any) -> None:
                        optimizer.zero_grad(set_to_none=True)
                        out = compiled(*x)
                        loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
                        loss.backward()
                        optimizer.step()

                    samp, peak_alloc, peak_res = _torch_block(run_compiled_step, inputs, warmup, repeats)
                elif mode == "backward_only":
                    def run_compiled_bwd(*x: Any) -> None:
                        compiled.zero_grad(set_to_none=True)
                        out = compiled(*x)
                        loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
                        loss.backward()

                    samp, peak_alloc, peak_res = _torch_block(run_compiled_bwd, inputs, warmup, repeats)
                else:
                    samp, peak_alloc, peak_res = _torch_block(compiled, inputs, warmup, repeats)

                st = _stats(samp)
                meas = Measurement(
                    median_seconds=st["median"] / 1e3,
                    samples_seconds=tuple(s / 1e3 for s in samp),
                    peak_memory_bytes=peak_alloc,
                    allocated_memory_bytes=peak_alloc,
                    reserved_memory_bytes=peak_res,
                )
                gap = analyze_gap(
                    cost,
                    meas,
                    hardware,
                    operations=prof.operations,
                )
                records.append(
                    _make_bench_record(
                        "PyTorch",
                        "compiled",
                        arch,
                        batch,
                        prof,
                        gap,
                        st,
                        scale=scale,
                        precision=precision,
                        mode=mode,
                        peak_alloc=peak_alloc,
                        peak_res=peak_res,
                    )
                )
            except Exception as exc:
                print(f"  PyTorch compiled {arch} B={batch}: {exc}")
    return records


def eval_jax(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    scale: str = "standard",
    architectures: list[str] | None = None,
    precision: str = "fp32",
    mode: str = "inference",
) -> list[BenchRecord]:
    import jax

    adapter = JaxAdapter()
    active_archs = architectures if architectures is not None else ARCHITECTURES
    records: list[BenchRecord] = []
    for arch in active_archs:
        for batch in batches:
            try:
                model, inputs = get_model(arch, framework="jax", scale=scale, batch=batch)
            except Exception as exc:
                print(f"  JAX get_model {arch} scale={scale}: {exc}")
                continue
            try:
                prof = profile_model(model, inputs, adapter)
            except Exception as exc:
                print(f"  JAX profile {arch} B={batch}: {exc}")
                continue
            flop_multiplier = get_mode_flops_multiplier(mode)
            cost = prof.cost
            if flop_multiplier != 1.0:
                cost = CostEstimate(
                    flops=int(prof.cost.flops * flop_multiplier),
                    read_bytes=prof.cost.read_bytes,
                    write_bytes=prof.cost.write_bytes,
                    operations=prof.cost.operations,
                )
            # Baseline (eager)
            try:
                samp, peak_alloc, peak_res = _jax_block(model, inputs, warmup, repeats)
                st = _stats(samp)
                meas = Measurement(
                    median_seconds=st["median"] / 1e3,
                    samples_seconds=tuple(s / 1e3 for s in samp),
                    peak_memory_bytes=peak_alloc,
                    allocated_memory_bytes=peak_alloc,
                    reserved_memory_bytes=peak_res,
                )
                gap = analyze_gap(
                    cost,
                    meas,
                    hardware,
                    operations=prof.operations,
                )
                records.append(
                    _make_bench_record(
                        "JAX",
                        "baseline",
                        arch,
                        batch,
                        prof,
                        gap,
                        st,
                        scale=scale,
                        precision=precision,
                        mode=mode,
                        peak_alloc=peak_alloc,
                        peak_res=peak_res,
                    )
                )
            except Exception as exc:
                print(f"  JAX baseline {arch} B={batch}: {exc}")
            # Optimised: jax.jit
            try:
                jit_model = jax.jit(model)
                wait = lambda v: [
                    leaf.block_until_ready()
                    for leaf in jax.tree.leaves(v)
                    if hasattr(leaf, "block_until_ready")
                ]
                wait(jit_model(*inputs))
                samp, peak_alloc, peak_res = _jax_block(jit_model, inputs, warmup, repeats)
                st = _stats(samp)
                meas = Measurement(
                    median_seconds=st["median"] / 1e3,
                    samples_seconds=tuple(s / 1e3 for s in samp),
                    peak_memory_bytes=peak_alloc,
                    allocated_memory_bytes=peak_alloc,
                    reserved_memory_bytes=peak_res,
                )
                gap = analyze_gap(
                    cost,
                    meas,
                    hardware,
                    operations=prof.operations,
                )
                records.append(
                    _make_bench_record(
                        "JAX",
                        "jit",
                        arch,
                        batch,
                        prof,
                        gap,
                        st,
                        scale=scale,
                        precision=precision,
                        mode=mode,
                        peak_alloc=peak_alloc,
                        peak_res=peak_res,
                    )
                )
            except Exception as exc:
                print(f"  JAX jit {arch} B={batch}: {exc}")
    return records


def eval_tensorflow(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    scale: str = "standard",
    architectures: list[str] | None = None,
    precision: str = "fp32",
    mode: str = "inference",
) -> list[BenchRecord]:
    import tensorflow as tf

    adapter = TensorFlowAdapter()
    active_archs = architectures if architectures is not None else ARCHITECTURES
    records: list[BenchRecord] = []
    for arch in active_archs:
        if arch not in TF_BUILDERS:
            continue
        for batch in batches:
            model, inputs = TF_BUILDERS[arch](batch)
            try:
                prof = profile_model(model, inputs, adapter)
            except Exception:
                continue
            flop_multiplier = get_mode_flops_multiplier(mode)
            cost = prof.cost
            if flop_multiplier != 1.0:
                cost = CostEstimate(
                    flops=int(prof.cost.flops * flop_multiplier),
                    read_bytes=prof.cost.read_bytes,
                    write_bytes=prof.cost.write_bytes,
                    operations=prof.cost.operations,
                )
            fn = lambda *a: model(*a, training=False)
            # Baseline (eager)
            try:
                samp, peak_alloc, peak_res = _tf_block(fn, inputs, warmup, repeats)
                st = _stats(samp)
                meas = Measurement(
                    median_seconds=st["median"] / 1e3,
                    samples_seconds=tuple(s / 1e3 for s in samp),
                    peak_memory_bytes=peak_alloc,
                    allocated_memory_bytes=peak_alloc,
                    reserved_memory_bytes=peak_res,
                )
                gap = analyze_gap(
                    cost,
                    meas,
                    hardware,
                    operations=prof.operations,
                )
                records.append(
                    _make_bench_record(
                        "TensorFlow",
                        "baseline",
                        arch,
                        batch,
                        prof,
                        gap,
                        st,
                        scale=scale,
                        precision=precision,
                        mode=mode,
                        peak_alloc=peak_alloc,
                        peak_res=peak_res,
                    )
                )
            except Exception as exc:
                print(f"  TF baseline {arch} B={batch}: {exc}")
            # Optimised: tf.function (XLA)
            try:
                tf_fn = tf.function(fn, jit_compile=False)
                for _ in range(3):
                    tf_fn(*inputs)
                samp, peak_alloc, peak_res = _tf_block(tf_fn, inputs, warmup, repeats)
                st = _stats(samp)
                meas = Measurement(
                    median_seconds=st["median"] / 1e3,
                    samples_seconds=tuple(s / 1e3 for s in samp),
                    peak_memory_bytes=peak_alloc,
                    allocated_memory_bytes=peak_alloc,
                    reserved_memory_bytes=peak_res,
                )
                gap = analyze_gap(
                    cost,
                    meas,
                    hardware,
                    operations=prof.operations,
                )
                records.append(
                    _make_bench_record(
                        "TensorFlow",
                        "tf.function",
                        arch,
                        batch,
                        prof,
                        gap,
                        st,
                        scale=scale,
                        precision=precision,
                        mode=mode,
                        peak_alloc=peak_alloc,
                        peak_res=peak_res,
                    )
                )
            except Exception as exc:
                print(f"  TF tf.function {arch} B={batch}: {exc}")
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

RUNNERS = {
    "torch": ("PyTorch", eval_torch),
    "jax": ("JAX", eval_jax),
    "tensorflow": ("TensorFlow", eval_tensorflow),
}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for collect_data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        choices=["micro", "standard", "production"],
        default="standard",
        help="Model scale tier (default: standard)",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        default=False,
        help="Include legacy recurrent architectures (RNN, LSTM)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "int8"],
        default="fp32",
        help="Target compute precision (default: fp32)",
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "train_step", "backward_only"],
        default="inference",
        help="Operational execution mode (default: inference)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Fewer batch sizes and repeats for fast iteration"
    )
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--peak-flops", type=float, default=None)
    parser.add_argument("--memory-bandwidth", type=float, default=None)
    parser.add_argument(
        "--no-fx", action="store_true", help="Disable PyTorch FX graph tracing fallback"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append and merge records into existing benchmark_data.json rather than overwriting",
    )
    parser.add_argument(
        "--framework",
        choices=["all", "torch", "jax", "tensorflow"],
        default="all",
        help="Target framework to benchmark (default: all)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    warmup = 5 if args.quick else args.warmup
    repeats = 15 if args.quick else args.repeats
    batches = get_batch_sizes(scale=args.scale, quick=args.quick)
    active_archs = get_benchmarked_architectures(include_legacy=args.include_legacy)

    print("Detecting hardware…", flush=True)
    hardware, detection = detect_hardware(precision=args.precision)
    if args.peak_flops:
        hardware = HardwareSpec(
            hardware.name,
            args.peak_flops,
            hardware.memory_bandwidth,
            caches=hardware.caches,
        )
    if args.memory_bandwidth:
        hardware = HardwareSpec(
            hardware.name,
            hardware.peak_flops,
            args.memory_bandwidth,
            caches=hardware.caches,
        )

    hw_meta = {
        "name": hardware.name,
        "peak_flops": hardware.peak_flops,
        "memory_bandwidth": hardware.memory_bandwidth,
        "ridge_point": hardware.ridge_point,
        "source": detection.source,
        "measured_bw_gb_s": detection.measured_bandwidth_gb_s,
    }
    print(
        f"  {hardware.name}  {hardware.peak_flops / 1e12:.2f} TFLOP/s  {hardware.memory_bandwidth / 1e9:.0f} GB/s"
    )
    print(
        f"  Scale={args.scale}  Precision={args.precision}  Mode={args.mode}  Batches={batches}  warmup={warmup}  repeats={repeats}\n"
    )

    all_records: list[BenchRecord] = []
    runners_to_run = (
        RUNNERS if args.framework == "all" else {args.framework: RUNNERS[args.framework]}
    )
    for pkg, (label, runner) in runners_to_run.items():
        if importlib.util.find_spec(pkg) is None:
            print(f"[skip] {label} not installed")
            continue
        print(f"── {label} ─────────────────────────────────")
        if pkg == "torch":
            recs = runner(
                hardware,
                batches,
                warmup,
                repeats,
                use_fx=not args.no_fx,
                scale=args.scale,
                architectures=active_archs,
                precision=args.precision,
                mode=args.mode,
            )
        else:
            recs = runner(
                hardware,
                batches,
                warmup,
                repeats,
                scale=args.scale,
                architectures=active_archs,
                precision=args.precision,
                mode=args.mode,
            )
        all_records.extend(recs)
        for r in recs:
            extra = []
            if r.fused_efficiency is not None:
                extra.append(f"fused: {r.fused_efficiency:.1%}")
            if r.cache_resident and r.cache_name:
                extra.append(f"resident: {r.cache_name}")
            if r.top_layer_bottleneck:
                extra.append(f"top: {r.top_layer_bottleneck}")
            extra_str = f"  [{', '.join(extra)}]" if extra else ""
            print(
                f"  {r.variant:<12} {r.architecture:<14} B={r.batch:<4}  "
                f"{r.latency_median_ms:7.2f} ms  {r.roofline_efficiency:.1%}  {r.bottleneck}{extra_str}"
            )
        print()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_data.json"

    records_to_save: list[dict] = []
    if args.append and out_path.exists():
        try:
            old_payload = json.loads(out_path.read_text())
            record_map = {
                (
                    r.get("framework"),
                    r.get("variant"),
                    r.get("architecture"),
                    r.get("batch"),
                    r.get("scale", "standard"),
                    r.get("precision", "fp32"),
                    r.get("mode", "inference"),
                ): r
                for r in old_payload.get("records", [])
            }
            for r in all_records:
                d = asdict(r)
                k = (
                    d.get("framework"),
                    d.get("variant"),
                    d.get("architecture"),
                    d.get("batch"),
                    d.get("scale", "standard"),
                    d.get("precision", "fp32"),
                    d.get("mode", "inference"),
                )
                record_map[k] = d
            records_to_save = list(record_map.values())
        except Exception:
            records_to_save = [asdict(r) for r in all_records]
    else:
        records_to_save = [asdict(r) for r in all_records]

    payload = {"hardware": hw_meta, "records": records_to_save}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(records_to_save)} records → {out_path}")


if __name__ == "__main__":
    main()
