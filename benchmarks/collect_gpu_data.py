"""GPU benchmark: collect performance data across architectures × frameworks × batch sizes on GPU.

Writes results/benchmark_gpu_data.json when complete.

Key differences vs collect_data.py (CPU):
  - PyTorch: tensors/model moved to CUDA; CUDA events used for precise kernel timing
  - JAX: arrays placed on GPU device; XLA JIT targets GPU backend
  - TensorFlow: GPU device explicitly selected; tf.function + XLA JIT compile enabled
  - Larger batch sizes are swept (GPU favours high arithmetic intensity)
  - CUDA synchronisation barriers ensure only kernel time is measured
  - Gracefully falls back to CPU if no GPU accelerator is found

New in v0.3.0:
  - --crossover: sweeps a fine batch-size grid (1..1024) to find where GPU first beats CPU.
    Reads the CPU baseline from benchmarks/results/benchmark_data.json if present.
  - JAX MPS driver check: detects whether jax-metal or jax-mps is installed and warns
    clearly if JAX falls back to CPU on Apple Silicon.
  - CUDA quality-of-life: enables TF32 matmul and cuDNN benchmark mode on Ampere+ GPUs.

Run:
    python benchmarks/collect_gpu_data.py [--quick]
    python benchmarks/collect_gpu_data.py --device cuda  # explicit CUDA
    python benchmarks/collect_gpu_data.py --device mps   # Apple Silicon GPU
    python benchmarks/collect_gpu_data.py --crossover    # find GPU break-even batch size
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

# Ensure repo root and src/ are importable when run from any directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.models import (
    get_available_models,
    get_model,
)

from neural_cost import (
    HardwareSpec,
    Measurement,
    analyze_gap,
    analyze_memory_gap,
    estimate_fused_operations,
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

# GPU benefits most from larger batches (more parallelism).
BATCH_SIZES = [8, 32, 128, 512]
QUICK_BATCH_SIZES = [32, 256]

# Fine-grained grid for GPU-vs-CPU crossover analysis.
# We sweep 1 → 1024 in a log scale to find the exact break-even point.
CROSSOVER_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Path to CPU benchmark JSON (written by collect_data.py).
_CPU_DATA_FILE = Path(__file__).parent / "results" / "benchmark_data.json"


# ---------------------------------------------------------------------------
# Result record (identical schema to CPU benchmark for easy comparison)
# ---------------------------------------------------------------------------


@dataclass
class BenchRecord:
    framework: str
    variant: str  # "baseline" | "compiled" | "jit" | "tf.function"
    architecture: str
    batch: int
    device: str  # "cuda:0" / "mps" / "cpu-fallback" / "gpu" / etc.
    flops: int
    param_bytes: int
    total_bytes: int
    arith_intensity: float
    latency_median_ms: float
    latency_mean_ms: float
    latency_stddev_ms: float
    latency_cv_pct: float
    latency_p95_ms: float
    roofline_efficiency: float
    achieved_gflops: float
    achieved_gbw: float
    bottleneck: str
    scale: str = "standard"
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


def _make_gpu_bench_record(
    framework: str,
    variant: str,
    arch: str,
    batch: int,
    device: str,
    prof: Any,
    gap: Any,
    st: dict[str, float],
    peak_alloc: int | None = None,
    peak_res: int | None = None,
    scale: str = "standard",
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

    return BenchRecord(
        framework=framework,
        variant=variant,
        architecture=arch,
        batch=batch,
        device=device,
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
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _cuda_event_block(
    fn: Callable,
    args: tuple,
    warmup: int,
    repeats: int,
    device: Any,
) -> tuple[list[float], int | None, int | None]:
    """Use CUDA events for sub-millisecond GPU kernel timing and collect allocator telemetry."""
    import torch

    # Reset peak stats before run
    torch.cuda.reset_peak_memory_stats(device)
    # Warmup — lets CUDA driver warm up and torch.compile finish tracing.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(end))  # milliseconds

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return samples, peak_allocated, peak_reserved


def _mps_block(
    fn: Callable,
    args: tuple,
    warmup: int,
    repeats: int,
) -> tuple[list[float], int | None, int | None]:
    """Apple MPS has no CUDA events — use perf_counter with MPS sync and allocator telemetry."""
    import torch

    for _ in range(warmup):
        fn(*args)
    torch.mps.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e6)

    peak_allocated = int(torch.mps.current_allocated_memory())
    peak_reserved = int(torch.mps.driver_allocated_memory())
    return samples, peak_allocated, peak_reserved


def _cpu_block(
    fn: Callable,
    args: tuple,
    warmup: int,
    repeats: int,
) -> tuple[list[float], int | None, int | None]:
    import torch

    for _ in range(warmup):
        fn(*args)
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append((time.perf_counter_ns() - t0) / 1e6)

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
    samples: list[float] = []
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
    samples: list[float] = []
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
# Detect best available torch device
# ---------------------------------------------------------------------------


def _detect_torch_device(requested: str | None) -> tuple[Any, str]:
    """Return (torch.device, device_label) for the best available GPU."""
    import torch

    if requested:
        dev = torch.device(requested)
        return dev, str(dev)
    if torch.cuda.is_available():
        dev = torch.device("cuda", 0)
        name = torch.cuda.get_device_name(0)
        return dev, f"cuda:0 ({name})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps (Apple Silicon GPU)"
    return torch.device("cpu"), "cpu (no GPU found)"


# ---------------------------------------------------------------------------
# PyTorch GPU model builders
# ---------------------------------------------------------------------------


def _torch_ff_dnn(batch: int, device: Any):
    import torch

    model = (
        torch.nn.Sequential(
            torch.nn.Linear(784, EMBED_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(EMBED_DIM),
            torch.nn.Linear(EMBED_DIM, EMBED_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(EMBED_DIM),
            torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
        )
        .eval()
        .to(device)
    )
    x = torch.randn(batch, 784, device=device)
    return model, (x,)


def _torch_cnn(batch: int, device: Any):
    import torch

    model = (
        torch.nn.Sequential(
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
        )
        .eval()
        .to(device)
    )
    x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=device)
    return model, (x,)


def _torch_rnn(batch: int, device: Any):
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.RNN(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1])

    m = M().eval().to(device)
    x = torch.randn(batch, 32, EMBED_DIM, device=device)
    return m, (x,)


def _torch_lstm(batch: int, device: Any):
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    m = M().eval().to(device)
    x = torch.randn(batch, 32, EMBED_DIM, device=device)
    return m, (x,)


def _torch_transformer(batch: int, device: Any):
    import torch

    class M(torch.nn.Module):
        def __init__(self):
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

        def forward(self, x):
            xt = x.permute(1, 0, 2)
            a1, _ = self.attn1(xt, xt, xt)
            xt = self.norm1a(xt + a1)
            xt = self.norm1b(xt + self.ff1b(torch.relu(self.ff1(xt))))
            a2, _ = self.attn2(xt, xt, xt)
            xt = self.norm2a(xt + a2)
            xt = self.norm2b(xt + self.ff2b(torch.relu(self.ff2(xt))))
            return self.head(xt.mean(0))

    m = M().eval().to(device)
    x = torch.randn(batch, 32, EMBED_DIM, device=device)
    return m, (x,)


def _torch_convnext(batch: int, device: Any):
    model, inputs = get_model("ConvNeXt", framework="torch", batch=batch)
    return model.to(device), tuple(x.to(device) for x in inputs)


def _torch_vit(batch: int, device: Any):
    model, inputs = get_model("ViT", framework="torch", batch=batch)
    return model.to(device), tuple(x.to(device) for x in inputs)


TORCH_GPU_BUILDERS = {
    "FF DNN": _torch_ff_dnn,
    "ConvNeXt": _torch_convnext,
    "ViT": _torch_vit,
    "Transformer": _torch_transformer,
    "CNN": _torch_cnn,
    "RNN": _torch_rnn,
    "LSTM": _torch_lstm,
}

ARCHITECTURES = ["FF DNN", "ConvNeXt", "ViT", "Transformer", "CNN", "RNN", "LSTM"]
FRAMEWORKS = ["PyTorch", "JAX", "TensorFlow"]


# ---------------------------------------------------------------------------
# JAX GPU model builders (identical math, jnp arrays on GPU device)
# ---------------------------------------------------------------------------


def _jax_ff_dnn_gpu(batch: int, jax_device: Any):
    import jax
    import jax.numpy as jnp

    def _put(x):
        return jax.device_put(x, jax_device)

    w1 = _put(jnp.ones((784, EMBED_DIM)))
    w2 = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    w3 = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))

    def model(x, _w1=w1, _w2=w2, _w3=w3):
        return jnp.tanh(jnp.tanh(x @ _w1) @ _w2) @ _w3

    return model, (_put(jnp.ones((batch, 784))), w1, w2, w3)


def _jax_cnn_gpu(batch: int, jax_device: Any):
    import jax
    import jax.lax as lax
    import jax.numpy as jnp

    def _put(x):
        return jax.device_put(x, jax_device)

    k1 = _put(jnp.ones((EMBED_DIM // 2, 3, 3, 3)))
    k2 = _put(jnp.ones((EMBED_DIM, EMBED_DIM // 2, 3, 3)))
    wfc = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))

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

    return model, (_put(jnp.ones((batch, 3, IMG_SIZE, IMG_SIZE))), k1, k2, wfc)


def _jax_rnn_gpu(batch: int, jax_device: Any):
    import jax
    import jax.numpy as jnp

    T = 32

    def _put(x):
        return jax.device_put(x, jax_device)

    wih = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    whh = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wfc = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))

    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], EMBED_DIM))
        for t in range(T):
            h = jnp.tanh(x[:, t] @ _wih + h @ _whh)
        return h @ _wfc

    return model, (_put(jnp.ones((batch, T, EMBED_DIM))), wih, whh, wfc)


def _jax_lstm_gpu(batch: int, jax_device: Any):
    import jax
    import jax.numpy as jnp

    T, G = 32, 4

    def _put(x):
        return jax.device_put(x, jax_device)

    wih = _put(jnp.ones((EMBED_DIM, G * EMBED_DIM)))
    whh = _put(jnp.ones((EMBED_DIM, G * EMBED_DIM)))
    wfc = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))

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

    return model, (_put(jnp.ones((batch, T, EMBED_DIM))), wih, whh, wfc)


def _jax_transformer_gpu(batch: int, jax_device: Any):
    import jax
    import jax.numpy as jnp

    T = 32
    hd = EMBED_DIM // NUM_HEADS

    def _put(x):
        return jax.device_put(x, jax_device)

    wq = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wk = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wv = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wo = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wf1 = _put(jnp.ones((EMBED_DIM, EMBED_DIM * 4)))
    wf2 = _put(jnp.ones((EMBED_DIM * 4, EMBED_DIM)))
    wfc = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))
    scale = hd**-0.5

    def attn(x, wq_, wk_, wv_, wo_):
        q, k, v = x @ wq_, x @ wk_, x @ wv_
        s = (q * scale) @ k.transpose(0, 2, 1)
        w = jnp.exp(s) / jnp.exp(s).sum(-1, keepdims=True)
        return (w @ v) @ wo_

    def model(x, _wq=wq, _wk=wk, _wv=wv, _wo=wo, _wf1=wf1, _wf2=wf2, _wfc=wfc):
        x = x + attn(x, _wq, _wk, _wv, _wo)
        x = x + jnp.tanh(x @ _wf1) @ _wf2
        return x.mean(1) @ _wfc

    return model, (_put(jnp.ones((batch, T, EMBED_DIM))), wq, wk, wv, wo, wf1, wf2, wfc)


def _jax_convnext_gpu(batch: int, jax_device: Any):
    import jax

    model, inputs = get_model("ConvNeXt", framework="jax", batch=batch)
    return model, tuple(jax.device_put(x, jax_device) for x in inputs)


def _jax_vit_gpu(batch: int, jax_device: Any):
    import jax

    model, inputs = get_model("ViT", framework="jax", batch=batch)
    return model, tuple(jax.device_put(x, jax_device) for x in inputs)


JAX_GPU_BUILDERS = {
    "FF DNN": _jax_ff_dnn_gpu,
    "ConvNeXt": _jax_convnext_gpu,
    "ViT": _jax_vit_gpu,
    "Transformer": _jax_transformer_gpu,
    "CNN": _jax_cnn_gpu,
    "RNN": _jax_rnn_gpu,
    "LSTM": _jax_lstm_gpu,
}


# ---------------------------------------------------------------------------
# TensorFlow GPU model builders
# ---------------------------------------------------------------------------


def _tf_ff_dnn_gpu(batch: int, tf_device: str):
    import tensorflow as tf

    with tf.device(tf_device):
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


def _tf_cnn_gpu(batch: int, tf_device: str):
    import tensorflow as tf

    with tf.device(tf_device):
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


def _tf_rnn_gpu(batch: int, tf_device: str):
    import tensorflow as tf

    with tf.device(tf_device):
        inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
        x = tf.keras.layers.GRU(EMBED_DIM, return_sequences=True)(inp)
        x = tf.keras.layers.GRU(EMBED_DIM)(x)
        out = tf.keras.layers.Dense(NUM_CLASSES)(x)
        m = tf.keras.Model(inp, out)
        xi = tf.ones((batch, 32, EMBED_DIM))
        m(xi)
    return m, (xi,)


def _tf_lstm_gpu(batch: int, tf_device: str):
    import tensorflow as tf

    with tf.device(tf_device):
        inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
        x = tf.keras.layers.LSTM(EMBED_DIM, return_sequences=True)(inp)
        x = tf.keras.layers.LSTM(EMBED_DIM)(x)
        out = tf.keras.layers.Dense(NUM_CLASSES)(x)
        m = tf.keras.Model(inp, out)
        xi = tf.ones((batch, 32, EMBED_DIM))
        m(xi)
    return m, (xi,)


def _tf_transformer_gpu(batch: int, tf_device: str):
    import tensorflow as tf

    with tf.device(tf_device):
        inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
        x = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(
            inp, inp
        )
        x = tf.keras.layers.LayerNormalization()(inp + x)
        ff = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x)
        ff = tf.keras.layers.Dense(EMBED_DIM)(ff)
        x = tf.keras.layers.LayerNormalization()(x + ff)
        x2 = tf.keras.layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS
        )(x, x)
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


TF_GPU_BUILDERS = {
    "FF DNN": _tf_ff_dnn_gpu,
    "CNN": _tf_cnn_gpu,
    "RNN": _tf_rnn_gpu,
    "LSTM": _tf_lstm_gpu,
    "Transformer": _tf_transformer_gpu,
}


# ---------------------------------------------------------------------------
# GPU device detection helpers
# ---------------------------------------------------------------------------


def _detect_jax_gpu() -> tuple[Any | None, str]:
    """Return (jax.Device, label) for the first available GPU/accelerator."""
    try:
        import jax

        # Try GPU/accelerators first: "gpu", "mps", "tpu", fall back to CPU
        for backend in ("gpu", "mps", "tpu", "cpu"):
            try:
                devs = jax.devices(backend)
                if devs:
                    dev = devs[0]
                    return dev, f"{backend}:{dev.id} ({dev.device_kind})"
            except (RuntimeError, ValueError):
                pass
        return None, "cpu-fallback"
    except ImportError:
        return None, "jax-not-installed"


def _detect_tf_gpu() -> tuple[str, str]:
    """Return (tf_device_str, label) for the first available GPU."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return "/GPU:0", f"GPU:0 ({gpus[0].name})"
        return "/CPU:0", "cpu-fallback (no TF GPU)"
    except ImportError:
        return "/CPU:0", "tf-not-installed"


# ---------------------------------------------------------------------------
# Per-framework GPU evaluation
# ---------------------------------------------------------------------------


def _make_gap(
    cost: Any,
    st: dict,
    hardware: HardwareSpec,
    operations: Any = None,
    peak_alloc: int | None = None,
    peak_res: int | None = None,
) -> Any:
    """Build a measurement object compatible with analyze_gap."""
    meas = Measurement(
        median_seconds=st["median"] / 1e3,
        samples_seconds=tuple(s / 1e3 for s in [st["median"]] * 2),
        peak_memory_bytes=peak_alloc,
        allocated_memory_bytes=peak_alloc,
        reserved_memory_bytes=peak_res,
    )
    return analyze_gap(
        cost,
        meas,
        hardware,
        operations=operations,
    )


def eval_torch_gpu(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    torch_device: Any,
    device_label: str,
    use_fx: bool = True,
    architectures: list[str] | None = None,
) -> list[BenchRecord]:
    import torch

    if use_fx:
        try:
            adapter: FrameworkAdapter = TorchFxAdapter()
        except Exception:
            adapter = TorchAdapter()
    else:
        adapter = TorchAdapter()

    records: list[BenchRecord] = []

    is_cuda = str(torch_device).startswith("cuda")
    is_mps = str(torch_device) == "mps"

    active_archs = architectures if architectures is not None else list(TORCH_GPU_BUILDERS.keys())

    for arch in active_archs:
        if arch not in TORCH_GPU_BUILDERS:
            continue
        for batch in batches:
            try:
                model, inputs = TORCH_GPU_BUILDERS[arch](batch, torch_device)
            except Exception as exc:
                print(f"  PyTorch GPU builder {arch} B={batch}: {exc}")
                continue

            # Profile on CPU copies (adapter hooks work on any device)
            try:
                cpu_model, cpu_inputs = TORCH_GPU_BUILDERS[arch](batch, torch.device("cpu"))
                prof = profile_model(cpu_model, cpu_inputs, adapter)
                cost = prof.cost
            except Exception:
                continue

            def _time(fn: Callable, args: tuple) -> tuple[list[float], int | None, int | None]:
                if is_cuda:
                    return _cuda_event_block(fn, args, warmup, repeats, torch_device)
                elif is_mps:
                    return _mps_block(fn, args, warmup, repeats)
                else:
                    return _cpu_block(fn, args, warmup, repeats)


            # Baseline (eager GPU)
            try:
                samp, peak_alloc, peak_res = _time(model, inputs)
                st = _stats(samp)
                gap = _make_gap(
                    cost,
                    st,
                    hardware,
                    operations=prof.operations,
                    peak_alloc=peak_alloc,
                    peak_res=peak_res,
                )
                records.append(
                    _make_gpu_bench_record(
                        "PyTorch",
                        "baseline",
                        arch,
                        batch,
                        device_label,
                        prof,
                        gap,
                        st,
                        peak_alloc,
                        peak_res,
                    )
                )
            except Exception as exc:
                print(f"  PyTorch GPU baseline {arch} B={batch}: {exc}")

            # Optimised: torch.compile (Inductor backend, GPU)
            try:
                compiled = torch.compile(model)
                # Trigger compilation
                if is_cuda:
                    for _ in range(max(3, warmup)):
                        compiled(*inputs)
                    torch.cuda.synchronize(torch_device)
                elif is_mps:
                    for _ in range(max(3, warmup)):
                        compiled(*inputs)
                    torch.mps.synchronize()
                else:
                    for _ in range(max(3, warmup)):
                        compiled(*inputs)
                samp, peak_alloc, peak_res = _time(compiled, inputs)
                st = _stats(samp)
                gap = _make_gap(
                    cost,
                    st,
                    hardware,
                    operations=prof.operations,
                    peak_alloc=peak_alloc,
                    peak_res=peak_res,
                )
                records.append(
                    _make_gpu_bench_record(
                        "PyTorch",
                        "compiled",
                        arch,
                        batch,
                        device_label,
                        prof,
                        gap,
                        st,
                        peak_alloc,
                        peak_res,
                    )
                )
            except Exception as exc:
                print(f"  PyTorch GPU compiled {arch} B={batch}: {exc}")

    return records


def eval_jax_gpu(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    jax_device: Any,
    device_label: str,
    architectures: list[str] | None = None,
) -> list[BenchRecord]:
    import jax

    adapter = JaxAdapter()
    records: list[BenchRecord] = []

    active_archs = architectures if architectures is not None else list(JAX_GPU_BUILDERS.keys())

    for arch in active_archs:
        if arch not in JAX_GPU_BUILDERS:
            continue
        for batch in batches:
            try:
                model, inputs = JAX_GPU_BUILDERS[arch](batch, jax_device)
            except Exception as exc:
                print(f"  JAX GPU builder {arch} B={batch}: {exc}")
                continue

            # Static profile uses CPU-side JAX adapter (shape-only)
            try:
                import jax.numpy as jnp

                cpu_model, cpu_inputs = JAX_GPU_BUILDERS[arch](batch, jax.devices("cpu")[0])
                prof = profile_model(cpu_model, cpu_inputs, adapter)
                cost = prof.cost
                mem = prof.memory
            except Exception:
                continue


            # Baseline (eager on GPU device)
            try:
                samp, peak_alloc, peak_res = _jax_block(model, inputs, warmup, repeats)
                st = _stats(samp)
                gap = _make_gap(
                    cost,
                    st,
                    hardware,
                    operations=prof.operations,
                    peak_alloc=peak_alloc,
                    peak_res=peak_res,
                )
                records.append(
                    _make_gpu_bench_record(
                        "JAX",
                        "baseline",
                        arch,
                        batch,
                        device_label,
                        prof,
                        gap,
                        st,
                        peak_alloc,
                        peak_res,
                    )
                )
            except Exception as exc:
                print(f"  JAX GPU baseline {arch} B={batch}: {exc}")

            # Optimised: jax.jit
            try:
                jit_model = jax.jit(model)
                # Trigger compilation
                wait = lambda v: [
                    leaf.block_until_ready()
                    for leaf in jax.tree.leaves(v)
                    if hasattr(leaf, "block_until_ready")
                ]
                wait(jit_model(*inputs))
                samp, peak_alloc, peak_res = _jax_block(jit_model, inputs, warmup, repeats)
                st = _stats(samp)
                gap = _make_gap(
                    cost,
                    st,
                    hardware,
                    operations=prof.operations,
                    peak_alloc=peak_alloc,
                    peak_res=peak_res,
                )
                records.append(
                    _make_gpu_bench_record(
                        "JAX", "jit", arch, batch, device_label, prof, gap, st, peak_alloc, peak_res
                    )
                )
            except Exception as exc:
                print(f"  JAX GPU jit {arch} B={batch}: {exc}")

    return records


def eval_tensorflow_gpu(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    tf_device: str,
    device_label: str,
    architectures: list[str] | None = None,
) -> list[BenchRecord]:
    import tensorflow as tf

    adapter = TensorFlowAdapter()
    records: list[BenchRecord] = []

    active_archs = architectures if architectures is not None else list(TF_GPU_BUILDERS.keys())

    for arch in active_archs:
        if arch not in TF_GPU_BUILDERS:
            continue
        for batch in batches:
            try:
                model, inputs = TF_GPU_BUILDERS[arch](batch, tf_device)
            except Exception as exc:
                print(f"  TF GPU builder {arch} B={batch}: {exc}")
                continue

            try:
                prof = profile_model(model, inputs, adapter)
                cost = prof.cost
            except Exception:
                continue

            fn = lambda *a: model(*a, training=False)

            # Baseline (eager GPU)
            try:
                samp, peak_alloc, peak_res = _tf_block(fn, inputs, warmup, repeats)
                st = _stats(samp)
                gap = _make_gap(
                    cost,
                    st,
                    hardware,
                    operations=prof.operations,
                    peak_alloc=peak_alloc,
                    peak_res=peak_res,
                )
                records.append(
                    _make_gpu_bench_record(
                        "TensorFlow",
                        "baseline",
                        arch,
                        batch,
                        device_label,
                        prof,
                        gap,
                        st,
                        peak_alloc,
                        peak_res,
                    )
                )
            except Exception as exc:
                print(f"  TF GPU baseline {arch} B={batch}: {exc}")

            # Optimised: tf.function + XLA JIT compile
            try:
                # Enable XLA JIT for GPU — significant gains on matmul/conv
                tf_fn = tf.function(fn, jit_compile=True)
                for _ in range(3):
                    tf_fn(*inputs)
                samp, peak_alloc, peak_res = _tf_block(tf_fn, inputs, warmup, repeats)
                st = _stats(samp)
                gap = _make_gap(
                    cost,
                    st,
                    hardware,
                    operations=prof.operations,
                    peak_alloc=peak_alloc,
                    peak_res=peak_res,
                )
                records.append(
                    _make_gpu_bench_record(
                        "TensorFlow",
                        "tf.function+XLA",
                        arch,
                        batch,
                        device_label,
                        prof,
                        gap,
                        st,
                        peak_alloc,
                        peak_res,
                    )
                )
            except Exception as exc:
                # Fallback to graph mode without XLA if XLA compile fails
                try:
                    tf_fn = tf.function(fn, jit_compile=False)
                    for _ in range(3):
                        tf_fn(*inputs)
                    samp, peak_alloc, peak_res = _tf_block(tf_fn, inputs, warmup, repeats)
                    st = _stats(samp)
                    gap = _make_gap(
                        cost,
                        st,
                        hardware,
                        operations=prof.operations,
                        peak_alloc=peak_alloc,
                        peak_res=peak_res,
                    )
                    records.append(
                        _make_gpu_bench_record(
                            "TensorFlow",
                            "tf.function",
                            arch,
                            batch,
                            device_label,
                            prof,
                            gap,
                            st,
                            peak_alloc,
                            peak_res,
                        )
                    )
                except Exception as exc2:
                    print(f"  TF GPU tf.function {arch} B={batch}: {exc2}")

    return records


# ---------------------------------------------------------------------------
# JAX MPS / Metal driver diagnostics
# ---------------------------------------------------------------------------


def _check_jax_mps_driver() -> None:
    """Warn clearly when JAX is running on CPU instead of Apple Silicon GPU.

    On Apple Silicon the user must install one of:
      - ``jax-metal``  (official Apple plugin):    pip install jax-metal
      - ``jax-mps``    (community MLX backend):    pip install jax-mps

    Without a plugin, JAX silently runs on CPU even on MPS-capable machines.
    This function prints a diagnostic so the benchmark output is honest about
    which device JAX is actually using.
    """
    try:
        import importlib

        import jax

        platforms = [d.platform for d in jax.devices()]
        if any(p in ("gpu", "tpu", "metal", "mps") for p in platforms):
            return  # JAX has a real accelerator — nothing to do

        # JAX is on CPU: check whether a Metal/MPS plugin is installed at all.
        has_jax_metal = importlib.util.find_spec("jax_metal") is not None
        has_jax_mps = importlib.util.find_spec("jax_mps") is not None

        print()
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  WARNING: JAX is running on CPU, not on the MPS/Metal GPU.  ║")
        if not has_jax_metal and not has_jax_mps:
            print("  ║  No JAX GPU plugin detected.  Install one of:               ║")
            print("  ║    pip install jax-metal          # Official Apple plugin    ║")
            print("  ║    pip install jax-mps            # Community MLX backend   ║")
        elif has_jax_metal:
            print("  ║  jax-metal is installed but JAX is still on CPU.            ║")
            print("  ║  Check jax/jaxlib version compatibility:                    ║")
            print("  ║    pip install --upgrade jax jaxlib jax-metal               ║")
            print("  ║  Or set: JAX_PLATFORMS=metal python ...                     ║")
        elif has_jax_mps:
            print("  ║  jax-mps is installed but JAX is still on CPU.              ║")
            print("  ║  Try: JAX_PLATFORMS=mps python ...                          ║")
            print("  ║  Or: pip install --upgrade jax-mps                          ║")
        print("  ║  JAX benchmark results below reflect CPU performance only.   ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        print()
    except ImportError:
        pass  # JAX not installed — handled elsewhere.


# ---------------------------------------------------------------------------
# CUDA quality-of-life setup
# ---------------------------------------------------------------------------


def _configure_cuda_for_benchmark(device: Any) -> None:
    """Enable TF32 and cuDNN benchmark mode for realistic throughput numbers.

    These settings match production defaults on Ampere+ (A100, RTX 3090+, etc.)
    and are safe for benchmarking.  They have no effect on MPS or CPU.
    """
    try:
        import torch

        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            return
        # TF32 matmul (enabled by default since PyTorch 1.11, but be explicit)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cuDNN benchmark mode: selects the fastest convolution algorithm
        # on the first run and caches it.  Adds ~30 s to the first iteration
        # (already covered by our warmup) but improves steady-state throughput.
        torch.backends.cudnn.benchmark = True
        print(
            f"  CUDA TF32={torch.backends.cuda.matmul.allow_tf32}  "
            f"cuDNN-benchmark={torch.backends.cudnn.benchmark}"
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GPU-vs-CPU crossover analysis
# ---------------------------------------------------------------------------


def _load_cpu_latencies() -> dict[tuple[str, str, int], float] | None:
    """Load CPU benchmark results from the companion JSON file, if present.

    Returns a dict keyed by (framework, architecture, batch) -> latency_ms,
    or None if the CPU data file is not found.
    """
    if not _CPU_DATA_FILE.exists():
        return None
    try:
        raw = json.loads(_CPU_DATA_FILE.read_text())
        out: dict[tuple[str, str, int], float] = {}
        best_variant = {"PyTorch": "compiled", "JAX": "jit", "TensorFlow": "tf.function"}
        fallback = {"PyTorch": "baseline", "JAX": "baseline", "TensorFlow": "baseline"}
        for r in raw.get("records", []):
            fw, arch, batch = r["framework"], r["architecture"], r["batch"]
            key = (fw, arch, batch)
            preferred = best_variant.get(fw, "baseline")
            # Accept the preferred variant, or fall back to baseline when preferred
            # isn't in the data for this key yet.
            if r["variant"] == preferred or (
                key not in out and r["variant"] in (preferred, fallback.get(fw, "baseline"))
            ):
                out[key] = r["latency_median_ms"]
        return out or None
    except Exception as exc:
        print(f"  [crossover] could not read CPU data: {exc}")
        return None


def compute_crossover(
    gpu_records: list[BenchRecord],
    cpu_latencies: dict[tuple[str, str, int], float],
) -> dict:
    """Find the first batch size where GPU latency drops below CPU latency.

    Returns a nested dict::

        {framework: {architecture: {
            'crossover_batch': int | None,
            'gpu_wins_any': bool,
            'data': [{batch, gpu_ms, cpu_ms, gpu_faster, variant}, ...]
        }}}
    """
    result: dict = {}
    active_archs = list(dict.fromkeys(r.architecture for r in gpu_records)) or ARCHITECTURES
    for fw in FRAMEWORKS:
        result[fw] = {}
        for arch in active_archs:
            # Collect GPU data points (best variant, sorted by batch)
            gpu_recs = sorted(
                [r for r in gpu_records if r.framework == fw and r.architecture == arch],
                key=lambda r: r.batch,
            )
            points = []
            for r in gpu_recs:
                cpu_ms = cpu_latencies.get((fw, arch, r.batch))
                gpu_faster = (cpu_ms is not None) and (r.latency_median_ms < cpu_ms)
                points.append(
                    {
                        "batch": r.batch,
                        "gpu_ms": r.latency_median_ms,
                        "cpu_ms": cpu_ms,
                        "gpu_faster": gpu_faster,
                        "variant": r.variant,
                    }
                )

            # Find the first batch at which GPU beats CPU
            crossover_batch = next((pt["batch"] for pt in points if pt["gpu_faster"]), None)

            result[fw][arch] = {
                "crossover_batch": crossover_batch,
                "gpu_wins_any": crossover_batch is not None,
                "data": points,
            }
    return result


def print_crossover_table(crossover: dict) -> None:
    """Print a human-readable GPU-vs-CPU crossover summary table."""
    print()
    print("  ┌─ GPU-vs-CPU Crossover Analysis ──────────────────────────────────────────")
    print("  │  Smallest batch size at which each architecture runs faster on GPU than CPU.")
    print("  │  'never' means GPU did not win at any tested batch size.")
    print("  │")
    header = f"  │  {'Architecture':<14} {'Framework':<12} {'Crossover batch':>16}  Notes"
    print(header)
    print("  │  " + "─" * (len(header) - 5))
    all_archs = sorted({a for fw_data in crossover.values() for a in fw_data.keys()})
    for fw in FRAMEWORKS:
        for arch in all_archs:
            info = crossover.get(fw, {}).get(arch, {})
            cb = info.get("crossover_batch")
            tag = f"≥ batch={cb}" if cb else "never"
            note = ""
            if cb:
                pt = next((p for p in info.get("data", []) if p["batch"] == cb), None)
                if pt and pt.get("cpu_ms") and pt.get("gpu_ms"):
                    ratio = pt["cpu_ms"] / pt["gpu_ms"]
                    note = f"GPU {ratio:.2f}× faster at batch={cb}"
            print(f"  │  {arch:<14} {fw:<12} {tag:>16}  {note}")
    print("  └──────────────────────────────────────────────────────────────────────────")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Fewer batches/repeats for fast iteration"
    )
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device override: 'cuda', 'cuda:1', 'mps', 'cpu'",
    )
    parser.add_argument(
        "--peak-flops", type=float, default=None, help="Override peak FLOP/s (e.g. 312e12 for A100)"
    )
    parser.add_argument(
        "--memory-bandwidth",
        type=float,
        default=None,
        help="Override memory bandwidth in bytes/s (e.g. 2.0e12 for A100)",
    )
    parser.add_argument(
        "--crossover",
        action="store_true",
        help=(
            "Sweep a fine batch-size grid (1..1024) to find where each architecture "
            "first runs faster on GPU than on CPU.  Reads CPU baseline from "
            "benchmarks/results/benchmark_data.json if present."
        ),
    )
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
        "--append",
        action="store_true",
        default=False,
        help="Append and merge records into existing benchmark_gpu_data.json rather than overwriting",
    )
    parser.add_argument(
        "--framework",
        choices=["all", "torch", "jax", "tensorflow"],
        default="all",
        help="Target framework to benchmark (default: all)",
    )
    parser.add_argument(
        "--no-fx",
        action="store_true",
        help="Disable PyTorch FX graph tracing fallback",
    )
    args = parser.parse_args()

    warmup = 5 if args.quick else args.warmup
    repeats = 15 if args.quick else args.repeats
    if args.crossover:
        batches = CROSSOVER_BATCH_SIZES
        print("Crossover mode: sweeping batch sizes", batches)
    else:
        batches = QUICK_BATCH_SIZES if args.quick else BATCH_SIZES

    active_archs = get_available_models(include_legacy=args.include_legacy)

    # ------------------------------------------------------------------
    # Detect GPU and build a HardwareSpec for the GPU if one is found.
    # The CPU-targeted detect_hardware() is used as the base; GPU specs
    # can be supplied on the command line or will be estimated from the
    # GPU name when a CUDA device is available.
    # ------------------------------------------------------------------
    print("Detecting hardware…", flush=True)
    hardware, detection = detect_hardware()

    # Attempt to read better GPU specs from nvidia-smi / torch.cuda
    gpu_name = hardware.name
    peak_flops = hardware.peak_flops
    mem_bw = hardware.memory_bandwidth

    if importlib.util.find_spec("torch") is not None:
        import torch

        torch_dev, torch_label = _detect_torch_device(args.device)
        if str(torch_dev).startswith("cuda") and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch_dev)
            gpu_name = props.name
            # SM count × 128 CUDA cores/SM (Ampere) × 2 FP32 ops/clock × clock Hz
            # This is an approximation; --peak-flops can override.
            clock_hz = props.max_clock_rate * 1000  # kHz → Hz
            n_sm = props.multi_processor_count
            peak_flops = n_sm * 128 * 2 * clock_hz
            mem_bw = props.memory_bandwidth  # bytes/s (PyTorch 2.x)
            print(f"  GPU detected: {gpu_name}")
            print(
                f"  SMs={n_sm}  clock={clock_hz / 1e9:.2f} GHz  "
                f"est. peak={peak_flops / 1e12:.1f} TFLOP/s  "
                f"bw={mem_bw / 1e9:.0f} GB/s"
            )
            _configure_cuda_for_benchmark(torch_dev)
        elif str(torch_dev) == "mps":
            gpu_name = torch_label
            # Apple Silicon — use detected values (chip table has GPU specs)
            print(f"  Device: {torch_label}")
        else:
            torch_label = f"cpu-fallback ({hardware.name})"
            print(f"  No GPU found — running on {hardware.name}")
    else:
        torch_dev, torch_label = None, "torch-not-installed"

    # JAX MPS driver diagnostics (runs even when torch is unavailable)
    _check_jax_mps_driver()

    if args.peak_flops:
        peak_flops = args.peak_flops
    if args.memory_bandwidth:
        mem_bw = args.memory_bandwidth

    gpu_hardware = HardwareSpec(gpu_name, peak_flops, mem_bw, caches=hardware.caches)

    hw_meta = {
        "name": gpu_hardware.name,
        "peak_flops": gpu_hardware.peak_flops,
        "memory_bandwidth": gpu_hardware.memory_bandwidth,
        "ridge_point": gpu_hardware.ridge_point,
        "source": detection.source,
        "measured_bw_gb_s": detection.measured_bandwidth_gb_s,
        "torch_device": torch_label,
    }
    print(
        f"\n  {gpu_hardware.name}  "
        f"{gpu_hardware.peak_flops / 1e12:.2f} TFLOP/s  "
        f"{gpu_hardware.memory_bandwidth / 1e9:.0f} GB/s"
    )
    print(f"  Batches={batches}  warmup={warmup}  repeats={repeats}\n")

    all_records: list[BenchRecord] = []

    def _print_recs(recs: list[BenchRecord]) -> None:
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
                f"  {r.variant:<16} {r.architecture:<14} B={r.batch:<4}  "
                f"{r.latency_median_ms:7.2f} ms  {r.roofline_efficiency:.1%}  {r.bottleneck}{extra_str}"
            )

    run_torch = args.framework in ("all", "torch")
    run_jax = args.framework in ("all", "jax")
    run_tf = args.framework in ("all", "tensorflow")

    # PyTorch GPU
    if run_torch and importlib.util.find_spec("torch") is not None and torch_dev is not None:
        print(f"── PyTorch ({torch_label}) ─────────────────────────────────")
        recs = eval_torch_gpu(
            gpu_hardware,
            batches,
            warmup,
            repeats,
            torch_dev,
            torch_label,
            use_fx=not args.no_fx,
            architectures=active_archs,
        )
        all_records.extend(recs)
        _print_recs(recs)
        print()
    elif run_torch:
        print("[skip] PyTorch not installed or device unavailable")

    # JAX GPU
    if run_jax and importlib.util.find_spec("jax") is not None:
        jax_dev, jax_label = _detect_jax_gpu()
        if jax_dev is not None:
            print(f"── JAX ({jax_label}) ─────────────────────────────────")
            recs = eval_jax_gpu(
                gpu_hardware,
                batches,
                warmup,
                repeats,
                jax_dev,
                jax_label,
                architectures=active_archs,
            )
            all_records.extend(recs)
            _print_recs(recs)
            print()
        else:
            print("[skip] JAX: no GPU/accelerator device available")
    elif run_jax:
        print("[skip] JAX not installed")

    # TensorFlow GPU
    if run_tf and importlib.util.find_spec("tensorflow") is not None:
        tf_dev, tf_label = _detect_tf_gpu()
        print(f"── TensorFlow ({tf_label}) ─────────────────────────────────")
        recs = eval_tensorflow_gpu(
            gpu_hardware,
            batches,
            warmup,
            repeats,
            tf_dev,
            tf_label,
            architectures=active_archs,
        )
        all_records.extend(recs)
        _print_recs(recs)
        print()
    elif run_tf:
        print("[skip] TensorFlow not installed")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_gpu_data.json"

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
                )
                record_map[k] = d
            records_to_save = list(record_map.values())
        except Exception:
            records_to_save = [asdict(r) for r in all_records]
    else:
        records_to_save = [asdict(r) for r in all_records]

    payload: dict = {"hardware": hw_meta, "records": records_to_save}

    # ------------------------------------------------------------------
    # Crossover analysis: compare GPU latencies to CPU baseline
    # ------------------------------------------------------------------
    if args.crossover:
        cpu_latencies = _load_cpu_latencies()
        if cpu_latencies:
            crossover = compute_crossover(all_records, cpu_latencies)
            print_crossover_table(crossover)
            payload["crossover"] = crossover
        else:
            print(
                f"\n  [crossover] No CPU baseline found at {_CPU_DATA_FILE}\n"
                "  Run `python benchmarks/collect_data.py` first to generate it.\n"
            )

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(records_to_save)} records → {out_path}")


if __name__ == "__main__":
    main()
