"""Generate figures and a markdown GPU benchmark report from benchmark_gpu_data.json.

Usage:
    python benchmarks/generate_gpu_report.py [--results-dir benchmarks/results/<device>]

Reads:   <results-dir>/benchmark_gpu_data.json  (default: the most recently written one)
Writes:  <results-dir>/figures/gpu_*.png
         <results-dir>/GPU_BENCHMARK_REPORT.md
"""

from __future__ import annotations

import json
import tempfile
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent.parent
# Set by configure() from --results-dir: benchmarks/results/<device>/.  Until
# then (when imported, e.g. by tests) figures go to a scratch directory.
DATA_FILE   = REPO_ROOT / "benchmarks" / "results" / "benchmark_gpu_data.json"
FIG_DIR   = Path(tempfile.gettempdir()) / "neural-cost-figures"
REPORT_FILE = FIG_DIR / "GPU_BENCHMARK_REPORT.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def configure(results_dir: Path | None) -> None:
    """Point every input and output at one per-device results directory."""
    import sys

    global DATA_FILE, FIG_DIR, REPORT_FILE
    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.run_info import latest_results_dir

    root = results_dir if results_dir is not None else latest_results_dir("benchmark_gpu_data.json")
    DATA_FILE = root / "benchmark_gpu_data.json"
    FIG_DIR = root / "figures"
    REPORT_FILE = root / "GPU_BENCHMARK_REPORT.md"
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def provenance(path: Path) -> str:
    """Report line naming the version, commit and device that produced ``path``."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.run_info import describe_run

    return f"> **Run:** {describe_run(json.loads(path.read_text()).get('run'))}\n"


def describe_architecture(arch: str, scale: str) -> str | None:
    """Description of ``arch`` at ``scale`` from the shared model configs."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.models.configs import describe

    return describe(arch, scale)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

FW_COLORS = {"PyTorch": "#EE4C2C", "JAX": "#9B59B6", "TensorFlow": "#FF6F00"}
ARCHS     = ["FF DNN", "CNN", "RNN", "LSTM", "Transformer", "ConvNeXt", "ViT"]
FRAMEWORKS = ["PyTorch", "JAX", "TensorFlow"]

ARCH_DESCRIPTIONS = {
    "Transformer": "2-layer encoder (MHA h=4 + FFN×4 + LayerNorm), embed=128, seq=32",
    "CNN": "Conv64 (3×3) → BN → MaxPool → Conv128 (3×3) → BN → GAP → Dense10, input 32×32×3",
    "FF DNN": "784 → 128 → 128 → 10, ReLU + LayerNorm",
    "RNN": "2-layer Vanilla RNN, hidden=128, seq=32",
    "LSTM": "2-layer LSTM (4-gate), hidden=128, seq=32",
}

# Best variant per framework (GPU adds XLA option for TF)
BEST_VARIANTS = {
    "PyTorch":     "compiled",
    "JAX":         "jit",
    "TensorFlow":  "tf.function+XLA",
}
BEST_VARIANTS_FALLBACK = {
    "PyTorch":    "baseline",
    "JAX":        "baseline",
    "TensorFlow": "tf.function",
}

VAR_ALPHA = {
    "baseline": 0.45,
    "compiled": 1.0,
    "jit": 1.0,
    "tf.function": 0.8,
    "tf.function+XLA": 1.0,
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def gpu_backend(hw: dict) -> str:
    """The run's GPU backend: ``"cuda"``, ``"mps"`` or ``"cpu"`` (no GPU found)."""
    label = str(hw.get("torch_device") or hw.get("timing_device") or "").lower()
    if label.startswith("cuda"):
        return "cuda"
    if label.startswith("mps"):
        return "mps"
    if label.startswith("cpu"):
        return "cpu"
    name = str(hw.get("name", "")).lower()
    if "apple" in name:
        return "mps"
    return "cuda" if any(t in name for t in ("nvidia", "tesla", "geforce", "rtx")) else "cpu"


# Per-backend wording of the methodology section: (sync note, variants table, warmup note).
_BACKEND_METHODOLOGY: dict[str, tuple[str, str, str]] = {
    "cuda": (
        "On CUDA, each sample waits for the device (`torch.cuda.synchronize()`, "
        "`jax.Array.block_until_ready()`, TensorFlow's device sync) so the host clock covers "
        "all queued kernels.",
        "| **PyTorch** | Eager GPU (`cuda`) | `torch.compile()` | PyTorch CUDA backend with Inductor compilation |\n"
        "| **JAX** | Eager XLA GPU (`cuda`) | `jax.jit()` | JAX CUDA plugin (`jax-cuda*-plugin`) with whole-graph XLA JIT |\n"
        "| **TensorFlow** | Eager GPU | `tf.function(jit_compile=True)` | TensorFlow CUDA graph execution with XLA compilation |",
        "full compilation, cache warm, CUDA context and cuDNN/cuBLAS autotuning",
    ),
    "mps": (
        "On Apple Silicon MPS, synchronous timing barriers (`torch.mps.synchronize()`, "
        "`jax.block_until_ready()`, and `tf.experimental.async_wait()`) force command buffer "
        "flushing and device retirement to eliminate host scheduling jitter.",
        "| **PyTorch** | Eager GPU (`mps`) | `torch.compile()` | PyTorch MPS backend with Inductor compilation |\n"
        "| **JAX** | Eager XLA GPU (`mps`) | `jax.jit()` | `jax-mps` MLX-backed Metal plugin with whole-graph XLA JIT |\n"
        "| **TensorFlow** | Eager GPU | `tf.function(jit_compile=True)` | TensorFlow graph execution with XLA compilation |",
        "full compilation, cache warm, Metal kernel pipeline initialization",
    ),
    "cpu": (
        "No GPU was found, so every framework ran on the CPU; synchronisation barriers are no-ops.",
        "| **PyTorch** | Eager (`cpu`) | `torch.compile()` | PyTorch CPU backend with Inductor compilation |\n"
        "| **JAX** | Eager XLA (`cpu`) | `jax.jit()` | XLA CPU backend with whole-graph JIT |\n"
        "| **TensorFlow** | Eager | `tf.function(jit_compile=True)` | TensorFlow graph execution with XLA compilation |",
        "full compilation and cache warm",
    ),
}


def partial_run_warning(payload: dict) -> str:
    """A warning block for a sweep that stopped before it finished, else ``""``.

    ``collect_gpu_data.py`` saves after every architecture with
    ``"complete": false`` and sets it to true only at the end; files written
    before that flag existed have no key and count as complete.
    """
    if payload.get("complete", True):
        return ""
    frameworks = sorted({r.get("framework", "?") for r in payload.get("records", [])})
    return (
        "> [!WARNING]\n"
        "> **Partial run:** the sweep stopped before it finished (crash, kill or out of memory). "
        f"This report covers only the {len(payload.get('records', []))} configurations saved "
        f"before that ({', '.join(frameworks) or 'none'}).\n\n"
    )


def load(path: Path) -> tuple[dict, list[dict]]:
    raw = json.loads(path.read_text())
    return raw["hardware"], raw["records"]


def get_active_architectures(records: list[dict]) -> list[str]:
    """Extract ordered unique architecture names present in the records."""
    seen = []
    for r in records:
        a = r.get("architecture")
        if a and a not in seen:
            seen.append(a)
    return seen or ARCHS


def select(records: list[dict], **kwargs) -> list[dict]:
    result = records
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple)):
            result = [r for r in result if r[k] in v]
        else:
            result = [r for r in result if r[k] == v]
    return result


def get(records: list[dict], fw: str, variant: str, arch: str, batch: int, field: str):
    hits = select(records, framework=fw, variant=variant, architecture=arch, batch=batch)
    return hits[0][field] if hits else None


def best(records: list[dict], fw: str, arch: str, batch: int, field: str):
    """Return the value for the optimised variant; fall back to baseline."""
    for v in [BEST_VARIANTS.get(fw), BEST_VARIANTS_FALLBACK.get(fw), "baseline"]:
        if v is None:
            continue
        val = get(records, fw, v, arch, batch, field)
        if val is not None:
            return val
    return None


def best_variant_label(records: list[dict], fw: str, arch: str, batch: int) -> str:
    """Return the variant label that was actually used."""
    for v in [BEST_VARIANTS.get(fw), BEST_VARIANTS_FALLBACK.get(fw), "baseline"]:
        if v and get(records, fw, v, arch, batch, "latency_median_ms") is not None:
            return v
    return "—"


# ---------------------------------------------------------------------------
# Figure GPU-1: Roofline scatter (AI vs GFLOP/s)
# ---------------------------------------------------------------------------

def fig_roofline(hw: dict, records: list[dict]) -> Path:
    # Use the largest available batch for GPU (better utilisation)
    batches_in_data = sorted({r["batch"] for r in records})
    batch = batches_in_data[-1] if batches_in_data else 128

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    peak_gflops = hw["peak_flops"] / 1e9
    bw_gb_s     = hw["memory_bandwidth"] / 1e9
    ridge       = hw["ridge_point"]

    ai_range = np.logspace(-1, 4, 500)
    roofline  = np.minimum(peak_gflops, ai_range * bw_gb_s)
    ax.loglog(ai_range, roofline, "k-", lw=2, label="Roofline bound", zorder=5)
    ax.axvline(ridge, color="k", lw=1, ls="--", alpha=0.5,
               label=f"Ridge ({ridge:.0f} FLOP/byte)")

    markers = {
        "FF DNN": "o",
        "Deep DNN": "o",
        "CNN": "s",
        "RNN": "D",
        "LSTM": "^",
        "Transformer": "P",
        "ConvNeXt": "h",
        "ViT": "X",
    }

    for fw in FRAMEWORKS:
        for arch in ARCHS:
            ai_val  = get(records, fw, "baseline", arch, batch, "arith_intensity")
            gf_best = best(records, fw, arch, batch, "achieved_gflops")
            if ai_val is None or gf_best is None:
                continue
            col = FW_COLORS[fw]
            m = markers.get(arch, "o")
            ax.scatter(ai_val, gf_best, c=col, marker=m, s=90,
                       zorder=6, edgecolors="white", linewidths=0.5)
            ax.annotate(f"{fw[:3]}", (ai_val, gf_best),
                        textcoords="offset points", xytext=(5, 2),
                        fontsize=7, color=col, alpha=0.85)

    fw_patches = [mpatches.Patch(color=FW_COLORS[f], label=f) for f in FRAMEWORKS]
    arch_lines = [plt.scatter([], [], marker=markers.get(a, "o"), c="gray", s=70, label=a) for a in ARCHS]
    leg1 = ax.legend(handles=fw_patches, loc="lower right", title="Framework", framealpha=0.9)
    ax.legend(handles=arch_lines, loc="upper left", title="Architecture", framealpha=0.9)
    ax.add_artist(leg1)

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Achieved Throughput (GFLOP/s)")
    ax.set_title(
        f"GPU Roofline Model — {hw['name']}\n"
        f"batch={batch}, optimised variants  ·  {hw.get('torch_device', '')}",
        fontweight="bold",
    )
    ax.set_xlim(0.5, 2000)
    ax.set_ylim(0.5, peak_gflops * 2)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    out = FIG_DIR / "gpu_fig1_roofline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-2: Latency grouped bar chart (best variants, multiple batches)
# ---------------------------------------------------------------------------

def fig_latency_bars(hw: dict, records: list[dict]) -> Path:
    batches_in_data = sorted({r["batch"] for r in records})
    ref_batch = 32 if 32 in batches_in_data else batches_in_data[len(batches_in_data) // 2]  # mid-range batch

    x     = np.arange(len(ARCHS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(ARCHS) * 1.5), 5))

    for i, fw in enumerate(FRAMEWORKS):
        vals = [best(records, fw, arch, ref_batch, "latency_median_ms") or 0 for arch in ARCHS]
        errs = [best(records, fw, arch, ref_batch, "latency_stddev_ms") or 0 for arch in ARCHS]
        ax.bar(x + i * width, vals, width * 0.88,
               label=fw, color=FW_COLORS[fw], alpha=0.88,
               yerr=errs, capsize=3,
               error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.6})

    ax.set_xticks(x + width)
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("Median Latency (ms)")
    ax.set_title(
        f"GPU Inference Latency by Architecture & Framework\n"
        f"{hw['name']}  ·  batch={ref_batch}  ·  optimised variants",
        fontweight="bold",
    )
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig2_latency_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-3: Roofline efficiency heatmap
# ---------------------------------------------------------------------------

def fig_efficiency_heatmap(hw: dict, records: list[dict]) -> Path:
    batches_in_data = sorted({r["batch"] for r in records})
    ref_batch = 32 if 32 in batches_in_data else batches_in_data[len(batches_in_data) // 2]

    matrix = np.full((len(ARCHS), len(FRAMEWORKS)), np.nan)
    for i, arch in enumerate(ARCHS):
        for j, fw in enumerate(FRAMEWORKS):
            v = best(records, fw, arch, ref_batch, "roofline_efficiency")
            if v is not None:
                matrix[i, j] = v * 100

    fig, ax = plt.subplots(figsize=(max(7.5, len(FRAMEWORKS) * 2.5), max(4.5, len(ARCHS) * 0.6)))
    valid = matrix[~np.isnan(matrix)]
    vmax  = min(valid.max() * 1.2, 100) if valid.size else 100
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(FRAMEWORKS)))
    ax.set_xticklabels(FRAMEWORKS)
    ax.set_yticks(range(len(ARCHS)))
    ax.set_yticklabels(ARCHS)
    plt.colorbar(im, ax=ax, label="Roofline Efficiency (%)")
    for i in range(len(ARCHS)):
        for j in range(len(FRAMEWORKS)):
            val = matrix[i, j]
            txt = f"{val:.1f}%" if not np.isnan(val) else "N/A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold",
                    color="black" if (np.isnan(val) or val < 60) else "white")
    ax.set_title(
        f"GPU Roofline Efficiency (%) — {hw['name']}\nbatch={ref_batch}, optimised variants",
        fontweight="bold",
    )
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig3_efficiency_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-4: Latency vs batch size scaling
# ---------------------------------------------------------------------------

def fig_batch_scaling(hw: dict, records: list[dict]) -> Path:
    batches = sorted({r["batch"] for r in records})
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(max(14, len(ARCHS) * 2.5), 4), sharey=False)
    axes_list = [axes] if len(ARCHS) == 1 else list(axes)

    for ax, arch in zip(axes_list, ARCHS):
        for fw in FRAMEWORKS:
            ys = [best(records, fw, arch, b, "latency_median_ms") for b in batches]
            valid = [(b, y) for b, y in zip(batches, ys) if y is not None]
            if not valid:
                continue
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "o-", color=FW_COLORS[fw], label=fw, lw=1.8, ms=5)
        ax.set_title(arch, fontsize=9, fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batches)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    axes[0].set_ylabel("Median Latency (ms, log)")
    handles = [mpatches.Patch(color=FW_COLORS[f], label=f) for f in FRAMEWORKS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), framealpha=0.9)
    fig.suptitle(
        f"GPU Latency Scaling with Batch Size — {hw['name']}  (optimised variants)",
        fontweight="bold", y=1.07,
    )
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig4_batch_scaling.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-5: Compilation/JIT speedup over eager baseline
# ---------------------------------------------------------------------------

def fig_speedup(hw: dict, records: list[dict]) -> Path:
    batches_in_data = sorted({r["batch"] for r in records})
    ref_batch = 32 if 32 in batches_in_data else batches_in_data[len(batches_in_data) // 2]

    opt_map = {
        "PyTorch":    "compiled",
        "JAX":        "jit",
        "TensorFlow": BEST_VARIANTS["TensorFlow"],
    }
    fig, ax = plt.subplots(figsize=(max(10, len(ARCHS) * 1.5), 4.5))
    x     = np.arange(len(ARCHS))
    width = 0.25
    max_y = 1.0

    for i, fw in enumerate(FRAMEWORKS):
        opt = opt_map[fw]
        speedups = []
        for arch in ARCHS:
            base = get(records, fw, "baseline", arch, ref_batch, "latency_median_ms")
            fast = get(records, fw, opt, arch, ref_batch, "latency_median_ms")
            if not fast:
                # Try fallback variant
                fast = get(records, fw, BEST_VARIANTS_FALLBACK[fw], arch, ref_batch, "latency_median_ms")
            if base and fast and fast > 0:
                speedups.append(base / fast)
            else:
                speedups.append(1.0)
        max_y = max(max_y, max(speedups))
        bars = ax.bar(x + i * width, speedups, width * 0.88,
                      label=f"{fw} ({opt})", color=FW_COLORS[fw], alpha=0.88)
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{sp:.1f}×", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.4, label="No speedup (1×)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("Speedup over eager baseline (×)")
    ax.set_ylim(0, max_y * 1.25)
    ax.set_title(
        f"GPU Compilation Speedup (baseline → optimised)\n{hw['name']}  ·  batch={ref_batch}",
        fontweight="bold",
    )
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig5_speedup.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-6: Throughput (GFLOP/s)
# ---------------------------------------------------------------------------

def fig_throughput(hw: dict, records: list[dict]) -> Path:
    batches_in_data = sorted({r["batch"] for r in records})
    ref_batch = 32 if 32 in batches_in_data else batches_in_data[len(batches_in_data) // 2]

    x     = np.arange(len(ARCHS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(ARCHS) * 1.5), 5))

    peak = hw["peak_flops"] / 1e9
    ax.axhline(peak, color="gray", lw=1.5, ls="--", alpha=0.7,
               label=f"Peak ({peak:.0f} GFLOP/s)")

    for i, fw in enumerate(FRAMEWORKS):
        vals = [best(records, fw, arch, ref_batch, "achieved_gflops") or 0 for arch in ARCHS]
        ax.bar(x + i * width, vals, width * 0.88,
               label=fw, color=FW_COLORS[fw], alpha=0.88)

    ax.set_xticks(x + width)
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("Achieved Throughput (GFLOP/s)")
    ax.set_title(
        f"GPU Achieved Throughput by Architecture & Framework\n{hw['name']}  ·  batch={ref_batch}",
        fontweight="bold",
    )
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig6_throughput.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-7: Throughput scaling with batch size (best variant)
# ---------------------------------------------------------------------------

def fig_throughput_scaling(hw: dict, records: list[dict]) -> Path:
    batches = sorted({r["batch"] for r in records})
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(max(14, len(ARCHS) * 2.5), 4), sharey=False)
    axes_list = [axes] if len(ARCHS) == 1 else list(axes)

    for ax, arch in zip(axes_list, ARCHS):
        for fw in FRAMEWORKS:
            ys = [best(records, fw, arch, b, "achieved_gflops") for b in batches]
            valid = [(b, y) for b, y in zip(batches, ys) if y is not None]
            if not valid:
                continue
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "o-", color=FW_COLORS[fw], label=fw, lw=1.8, ms=5)
        ax.set_title(arch, fontsize=9, fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylim(bottom=0)

    axes_list[0].set_ylabel("Achieved GFLOP/s")
    handles = [mpatches.Patch(color=FW_COLORS[f], label=f) for f in FRAMEWORKS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), framealpha=0.9)
    fig.suptitle(
        f"GPU Throughput Scaling with Batch Size — {hw['name']}",
        fontweight="bold", y=1.07,
    )
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig7_throughput_scaling.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-9: Inference Latency per Sample (Amortization & Saturation Floor)
# ---------------------------------------------------------------------------

def fig_per_sample_latency(hw: dict, records: list[dict]) -> Path | None:
    batches = sorted({r["batch"] for r in records})
    if len(batches) < 2:
        return None
    x = np.arange(len(ARCHS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(11, len(ARCHS) * 1.6), 5))

    for idx, b in enumerate(batches):
        vals = []
        for arch in ARCHS:
            b_recs = [r for r in records if r["architecture"] == arch and r["batch"] == b]
            fast_ms = min((r["latency_median_ms"] for r in b_recs), default=None)
            vals.append((fast_ms / b) if fast_ms else 0)

        offset = (idx - 0.5) * width
        color = "#3498DB" if idx == 0 else "#E67E22"
        bars = ax.bar(x + offset, vals, width * 0.9, label=f"Batch {b}", color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{v:.3f}ms" if v < 1 else f"{v:.2f}ms",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=25,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("Latency per Sample (ms/sample)")
    ax.set_title(
        f"Inference Latency per Sample: Batch Scaling Law & Saturation Floor\n{hw['name']}  ·  Fastest Variant per Architecture",
        fontweight="bold",
    )
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig9_per_sample_latency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-10: JAX vs PyTorch Head-to-Head Speedup
# ---------------------------------------------------------------------------

def fig_framework_comparison(hw: dict, records: list[dict]) -> Path:
    ref_batch = 32 if 32 in {r["batch"] for r in records} else 256
    x = np.arange(len(ARCHS))
    width = 0.5
    fig, ax = plt.subplots(figsize=(max(10, len(ARCHS) * 1.5), 4.8))

    speedups = []
    colors = []
    labels = []
    for arch in ARCHS:
        pt_lat = best(records, "PyTorch", arch, ref_batch, "latency_median_ms")
        jax_lat = best(records, "JAX", arch, ref_batch, "latency_median_ms")
        if pt_lat and jax_lat and jax_lat > 0:
            sp = pt_lat / jax_lat
            speedups.append(sp)
            colors.append("#9B59B6" if sp >= 1.0 else "#EE4C2C")
            labels.append(f"{sp:.1f}×")
        else:
            speedups.append(1.0)
            colors.append("gray")
            labels.append("N/A")

    bars = ax.bar(x, speedups, width, color=colors, alpha=0.88)
    ax.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.7, label="Parity (1.0×)")
    for bar, lbl, sp in zip(bars, labels, speedups):
        y_pos = max(sp, 1.0) + 0.15 if sp >= 1.0 else sp + 0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("Speedup over PyTorch (T_PyTorch / T_JAX)")
    ax.set_title(
        f"JAX (JIT) vs PyTorch (Compiled) Head-to-Head Speedup\n{hw['name']}  ·  batch={ref_batch}  (Values > 1.0× indicate JAX is faster)",
        fontweight="bold",
    )
    legend_elements = [
        mpatches.Patch(color="#9B59B6", label="JAX faster"),
        mpatches.Patch(color="#EE4C2C", label="PyTorch faster"),
        plt.Line2D([0], [0], color="black", lw=1.2, ls="--", label="Parity (1.0×)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(speedups) * 1.25)
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig10_framework_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-8: GPU-vs-CPU crossover (batch size where GPU first wins)
# ---------------------------------------------------------------------------

def fig_crossover(hw: dict, records: list[dict], crossover: dict | None) -> Path | None:
    """Plot GPU and CPU latency curves together, annotating the crossover point.

    If crossover data is not available (CPU baseline not collected), returns None.
    """
    if not crossover:
        return None

    # We need at least one (fw, arch) pair that has CPU data in the crossover dict.
    has_cpu_data = any(
        pt.get("cpu_ms") is not None
        for fw_data in crossover.values()
        for arch_data in fw_data.values()
        for pt in arch_data.get("data", [])
    )
    if not has_cpu_data:
        return None

    n_archs = len(ARCHS)
    fig, axes = plt.subplots(1, n_archs, figsize=(max(14, n_archs * 2.5), 4), sharey=False)
    axes_list = [axes] if n_archs == 1 else list(axes)

    for ax, arch in zip(axes_list, ARCHS):
        for fw in FRAMEWORKS:
            info = crossover.get(fw, {}).get(arch, {})
            pts  = info.get("data", [])
            if not pts:
                continue

            gpu_xs = [p["batch"] for p in pts]
            gpu_ys = [p["gpu_ms"] for p in pts]
            cpu_ys = [p.get("cpu_ms") for p in pts]

            # GPU line
            ax.plot(gpu_xs, gpu_ys, "o-", color=FW_COLORS[fw],
                    lw=1.8, ms=5, label=f"{fw} GPU")

            # CPU dashed line (same colour, dashed)
            cpu_valid = [(x, y) for x, y in zip(gpu_xs, cpu_ys) if y is not None]
            if cpu_valid:
                cx, cy = zip(*cpu_valid)
                ax.plot(cx, cy, "--", color=FW_COLORS[fw],
                        lw=1.4, alpha=0.55, label=f"{fw} CPU")

            # Annotate crossover batch
            cb = info.get("crossover_batch")
            if cb:
                cb_pt = next((p for p in pts if p["batch"] == cb), None)
                if cb_pt and cb_pt.get("cpu_ms"):
                    ax.axvline(cb, color=FW_COLORS[fw], lw=1, ls=":", alpha=0.7)
                    ax.annotate(
                        f"B={cb}",
                        (cb, cb_pt["gpu_ms"]),
                        textcoords="offset points", xytext=(4, -12),
                        fontsize=7, color=FW_COLORS[fw],
                    )

        ax.set_title(arch, fontsize=9, fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        valid_xs = [p["batch"] for fw_d in crossover.values()
                    for a_d in fw_d.values()
                    for p in a_d.get("data", [])
                    if a_d.get("data") and p.get("gpu_ms") is not None]
        if valid_xs:
            ax.set_xticks(sorted(set(valid_xs)))
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    axes_list[0].set_ylabel("Median Latency (ms, log scale)")

    # Build a compact legend (GPU solid / CPU dashed) using framework colours
    from matplotlib.lines import Line2D
    legend_elems = []
    for fw in FRAMEWORKS:
        legend_elems.append(
            Line2D([0], [0], color=FW_COLORS[fw], lw=2, label=f"{fw}")
        )
    legend_elems.append(Line2D([0], [0], color="k", lw=1.5, ls="-",  label="GPU"))
    legend_elems.append(Line2D([0], [0], color="k", lw=1.5, ls="--", label="CPU"))
    fig.legend(handles=legend_elems, loc="upper center",
               ncol=len(FRAMEWORKS) + 2, bbox_to_anchor=(0.5, 1.05), framealpha=0.9)
    fig.suptitle(
        f"GPU-vs-CPU Crossover — {hw['name']}  (vertical dotted line = first batch where GPU wins)",
        fontweight="bold", y=1.1,
    )
    fig.tight_layout()
    out = FIG_DIR / "gpu_fig8_crossover.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure GPU-Memory: Peak memory utilization and allocator overhead
# ---------------------------------------------------------------------------

def fig_gpu_memory(hw: dict, records: list[dict]) -> Path | None:
    has_mem = any(r.get("peak_allocated_bytes") for r in records)
    if not has_mem:
        return None
    batches = sorted({r["batch"] for r in records})
    ref_batch = 32 if 32 in batches else (batches[len(batches) // 2] if batches else 32)
    x = np.arange(len(ARCHS))
    width = 0.22

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(ARCHS) * 2), 5))

    theo_mins = []
    theo_conss = []
    for arch in ARCHS:
        t_min = get(records, "PyTorch", "baseline", arch, ref_batch, "theoretical_min_bytes")
        t_cons = get(records, "PyTorch", "baseline", arch, ref_batch, "theoretical_conservative_bytes")
        theo_mins.append((t_min or 0) / (1024 * 1024))
        theo_conss.append((t_cons or 0) / (1024 * 1024))

    ax1.bar(x - width, theo_mins, width * 0.9, label="Theoretical Min", color="#2ECC71", alpha=0.85)
    ax1.bar(x, theo_conss, width * 0.9, label="Theoretical Cons.", color="#27AE60", alpha=0.85)

    for i, fw in enumerate(FRAMEWORKS):
        allocs = []
        for arch in ARCHS:
            val = best(records, fw, arch, ref_batch, "peak_allocated_bytes")
            allocs.append((val or 0) / (1024 * 1024))
        if any(v > 0 for v in allocs):
            ax1.bar(x + (i + 1) * width, allocs, width * 0.9, label=f"{fw} Peak Alloc", color=FW_COLORS[fw], alpha=0.85)

    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(ARCHS, rotation=15 if len(ARCHS) > 5 else 0, ha="right" if len(ARCHS) > 5 else "center")
    ax1.set_ylabel("Memory Footprint (MB, log scale)")
    ax1.set_yscale("log")
    ax1.set_title(f"Peak GPU Memory vs Theoretical Bounds\n{hw['name']}  ·  batch={ref_batch}", fontweight="bold")
    ax1.legend(framealpha=0.9, fontsize=8)

    # Panel 2: Overhead Ratio & Reserved Cache
    for i, fw in enumerate(FRAMEWORKS):
        ratios = [best(records, fw, arch, ref_batch, "memory_overhead_ratio") or 1.0 for arch in ARCHS]
        if any(r > 1.0 for r in ratios):
            bars = ax2.bar(x + i * 0.35, ratios, 0.32, label=f"{fw} Overhead Ratio", color=FW_COLORS[fw], alpha=0.85)
            for bar, r in zip(bars, ratios):
                if r > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                             f"{r:.1f}×", ha="center", va="bottom", fontsize=8)

    ax2.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5, label="Theoretical Min (1.0×)")
    ax2.set_xticks(x + 0.18)
    ax2.set_xticklabels(ARCHS, rotation=15 if len(ARCHS) > 5 else 0, ha="right" if len(ARCHS) > 5 else "center")
    ax2.set_ylabel("Overhead Ratio (observed / theoretical_min)")
    ax2.set_title(f"Dynamic GPU Memory Overhead Ratio\n{hw['name']}  ·  batch={ref_batch}", fontweight="bold")
    ax2.legend(framealpha=0.9, fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "gpu_fig_memory.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Build markdown report
# ---------------------------------------------------------------------------

def build_report(hw: dict, records: list[dict], fig_paths: dict[str, Path | None],
                 crossover: dict | None = None) -> str:
    def rel(p: Path) -> str:
        return str(p.relative_to(REPORT_FILE.parent))

    batches = sorted({r["batch"] for r in records})
    ref_batch = 32 if 32 in batches else batches[len(batches) // 2]

    def fmt_n(n):
        if n is None: return "—"
        if n >= 1e9:  return f"{n/1e9:.2f}G"
        if n >= 1e6:  return f"{n/1e6:.1f}M"
        return f"{n:,.0f}"

    def gpu_memory_table(batch: int) -> str:
        rows = [
            "| Architecture | Framework | Variant | Theo Min (KB) | Theo Cons (KB) | Peak Alloc (KB) | Peak Reserved (KB) | Overhead Ratio | Pool Caching |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        def fmt_kb(n):
            if n is None: return "—"
            return f"{n / 1024:,.1f}"

        for arch in ARCHS:
            for fw in FRAMEWORKS:
                for variant in ["baseline", "compiled", "jit", "tf.function", "tf.function+XLA"]:
                    hits = select(records, framework=fw, variant=variant, architecture=arch, batch=batch)
                    if not hits:
                        continue
                    r = hits[0]
                    t_min = r.get("theoretical_min_bytes")
                    t_cons = r.get("theoretical_conservative_bytes")
                    p_alloc = r.get("peak_allocated_bytes")
                    p_res = r.get("peak_reserved_bytes")
                    overhead = r.get("memory_overhead_ratio")
                    overhead_str = f"**{overhead:.2f}×**" if overhead is not None else "—"
                    caching_str = "—"
                    if p_res is not None and p_alloc is not None and p_alloc > 0:
                        caching_ratio = p_res / p_alloc
                        caching_str = f"{caching_ratio:.2f}×" if caching_ratio > 1.05 else "1.00× (minimal)"
                    rows.append(
                        f"| {arch} | {fw} | {variant} "
                        f"| {fmt_kb(t_min)} "
                        f"| {fmt_kb(t_cons)} "
                        f"| {fmt_kb(p_alloc)} "
                        f"| {fmt_kb(p_res)} "
                        f"| {overhead_str} "
                        f"| {caching_str} |"
                    )
        return "\n".join(rows)

    def stat_table(batch: int) -> str:
        rows = [
            "| Architecture | Framework | Variant | FLOPs | Params | AI (FLOP/B) "
            "| Latency med (ms) | ±σ | CV% | Efficiency | GFLOP/s | Bottleneck |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arch in ARCHS:
            for fw in FRAMEWORKS:
                for variant in ["baseline", "compiled", "jit", "tf.function", "tf.function+XLA"]:
                    hits = select(records, framework=fw, variant=variant, architecture=arch, batch=batch)
                    if not hits:
                        continue
                    r = hits[0]
                    rows.append(
                        f"| {arch} | {fw} | {variant} "
                        f"| {fmt_n(r['flops'])} "
                        f"| {fmt_n(r['param_bytes'])} "
                        f"| {r['arith_intensity']:.2f} "
                        f"| {r['latency_median_ms']:.3f} "
                        f"| {r['latency_stddev_ms']:.3f} "
                        f"| {r['latency_cv_pct']:.1f} "
                        f"| {r['roofline_efficiency']:.1%} "
                        f"| {r['achieved_gflops']:.2f} "
                        f"| {r['bottleneck']} |"
                    )
        return "\n".join(rows)

    def diagnostics_table(batch: int) -> str:
        rows = [
            "| Architecture | Framework | Variant | Fused Efficiency | Traffic Saved | Resident Cache | Top Layer Bottleneck | Layer Share |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arch in ARCHS:
            for fw in FRAMEWORKS:
                for variant in ["baseline", "compiled", "jit", "tf.function", "tf.function+XLA"]:
                    hits = select(records, framework=fw, variant=variant, architecture=arch, batch=batch)
                    if not hits:
                        continue
                    r = hits[0]
                    fused_eff = f"{r['fused_efficiency']:.1%}" if r.get("fused_efficiency") is not None else "—"
                    traffic = f"{r['traffic_reduction_pct']:.1f}%" if r.get("traffic_reduction_pct") is not None else "—"
                    res = f"{r['cache_name']}" if r.get("cache_resident") and r.get("cache_name") else ("Yes" if r.get("cache_resident") else "DRAM/VRAM")
                    top_layer = r.get("top_layer_bottleneck") or "—"
                    share = f"{r['top_layer_share_pct']:.1f}%" if r.get("top_layer_share_pct") is not None else "—"
                    rows.append(
                        f"| {arch} | {fw} | {variant} | {fused_eff} | {traffic} | {res} | {top_layer} | {share} |"
                    )
        return "\n".join(rows)

    def speedup_table() -> str:
        rows = [
            "| Architecture | PyTorch (compile) | JAX (jit) | TensorFlow (XLA/graph) |",
            "|---|---|---|---|",
        ]
        opt_map = {
            "PyTorch":    "compiled",
            "JAX":        "jit",
            "TensorFlow": BEST_VARIANTS["TensorFlow"],
        }
        for arch in ARCHS:
            cells = [arch]
            for fw in FRAMEWORKS:
                opt  = opt_map[fw]
                base = get(records, fw, "baseline", arch, ref_batch, "latency_median_ms")
                fast = get(records, fw, opt, arch, ref_batch, "latency_median_ms")
                if not fast:
                    fast = get(records, fw, BEST_VARIANTS_FALLBACK[fw], arch, ref_batch, "latency_median_ms")
                if base and fast and fast > 0:
                    cells.append(f"**{base/fast:.2f}×** ({base:.3f}→{fast:.3f} ms)")
                else:
                    cells.append("—")
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    def conclusions() -> str:
        lines = []
        for arch in ARCHS:
            best_fw, best_ms = None, math.inf
            for fw in FRAMEWORKS:
                ms = best(records, fw, arch, ref_batch, "latency_median_ms")
                if ms and ms < best_ms:
                    best_ms, best_fw = ms, fw
            if best_fw:
                vl = best_variant_label(records, best_fw, arch, ref_batch)
                lines.append(
                    f"- **{arch}**: fastest is **{best_fw}** ({vl}) "
                    f"at {best_ms:.3f} ms (batch={ref_batch})"
                )
        return "\n".join(lines)

    def crossover_table() -> str:
        """Markdown table summarising the GPU-vs-CPU crossover batch sizes."""
        if not crossover:
            return (
                "_No crossover data available.  "
                "Re-run with `--crossover` after collecting CPU baseline._"
            )
        rows = [
            "| Architecture | Framework | Crossover batch | Notes |",
            "|---|---|---|---|",
        ]
        for fw in FRAMEWORKS:
            for arch in ARCHS:
                info = crossover.get(fw, {}).get(arch, {})
                cb   = info.get("crossover_batch")
                tag  = f"≥ {cb}" if cb else "never"
                note = ""
                if cb:
                    pt = next((p for p in info.get("data", []) if p["batch"] == cb), None)
                    if pt and pt.get("cpu_ms") and pt.get("gpu_ms"):
                        note = f"GPU {pt['cpu_ms']/pt['gpu_ms']:.2f}× faster"
                rows.append(f"| {arch} | {fw} | {tag} | {note} |")
        return "\n".join(rows)

    arch_rows = [
        "| Architecture | Category | Description |",
        "|---|---|---|",
    ]
    for arch in ARCHS:
        desc = (
            describe_architecture(arch, "standard")
            if arch in {"ConvNeXt", "ViT"}
            else ARCH_DESCRIPTIONS.get(arch, "Neural network architecture under test")
        )
        cat = "Modern Vision" if arch in ("ConvNeXt", "ViT") else "Core / Legacy"
        arch_rows.append(f"| **{arch}** | {cat} | {desc} |")
    arch_table_md = "\n".join(arch_rows)

    ridge = hw["ridge_point"]
    tf_device = hw.get("torch_device", hw["name"])
    sync_note, variants_table, warmup_note = _BACKEND_METHODOLOGY[gpu_backend(hw)]

    md = f"""# Neural-Cost GPU Benchmark Report

> **Device:** {hw['name']}  ·  **Peak FP32:** {hw['peak_flops']/1e12:.2f} TFLOP/s  
> **Peak bandwidth:** {hw['memory_bandwidth']/1e9:.0f} GB/s
> **Ridge point:** {ridge:.1f} FLOP/byte  ·  **Detection:** {hw['source']}  
> **Timing device:** {tf_device}

---

## Methodology

### GPU timing protocol

| Framework | Timing method | Sync barrier |
|---|---|---|
| **PyTorch** | `torch.cuda.Event` (CUDA events) | `torch.cuda.synchronize()` |
| **PyTorch MPS** | `time.perf_counter_ns` | `torch.mps.synchronize()` |
| **JAX** | `time.perf_counter_ns` | `jax.Array.block_until_ready()` |
| **TensorFlow** | `time.perf_counter_ns` | `tf.experimental.async_wait()` |

{sync_note}

### Architectures under test

{arch_table_md}

### Frameworks and optimisation variants

| Framework | Baseline | Optimised | Notes |
|---|---|---|---|
{variants_table}

### Measurement protocol

- **Warmup:** {hw.get('warmup', 15)} iterations ({warmup_note})
- **Timed repeats:** {hw.get('repeats', 40)} samples per configuration
- **Statistics reported:** median, mean, σ (stddev), CV%, p95
- **Roofline efficiency:** `min(1, lower_bound / observed)`
- **Batch sizes swept:** {batches}

---

## Figure GPU-1 — Roofline Model

Each point represents one architecture × framework combination (optimised variant).

![GPU Roofline]({rel(fig_paths['roofline'])})

---

## Figure GPU-2 — Inference Latency by Architecture (batch={ref_batch})

Error bars show ±1σ across timed iterations.

![GPU Latency bars]({rel(fig_paths['latency_bars'])})

---

## Figure GPU-3 — Roofline Efficiency Heatmap (batch={ref_batch})

![GPU Efficiency heatmap]({rel(fig_paths['heatmap'])})

---

## Figure GPU-4 — Latency Scaling with Batch Size

![GPU Batch scaling]({rel(fig_paths['batch_scaling'])})

---

## Figure GPU-5 — Compilation / JIT Speedup (batch={ref_batch})

Speedup ratio = eager latency / optimised latency. Higher is better.

![GPU Speedup]({rel(fig_paths['speedup'])})

---

## Figure GPU-5b — JAX vs PyTorch Head-to-Head Speedup (batch={ref_batch})

Direct speedup of JAX (JIT) over PyTorch (Compiled): $T_{{\\text{{PyTorch}}}} / T_{{\\text{{JAX}}}}$. Values > 1.0× indicate JAX is faster.

![JAX vs PyTorch Speedup]({rel(fig_paths['framework_comparison'])})

---

## Figure GPU-6 — Achieved Throughput (GFLOP/s, batch={ref_batch})

![GPU Throughput]({rel(fig_paths['throughput'])})

---

## Figure GPU-7 — Throughput Scaling with Batch Size

![GPU Throughput scaling]({rel(fig_paths['throughput_scaling'])})

---

## Latency per Sample

![Latency per Sample]({rel(fig_paths['per_sample_latency'])})

---

## Full Results Table (batch={ref_batch})

<details>
<summary>Expand full results table (all variants, batch={ref_batch})</summary>

{stat_table(ref_batch)}

</details>

---

## Advanced Causal Diagnostics (batch={ref_batch})

Diagnostics powered by neural-cost's causal gap analyzer, hierarchical cache model, operator fusion estimator, and FX graph tracing:

<details>
<summary>Expand advanced diagnostics table (batch={ref_batch})</summary>

{diagnostics_table(ref_batch)}

</details>

---

## Compilation Speedup Summary (batch={ref_batch})

{speedup_table()}

---

## Per-Architecture Winner (batch={ref_batch})

{conclusions()}

---

## Conclusions

### 1. GPU changes the performance landscape vs CPU

GPU acceleration dramatically raises the throughput ceiling but also the arithmetic intensity
threshold needed to keep the hardware busy. Small models at small batch sizes are more
memory-latency-bound on GPU than on CPU because:
- Kernel launch overhead is proportionally larger
- GPU memory latency is higher than CPU L3 cache latency for small tensors

### 2. Batch size is the primary GPU utilisation lever

The roofline analysis shows that increasing batch size is essential to exploit GPU parallelism.
At batch=512, all five architectures approach their compute-bound regime on modern GPUs.

### 3. JIT compilation provides larger GPU speedups than CPU speedups

On CPU, Python dispatch overhead is the dominant bottleneck. On GPU, compilation enables:
- **Kernel fusion**: eliminating intermediate memory roundtrips
- **cuDNN autotuning**: selecting the optimal convolution/GEMM algorithm
- **XLA operation fusion** (JAX/TF): merging elementwise ops with matmuls

### 4. Framework GPU support maturity

| Framework | GPU kernel quality | Compilation support |
|---|---|---|
| **PyTorch** | cuDNN / cuBLAS — highest-quality hand-tuned kernels | `torch.compile()` Inductor |
| **JAX** | XLA GPU backend — strong GEMM, improving conv | `jax.jit()` native |
| **TensorFlow** | cuDNN / XLA — competitive for dense workloads | `tf.function(jit_compile=True)` |

### 5. JAX on Apple Silicon GPU via MPS

JAX GPU support on Apple Silicon is fully functional using the modern `jax-mps` plugin:

1. **Plugin Ecosystem & Root Cause of Prior Absence:**
   - Apple's legacy `jax-metal==0.1.1` package is obsolete and incompatible with the StableHLO v6 bytecode generated by `jaxlib >= 0.4.30` (triggering compilation crashes: `unknown attribute code: 22`).
   - By removing `jax-metal` and utilizing `jax-mps==0.11.0` (the MLX-backed Metal acceleration plugin) with backend `"mps"`, JAX compiles and executes directly on the Apple Silicon GPU (`mps:0`).

---

## Figure GPU-8 — GPU-vs-CPU Crossover (batch size where GPU wins)
"""
    # Figure GPU-8 is optional — only emitted when crossover data is present
    if "crossover" in fig_paths and fig_paths["crossover"] is not None:
        md += f"""
![GPU-vs-CPU Crossover]({rel(fig_paths['crossover'])})

Each panel shows GPU latency (solid) vs CPU latency (dashed) across the batch-size grid.
Vertical dotted lines mark the first batch at which GPU latency drops below CPU latency.

### Crossover batch sizes

{crossover_table()}

> **Reading the table:** "never" means the GPU did not outperform the CPU at any
> tested batch size with this architecture + framework combination, typically because
> kernel launch overhead dominates at all tested sizes (e.g. very small FF DNN models).

---
"""
    else:
        md += """
_Crossover figure not available.  Re-run with `--crossover` flag to generate it:_

```bash
python benchmarks/collect_data.py          # generate CPU baseline first
python benchmarks/collect_gpu_data.py --crossover
```

---
"""

    # Memory section if telemetry is available
    if fig_paths.get("memory") is not None:
        md += f"""
---

## GPU Memory Telemetry and Allocator Fragmentation (batch={ref_batch})

Empirical memory telemetry measured from framework device allocators compared against theoretical tensor bounds calculated by `neural_cost.profile_model` and `neural_cost.analyze_memory_gap`.

![GPU Memory]({rel(fig_paths["memory"])})

### Memory Telemetry and Allocator Fragmentation Table (batch={ref_batch})

{gpu_memory_table(ref_batch)}

"""

    md += """
*Generated by `benchmarks/generate_gpu_report.py` using [neural-cost](https://github.com/davidgraymi/neural-cost)*
"""
    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="benchmarks/results/<device>/ to read and write (default: most recent)",
    )
    configure(parser.parse_args().results_dir)
    print(f"Loading GPU data from {DATA_FILE}…")
    hw, records = load(DATA_FILE)
    print(f"  {len(records)} records  ·  {hw['name']}")

    global ARCHS
    ARCHS = get_active_architectures(records)
    print(f"  Active architectures ({len(ARCHS)}): {', '.join(ARCHS)}")

    # Load crossover data if it was collected with --crossover
    raw_json = json.loads(DATA_FILE.read_text())
    crossover: dict | None = raw_json.get("crossover")
    if crossover:
        print("  crossover data present in JSON")

    print("Generating GPU figures…")
    fig_paths: dict[str, Path | None] = {
        "roofline":             fig_roofline(hw, records),
        "latency_bars":         fig_latency_bars(hw, records),
        "heatmap":              fig_efficiency_heatmap(hw, records),
        "batch_scaling":        fig_batch_scaling(hw, records),
        "speedup":              fig_speedup(hw, records),
        "framework_comparison": fig_framework_comparison(hw, records),
        "throughput":           fig_throughput(hw, records),
        "throughput_scaling":   fig_throughput_scaling(hw, records),
        "per_sample_latency":   fig_per_sample_latency(hw, records),
        "crossover":            fig_crossover(hw, records, crossover),
        "memory":               fig_gpu_memory(hw, records),
    }
    for name, p in fig_paths.items():
        if p is not None:
            print(f"  {name}: {p}")
        else:
            print(f"  {name}: (skipped — no data)")

    print("Building GPU report…")
    report = build_report(hw, records, fig_paths, crossover=crossover)
    title, _, body = report.partition("\n")
    REPORT_FILE.write_text(
        f"{title}\n\n{partial_run_warning(raw_json)}{provenance(DATA_FILE)}{body}"
    )
    print(f"  Report written → {REPORT_FILE}")


if __name__ == "__main__":
    main()

