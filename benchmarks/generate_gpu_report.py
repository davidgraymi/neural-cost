"""Generate figures and a markdown GPU benchmark report from benchmark_gpu_data.json.

Usage:
    python benchmarks/generate_gpu_report.py

Reads:   benchmarks/results/benchmark_gpu_data.json
Writes:  benchmarks/results/figures/gpu_*.png
         GPU_BENCHMARK_REPORT.md  (repo root)
"""

from __future__ import annotations

import json
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
DATA_FILE   = REPO_ROOT / "benchmarks" / "results" / "benchmark_gpu_data.json"
FIG_DIR     = REPO_ROOT / "benchmarks" / "results" / "figures"
REPORT_FILE = REPO_ROOT / "GPU_BENCHMARK_REPORT.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    "ConvNeXt": "ConvNeXt-Tiny 4-stage hierarchy (7×7 depthwise conv, 1×1 pointwise inverted bottleneck, LayerNorm, GELU, input 224×224×3)",
    "ViT": "Vision Transformer Tiny (16×16 patch embedding, 4-layer Transformer encoder with 4 heads, MLP ratio 4, seq_len 197, input 224×224×3)",
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
        return str(p.relative_to(REPO_ROOT))

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
        desc = ARCH_DESCRIPTIONS.get(arch, "Neural network architecture under test")
        cat = "Modern Vision" if arch in ("ConvNeXt", "ViT") else "Core / Legacy"
        arch_rows.append(f"| **{arch}** | {cat} | {desc} |")
    arch_table_md = "\n".join(arch_rows)

    ridge = hw["ridge_point"]
    tf_device = hw.get("torch_device", hw["name"])

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

CUDA event timing measures only GPU kernel execution time on CUDA devices. On Apple Silicon MPS,
synchronous timing barriers (`torch.mps.synchronize()`, `jax.block_until_ready()`, and `tf.experimental.async_wait()`)
force command buffer flushing and device retirement to eliminate host scheduling jitter.

### Architectures under test

{arch_table_md}

### Frameworks and optimisation variants

| Framework | Baseline | Optimised | Notes |
|---|---|---|---|
| **PyTorch** | Eager GPU (`mps`) | `torch.compile()` | PyTorch MPS backend with Inductor compilation |
| **JAX** | Eager XLA GPU (`mps`) | `jax.jit()` | `jax-mps` MLX-backed Metal plugin with whole-graph XLA JIT |
| **TensorFlow** | Eager GPU | `tf.function(jit_compile=True)` | TensorFlow graph execution with XLA compilation |

### Measurement protocol

- **Warmup:** {hw.get('warmup', 15)} iterations (full compilation, cache warm, Metal kernel pipeline initialization)
- **Timed repeats:** {hw.get('repeats', 40)} samples per configuration
- **Statistics reported:** median, mean, σ (stddev), CV%, p95
- **Roofline efficiency:** `min(1, lower_bound / observed)`
- **Batch sizes swept:** {batches}

---

## Figure GPU-1 — Roofline Model

Each point represents one architecture × framework combination (optimised variant).

![GPU Roofline]({rel(fig_paths['roofline'])})

**Key observations:**
- **Hardware Ridge Point ({ridge:.1f} FLOP/B):** The M1 GPU's ridge point is much higher than typical CPUs (5–10 FLOP/B). Workloads with arithmetic intensity below {ridge:.1f} FLOP/B are fundamentally memory-bandwidth bound.
- **Modern High-AI Architectures:** Modern vision models cross into the compute-bound regime:
  - **ViT** achieves $I = 60.3$ to $89.0$ FLOP/B, well past the ridge point, attaining up to **1,297.5 GFLOP/s** (~50.0% of theoretical peak) in JAX JIT.
  - **ConvNeXt** sits right on the ridge point ($I = 35.5$ to $38.3$ FLOP/B), delivering **732.6–763.2 GFLOP/s** in JAX JIT and **738.3 GFLOP/s** in PyTorch.
- **Memory-Bound Legacy Workloads:** Small models like FF DNN ($I = 6.1$ to $21.6$ FLOP/B) and recurrent nets (RNN/LSTM, $I = 2.9$ to $11.7$ FLOP/B) sit deep on the memory-bandwidth slope where performance is throttled by memory roundtrips and kernel dispatch overhead.
- **XLA and Compiler Fusion:** Whole-graph compilation (`jax.jit()` and `torch.compile()`) eliminates intermediate tensor writes to unified memory, raising operational arithmetic intensity and shifting points closer to the roofline ceiling.

---

## Figure GPU-2 — Inference Latency by Architecture (batch={ref_batch})

Error bars show ±1σ across timed iterations.

![GPU Latency bars]({rel(fig_paths['latency_bars'])})

**Key observations:**
- **Modern Vision Workloads:** Handling full $224 \\times 224$ images at batch={ref_batch}:
  - **ConvNeXt:** JAX JIT finishes in **48.7 ms** vs PyTorch baseline at **387.3 ms** (and PyTorch compiled at 406.7 ms).
  - **ViT:** JAX JIT completes in **21.5 ms** (1,203 GFLOP/s) vs PyTorch compiled at **132.3 ms** and PyTorch baseline at **232.9 ms**.
- **Recurrent Network Fusion:** Uncompiled sequential loops suffer catastrophic command buffer dispatch overhead. JAX JIT fuses all 32 sequential steps into a single Metal command buffer, reducing **LSTM latency from 201.8 ms to 4.01 ms** (50.3× speedup) and **RNN from 68.5 ms to 1.68 ms** (40.8× speedup).
- **Small Model Dispatch Floor:** For FF DNN, median execution time is sub-millisecond (0.23–0.58 ms across frameworks). At this scale, Metal command buffer encoding and host-device synchronization latency dominate actual GPU ALU execution.

---

## Figure GPU-3 — Roofline Efficiency Heatmap (batch={ref_batch})

![GPU Efficiency heatmap]({rel(fig_paths['heatmap'])})

**Key observations:**
- **Peak Utilization in Attention & Dense Convolutions:** ViT reaches the highest roofline efficiency (**46.3%** in JAX JIT, **43.2%** in PyTorch compiled at batch=32, reaching **49.9%** at batch=256). Large GEMM projections and multi-head attention matrix multiplications effectively saturate the M1 GPU's 8 cores and 128 execution units.
- **ConvNeXt Efficiency:** ConvNeXt achieves **30.3%** efficiency in JAX JIT and **28.4%** in PyTorch baseline. The 7×7 depthwise convolutions have lower arithmetic intensity than standard convolutions, slightly tempering peak efficiency.
- **Recurrent Model Contrast:** Eager PyTorch and TensorFlow exhibit < 1% roofline efficiency on RNN/LSTM due to sequential kernel launch starvation. JAX JIT elevates LSTM to **23.0%** efficiency at batch=32.

---

## Figure GPU-4 — Latency Scaling with Batch Size

![GPU Batch scaling]({rel(fig_paths['batch_scaling'])})

**Key observations:**
- **Sub-linear Scaling at Small Batches ($B < 32$):** Latency grows sub-linearly because constant kernel launch costs and weight memory fetch are amortized across batch items.
- **Linear Scaling in Compute-Bound Regime ($B \\ge 32$):** For dense models like ViT and ConvNeXt, once GPU execution units are fully saturated, latency scales linearly with batch size ($T(B) \\propto B$), meaning throughput plateaus.
- **Memory Wall and Divergence at $B=256$:**
  - **PyTorch ConvNeXt OOM:** At batch=256, PyTorch ConvNeXt exceeds the Metal allocation limit (`20.13 GiB max allowed`) and aborts with OutOfMemory, whereas JAX JIT completes batch=256 in **373.7 ms** with predictable memory allocation.
  - **PyTorch ViT Swapping:** PyTorch ViT latency degrades from 232.9 ms (B=32) to **2,204 ms** (B=256 baseline) and **3,568 ms** (B=256 compiled) due to unified memory swapping and Metal allocator thrashing. Meanwhile, JAX JIT scales gracefully to **159.7 ms** (1,297.5 GFLOP/s).

---

## Figure GPU-5 — Compilation / JIT Speedup (batch={ref_batch})

Speedup ratio = eager latency / optimised latency. Higher is better.

![GPU Speedup]({rel(fig_paths['speedup'])})

**Key observations:**
- **JAX JIT Loop Fusion:** JAX JIT yields massive speedups on sequential models: **50.3× on LSTM** (201.8 ms → 4.01 ms) and **40.8× on RNN** (68.5 ms → 1.68 ms), plus **4.17× on ViT** (89.8 ms → 21.5 ms) and **2.85× on ConvNeXt** (138.7 ms → 48.7 ms).
- **PyTorch Inductor on MPS:** `torch.compile()` provides a **1.76× speedup on ViT** (232.9 ms → 132.3 ms) and **1.58× on Transformer** (3.66 ms → 2.32 ms) through operator fusion and pointwise kernel codegen. However, it shows no speedup on ConvNeXt where depthwise convolutions already dispatch via MPSGraph.
- **TensorFlow XLA:** `tf.function(jit_compile=True)` achieves **13.5× on FF DNN** (3.07 ms → 0.23 ms) and **7.8×–12.0× on recurrent models**, eliminating Python graph traversal overhead.

---

## Figure GPU-6 — Achieved Throughput (GFLOP/s, batch={ref_batch})

![GPU Throughput]({rel(fig_paths['throughput'])})

**Key observations:**
- **Hardware Ceiling:** M1 GPU theoretical peak FP32 throughput is 2,600 GFLOP/s.
- **Top Performers:** JAX JIT ViT leads all models with **1,203 GFLOP/s** at batch=32 (and **1,297 GFLOP/s** at batch=256), followed closely by PyTorch compiled ViT (**1,124 GFLOP/s**) and ConvNeXt (**738 GFLOP/s**).
- **Legacy Models:** CNN achieves 300–498 GFLOP/s, Transformer achieves 193–384 GFLOP/s, while FF DNN achieves 17–33 GFLOP/s due to memory bandwidth limits.

---

## Figure GPU-7 — Throughput Scaling with Batch Size

![GPU Throughput scaling]({rel(fig_paths['throughput_scaling'])})

**Key observations:**
- **Throughput Saturation Plateau:** Throughput rises steeply between batch=1 and batch=32 as execution units fill, then asymptotes between batch=32 and batch=256. For example, JAX ViT scales from 1,203 GFLOP/s (B=32) to 1,297 GFLOP/s (B=256) — an increase of only 7.8% despite an 8× increase in batch size.
- **Stability under Load:** JAX JIT maintains monotonic throughput scaling up to batch=256 across all architectures, whereas PyTorch exhibits performance degradation at batch=256 due to memory subsystem pressure.

---

## Deep Dive: Batch Size Scaling Limits — Can Linear Performance Scaling Continue?

A natural question when looking at batch scaling curves is: **"Since we see linear scaling on batch size, can we continue scaling and see the same linear performance?"**

The data and computer architecture principles show definitively that **linear performance scaling cannot continue indefinitely**. The illusion of "linear scaling" at small batch sizes reflects the amortization of constant overheads, which transitions into a strict physical ceiling governed by the Roofline Model, followed by catastrophic memory degradation.

1. **The Amortization Regime (B < 32): The Illusion of Linear Scaling**
   - At small batch sizes, total execution time T(B) is dominated by constant overheads: Python runtime dispatch, command buffer encoding, Metal kernel launch latency (~20–50 µs), and cold parameter DRAM transfers:
     `T(B) ≈ T_0 + c · B ≈ T_0`
   - Since arithmetic work (FLOPs) scales as O(B) while latency remains nearly flat (T_0), throughput appears to scale linearly:
     `Throughput(B) = FLOPs(B) / T(B) ∝ B / T_0 ∝ B`
   - This is **not** true hardware scaling; it is merely paying down fixed host-driver latency overhead.

2. **The Roofline Saturation Plateau (B = 32 to 256): Hardware Limit Reached**
   - Once batch size is sufficient to saturate all 8 GPU cores and 128 Execution Units (EUs), and operational intensity crosses the ridge point (I ≥ 38.2 FLOP/B), the GPU enters the **compute-bound plateau**.
   - In this regime, execution time scales directly with batch size (T(B) ∝ B).
   - As a result, throughput strictly plateaus:
     `Throughput(B) = O(B) / O(B) ≈ Peak GFLOP/s = constant`
   - In our empirical results, JAX ViT achieves 1,203 GFLOP/s at B=32 and 1,297 GFLOP/s at B=256. An **8× increase in batch size** yielded only a **1.08× throughput gain**, demonstrating that the hardware is almost completely saturated.

3. **The Memory Wall and Allocation Cliff (B > 256): Severe Degradation and OOM**
   - While parameter memory is constant, activation tensor footprint scales as O(B · L · H).
   - **Metal Buffer Limit (OOM):** On Apple Silicon unified memory, PyTorch ConvNeXt at B=256 exhausted the Metal single-buffer watermark (`20.13 GiB max allowed`), causing a hard OutOfMemory crash.
   - **Unified Memory Swapping:** For PyTorch ViT at B=256, activation footprint exceeded available physical RAM, triggering macOS unified memory swapping to the internal NVMe SSD. Latency exploded from **232.9 ms (B=32) to 2,204 ms (eager) and 3,568 ms (compiled)** — a 15× latency slowdown and a collapse in achieved throughput.
   - **Activation DRAM Bandwidth Saturation:** At large batch sizes, intermediate activation spills overwhelm the GPU cache hierarchy and saturate the 68 GB/s unified memory bus, pulling even high-AI models back down into a memory-bandwidth-bound bottleneck.

**Conclusion:** Scaling batch size yields diminishing throughput returns up to the hardware roofline ceiling, followed by severe latency penalties or hard out-of-memory crashes. The optimal batch size for throughput on Apple Silicon M1 GPU lies between **B=32 and B=64**.

---

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
2. **Key Performance Findings:**
   - **Peak Throughput on ViT:** JAX JIT achieved **1,203.1 GFLOP/s** at batch=32 and **1,297.5 GFLOP/s** at batch=256 (**49.9% roofline efficiency**), setting the benchmark's highest observed FP32 throughput on the M1 GPU.
   - **Massive Recurrent Speedups:** Whole-graph XLA JIT unrolls and fuses sequential time steps into a single Metal command buffer, delivering **50.3× speedup on LSTM** (201.8 ms → 4.01 ms) and **40.8× on RNN** (68.5 ms → 1.68 ms).
   - **Memory Allocator Resilience:** Unlike PyTorch MPS, which hit the Metal buffer watermark (`20.13 GiB max allowed`) causing OOM on ConvNeXt at batch=256, JAX JIT successfully executed both ConvNeXt (373.7 ms) and ViT (159.7 ms) without memory fragmentation or swapping.

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

**Key observations:**
- **Dynamic Overhead Ratio:** Observed peak device memory exceeds theoretical minimum tensor storage due to kernel scratchpads, GEMM workspaces, activation retention, and framework runtime contexts. Modern vision models exhibit lower overhead ratios because large parameter and activation weights dominate framework overhead.
- **Allocator Caching and Pooling:** PyTorch MPS aggressively pools allocations to amortize Metal command buffer allocation costs. However, at batch=256 this caching policy causes severe fragmentation on large models (triggering OOM on ConvNeXt and memory swapping on ViT). JAX with MLX backend manages unified memory allocations with tighter recycling.
"""

    md += """
*Generated by `benchmarks/generate_gpu_report.py` using [neural-cost](https://github.com/davidgraymi/neural-cost)*
"""
    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
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
        "roofline":           fig_roofline(hw, records),
        "latency_bars":       fig_latency_bars(hw, records),
        "heatmap":            fig_efficiency_heatmap(hw, records),
        "batch_scaling":      fig_batch_scaling(hw, records),
        "speedup":            fig_speedup(hw, records),
        "throughput":         fig_throughput(hw, records),
        "throughput_scaling": fig_throughput_scaling(hw, records),
        "crossover":          fig_crossover(hw, records, crossover),
        "memory":             fig_gpu_memory(hw, records),
    }
    for name, p in fig_paths.items():
        if p is not None:
            print(f"  {name}: {p}")
        else:
            print(f"  {name}: (skipped — no data)")

    print("Building GPU report…")
    report = build_report(hw, records, fig_paths, crossover=crossover)
    REPORT_FILE.write_text(report)
    print(f"  Report written → {REPORT_FILE}")


if __name__ == "__main__":
    main()

