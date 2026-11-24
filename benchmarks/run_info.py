"""Shared plumbing for the benchmark scripts.

Every script times through ``adapter.benchmark()`` (the same code users run),
stamps its results with :func:`run_metadata`, and writes them under
``benchmarks/results/<device>/`` so data from different hardware never mixes.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"

#: Distributions whose versions are recorded with every run (read from package
#: metadata, so recording them never imports a framework).
FRAMEWORK_DISTRIBUTIONS = ("torch", "jax", "jaxlib", "tensorflow", "keras", "numpy")


def git_commit() -> tuple[str | None, bool | None]:
    """Commit the benchmarked code came from, and whether it had local changes.

    Tries, in order: ``NEURAL_COST_COMMIT`` (set it when running from a copy
    with no ``.git``), the repository checkout, then the commit pip recorded
    when neural-cost was installed from a git URL.  The dirty flag is ``None``
    when it cannot be known.
    """
    env = os.environ.get("NEURAL_COST_COMMIT")
    if env:
        return env, None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        direct_url = metadata.distribution("neural-cost").read_text("direct_url.json")
        commit = json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
        return commit, None
    except (metadata.PackageNotFoundError, ValueError):
        return None, None


def git_describe() -> str | None:
    """``git describe`` of the checkout (``v0.12.2-12-g4f11804-dirty``), if any.

    Editable installs keep the version computed when they were built, so a
    long-lived environment can report a release far older than the code it
    runs; the checkout's own description is the reliable one.
    """
    try:
        return subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def neural_cost_version() -> str:
    from neural_cost import __version__

    return __version__


def run_metadata(device: str, **extra: Any) -> dict[str, Any]:
    """Where and with what a result was produced: attach it to every result file."""
    commit, dirty = git_commit()
    return {
        "neural_cost_version": neural_cost_version(),
        "git_describe": git_describe(),
        "git_commit": commit,
        "git_dirty": dirty,
        "device": device,
        "device_slug": device_slug(device),
        "frameworks": {
            name: version
            for name in FRAMEWORK_DISTRIBUTIONS
            if (version := _version(name)) is not None
        },
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }


def describe_run(run: dict[str, Any] | None) -> str:
    """One-line provenance for report headers."""
    if not run:
        return "provenance not recorded (results predate run metadata)"
    commit = (run.get("git_commit") or "unknown commit")[:12]
    if run.get("git_dirty"):
        commit += " + local changes"
    elif run.get("git_dirty") is None and run.get("git_commit"):
        commit += " (clean state unknown)"
    version = run.get("git_describe") or run.get("neural_cost_version", "?")
    frameworks = ", ".join(f"{k} {v}" for k, v in sorted(run.get("frameworks", {}).items()))
    return (
        f"neural-cost {version} @ {commit}"
        f" on {run.get('device', '?')}, {run.get('created_utc', '?')}"
        f" (Python {run.get('python', '?')}; {frameworks or 'no frameworks'})"
    )


def device_slug(name: str) -> str:
    """Directory-safe machine name: ``"Apple M3 (8-core GPU)"`` → ``"apple-m3"``.

    Parenthesised details are dropped so the CPU and GPU scripts on one chip
    share a directory, while ``"Intel(R) Xeon(R)"`` keeps its name.
    """
    name = re.sub(r"\s+\([^)]*\)", "", name)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "unknown-device"


def results_dir(device: str, override: Path | None = None) -> Path:
    """``benchmarks/results/<device-slug>/`` (created), or ``override`` when given."""
    path = override if override is not None else RESULTS_ROOT / device_slug(device)
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_results_dir(filename: str) -> Path:
    """The per-device results directory whose ``filename`` was written most recently."""
    candidates = sorted(
        RESULTS_ROOT.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            f"no benchmarks/results/<device>/{filename}; run the collector first or pass --results-dir"
        )
    return candidates[0].parent


def accelerator_name(torch_device: Any = None) -> str | None:
    """Marketing name of a CUDA GPU (``"NVIDIA Tesla T4"``) if one is in use."""
    if torch_device is None or not str(torch_device).startswith("cuda"):
        return None
    import torch

    return torch.cuda.get_device_name(torch_device)


def timing_stats(measurement: Any) -> dict[str, float | int | None]:
    """Latency summary in milliseconds from a ``Measurement``."""
    samples = sorted(s * 1e3 for s in measurement.samples_seconds)
    n = len(samples)
    avg = mean(samples)
    sd = stdev(samples) if n > 1 else 0.0
    overhead = getattr(measurement, "sync_overhead_seconds", None)
    # The profiler reports busy time summed over its ``repeats`` iterations
    # (one per sample); on CPU it is host op time under the profiler, not
    # accelerator time, so it is left out.
    device_time = getattr(measurement, "trace_device_time_seconds", None)
    if device_time is not None:
        on_cpu = str(getattr(measurement, "device", "") or "cpu").lower().startswith("cpu")
        device_time = None if on_cpu else device_time / n
    return {
        "median": median(samples),
        "mean": avg,
        "stddev": sd,
        "cv": 100 * sd / avg if avg > 0 else 0.0,
        "p95": samples[min(int(0.95 * n), n - 1)],
        "min": samples[0],
        "relative_spread": getattr(measurement, "relative_spread", None),
        "calls_per_sample": getattr(measurement, "calls_per_sample", 1),
        "sync_overhead_us": overhead * 1e6 if overhead is not None else None,
        "trace_device_ms": device_time * 1e3 if device_time is not None else None,  # per call
    }
