"""validate.py keeps JAX and TensorFlow from reserving GPU memory (issue #95)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROBE = """
import json, os, sys
sys.argv = ["validate.py", *sys.argv[1:]]
sys.path.insert(0, "benchmarks")
import validate
def stop(*a, **k):
    print(json.dumps({k: os.environ.get(k) for k in
        ("XLA_PYTHON_CLIENT_PREALLOCATE", "TF_FORCE_GPU_ALLOW_GROWTH", "JAX_PLATFORMS")}))
    raise SystemExit(0)
validate.detect_hardware = stop
validate.main()
"""


def _env_after_start(*backends: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS",)}
    env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    env.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)
    out = subprocess.run(
        [sys.executable, "-c", PROBE, *(["--backends", *backends] if backends else [])],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_no_preallocation_and_cpu_only_jax():
    env = _env_after_start("torch-cpu", "jax-cpu")
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert env["JAX_PLATFORMS"] == "cpu"


def test_gpu_jax_backend_keeps_all_platforms():
    assert _env_after_start("jax-cpu", "jax-mps")["JAX_PLATFORMS"] is None
    assert _env_after_start()["JAX_PLATFORMS"] is None
