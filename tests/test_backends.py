"""Backend comparison across JAX platforms in subprocesses (issue #57)."""

import json
import textwrap

import pytest

from neural_cost import BackendComparison, BackendResult, compare_backends
from neural_cost.backends import _error_line


def comparison():
    return BackendComparison(
        (
            BackendResult(
                "cpu",
                "cpu:0",
                0.020,
                hardware_name="Apple M3 (CPU)",
                peak_flops=6e11,
                memory_bandwidth=1e11,
                efficiency=0.2,
                bottleneck="memory",
                coverage=1.0,
            ),
            BackendResult("mps", "mps:0", 0.005, hardware_name="Apple M3 (10-core GPU)"),
            BackendResult("cuda", error="worker failed (1): RuntimeError: no cuda"),
        ),
        baseline="cpu",
    )


def test_speedups_render_and_dict():
    result = comparison()
    assert result.speedup("mps") == pytest.approx(4.0)
    assert result.speedup("cpu") == pytest.approx(1.0)
    assert result.speedup("cuda") is None and result.speedup("tpu") is None
    assert result.fastest().platform == "mps"
    text = result.render()
    assert "speedup vs cpu" in text and "4.00x" in text
    assert "0.60 TFLOP/s, 100 GB/s" in text and "20.0% (memory-bound)" in text
    assert "cuda: worker failed (1): RuntimeError: no cuda" in text
    data = result.to_dict()
    assert data["results"][1]["speedup"] == pytest.approx(4.0)
    assert json.dumps(data)


def test_error_line_prefers_exception():
    stderr = "Traceback...\nRuntimeError: backend missing\nFor simplicity, JAX has removed frames"
    assert _error_line(stderr) == "RuntimeError: backend missing"
    assert _error_line("") == "no output"
    assert _error_line("x" * 500).endswith("…")


TARGET = """
def build():
    import jax, jax.numpy as jnp
    return jax.jit(lambda x, w: jnp.tanh(x @ w)), (jnp.ones((8, 16)), jnp.ones((16, 16)))

def bad():
    return 42
"""


@pytest.fixture
def target(tmp_path):
    pytest.importorskip("jax")
    path = tmp_path / "bench.py"
    path.write_text(textwrap.dedent(TARGET))
    return path


def test_compare_backends_runs_each_platform_in_subprocess(target):
    result = compare_backends(f"{target}:build", ["cpu", "no_such_platform"], warmup=1, repeats=2)
    cpu, missing = result.results
    assert cpu.ok and cpu.device and cpu.hardware_device_type == "cpu"
    assert cpu.efficiency is not None and cpu.coverage == 1.0
    assert not missing.ok and "no_such_platform" in missing.error
    assert result.baseline == "cpu"


def test_bad_factory_reports_error(target):
    result = compare_backends(f"{target}:bad", ["cpu"], warmup=0, repeats=1)
    assert "TypeError" in result.results[0].error
    with pytest.raises(ValueError):
        compare_backends(f"{target}:build", [" "])


def test_cli_compare_platforms_json(target, capsys):
    from neural_cost.cli.root import build_parser

    args = build_parser().parse_args(
        ["compare", f"{target}:build", "--platforms", "cpu", "--repeats", "2", "--json"]
    )
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline"] == "cpu" and payload["results"][0]["speedup"] == 1.0

    no_platforms = build_parser().parse_args(["compare", f"{target}:build"])
    assert no_platforms.func(no_platforms) == 2
