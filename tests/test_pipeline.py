"""Pipeline profiling with call counts and unexplained remainder (issue #54)."""

import json
import textwrap

import pytest

from neural_cost import (
    HardwareSpec,
    Measurement,
    Operation,
    Pipeline,
    Stage,
)
from neural_cost.adapters.base import FrameworkAdapter


class FakeAdapter(FrameworkAdapter):
    """Returns a fixed time per function so the arithmetic is exact."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.calls = []

    def operations(self, model, example_inputs):
        return [Operation("mm", "matmul", ((4, 8), (8, 8)), (4, 8))]

    def benchmark(self, function, *args, warmup=3, repeats=10, **kwargs):
        self.calls.append((function, args, kwargs))
        return Measurement(self.seconds[function], (self.seconds[function],), device="cpu:0")


def tree():
    pass


def env(state, action=None):
    pass


def net(obs):
    pass


def search():
    pass


TIMES = {tree: 0.300, env: 0.0116, net: 0.0057, search: 0.614}


def test_stage_split_and_unexplained_remainder():
    pipeline = Pipeline(
        {"tree": tree, "env": (env, (1,), 16), "net": Stage(net, (2,), calls=16)},
        total=search,
    )
    report = pipeline.run(adapter=FakeAdapter(TIMES))
    stages = {s.name: s for s in report.stages}
    assert stages["env"].total_seconds == pytest.approx(16 * 0.0116)
    assert stages["net"].share == pytest.approx(16 * 0.0057 / 0.614)
    expected_sum = 0.300 + 16 * 0.0116 + 16 * 0.0057
    assert report.stage_sum_seconds == pytest.approx(expected_sum)
    assert report.unexplained_seconds == pytest.approx(0.614 - expected_sum)
    assert report.unexplained_ratio == pytest.approx((0.614 - expected_sum) / 0.614)
    assert report.dominant().name == "tree"
    text = report.render()
    assert "measured total 614.000 ms on cpu:0" in text
    assert text.index("tree") < text.index("env") < text.index("net")  # sorted by total
    assert "unexplained" in text and "dispatch / loop overhead" in text


def test_without_total_shares_sum_to_one_and_kwargs_pass_through():
    adapter = FakeAdapter(TIMES)
    report = Pipeline({"env": Stage(env, (1,), 2, kwargs={"action": 3}), "net": net}).run(
        adapter=adapter
    )
    assert sum(s.share for s in report.stages) == pytest.approx(1.0)
    assert report.unexplained_seconds is None and report.unexplained_ratio is None
    assert (env, (1,), {"action": 3}) in adapter.calls
    assert "sum" in report.render()


def test_overlap_note_when_stages_exceed_total():
    report = Pipeline({"env": (env, (1,), 100)}, total=search).run(adapter=FakeAdapter(TIMES))
    assert report.unexplained_seconds < 0
    assert "stages may overlap" in report.render()


def test_per_stage_roofline_with_hardware():
    hardware = HardwareSpec("test", 1e12, 1e11, device_type="cpu")
    ops = [Operation("e", "elementwise", ((1000,),), (1000,))]
    pipeline = Pipeline({"traced": net, "given": Stage(tree, operations=ops)})
    report = pipeline.run(adapter=FakeAdapter(TIMES), hardware=hardware)
    assert all(stage.gap is not None for stage in report.stages)
    assert "-bound" in report.render()
    untraced = Pipeline({"net": (net, (1,))}).run(hardware=hardware, warmup=0, repeats=1)
    assert untraced.stages[0].gap is None and "no operations" in untraced.stages[0].note


def test_real_benchmark_default_and_validation():
    report = Pipeline({"noop": (lambda: None, (), 3)}).run(warmup=0, repeats=2)
    assert report.stages[0].calls == 3
    with pytest.raises(ValueError):
        Pipeline({})
    with pytest.raises(ValueError):
        Stage(net, calls=-1)
    with pytest.raises(TypeError):
        Pipeline({"bad": 42})


def test_cli_pipeline_json(tmp_path, capsys):
    from neural_cost.cli import main

    script = tmp_path / "pipe.py"
    script.write_text(
        textwrap.dedent(
            """
            from neural_cost import Pipeline, Stage

            def build():
                return Pipeline({"a": Stage(lambda: None, calls=4)}, total=lambda: None)
            """
        )
    )
    main(["pipeline", f"{script}:build", "--warmup", "0", "--repeats", "2", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"][0]["name"] == "a" and payload["stages"][0]["calls"] == 4
    assert payload["measured_total_seconds"] > 0
