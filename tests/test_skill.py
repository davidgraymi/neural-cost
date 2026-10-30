"""The agent skill stays correct: frontmatter, runnable examples, real CLI commands (issue #59)."""

import json
import re
from pathlib import Path

import pytest

from neural_cost.cli.root import build_parser
from neural_cost.skill import install_skill, skill_body, skill_text

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "neural-cost" / "SKILL.md"
PYTHON_BLOCKS = re.findall(r"```python\n(.*?)```", SKILL.read_text(), flags=re.DOTALL)


def test_frontmatter_and_bundled_copy():
    text = SKILL.read_text()
    front = text.split("---")[1]
    assert re.search(r"^name: neural-cost$", front, flags=re.MULTILINE)
    description = re.search(r"^description: (.+)$", front, flags=re.MULTILINE).group(1)
    assert 50 < len(description) <= 1024
    assert skill_text() == text
    assert not skill_body().startswith("---") and skill_body().startswith("# neural-cost")


@pytest.mark.parametrize("block", PYTHON_BLOCKS, ids=lambda b: b.split("\n")[0][:40])
def test_python_examples_run(block):
    for framework in ("torch", "jax", "tensorflow"):
        if re.search(rf"^import {framework}|^from {framework}", block, flags=re.MULTILINE):
            pytest.importorskip(framework)
    exec(compile(block, str(SKILL), "exec"), {"__name__": "skill_example"})  # noqa: S102


def test_cli_commands_in_skill_exist():
    commands = set(re.findall(r"`neural-cost (\w+)", SKILL.read_text()))
    assert commands >= {"hardware", "profile", "compare", "pipeline", "audit", "benchmark"}
    subparsers = next(
        action for action in build_parser()._actions if action.dest == "command"
    ).choices
    assert commands <= set(subparsers), commands - set(subparsers)


def test_install_and_cli(tmp_path, capsys):
    path = install_skill(tmp_path)
    assert path == tmp_path / "neural-cost" / "SKILL.md" and path.read_text() == skill_text()
    parser = build_parser()
    args = parser.parse_args(["skill", "--agents-md"])
    assert args.func(args) == 0
    assert capsys.readouterr().out.startswith("# neural-cost")


def test_plugin_manifests():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    marketplace = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert plugin["name"] == "neural-cost"
    assert marketplace["plugins"][0]["name"] == plugin["name"]
    assert marketplace["plugins"][0]["source"] == "./"
