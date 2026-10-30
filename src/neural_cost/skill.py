"""Access the bundled agent skill (``skills/neural-cost/SKILL.md``).

The skill ships inside the wheel, so its version always matches the
installed package.  ``neural-cost skill`` prints it, installs it into an
agent's skills directory, or prints the body without frontmatter for pasting
into an ``AGENTS.md``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

SKILL_NAME = "neural-cost"


def skill_text() -> str:
    """Return the full ``SKILL.md`` (YAML frontmatter plus body)."""
    bundled = resources.files("neural_cost").joinpath("agent_skill", "SKILL.md")
    if bundled.is_file():
        return bundled.read_text(encoding="utf-8")
    # Source checkout / editable install: read the canonical file from the repo.
    source = Path(__file__).resolve().parents[2] / "skills" / SKILL_NAME / "SKILL.md"
    if source.is_file():
        return source.read_text(encoding="utf-8")
    raise FileNotFoundError("the neural-cost skill is not bundled with this installation")


def skill_body() -> str:
    """``SKILL.md`` without its frontmatter: usable as an ``AGENTS.md`` section."""
    text = skill_text()
    if text.startswith("---"):
        _, _, rest = text[3:].partition("\n---")
        return rest.lstrip("\n")
    return text


def install_skill(skills_dir: str | Path) -> Path:
    """Write ``SKILL.md`` to ``<skills_dir>/neural-cost/SKILL.md`` and return the path.

    For Claude Code use ``~/.claude/skills`` (personal) or ``.claude/skills``
    (project).
    """
    target = Path(skills_dir).expanduser() / SKILL_NAME / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(skill_text(), encoding="utf-8")
    return target
