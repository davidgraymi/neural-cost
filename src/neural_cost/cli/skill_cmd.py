"""Print or install the bundled neural-cost agent skill."""

from __future__ import annotations

import argparse

from neural_cost.skill import install_skill, skill_body, skill_text


def register_skill_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "skill",
        help="Print or install the neural-cost agent skill (SKILL.md).",
        description=(
            "Print the agent skill that teaches AI agents to use neural-cost, install it into "
            "a skills directory (e.g. ~/.claude/skills), or print it as an AGENTS.md section."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--install",
        metavar="SKILLS_DIR",
        help="Write SKILLS_DIR/neural-cost/SKILL.md (e.g. ~/.claude/skills or .claude/skills).",
    )
    group.add_argument(
        "--agents-md",
        action="store_true",
        help="Print the skill body without frontmatter, for pasting into AGENTS.md.",
    )
    parser.set_defaults(func=run_skill)


def run_skill(args: argparse.Namespace) -> int:
    if args.install:
        print(f"Installed {install_skill(args.install)}")
    elif args.agents_md:
        print(skill_body())
    else:
        print(skill_text())
    return 0
