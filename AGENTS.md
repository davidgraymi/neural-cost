# AGENTS.md

Guidance for AI coding agents working in this repository, and for agents
*using* neural-cost elsewhere.

## Using neural-cost (any project)

The full usage guide is the agent skill at
[`skills/neural-cost/SKILL.md`](skills/neural-cost/SKILL.md). It is also bundled
with the installed package:

```bash
neural-cost skill --install ~/.claude/skills   # Claude Code (personal); or .claude/skills
neural-cost skill --agents-md >> AGENTS.md     # paste-able section for other agents
```

In Claude Code you can also install it as a plugin:
`/plugin marketplace add davidgraymi/neural-cost`, then
`/plugin install neural-cost@neural-cost`.

Reporting rule: always state the device, batch size, number of timed runs and
coverage. Never quote a roofline efficiency without saying how much of the
workload it covers.

## Working on this repository

- Layout: library in `src/neural_cost/` (adapters in `adapters/`, CLI in `cli/`),
  tests in `tests/`, examples in `examples/`.
- The core package must keep **zero mandatory dependencies**. Import torch, jax,
  tensorflow and numpy lazily, and guard tests with `pytest.importorskip`.
- Format and lint: `uv run --extra dev python scripts/format.py` (CI runs it with `--check`).
- Tests: `uv run --extra dev --extra torch --extra jax python -m pytest tests/`.
- When you change a public API, update `README.md` and `skills/neural-cost/SKILL.md`.
  `tests/test_skill.py` executes the skill's Python examples and checks its CLI
  commands exist, so a stale skill fails CI.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:`). They
  drive the semantic-version release pipeline.
