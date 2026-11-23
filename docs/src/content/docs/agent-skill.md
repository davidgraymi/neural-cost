---
title: "AI agent skill"
description: "Install the bundled skill so coding agents can profile models with neural-cost."
section: "Reference"
order: 4
---

neural-cost ships an agent skill ([`skills/neural-cost/SKILL.md`](https://github.com/davidgraymi/neural-cost/blob/main/skills/neural-cost/SKILL.md)).
It covers when to use the library, which extra to install, per-framework
recipes, how to read the report, caveats, and a reporting rule. The skill is
bundled with the package, so its version matches the API:

```bash
neural-cost skill --install ~/.claude/skills   # Claude Code; or .claude/skills per project
neural-cost skill --agents-md                  # print as an AGENTS.md section for other agents
```

Or as a Claude Code plugin: `/plugin marketplace add davidgraymi/neural-cost`
then `/plugin install neural-cost@neural-cost`. CI runs the skill's examples
(`tests/test_skill.py`).
