---
title: "Contributing"
description: "Development setup, formatting and the semantic-version release pipeline."
section: "Reference"
order: 5
---

For local development or running benchmarks directly from the source repository:

```bash
git clone https://github.com/davidgraymi/neural-cost.git
cd neural-cost
pip install -e '.[dev,torch,jax,tensorflow]'
```

## Development

Format and lint the codebase with a single command:

```bash
# Auto-format and fix linting in place:
python scripts/format.py
# Or using the shell wrapper:
./scripts/format.sh
# Or via Make:
make format

# Check formatting and linting without modifying files:
python scripts/format.py --check
# Or:
make check
```

## Automated Semantic Versioning & Release Pipeline

`neural-cost` uses an automated semantic versioning and release pipeline in GitHub Actions (`.github/workflows/ci.yml` and `.github/workflows/publish.yml`) powered by [`paulhatch/semantic-version`](https://github.com/paulhatch/semantic-version) with path filtering and PyPI trusted publishing.

Version upgrades are determined strictly by Conventional Commits that modify files in the distributed library (`src/`, `pyproject.toml`):

| Change Kind | Commit Conventional Type | Affects Library (`src/`)? | Version Increment | Release Action |
|---|---|---|---|---|
| **Breaking change** | `feat!:`, `fix!:`, `BREAKING CHANGE:` | ✅ Yes | **Major** (`X.0.0`) | Create tag & publish PyPI release |
| **New feature** | `feat:`, `feat(scope):` | ✅ Yes | **Minor** (`0.X.0`) | Create tag & publish PyPI release |
| **Bug fix / performance** | `fix:`, `perf:`, `refactor:` | ✅ Yes | **Patch** (`0.0.X`) | Create tag & publish PyPI release |
| **Non-library changes** | `docs:`, `chore:`, `ci:`, `test:`, benchmarks, workflows | ❌ No | **None** (unchanged) | No tag created; PyPI publish skipped |
| **Doc updates to library** | `docs:`, `style:` (docstrings only) | ✅ Yes | **None** (unchanged) | No tag created; PyPI publish skipped |

- **In CI (PRs)**: The `semver-check` workflow job automatically analyzes commits and library paths in the PR and prints a summary indicating whether a version bump is required and what the new version will be.
- **On Push to `main`**: The `publish` workflow triggers on changes to `src/**` or `pyproject.toml`. If a version increment is detected, it automatically creates and pushes the git tag, generates the GitHub Release, builds sdist & wheel, and publishes to PyPI.
