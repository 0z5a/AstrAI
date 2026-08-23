# Contributing to AstrAI

Thank you for your interest in contributing! This document provides step-by-step guidelines.

## Quick Start

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e ".[dev]"     # install with dev dependencies (pytest, ruff)
```

## Before You Commit

Run the following checks **in order** — CI will reject if any fail.

### 1. Format

```bash
ruff format .
```

### 2. Import sorting

```bash
ruff check . --select I
```

If this fails, **manually fix** import ordering (ruff does not auto-fix in this project's CI):

```bash
ruff check . --select I --fix .
ruff format .    # re-format after fix
```

### 3. Run tests

```bash
python -u -m pytest tests/ -v
```

> Failed tests may leave orphan tempdirs under the system temp directory
> (`$TMPDIR` on Linux/macOS, `%TEMP%` on Windows). Clean them manually if needed.

### 4. (Optional) Full pre-commit check script

If you have `bash` available (Git Bash on Windows works too):

```bash
bash scripts/pre_commit.sh
```

The script installs development dependencies by default, then runs the format
check, import sort check, and tests. If dependencies are already installed, use:

```bash
bash scripts/pre_commit.sh --skip-deps
```

## Commit Style

```
type: short description (~50 chars)

- bullet point body (each ~60 chars)
```

- **Type** must be one of: `fix`, `feat`, `chore`, `docs`, `refactor`, `perf`, `test`, `style`, `ci`, `build`, `revert`.
- **Subject line** ends with no period.
- **Body** uses bullet points starting with `-`.
- No `(scope)` parentheses.

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `ruff check --select I` fails | Wrong import order | `ruff check . --select I --fix .` then `ruff format .` |
| `ruff format` changed many files | Not formatted before commit | Review diff carefully before staging |
| Pre-commit check script fails | Dependency install, tests, or lint failed | Fix the failing step; use `--skip-deps` only when dependencies are already installed |
| Tests fail with tempdir left | Test crash | Clean `%TEMP%` manually |

## Submitting Changes

1. Fork the repo.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make changes following the steps above.
4. Commit with the commit style above.
5. Push: `git push origin feat/my-feature`
6. Open a Pull Request against `main`.

## Code Review

- All PRs are reviewed. We may request changes.
- CI runs `ruff format --check .` then `ruff check . --select I` (no `--fix` in CI).
- Ensure all tests pass.

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).

---

Questions? Ask in [GitHub Discussions](https://github.com/ViperEkura/AstrAI/discussions) or open an issue.
