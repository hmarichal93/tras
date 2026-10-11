---
name: github-workflow
description: Land TRAS work through git and GitHub - one branch per task, task board in docs/product-roadmap.md, commits, PR with evidence, CI, merge. Use when starting or finishing a task, committing, opening/updating a PR, checking CI, or when the user says "land it", "open a PR", "merge", "next task". Adapted from polar-tree-ring-tracking's research-loop.
---

# GitHub workflow (TRAS)

One task = one branch = one PR. The task board lives in `docs/product-roadmap.md`
("Task board"); nowhere else.

## Permissions

- Always allowed: create branches, commit, run tests/lint/benchmarks locally.
- **Push, open PRs, comment on issues**: only after the user has authorized it for the
  session (e.g. "you can push", "full autonomy including PRs"). Otherwise prepare the PR
  text and ask once.
- **Merge to `main`, delete branches, force-push, releases**: only with explicit approval.

## Per-task loop

| Step | Do |
|------|----|
| 1. Branch | From fresh `main` (or the current integration branch the user named): `git checkout -b <type>/<slug>` (`feat/`, `fix/`, `docs/`, `refactor/`) |
| 2. Pre-register | Add a row to the task board: task, user job, metric, done-when. Commit it first |
| 3. Build | Delegated by the PI (`cv-pi`) to the role skills |
| 4. Verify | `uv run ruff format <changed>`; `make lint_strict`; `uv run pytest tests/ -q`; benchmark if detection changed (`tras-benchmark`) |
| 5. Commit | Small commits, imperative subject ≤ 72 chars (`Cores: export edited boundaries`), body says why. Stage files by name, never `git add -A` (keeps `uv.lock`, scratch and data out) |
| 6. PR | `gh pr create` with the template below (written with `technical-writing`) |
| 7. CI | `gh pr checks --watch`; fix failures on the branch, never skip hooks |
| 8. Land | After approval: `gh pr merge --merge`; update the board (state, outcome, link), memory |

## PR template

```markdown
<What and why, 2–3 lines.>

## Changes
- …

## Evidence
- Tests: `uv run pytest tests/ -q` → N passed
- Lint: `make lint_strict` → clean
- Benchmark (if detection changed): table before/after, command

## Review focus
- …

Closes #N
```

## Never

- Commit secrets, datasets, model weights or `tests/data`.
- Rewrite published history or push to `main` directly.
- Mix unrelated changes in one PR; split them.
