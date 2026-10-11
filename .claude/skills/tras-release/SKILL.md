---
name: tras-release
description: Prepare and cut a TRAS release (checks, docs, version bump through the GitHub release workflow). Use when the user asks to release, publish, bump the version, or prepare release notes.
---

# Release TRAS

1. On a branch, run the gates: `make test` and `make lint_strict` (CI runs both).
2. Re-run the benchmarks touched by the release (`tras-benchmark` skill); update
   `docs/benchmarks.md` and the README accuracy table if numbers changed.
3. Check docs for new commands/options: README quick start, `docs/cores.md`,
   `examples/cli/process_config.yml`, `CLAUDE.md` key-module table.
4. Merge to `main` via PR (description: what/why, evidence, how tested; end with the
   Claude Code attribution line when Claude wrote it).
5. Version bump and tag are automated: GitHub → Actions → **Create Release** →
   choose patch/minor/major and paste release notes. It edits `pyproject.toml`, tags and
   publishes. Do not bump the version by hand.
6. Model weights are release assets; if they changed, upload them and update
   `tools/download_release_assets.py`.

Release notes: lead with user-visible changes (new commands, formats, accuracy numbers),
then fixes, then developer changes.
