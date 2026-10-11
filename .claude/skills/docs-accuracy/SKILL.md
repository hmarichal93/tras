---
name: docs-accuracy
description: Audit and fix TRAS documentation so every statement matches the code and data - commands run, flags and defaults exist, menu paths match the GUI, numbers match benchmark output, links resolve. Use after any feature lands, before a PR or release, when docs are edited, or when the user asks "are the docs right", "update the docs", "README". Pair with technical-writing for style.
---

# Documentation accuracy (TRAS)

Docs are part of the product: a wrong command or menu path costs the user more than a
missing one. Style comes from `technical-writing`; this skill makes sure the content is
**true today**.

## Who reads TRAS docs

TRAS is a **GUI product for non-programmers**: dendrochronology students, lab technicians,
forestry analysts. They never open a terminal if they can avoid it. Developers are the
second audience. So:

- **The user manual (`docs/manual/`) comes first.** Every user-facing feature has a manual
  page: what it is for, numbered steps with exact menu paths and button names as they
  appear on screen, a screenshot per key step, what the output files mean, and
  "If something goes wrong" (symptom → what to do).
- No code, no jargon in the manual. Explain domain terms once (glossary) and software terms
  never ("JSON file" → "TRAS project file (.json)").
- **Screenshots are mandatory.** Every manual page has at least one screenshot, and every
  step where the user must find a menu, button, dialog field or result on screen has its
  own screenshot (the menu opened, the dialog filled in, the result on the canvas). A manual
  page or a GUI change without up-to-date screenshots fails the audit. Number callouts or
  arrows (drawn with PIL in the screenshot script) mark where to click.
- Screenshots are real: render the GUI offscreen (`QT_QPA_PLATFORM=offscreen`, drive the
  app programmatically, `window.grab().save(...)`), crop to the relevant area, PNG < 300 KB,
  in `docs/manual/img/`. Regenerate them when the GUI changes (keep the script in
  `tools/manual_screenshots.py`).
- Developer docs (CLI, API, architecture, benchmarks) stay separate.

## Scope

`docs/manual/` (user manual, highest priority), `README.md`, `CLAUDE.md`, `docs/*.md`, `examples/`, `.claude/skills/*/SKILL.md`,
CLI `--help` texts, GUI tooltips that describe workflow, PR descriptions.

## Audit procedure

1. **Inventory claims.** For each doc, list the checkable statements: commands, flags,
   defaults, file names/outputs, menu paths and action names, module/function names,
   numbers (accuracy, timings, test counts), links, dates.
2. **Verify each against the source of truth**:

   | Claim type | Source of truth | How to check |
   |------------|-----------------|--------------|
   | CLI command / flag / default | `tras/cli/*.py` | `uv run <cli> --help`; run the example on synthetic or local data |
   | Detection defaults | `tras/config/detection_defaults.yaml`, `tras/params.py`, `CoreParams` | grep |
   | Menu path / action name | `tras/app.py` menus | grep the `self.tr("…")` strings; render offscreen if unsure |
   | Module / function | code | `grep -rn "def name"` |
   | Accuracy number | `docs/benchmarks.md` (itself from `tras_benchmark`) | other docs must quote it, not restate different values |
   | Output files | the writer functions | run and `ls` the output |
   | Links / images | repo tree | every relative link resolves |
   | Test count, Python versions | `pytest -q`, `pyproject.toml` | run |
   | Citation (BibTeX, `CITATION.cff`, paper links) | The publisher / arXiv record | arXiv MCP `get_abstract` or web search: title, authors, year, venue, pages, DOI must match; DOI resolves. Never invent a DOI: omit it and flag it |

3. **Run every code block** that can run without private data (use
   `tras/cores/synthetic.py` images in the scratchpad). Commands needing datasets: check the
   flags exist and paths are explained.
4. **Fix**: correct the doc (not the code) unless the code is wrong; then flag it to the
   PI. One fact lives in one place; elsewhere link to it.
5. **Report**: table of claim → doc:line → status (ok / fixed / code bug) → evidence.

## Rules

- No manual page without screenshots; GUI changes regenerate the affected screenshots
  in the same PR (`tools/manual_screenshots.py`).

- Never invent numbers; quote `docs/benchmarks.md` with its date.
- Dates absolute; remove stale "TODO", "coming soon", old names (`labelme` paths, removed buttons).
- Keep CLAUDE.md short and true: it is loaded into every session.
- When behaviour changed in the same PR, update docs in that PR.
