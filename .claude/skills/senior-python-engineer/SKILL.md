---
name: senior-python-engineer
description: Act as a senior Python developer for TRAS. Use when writing, refactoring or reviewing Python code in tras/ (analysis, cores, io, cli, utils, api), designing a module or function API, choosing between class/function/dataclass, fixing typing/lint/test failures, improving performance with NumPy, or when the user asks for "clean code", "refactor", "code review", "make it production quality".
---

# Senior Python developer (TRAS)

You write code that a scientist can read top-down like the algorithm description, that is
typed, tested and fast enough, and that fits the existing architecture
(`docs/architecture.md`). For GUI widgets also follow `tras-gui-feature`; for new detectors
`tras-add-detector`.

## Before writing code

1. Read the module you are changing and its tests; match its naming, docstring density and
   idioms (`from __future__ import annotations`, one import per line, loguru `{}` formatting).
2. Find the **owner** of the decision: defaults → `detection_defaults.yaml`/`params.py`; ring
   metrics → `tras/analysis/rings.py`; file syntax → `tras/io/`; core detection →
   `tras/cores/`. Extend the owner; never re-implement its math elsewhere.
3. Write or update the test first when fixing a bug (reproduce, then fix).

## Code rules

- **Top-level reads like the algorithm.** Each pipeline step is a named function; the
  entry point is a short list of calls (see `tras/api.py::detect`,
  `tras/cores/detect.py::detect_core_rings`).
- **One job per function**, ≤ ~20 lines, nesting ≤ 2 levels, named after what it returns
  or does (`extract_ribbon`, `pick_boundaries`).
- **No magic numbers**: module-level `UPPER_CASE` constants with a comment giving the unit
  and where the value came from (e.g. "tuned on UruDendro tune split").
- **Functions by default; classes only for state.** Results and parameters are
  `@dataclass(frozen=True)`; dicts received from outside (JSON, YAML) are `TypedDict`.
  Shared types go in a `types.py`/the owning module, not in a script.
- **Typed public API**: full annotations, `npt.NDArray[np.float64]` for arrays, document the
  array shape and units in the docstring (`(N, 2) x/y in px`).
- **Qt-free logic.** Nothing in `analysis/`, `cores/`, `io/`, `evaluation/`, `api.py`
  imports Qt. Torch and model packages are imported lazily inside functions.
- **Vectorise with NumPy** over Python loops on pixels/points/rays; measure before and after
  (`time.perf_counter` or `pytest --durations`) and keep results identical (assert in a test).
- **Errors**: validate at the boundary (CLI, API, file readers) with clear messages naming
  the bad value; inside, trust the types. Never swallow exceptions silently; batch code logs
  and counts per-item failures.
- **Units are explicit** in names (`widths_px`, `mm_per_px`, `half_width` in samples) and
  conversions happen in one place.
- **No hidden I/O or global state.** Functions take paths/arrays and return values; writing
  files is a separate, named step. Config is passed in, not read from globals.

## Testing rules

- Pytest, tests mirror the package (`tests/cores_tests/…`). Fixtures are generated
  (`tras/cores/synthetic.py`, `tmp_path`); never depend on `tests/data`.
- Test behaviour and invariants (round-trip write→read, known synthetic boundaries, units),
  not implementation details. Don't mock the maths; mock only I/O, models and Qt dialogs.
- Every bug fix gets a regression test named after the bug.

## Verify before claiming done

```bash
uv run ruff format <changed files>
make lint_strict
uv run pytest tests/ -q -x
```

For detector changes, also run the `tras-benchmark` protocol. Report the exact commands run
and their results; if something fails, show the output.

## Review output format

Findings ranked by severity: **bug** (wrong result/crash, with a concrete input) →
**design** (wrong owner, duplicated logic, leaky abstraction) → **readability** →
**nit**. Each finding: location `file:line`, problem, fix. Then a short "what is good" note
so the author knows what to keep.
