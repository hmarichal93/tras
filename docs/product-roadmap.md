# TRAS product roadmap: cross-sections first, then increment cores

Last updated: 2026-10-10 · Branch: `claude-full-autonomus`

**Goal.** Make TRAS the reference tool for automatic dendrochronology from images. It should
measure rings on wood cross-sections (our current strength) and on increment cores (the
CooRecorder use case). Every number it reports should be reproducible and checked against
ground truth.

**Order of work.** Fix and harden the cross-section pipeline first (correctness, one shared
analysis core, evaluation). Then add core detection on top of the same building blocks.

## Problems found in the current code (2026-10-10 audit)

| # | Problem | Where | Effect |
|---|---------|-------|--------|
| 1 | Single-image CLI hardcodes its own defaults (`alpha=45`, `rotations=5`, `threshold=0.5`) | `tras/cli/detect.py` | Disagrees with `detection_defaults.yaml` (`60`, `8`, `0.05`); CLI and GUI give different rings for the same image |
| 2 | The `.pos` export is not CooRecorder's format | `tras/utils/pos_exporter.py` | CooRecorder/CDendro cannot open TRAS output |
| 3 | Ring metrics computed twice, differently | `tras/app.py::_prepare_ring_properties_data` vs `tras/utils/ring_properties.py` | GUI handles exclusion zones; CLI does not. Report numbers differ |
| 4 | `logger.info("... %s", x)` with loguru | `tras/api.py` and others | loguru uses `{}`; values never appear in logs |
| 5 | `detect()` takes 25 flat keyword arguments | `tras/api.py`, `tras/cli/detect.py` | Each new parameter touches 4 files; easy to drift (see #1) |
| 6 | Pure-Python loops for polygon area/perimeter and ray resampling | `ring_properties.py`, `ring_sampling.py` | Slow on large batches; fallback branches hide failures |
| 7 | No accuracy evaluation in the repo | — | No way to tell whether a change helps or hurts |
| 8 | No CI for tests/lint | `.github/workflows/` has only `release.yml` | Regressions reach `main` |

## Plan

### Phase 1: Cross-section hardening

1. **Typed detection parameters.** `DetectionParams` dataclass built from
   `detection_defaults.yaml`. `api.detect()`, the CLI, batch and GUI all build one. Fixes #1, #5.
2. **One Qt-free analysis package** (`tras/analysis/`): vectorized polygon metrics, ring
   ordering, radial widths along a line, exclusion zones, eccentricity. The GUI and CLI both call
   it. Fixes #3, #6.
3. **Quality control (QC) flags.** Find rings that cross their neighbours, have outlier widths,
   or are not closed. Show the flags in reports so users know which rings to review.
4. **Standard exports.** Correct CooRecorder `.pos` (writer and reader), Tucson `.rwl`, CSV.
   Fixes #2.
5. **Evaluation.** `tras_benchmark` scores detections against annotations: ring
   precision/recall/F1 for cross-sections, boundary-position error for paths. Fixes #7.

### Phase 2: Increment cores (CooRecorder-style)

A core image is a long, thin strip from pith (or near it) to bark. Rings cross it as roughly
parallel boundaries. The same engine also measures a path drawn on a cross-section.

1. **Path sampling.** Straighten the image along a polyline path (auto-estimated core axis, or
   user-drawn) into a rectified strip.
2. **Boundary evidence.** Two interchangeable sources:
   - classical: oriented intensity gradient across the strip (latewood → earlywood transition);
   - learned: the DeepCS-TRD U-Net boundary probability map, run tiled on the strip.
3. **Boundary picking.** Consensus over several parallel lanes across the core width (robust
   to cracks and resin), then 1-D peak detection with a minimum ring width.
4. **Width perpendicular to the boundary.** Estimate each boundary's local tilt and correct the
   width, as CooRecorder does.
5. **Dating and export.** Assign calendar years from the outermost year, export `.pos`, `.rwl`,
   CSV, and a TRAS label file the GUI can edit.
6. **Benchmark without a core dataset.** Cut virtual cores from the 64 annotated UruDendro
   cross-sections along pith→bark rays. The annotated rings give exact boundary positions.
7. **GUI and CLI.** `tras_detect --mode core`, and a core dialog in the GUI.

### Phase 3: Product polish

- CI workflow (lint, mypy, non-GUI tests) on every push and PR.
- Docs: architecture, cores how-to, benchmark report. Updated README.
- Project Claude skills in `.claude/skills/` for recurring dev tasks (add a detector, run the
  benchmark, release).

## How progress is checked

- Every phase lands with unit tests; `make test` and `make check` must pass.
- Detector changes are judged by `tras_benchmark` numbers, recorded in `docs/benchmarks.md`.
