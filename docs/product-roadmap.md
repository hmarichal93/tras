# TRAS product roadmap: cross-sections first, then increment cores

Last updated: 2026-10-10 · Branch: `claude-full-autonomus`

**Goal.** Make TRAS the reference tool for automatic dendrochronology from images. It should
measure rings on wood cross-sections (our current strength) and on increment cores (the
CooRecorder use case). Every number it reports should be reproducible and checked against
ground truth.

**Order of work.** Fix and harden the cross-section pipeline first (correctness, one shared
analysis core, evaluation). Then add core detection on top of the same building blocks.

## Problems found in the 2026-10-10 audit (all fixed in T1)

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
   - learned: the DeepCS-TRD U-Net boundary probability map, run on the natural image crop
     around the path, then straightened into the strip.
3. **Boundary picking.** Consensus over several parallel lanes across the core width (robust
   to cracks and resin), then 1-D peak detection with a minimum ring width.
4. **Width perpendicular to the boundary.** Estimate each boundary's local tilt and correct the
   width, as CooRecorder does.
5. **Dating and export.** Assign calendar years from the outermost year, export `.pos`, `.rwl`,
   CSV, and a TRAS label file the GUI can edit.
6. **Benchmark without a core dataset.** Cut virtual cores from the 64 annotated UruDendro
   cross-sections along pith→bark rays. The annotated rings give exact boundary positions.
7. **GUI and CLI.** `tras_core` CLI and the Core Analysis dialog (done).

### Phase 3: Product polish

- CI workflow (lint, mypy, non-GUI tests) on every push and PR.
- Docs: architecture, cores how-to, benchmark report. Updated README.
- Project Claude skills in `.claude/skills/` for recurring dev tasks (add a detector, run the
  benchmark, release).

## Task board

Managed by the `tras-pi` orchestrator (see `CLAUDE.md`, "How work is organized").

| # | Task | User job | Metric / done when | State |
|---|------|----------|--------------------|-------|
| T1 | Phases 1–3 above | All | 135 tests at close (2026-10-11), benchmarks in `docs/benchmarks.md` | Done (2026-10-11) |
| T2 | Export core measurements from the edited canvas (not the raw detection); *Draw Core Boundary* tool | Lab researcher correcting cores | Dragged/added/deleted boundaries change .pos/.rwl/CSV; years relabelled; GUI test | Done 2026-10-11 (8a1487b; QA fixes 59724dd; hand-drawn lines kept on re-run daeca21) |
| T3 | Crossdating entry point (`tras_dating` CLI + report): series vs master, COFECHA-style flags | Lab researcher dating series | Flags a shifted series in a synthetic set; reads TRAS and dplR `.rwl` | Next |
| T4 | QC flags visible on the canvas and in the ring list | Anyone reviewing detections | Flagged rings highlighted; GUI test | Next |
| T5 | Incomplete outer ring: `core<k>_bark` marker; exported to CSV/.pos, left out of .rwl | Lab researcher (Schulman dating) | Bark ring never appears as a full width in .rwl | Next |
| T6 | Missing rings (0-width) and crack/gap pairs, written as CooRecorder gap markers in .pos | Lab researcher with damaged cores | Round-trip with CooRecorder conventions; domain review | Next |
| T7 | Provenance: boundaries dragged by hand marked user-edited (not detector strength) | QA of corrected series | Edited lines carry no `strength=` | Next |
| T8 | **User manual** for non-programmers (`docs/manual/`): install, first disc, correcting rings, cores, batch, exports explained, troubleshooting, glossary; real screenshots from `tools/manual_screenshots.py` | Dendro student / lab technician | A novice follows it start to finish without help; every menu path matches the GUI | Next (after T2) |
| T9 | **Installer without terminal**. Decision (2026-10-11): conda `constructor` installer per OS built in GitHub Actions, CPU-only torch, weights downloaded in-app on first use. Prerequisites: T9a user-writable model store + download dialog (`tras/utils/model_store.py`, replaces labelme `widgets/download.py`); T9b bundled-install update path (link, not git); T9c `packaging/constructor/` + `installers.yml` with silent-install smoke test | Dendro student on Windows | Clean Windows 11 VM: download → DeepCS-TRD result with 0 terminal commands, ≤ 5 clicks, ≤ 15 min | T9a next. PI decisions 2026-10-11: v1 CPU-only (GPU build later); offline installer (~1 GB) accepted, students often have poor lab internet; move to shapely 2 if `tras_benchmark` is unchanged (reverse if F1 drops > 0.005); ship unsigned v1 with "More info → Run anyway" steps + screenshot in the manual. Escalated to Henry: code-signing certificate (SignPath Foundation, free for OSS, needs his application) |
| T10 | Ring overlay readability: thick filled ring strokes hide the wood at fit-to-window zoom, so corrections are hard; scale line width with zoom, thin outlines by default | Anyone correcting rings | Wood visible between rings at 25 % zoom; GUI-designer review with screenshots | Done 2026-10-11 (bdaa8ff) |
| T11 | **Fix the record**: cite the *Forestry* version of record (99(4) cpag046, doi:10.1093/forestry/cpag046, verified) in CITATION.cff + README; remove "no public core dataset" claims (two CC BY 4.0 core datasets exist) | Anyone citing or evaluating TRAS | Citations resolve; docs agree with `docs/research/competitive-landscape.md` | Done 2026-10-11 (13b6620) |
| T14 | Heidelberg `.fh` reader/writer (TSAP-Win, LignoVision, CDendro) | Labs using Rinntech tools | Round-trip tests; domain review | Next |
| T15 | Earlywood/latewood boundaries and widths: port `hmarichal93/cstrd` branch `early_late_wood` (check its licence first), EW/LW in cores and `.rwl` | Lab researchers (paper already claims EW/LW areas) | EW/LW widths exported; benchmarked where GT exists | Later |
| T16 | Blue intensity along core paths (after domain review) | Dendroclimatology labs | Matches CooRecorder BI on a shared sample | Later |

Priority order set by the PI on 2026-10-11 from `docs/research/competitive-landscape.md`: T11 → T8 (manual) → T10 (overlay) → T9 (installer) → T3 (crossdating; note Fritts, a new MIT PyQt6 app, overlaps) → T14 → T5/T6 → T15 → T16. TRAS is a GUI product: no benchmarking or detector-tuning tasks (Henry, 2026-10-11; former T12/T13 dropped).

## How progress is checked

- Every phase lands with unit tests; `make test` and `make check` must pass.
- Detector changes are judged by `tras_benchmark` numbers, recorded in `docs/benchmarks.md`.
