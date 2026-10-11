# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRAS (Tree Ring Analyzer Suite) is a dendrochronology tool for automatic tree ring detection and measurement on wood cross-sections and increment cores. It provides a PyQt5 GUI (`tras`), CLIs (`tras_detect` for cross-sections, `tras_core` for cores, `tras_benchmark` for accuracy) and a Python API. Code map and design rules: `docs/architecture.md`.

## Commands

```bash
make setup       # Install all dependencies (uv sync --dev)
make format      # Format code with ruff
make lint        # ruff format check + ruff lint (no type check)
make check       # Run all checks (lint + mypy + translations); legacy code still fails
make lint_strict # Format/lint/type gate on STRICT_PATHS (what CI enforces)
make test        # Run all tests (headless; no datasets or model weights needed)
make build       # Build wheel
```

Run a single test:
```bash
uv run pytest tests/utils_tests/test_ring_sampling.py::test_build_rays_xy -v
uv run pytest tests/ -v -m "not gui"   # skip GUI tests
```

The project uses `uv` as the package manager — always prefer `uv run` over bare `python`.

## Architecture

### Core pipeline (data flow)
```
Image → Preprocessing (crop/resize/bg removal)
      → Pith Detection (APD family or manual)
      → Ring Detection (CS-TRD / DeepCS-TRD / INBD)
      → Ring Resampling (radial ray casting, 360 points)
      → Ring Properties (area, perimeter, radial width)
      → Export (JSON / CSV / PDF / .POS / .rwl)
```

### Key modules

| Path | Role |
|------|------|
| `tras/app.py` | PyQt5 main window — integrates all detection methods and annotation UI |
| `tras/api.py` | `detect()` function and `DetectionResult` dataclass — cross-section pipeline entry point |
| `tras/params.py` | `DetectionParams` frozen dataclass; legacy flat kwargs (`cstrd_sigma=`) map onto it |
| `tras/analysis/` | Qt-free geometry, `measure_rings` (areas, per-ray widths, exclusions), QC flags, crossdating (`dating.py`) |
| `tras/cores/` | Increment-core / path ring detection (`analyze_core_image`), exports to .pos/.rwl/CSV |
| `tras/io/` | CooRecorder `.pos` and Tucson `.rwl` readers/writers |
| `tras/evaluation/` | Metrics, dataset loaders (UruDendro, CooRecorder), core benchmark |
| `tras/cli/core.py`, `tras/cli/benchmark.py` | `tras_core` and `tras_benchmark` CLIs |
| `tras/cli/detect.py` | `tras_detect` CLI (single image or batch folder via YAML config) |
| `tras/tree_ring_methods/` | Detection algorithm implementations (see below) |
| `tras/utils/` | Shared helpers: preprocessing, ring sampling, export, report generation, batch runner (`batch_helper.py`) |
| `tras/widgets/` | Individual PyQt5 dialog widgets (canvas, export, preprocessing, scale, metadata, batch) |
| `tras/_label_file.py` | JSON annotation file format with base64-embedded image and shape/flag metadata |
| `tras/config/default_config.yaml` | Default UI settings and wood-themed color palette |
| `tras/config/detection_defaults.yaml` | Single source of truth for detection defaults (method, pith, sampling, per-method params) |

### Detection defaults (single source of truth)

`tras/config/detection_defaults.yaml` holds the default detection parameters. Load it via
`tras/config/detection_defaults.py::get_detection_defaults()` (cached). The GUI (`tree_ring_dialog.py`,
`batch_dialog.py`), the API (`api.py`), and the CLI config layer (`cli_config.py`) all read from it —
when adding or changing a detection parameter, update this YAML so every entry point stays in sync.

### Detection methods (`tras/tree_ring_methods/`)

- **APD** (`apd/`) — Automatic pith detection via local orientation estimation + accumulation. Variants: `apd`, `apd_pcl` (parallel coordinates), `apd_dl` (YOLO-based)
- **CS-TRD** (`cstrd/`) — Classical edge-based ring detection (131 s/disc CPU; `docs/benchmarks.md`)
- **DeepCS-TRD** (`deepcstrd/`) — U-Net deep learning approach (45 s/disc on a GTX 1650)
- **INBD** (`inbd/`) — Iterative Next Boundary Detection (CVPR 2023, GPU)
- **U2Net** (`urudendro/`) — Background segmentation via U-Net

Each method has a corresponding helper in `tras/utils/` (`apd_helper.py`, `cstrd_helper.py`, etc.) that bridges the method implementation to the API/CLI.

### Annotation format

Label files are JSON, written/read by `tras/_label_file.py`. Shapes use `labelme`-compatible format: rings are polygons, the pith is a point. Preprocessing metadata (scale factor, background mask) is stored in `flags`.

### Batch processing

The Tools menu has **Cross-section** and **Increment Core** submenus (per-image workflows; scale and metadata appear in both) and **Batch Processing (Cross-sections)…**. Core paths are linestrips labelled `core_path` (drawn via *Draw Core Path*); other linestrips are open rings. Core exports (*Export Core Measurements…*) are rebuilt from the canvas shapes by `tras/cores/edit.py` (`core_path_<k>` + boundary lines; hand-drawn lines are labelled `core_boundary`), so user corrections reach the `.pos`/`.rwl`/CSV.

- **CLI**: `tras_detect` accepts a YAML config file for batch jobs. See `examples/cli/process_config.yml` for a template covering scale, preprocessing, postprocessing, pith/ring method selection, and method-specific parameters.
- **GUI**: `tras/widgets/batch_dialog.py` (`BatchProcessDialog`) configures settings inline and runs `tras/utils/batch_helper.py::run_batch` in a worker thread. It iterates every image in the selected folder and **writes outputs into that same input folder**: one `.json` label file per image, a combined `summary.pdf` (one image per page, ring overlay, filename as title), a CLI-compatible `batch_config.yml` of the settings used, and `radii.rwl` (N/E/S/W radii; + `radii_ids.csv` on series-ID collisions) when an outer year and a scale are set. Images that already have a `.json` are skipped, so a batch can be resumed. `batch_helper.py` is Qt-free and reuses `api.detect()`, `cli_config.get_detection_params()`, and `report_pdf.generate_summary_pdf()`.

### Design rules

- Logic belongs in the Qt-free modules; widgets only gather parameters and display results.
- One owner per decision (defaults → YAML/`params.py`, ring metrics → `analysis/rings.py`,
  file syntax → `tras/io/`). Do not re-implement area/width math in widgets.
- Import torch/model packages lazily inside functions.
- Detector changes are judged with `tras_benchmark` (numbers in `docs/benchmarks.md`); tune on
  `--split tune`, report on `--split test` and the CooRecorder set.
- UruDendro's `pith_location.csv` header says `cy,cx` but the columns are x, y.

## How work is organized (team of skills)

The `cv-pi` skill is the orchestrator: it decides what to build, delegates each job to a
**fresh subagent** of the matching role, judges the results and lands them. Start every
autonomous session by invoking `cv-pi`.

| Role agent (`.claude/agents/`) | Skill (`.claude/skills/`) | Job |
|--------------------------------|---------------------------|-----|
| — (main session) | `cv-pi` | Plan, decide, delegate, judge, land |
| `product-engineer` | `senior-product-engineer` | Problem, metric, options, MVP, done-when |
| `domain-expert` | `dendro-domain-expert` | Dating, units, formats, lab practice |
| `gui-designer` | `senior-gui-designer` | Screens, menus, workflow, screenshots |
| `python-engineer` | `senior-python-engineer` | Code, API, refactors, tests |
| `qa-engineer` | `senior-qa-engineer` | Test plan, edge cases, merge gate |
| `benchmark-analyst` | `tras-benchmark` | Accuracy numbers |
| `docs-writer` | `docs-accuracy`, `technical-writing` | Docs match the code; commands run |
| — | `github-workflow` | Branch → PR → CI → merge, task board |

The task board is in `docs/product-roadmap.md`. Push/PR/merge need the user's authorization
(see `github-workflow`). A missing role gets a new skill and agent before work is delegated.

## Testing conventions

- Tests live in `tests/`, mirroring the package structure
- GUI tests are marked `@pytest.mark.gui` and require `pytest-qt`
- Use `unittest.mock` (MagicMock, patch) for isolation; avoid mocking core math utilities
- `tmp_path` fixture for temporary file I/O
- Tests generate their own fixtures (`tras/cores/synthetic.py`); never depend on `tests/data` (gitignored)
- `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` and `TRAS_NO_UPDATE_CHECK=1`

## Tooling notes

- **Linter**: ruff (E/F/I/UP checks, line length 100); isort via ruff; `tras/tree_ring_methods` (vendored) excluded
- **CI**: `.github/workflows/ci.yml` runs `make lint_strict` and the full test suite
- **Type checker**: mypy with `check_untyped_defs = true`; missing imports are ignored
- **Versioning**: managed in `pyproject.toml`; release automation via `.github/workflows/release.yml`
- **Python**: ≥3.9, targets 3.9–3.13
