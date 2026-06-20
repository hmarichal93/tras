# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRAS (Tree Ring Analyzer Suite) is a dendrochronology tool for automatic tree ring detection and measurement. It provides both a PyQt5 GUI (`tras`) and a unified CLI (`tras_detect`) for processing cross-section images.

## Commands

```bash
make setup       # Install all dependencies (uv sync --dev)
make format      # Format code with ruff
make lint        # Lint/type-check check
make check       # Run all checks (lint + mypy + translations)
make test        # Run all tests
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
      → Export (JSON / CSV / PDF / .POS)
```

### Key modules

| Path | Role |
|------|------|
| `tras/app.py` | PyQt5 main window — integrates all detection methods and annotation UI |
| `tras/api.py` | `detect()` function and `DetectionResult` dataclass — programmatic pipeline entry point |
| `tras/cli/detect.py` | `tras_detect` CLI (single image or batch folder via YAML config) |
| `tras/tree_ring_methods/` | Detection algorithm implementations (see below) |
| `tras/utils/` | Shared helpers: preprocessing, ring sampling, export, report generation |
| `tras/widgets/` | Individual PyQt5 dialog widgets (canvas, export, preprocessing, scale, metadata) |
| `tras/_label_file.py` | JSON annotation file format with base64-embedded image and shape/flag metadata |
| `tras/config/default_config.yaml` | Default UI settings and wood-themed color palette |

### Detection methods (`tras/tree_ring_methods/`)

- **APD** (`apd/`) — Automatic pith detection via local orientation estimation + accumulation. Variants: `apd`, `apd_pcl` (parallel coordinates), `apd_dl` (YOLO-based)
- **CS-TRD** (`cstrd/`) — Classical edge-based ring detection (~73 s CPU)
- **DeepCS-TRD** (`deepcstrd/`) — U-Net deep learning approach (~101 s GPU)
- **INBD** (`inbd/`) — Iterative Next Boundary Detection (CVPR 2023, GPU)
- **U2Net** (`urudendro/`) — Background segmentation via U-Net

Each method has a corresponding helper in `tras/utils/` (`apd_helper.py`, `cstrd_helper.py`, etc.) that bridges the method implementation to the API/CLI.

### Annotation format

Label files are JSON, written/read by `tras/_label_file.py`. Shapes use `labelme`-compatible format: rings are polygons, the pith is a point. Preprocessing metadata (scale factor, background mask) is stored in `flags`.

### CLI batch processing

`tras_detect` accepts a YAML config file for batch jobs. See `examples/cli/process_config.yml` for the full template covering scale, preprocessing, postprocessing, pith/ring method selection, and method-specific parameters.

## Testing conventions

- Tests live in `tests/`, mirroring the package structure
- GUI tests are marked `@pytest.mark.gui` and require `pytest-qt`
- Use `unittest.mock` (MagicMock, patch) for isolation; avoid mocking core math utilities
- `tmp_path` fixture for temporary file I/O

## Tooling notes

- **Linter**: ruff (E/F/I/UP checks); isort via ruff
- **Type checker**: mypy with `check_untyped_defs = true`; missing imports are ignored
- **Versioning**: managed in `pyproject.toml`; release automation via `.github/workflows/release.yml`
- **Python**: ≥3.9, targets 3.9–3.13
