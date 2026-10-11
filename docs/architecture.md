# TRAS architecture

How the code is organised, which module owns which decision, and the rules that keep
the GUI, CLI and Python API in agreement. Read this before adding a detector, a
measurement or an export format.

## Layers

```text
            GUI (tras/app.py, tras/widgets/)      CLIs (tras/cli/)       Python users
                         │                              │                     │
                         ▼                              ▼                     ▼
      ┌──────────────── Qt-free core (importable without a display) ───────────────┐
      │ tras/api.py          cross-section pipeline: detect()                       │
      │ tras/cores/          increment-core / path pipeline: analyze_core_image()   │
      │ tras/params.py       typed detection parameters (DetectionParams)           │
      │ tras/analysis/       geometry, ring measurements, QC flags, crossdating     │
      │ tras/io/             dendro file formats: CooRecorder .pos, Tucson .rwl     │
      │ tras/evaluation/     metrics, datasets, benchmarks                          │
      │ tras/utils/          detector bridges (*_helper.py), batch runner, reports   │
      └─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
      tras/tree_ring_methods/   vendored research code (APD, CS-TRD, DeepCS-TRD, INBD,
                                UruDendro); upstream style, not linted
```

**Rule 1: logic lives in the Qt-free core.** Widgets collect parameters, call a core
function in a worker thread, and display the result. A measurement computed in a widget
cannot be tested headless and drifts from the CLI. (Before 2026-10 the GUI and CLI had two
different ring-area implementations; `tras/analysis/rings.py` replaced both.)

**Rule 2: one source of truth per decision.**

| Decision | Owner |
|----------|-------|
| Detection defaults | `tras/config/detection_defaults.yaml` → `tras/params.py` |
| Ring order, areas, widths | `tras/analysis/rings.py::measure_rings` |
| Ray/polygon geometry | `tras/analysis/geometry.py` |
| Core boundary detection | `tras/cores/detect.py::detect_core_rings` |
| Core measurements from edited canvas shapes | `tras/cores/edit.py::cores_from_shapes` |
| `.pos` / `.rwl` syntax | `tras/io/pos.py`, `tras/io/tucson.py` |
| Batch outputs (GUI and CLI) | `tras/utils/batch_helper.py::run_batch` |

**Rule 3: heavy imports are lazy.** torch, the U-Net and the method packages are imported
inside the functions that need them, so `import tras.api` and `tras_core --help` are fast
and the classical core detector works without a GPU stack.

## Cross-section pipeline (`tras.api.detect`)

```text
load_image → preprocess (resize, U2Net background) → locate_pith (manual | APD)
          → detect_rings (CS-TRD | DeepCS-TRD | INBD) → resample_rings_by_rays → save
```

Parameters: `DetectionParams` (frozen dataclass). The legacy flat keywords
(`cstrd_sigma=...`) still work and map onto it. Unknown keywords to `detect()` raise;
unknown keys in a YAML config are rejected with an error that names the key and
suggests the closest valid one.

## Core pipeline (`tras.cores`)

```text
find_cores (axis.py) or user path → extract_ribbon (ribbon.py)
  → boundary evidence: classical (evidence.py) and/or U-Net (deep.py)
  → orientation: curvature (orientation.py), else edge polarity
  → peaks (sub-pixel) → CoreResult (perpendicular widths, years) → export.py
```

Design choices and their evidence are in [cores.md](cores.md) and
[benchmarks.md](benchmarks.md).

## Data model

| Type | Kind | Why |
|------|------|-----|
| `DetectionParams`, `CoreParams` | frozen dataclass | Constructed by us; immutable configs are safe to share across threads |
| `RingMeasurement`, `CoreBoundary`, `CoreResult` | frozen dataclass | Value snapshots of a measurement |
| `PosFile`, `RingWidthSeries` | frozen dataclass | In-memory form of a file; writers and readers round-trip |
| Label-file shapes | `dict` (labelme schema) | Received from / written to JSON; shared with the GUI canvas |

## Testing

- `make test` runs everything headless (`QT_QPA_PLATFORM=offscreen`,
  `TRAS_NO_UPDATE_CHECK=1`; see `tests/conftest.py`). Tests generate their own fixtures,
  so a clean clone passes without model weights or datasets.
- `make lint_strict` (run by CI) formats, lints and type-checks the modules listed in
  `STRICT_PATHS` in the `Makefile`. Add a module there once it is clean.
- Accuracy is measured, not unit-tested: `tras_benchmark` (see
  [benchmarks.md](benchmarks.md)). Re-run it when changing a detector.

## Add a new ring detector

1. Implement it under `tras/tree_ring_methods/<name>/` (vendored) or `tras/<package>/`.
2. Write a bridge `tras/utils/<name>_helper.py` returning rings as `(N, 2)` x/y arrays.
3. Add its parameters to `detection_defaults.yaml` and a `<Name>Params` dataclass in
   `tras/params.py`; extend `RING_METHODS`.
4. Dispatch it in `tras/api.py::detect_rings` (lazy import).
5. Run `tras_benchmark sections <UruDendro> --method <name>` and add the numbers to
   [benchmarks.md](benchmarks.md).
