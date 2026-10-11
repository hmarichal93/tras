---
name: tras-add-detector
description: Add or modify a ring/pith detection method in TRAS (cross-section or core) so the API, CLI, GUI, batch runner and benchmarks all pick it up. Use when integrating a new model or algorithm, adding a detector parameter, or changing detection defaults.
---

# Add a detector to TRAS

Read `docs/architecture.md` first. The rule: one owner per decision, logic in the Qt-free core.

## Cross-section ring detector

1. Code: vendored research code under `tras/tree_ring_methods/<name>/` (not linted) or a
   clean package under `tras/`.
2. Bridge: `tras/utils/<name>_helper.py::detect_rings_<name>(image, center_xy, ...) -> list[(N,2) x/y arrays]`.
3. Defaults: add a section to `tras/config/detection_defaults.yaml`.
4. Params: add `<Name>Params` frozen dataclass in `tras/params.py` (defaults read from the
   YAML), a field on `DetectionParams`, extend `RING_METHODS`, add flat keys to `_FLAT_KEYS`
   so `detect(<name>_param=...)` and CLI flags work.
5. Dispatch: `tras/api.py::detect_rings`, importing the helper **inside** the function.
6. CLI: add `--<name>-<param>` options in `tras/cli/detect.py` and pass them in
   `detection_flags`.
7. GUI: `tras/widgets/tree_ring_dialog.py` and `batch_dialog.py` read defaults via
   `get_detection_defaults()`.
8. Tests: params round-trip (`tests/test_params.py`), CLI flag mapping
   (`tests/cli_tests/test_detect.py`). Mock the helper; do not run models in unit tests.
9. Accuracy: `tras_benchmark sections ... --method <name>` → `docs/benchmarks.md`
   (use the `tras-benchmark` skill).

## Core boundary evidence

1. Add an evidence function returning `Evidence(score, tilt)` on the ribbon grid
   (see `tras/cores/detect.py::deep_evidence`).
2. Register the name in `EVIDENCE_SOURCES`, combine in `combine_evidence`, set tuned
   defaults in `tras/cores/pipeline.py`.
3. Synthetic tests in `tests/cores_tests/` (use `tras.cores.synthetic`), then benchmark.

## Gate

`make test` and `make lint_strict` must pass; add any new clean module to `STRICT_PATHS`
in the `Makefile`.
