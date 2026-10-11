---
name: tras-gui-feature
description: Add or change a PyQt5 dialog or menu action in the TRAS GUI following the project's pattern (Qt-free core function, worker thread, editable shapes, headless pytest-qt test). Use for any change under tras/widgets/ or tras/app.py.
---

# Add a GUI feature to TRAS

## Pattern (see `tras/widgets/core_dialog.py` and `batch_dialog.py`)

1. **Core first.** Put the computation in a Qt-free function (`tras/analysis`,
   `tras/cores`, `tras/utils`). Unit-test it without Qt.
2. **Dialog** in `tras/widgets/<name>_dialog.py`: builds widgets, assembles parameters,
   runs the core function in a `QObject` worker moved to a `QThread`
   (`finished`/`error` signals), and tears the thread down in `closeEvent`.
   Export it from `tras/widgets/__init__.py`.
3. **Action** in `tras/app.py`: create it with `action(...)`, add it to the
   `Tools → Single` menu list and to `self.on_load_active_actions` if it needs an image.
   The handler takes `self.imageArray` (or `qimage_to_numpy(self.image)`), opens the dialog,
   and on accept converts shape dicts to `Shape` objects and calls
   `self.loadShapes(shapes, replace=False)` + `self.setDirty()`.
4. **Scale/metadata**: `self.image_scale` (`{'value', 'unit'}`; convert with
   `tras.utils.pos_exporter.mm_per_pixel`), `self.sample_metadata['harvested_year']`.
5. **Strings** wrapped in `self.tr(...)`; after adding strings run `make update_translate`.

## Test

```python
@pytest.mark.gui
def test_my_dialog(qtbot): ...
```

Use `qtbot.waitUntil(...)` for worker results. Tests run headless
(`tests/conftest.py`: offscreen Qt, no update check). Never rely on simulated mouse
click sequences on the canvas; build `Shape` objects directly (see
`tests/labelme_tests/test_app.py::test_MainWindow_annotate_jpg`).

## Manual check

`uv run tras <image>` and exercise the dialog; modal dialogs block headless runs, so keep
network/update dialogs behind `TRAS_NO_UPDATE_CHECK`.
