"""GUI tests for the core analysis dialog."""

import numpy as np
import pytest

from tras.cores.synthetic import centre_path as horizontal_path
from tras.cores.synthetic import synthetic_core
from tras.widgets.core_dialog import CoreDialog


@pytest.mark.gui
def test_core_dialog_runs_on_drawn_path(qtbot, tmp_path):
    image = synthetic_core()
    dialog = CoreDialog(image, [horizontal_path(image)], mm_per_px=0.01, outer_year=2020)
    qtbot.addWidget(dialog)
    assert dialog.drawn.isChecked()
    dialog.evidence.setCurrentIndex(dialog.evidence.findData("classical"))
    dialog.min_ring_width.setValue(5)

    with qtbot.waitSignal(dialog.apply_button.clicked, timeout=1, raising=False):
        dialog._on_run()
    qtbot.waitUntil(lambda: dialog.apply_button.isEnabled(), timeout=20000)

    assert len(dialog.results) == 1
    assert len(dialog.results[0].widths_px) == 9
    shapes = dialog.label_shapes()
    assert shapes[0]["shape_type"] == "linestrip"
    assert shapes[-1]["label"] == "core1_2020"
    assert dialog.outer_ring_year() == 2020
    assert dialog.apply_button.isDefault()
    assert not hasattr(dialog, "export_button")  # export happens after canvas edits
    assert "Export Core Measurements" in dialog.status_label.text()


@pytest.mark.gui
def test_core_dialog_defaults_to_auto_without_paths(qtbot):
    dialog = CoreDialog(np.zeros((50, 50, 3), np.uint8), [], mm_per_px=None, outer_year=None)
    qtbot.addWidget(dialog)
    assert dialog.auto_paths.isChecked()
    assert not dialog.drawn.isEnabled()
    assert dialog.analysis_kwargs()["paths"] is None
    assert dialog.outer_ring_year() is None
