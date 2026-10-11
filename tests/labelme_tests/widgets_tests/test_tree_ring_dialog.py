"""GUI tests for the cross-section detection dialog (detector stubbed)."""

import numpy as np
import pytest
from PyQt5 import QtWidgets

from tras.widgets import tree_ring_dialog
from tras.widgets.tree_ring_dialog import TreeRingDialog


def _square(r, c=50.0):
    return np.array(
        [[c - r, c - r], [c + r, c - r], [c + r, c + r], [c - r, c + r]], dtype=np.float32
    )


@pytest.mark.gui
def test_dialog_runs_api_detection_in_worker(qtbot, monkeypatch):
    calls = {}

    def fake_detect_rings(image, pith_xy, params):
        calls["params"] = params
        calls["pith"] = pith_xy
        return [_square(10), _square(20)], pith_xy

    monkeypatch.setattr(tree_ring_dialog, "detect_rings", fake_detect_rings)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    dialog = TreeRingDialog(100, 100, image_np=image, initial_cx=50, initial_cy=50)
    qtbot.addWidget(dialog)
    dialog.remove_bg_checkbox.setChecked(False)  # do not depend on saved user settings
    dialog.cstrd_sigma.setValue(2.5)
    dialog.sampling_nr.setValue(36)

    dialog._on_cstrd()

    params = calls["params"]
    assert params.ring_method == "cstrd"
    assert params.cstrd.sigma == pytest.approx(2.5)
    assert calls["pith"] == (50.0, 50.0)
    rings = dialog.get_cstrd_rings()
    assert len(rings) == 2 and rings[0].shape == (36, 2)
    assert dialog.result() == QtWidgets.QDialog.Accepted


@pytest.mark.gui
def test_dialog_reports_detector_errors(qtbot, monkeypatch):
    def failing(*args, **kwargs):
        raise RuntimeError("boom")

    shown = []
    monkeypatch.setattr(tree_ring_dialog, "detect_rings", failing)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: shown.append(a))

    dialog = TreeRingDialog(
        100, 100, image_np=np.zeros((100, 100, 3), np.uint8), initial_cx=50, initial_cy=50
    )
    qtbot.addWidget(dialog)
    dialog.remove_bg_checkbox.setChecked(False)
    dialog._on_deepcstrd()

    assert shown, "an error dialog must be shown"
    assert dialog.get_deepcstrd_rings() is None
