"""Core analysis dialog: automatic ring detection on increment cores (CooRecorder-style).

The user chooses where to measure (cores found automatically, or paths drawn as
linestrips), the outer year and detector settings. Detection runs in a worker thread.
Results are returned to the main window as label shapes (one ``linestrip`` path and one
``line`` per boundary, all editable) and can be exported to CooRecorder ``.pos``,
Tucson ``.rwl`` and CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from tras.cores import CoreResult
from tras.cores import analyze_core_image
from tras.cores import default_core_params
from tras.cores import export_results
from tras.cores import to_label_shapes

_EVIDENCE = [
    ("", "Best available (deep model if installed)"),
    ("deep", "Deep (DeepCS-TRD U-Net, GPU recommended)"),
    ("classical", "Classical (intensity edges, CPU)"),
    ("hybrid", "Hybrid (deep + classical)"),
]
_PITH_SIDES = [
    ("auto", "Automatic"),
    ("start", "Start of path / left or top of core"),
    ("end", "End of path / right or bottom of core"),
]


class CoreWorker(QtCore.QObject):
    """Runs :func:`analyze_core_image` off the GUI thread."""

    finished = QtCore.pyqtSignal(object)  # list[CoreResult]
    error = QtCore.pyqtSignal(str)

    def __init__(self, image: np.ndarray, kwargs: dict[str, Any]):
        super().__init__()
        self._image = image
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            self.finished.emit(analyze_core_image(self._image, **self._kwargs))
        except Exception as exc:  # pragma: no cover - surfaced to the dialog
            logger.exception("Core analysis failed")
            self.error.emit(str(exc))


class CoreDialog(QtWidgets.QDialog):
    """Configure and run core ring detection on the open image."""

    def __init__(
        self,
        image: np.ndarray,
        drawn_paths: list[np.ndarray],
        mm_per_px: Optional[float],
        outer_year: Optional[int],
        image_name: str = "core",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Core Analysis"))
        self.setMinimumWidth(480)
        self._image = image
        self._drawn_paths = drawn_paths
        self._mm_per_px = mm_per_px
        self._image_name = image_name
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[CoreWorker] = None
        self.results: list[CoreResult] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_path_group())
        layout.addWidget(self._build_dating_group(outer_year))
        layout.addWidget(self._build_detector_group())
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addLayout(self._build_buttons())

    # ----- layout ----------------------------------------------------------

    def _build_path_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(self.tr("Where to measure"))
        form = QtWidgets.QVBoxLayout(group)
        self.auto_paths = QtWidgets.QRadioButton(self.tr("Find cores automatically"))
        self.drawn = QtWidgets.QRadioButton(
            self.tr("Use drawn paths (%d linestrip shape(s))") % len(self._drawn_paths)
        )
        self.drawn.setEnabled(bool(self._drawn_paths))
        (self.drawn if self._drawn_paths else self.auto_paths).setChecked(True)
        hint = QtWidgets.QLabel(
            self.tr("Tip: draw a path with Edit → Create LineStrip, from pith to bark.")
        )
        hint.setStyleSheet("color: gray")
        for widget in (self.auto_paths, self.drawn, hint):
            form.addWidget(widget)
        return group

    def _build_dating_group(self, outer_year: Optional[int]) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(self.tr("Dating and scale"))
        form = QtWidgets.QFormLayout(group)
        self.outer_year = QtWidgets.QSpinBox()
        self.outer_year.setRange(0, 3000)
        self.outer_year.setSpecialValueText(self.tr("undated"))
        self.outer_year.setValue(outer_year or 0)
        form.addRow(self.tr("Year of outermost ring:"), self.outer_year)
        scale_text = (
            self.tr("%.5f mm/px") % self._mm_per_px if self._mm_per_px
            else self.tr("not set (Tools → Set Scale); widths in px, no .pos")
        )
        form.addRow(self.tr("Scale:"), QtWidgets.QLabel(scale_text))
        self.pith_side = QtWidgets.QComboBox()
        for value, label in _PITH_SIDES:
            self.pith_side.addItem(self.tr(label), value)
        form.addRow(self.tr("Pith side:"), self.pith_side)
        return group

    def _build_detector_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(self.tr("Detector"))
        form = QtWidgets.QFormLayout(group)
        self.evidence = QtWidgets.QComboBox()
        for value, label in _EVIDENCE:
            self.evidence.addItem(self.tr(label), value)
        form.addRow(self.tr("Boundary evidence:"), self.evidence)
        self.min_ring_width = QtWidgets.QDoubleSpinBox()
        self.min_ring_width.setRange(1.0, 500.0)
        self.min_ring_width.setValue(5.0)
        self.min_ring_width.setSuffix(" px")
        self.min_ring_width.setToolTip(self.tr("Closest allowed spacing of two boundaries"))
        form.addRow(self.tr("Minimum ring width:"), self.min_ring_width)
        self.sensitivity = QtWidgets.QDoubleSpinBox()
        self.sensitivity.setRange(0.5, 10.0)
        self.sensitivity.setSingleStep(0.25)
        self.sensitivity.setValue(2.5)
        self.sensitivity.setToolTip(self.tr("Classical threshold; lower finds more (and weaker) rings"))
        form.addRow(self.tr("Sensitivity (classical):"), self.sensitivity)
        return group

    def _build_buttons(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton(self.tr("Detect rings"))
        self.run_button.clicked.connect(self._on_run)
        self.export_button = QtWidgets.QPushButton(self.tr("Export .pos / .rwl / .csv…"))
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        self.apply_button = QtWidgets.QPushButton(self.tr("Add to image"))
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.accept)
        close = QtWidgets.QPushButton(self.tr("Close"))
        close.clicked.connect(self.reject)
        for button in (self.run_button, self.export_button, self.apply_button, close):
            row.addWidget(button)
        return row

    # ----- run ---------------------------------------------------------------

    def analysis_kwargs(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            "pith_side": self.pith_side.currentData(),
            "min_ring_width": float(self.min_ring_width.value()),
            "sensitivity": float(self.sensitivity.value()),
        }
        if self.evidence.currentData():
            overrides["evidence"] = self.evidence.currentData()
        return {
            "paths": self._drawn_paths if self.drawn.isChecked() else None,
            "params": default_core_params(**overrides),
            "mm_per_px": self._mm_per_px,
            "outer_year": self.outer_year.value() or None,
        }

    def _on_run(self) -> None:
        self.run_button.setEnabled(False)
        self.status_label.setText(self.tr("Detecting…"))
        self._thread = QtCore.QThread(self)
        self._worker = CoreWorker(self._image, self.analysis_kwargs())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_finished(self, results: list[CoreResult]) -> None:
        self._teardown_thread()
        self.results = results
        self.run_button.setEnabled(True)
        if not results:
            self.status_label.setText(self.tr("No core found. Draw a path (linestrip) and retry."))
            return
        lines = [
            self.tr("Core %d: %d rings") % (k, len(r.widths_px)) for k, r in enumerate(results, 1)
        ]
        self.status_label.setText("\n".join(lines))
        self.export_button.setEnabled(True)
        self.apply_button.setEnabled(True)

    def _on_error(self, message: str) -> None:
        self._teardown_thread()
        self.run_button.setEnabled(True)
        self.status_label.setText(self.tr("Failed: %s") % message)

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def closeEvent(self, event: Any) -> None:
        self._teardown_thread()
        super().closeEvent(event)

    # ----- results -------------------------------------------------------------

    def label_shapes(self) -> list[dict[str, Any]]:
        shapes = []
        for k, result in enumerate(self.results, start=1):
            shapes += to_label_shapes(result, core_index=k)
        return shapes

    def _on_export(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Export folder"))
        if not folder:
            return
        written = export_results(self.results, Path(folder), self._image_name)
        QtWidgets.QMessageBox.information(
            self, self.tr("Export complete"), "\n".join(str(p) for p in written)
        )
