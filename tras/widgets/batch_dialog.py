"""Batch processing dialog for the TRAS GUI.

Lets the user pick an input folder, configure detection settings inline, and run ring
detection over every image. Outputs one ``.json`` per image plus a combined
``summary.pdf`` and a CLI-compatible ``batch_config.yml``. Detection runs in a worker
thread so the GUI stays responsive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from tras.config.detection_defaults import get_detection_defaults
from tras.utils.batch_helper import BatchSummary
from tras.utils.batch_helper import find_images
from tras.utils.batch_helper import run_batch

# DeepCS-TRD predefined model ids (id, display label); mirrors TreeRingDialog.
_DEEPCSTRD_MODELS = [
    ("generic", "generic (all species)"),
    ("pinus_v1", "pinus_v1 (Pinus taeda v1)"),
    ("pinus_v2", "pinus_v2 (Pinus taeda v2)"),
    ("gleditsia", "gleditsia (Gleditsia triacanthos)"),
    ("salix", "salix (Salix glauca)"),
]

_INBD_MODELS = [
    ("INBD_EH", "INBD_EH (Empetrum hermaphroditum)"),
    ("INBD_DO", "INBD_DO (Dryas octopetala)"),
    ("INBD_VM", "INBD_VM (Vaccinium myrtillus)"),
    ("INBD_UruDendro1", "INBD_UruDendro1 (Pinus taeda)"),
]

_RING_METHODS = [
    ("deepcstrd", "DeepCS-TRD (deep learning, GPU)"),
    ("cstrd", "CS-TRD (classical, CPU)"),
    ("inbd", "INBD (instance segmentation, GPU)"),
]


class BatchWorker(QtCore.QObject):
    """Runs :func:`run_batch` in a worker thread, reporting progress via signals."""

    progress = QtCore.pyqtSignal(int, int, str)  # current, total, name
    finished = QtCore.pyqtSignal(object)  # BatchSummary
    error = QtCore.pyqtSignal(str)

    def __init__(self, input_dir: Path, output_dir: Path, config: dict[str, Any]):
        super().__init__()
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._config = config

    def run(self) -> None:
        try:
            summary = run_batch(
                self._input_dir,
                self._output_dir,
                self._config,
                progress=lambda i, n, name: self.progress.emit(i, n, name),
            )
            self.finished.emit(summary)
        except Exception as exc:  # pragma: no cover - surfaced to the dialog
            logger.exception("Batch processing failed")
            self.error.emit(str(exc))


class BatchProcessDialog(QtWidgets.QDialog):
    """Configure and run folder-wide tree ring detection."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch Processing"))
        self.setMinimumWidth(520)
        self._defaults = get_detection_defaults()
        self._settings = QtCore.QSettings("TRAS", "TRAS")
        self._thread: QtCore.QThread | None = None
        self._worker: BatchWorker | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_folders_group())
        layout.addWidget(self._build_scale_group())
        layout.addWidget(self._build_preprocess_group())
        layout.addWidget(self._build_pith_group())
        layout.addWidget(self._build_rings_group())
        layout.addWidget(self._build_run_section())

        self._load_settings()

    # ----- UI construction -------------------------------------------------

    def _build_folders_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(self.tr("Folder"))
        form = QtWidgets.QFormLayout(group)

        self.input_edit = QtWidgets.QLineEdit()
        in_browse = QtWidgets.QPushButton(self.tr("Browse..."))
        in_browse.clicked.connect(self._pick_input_dir)
        form.addRow(
            self.tr("Input folder:"), self._with_button(self.input_edit, in_browse)
        )
        # Outputs (.json, summary.pdf, batch_config.yml) go into this same folder.
        note = QtWidgets.QLabel(self.tr("Results are written into the input folder."))
        note.setStyleSheet("color: gray;")
        form.addRow("", note)
        return group

    def _build_scale_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(self.tr("Physical scale (optional)"))
        form = QtWidgets.QFormLayout(group)
        self.scale_value = QtWidgets.QDoubleSpinBox()
        self.scale_value.setDecimals(6)
        self.scale_value.setRange(0.0, 1000.0)
        self.scale_value.setSingleStep(0.001)
        self.scale_value.setToolTip(
            self.tr("Unit per pixel. Leave at 0 for pixel units.")
        )
        self.scale_unit = QtWidgets.QComboBox()
        self.scale_unit.addItems(["mm", "cm", "um"])
        form.addRow(self.tr("Value (unit/px):"), self.scale_value)
        form.addRow(self.tr("Unit:"), self.scale_unit)
        return group

    def _build_preprocess_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(self.tr("Preprocessing"))
        form = QtWidgets.QFormLayout(group)
        self.resize_scale = QtWidgets.QDoubleSpinBox()
        self.resize_scale.setRange(0.1, 1.0)
        self.resize_scale.setSingleStep(0.1)
        self.resize_scale.setValue(1.0)
        self.resize_scale.setToolTip(self.tr("Resize factor (1.0 = no resize)."))
        self.remove_bg = QtWidgets.QCheckBox(self.tr("Remove background (U2Net)"))
        form.addRow(self.tr("Resize scale:"), self.resize_scale)
        form.addRow("", self.remove_bg)
        return group

    def _build_pith_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(self.tr("Pith detection"))
        form = QtWidgets.QFormLayout(group)
        self.pith_auto = QtWidgets.QCheckBox(self.tr("Auto-detect pith (APD)"))
        self.pith_auto.setChecked(self._defaults["pith"]["auto"])
        self.pith_method = QtWidgets.QComboBox()
        self.pith_method.addItems(["apd", "apd_pcl", "apd_dl"])
        self._select_data(
            self.pith_method, self._defaults["pith"]["method"], by_text=True
        )
        form.addRow("", self.pith_auto)
        form.addRow(self.tr("APD method:"), self.pith_method)
        return group

    def _build_rings_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(self.tr("Ring detection"))
        outer = QtWidgets.QVBoxLayout(group)

        top = QtWidgets.QFormLayout()
        self.ring_method = QtWidgets.QComboBox()
        for method_id, label in _RING_METHODS:
            self.ring_method.addItem(label, method_id)
        self._select_data(self.ring_method, self._defaults["method"])
        self.ring_method.currentIndexChanged.connect(self._on_method_changed)

        self.sampling_nr = QtWidgets.QSpinBox()
        self.sampling_nr.setRange(36, 720)
        self.sampling_nr.setValue(self._defaults["sampling"]["nr"])
        top.addRow(self.tr("Method:"), self.ring_method)
        top.addRow(self.tr("Sampling NR:"), self.sampling_nr)
        outer.addLayout(top)

        self.method_stack = QtWidgets.QStackedWidget()
        self.method_stack.addWidget(self._build_deepcstrd_page())  # index 0
        self.method_stack.addWidget(self._build_cstrd_page())  # index 1
        self.method_stack.addWidget(self._build_inbd_page())  # index 2
        outer.addWidget(self.method_stack)
        self._on_method_changed()
        return group

    def _build_cstrd_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        d = self._defaults["cstrd"]
        self.cstrd_sigma = self._dspin(0.5, 10.0, 0.1, d["sigma"])
        self.cstrd_th_low = self._dspin(0.0, 255.0, 1.0, d["th_low"])
        self.cstrd_th_high = self._dspin(0.0, 255.0, 1.0, d["th_high"])
        self.cstrd_alpha = self._ispin(1, 180, d["alpha"])
        self.cstrd_nr = self._ispin(36, 720, d["nr"])
        form.addRow(self.tr("Sigma:"), self.cstrd_sigma)
        form.addRow(self.tr("Threshold low:"), self.cstrd_th_low)
        form.addRow(self.tr("Threshold high:"), self.cstrd_th_high)
        form.addRow(self.tr("Alpha:"), self.cstrd_alpha)
        form.addRow("nr:", self.cstrd_nr)
        return page

    def _build_deepcstrd_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        d = self._defaults["deepcstrd"]
        self.deepcstrd_model = QtWidgets.QComboBox()
        for model_id, label in _DEEPCSTRD_MODELS:
            self.deepcstrd_model.addItem(label, model_id)
        self._select_data(self.deepcstrd_model, d["model"])
        self.deepcstrd_tile_size = QtWidgets.QComboBox()
        self.deepcstrd_tile_size.addItem("0 (Full image)", 0)
        self.deepcstrd_tile_size.addItem("256 (Tiled)", 256)
        self.deepcstrd_alpha = self._ispin(1, 180, d["alpha"])
        self.deepcstrd_nr = self._ispin(36, 720, d["nr"])
        self.deepcstrd_rotations = self._ispin(1, 10, d["rotations"])
        self.deepcstrd_threshold = self._dspin(0.0, 1.0, 0.01, d["threshold"])
        self.deepcstrd_width = self._ispin(0, 8192, d["width"])
        self.deepcstrd_height = self._ispin(0, 8192, d["height"])
        form.addRow(self.tr("Model:"), self.deepcstrd_model)
        form.addRow(self.tr("Tile size:"), self.deepcstrd_tile_size)
        form.addRow(self.tr("Alpha:"), self.deepcstrd_alpha)
        form.addRow("nr:", self.deepcstrd_nr)
        form.addRow(self.tr("Rotations:"), self.deepcstrd_rotations)
        form.addRow(self.tr("Threshold:"), self.deepcstrd_threshold)
        form.addRow(self.tr("Width:"), self.deepcstrd_width)
        form.addRow(self.tr("Height:"), self.deepcstrd_height)
        return page

    def _build_inbd_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        d = self._defaults["inbd"]
        self.inbd_model = QtWidgets.QComboBox()
        for model_id, label in _INBD_MODELS:
            self.inbd_model.addItem(label, model_id)
        self._select_data(self.inbd_model, d["model"])
        self.inbd_auto_pith = QtWidgets.QCheckBox(self.tr("Auto-detect pith (INBD)"))
        self.inbd_auto_pith.setChecked(d["auto_pith"])
        form.addRow(self.tr("Model:"), self.inbd_model)
        form.addRow("", self.inbd_auto_pith)
        return page

    def _build_run_section(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(widget)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QtWidgets.QLabel("")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.run_button = QtWidgets.QPushButton(self.tr("Run"))
        self.run_button.clicked.connect(self._on_run)
        self.close_button = QtWidgets.QPushButton(self.tr("Close"))
        self.close_button.clicked.connect(self.reject)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.close_button)
        v.addWidget(self.progress_bar)
        v.addWidget(self.status_label)
        v.addLayout(buttons)
        return widget

    # ----- small widget helpers -------------------------------------------

    @staticmethod
    def _with_button(
        edit: QtWidgets.QWidget, button: QtWidgets.QWidget
    ) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, stretch=3)
        row.addWidget(button, stretch=1)
        return container

    @staticmethod
    def _dspin(
        lo: float, hi: float, step: float, value: float
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _ispin(lo: int, hi: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        return spin

    @staticmethod
    def _select_data(
        combo: QtWidgets.QComboBox, value: Any, by_text: bool = False
    ) -> None:
        idx = combo.findText(str(value)) if by_text else combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_method_changed(self) -> None:
        method = self.ring_method.currentData()
        index = {"deepcstrd": 0, "cstrd": 1, "inbd": 2}.get(method, 0)
        self.method_stack.setCurrentIndex(index)

    # ----- folder picker ---------------------------------------------------

    def _pick_input_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select input folder"), self.input_edit.text()
        )
        if directory:
            self.input_edit.setText(directory)

    # ----- config assembly -------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Assemble a config dict in the schema ``get_detection_params`` reads."""
        config: dict[str, Any] = {}

        if self.scale_value.value() > 0:
            config["physical_scale"] = {
                "value": float(self.scale_value.value()),
                "unit": self.scale_unit.currentText(),
            }

        preprocess: dict[str, Any] = {"remove_background": self.remove_bg.isChecked()}
        if self.resize_scale.value() < 1.0:
            preprocess["resize_scale"] = float(self.resize_scale.value())
        config["preprocess"] = preprocess

        config["postprocess"] = {"sampling_nr": int(self.sampling_nr.value())}
        config["pith"] = {
            "auto": self.pith_auto.isChecked(),
            "method": self.pith_method.currentText(),
        }

        method = self.ring_method.currentData()
        rings: dict[str, Any] = {"method": method}
        if method == "cstrd":
            rings["cstrd"] = {
                "sigma": float(self.cstrd_sigma.value()),
                "th_low": float(self.cstrd_th_low.value()),
                "th_high": float(self.cstrd_th_high.value()),
                "alpha": int(self.cstrd_alpha.value()),
                "nr": int(self.cstrd_nr.value()),
            }
        elif method == "deepcstrd":
            rings["deepcstrd"] = {
                "model": self.deepcstrd_model.currentData(),
                "tile_size": int(self.deepcstrd_tile_size.currentData()),
                "alpha": int(self.deepcstrd_alpha.value()),
                "nr": int(self.deepcstrd_nr.value()),
                "rotations": int(self.deepcstrd_rotations.value()),
                "threshold": float(self.deepcstrd_threshold.value()),
                "width": int(self.deepcstrd_width.value()),
                "height": int(self.deepcstrd_height.value()),
            }
        elif method == "inbd":
            rings["inbd"] = {
                "model": self.inbd_model.currentData(),
                "auto_pith": self.inbd_auto_pith.isChecked(),
            }
        config["rings"] = rings
        return config

    # ----- run lifecycle ---------------------------------------------------

    def _on_run(self) -> None:
        input_dir = Path(self.input_edit.text().strip())
        if not input_dir.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Invalid input"),
                self.tr("Please select a valid input folder."),
            )
            return
        if not find_images(input_dir):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("No images"),
                self.tr("No supported images found in the input folder."),
            )
            return

        # Outputs are written into the input folder itself.
        output_dir = input_dir

        self._save_settings()
        self._set_running(True)

        self._thread = QtCore.QThread(self)
        self._worker = BatchWorker(input_dir, output_dir, self.get_config())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, current: int, total: int, name: str) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(
            self.tr("Processing %s (%d/%d)") % (name, current, total)
        )

    def _on_finished(self, summary: BatchSummary) -> None:
        self._teardown_thread()
        self._set_running(False)
        self.status_label.setText(self.tr("Done."))
        message = self.tr(
            "Processed: %d\nSkipped (already done): %d\nErrors: %d\n\n"
            "Summary: %s\nConfig: %s"
        ) % (
            summary.processed,
            summary.skipped,
            summary.errors,
            summary.summary_pdf,
            summary.config_path,
        )
        QtWidgets.QMessageBox.information(self, self.tr("Batch complete"), message)

    def _on_error(self, message: str) -> None:
        self._teardown_thread()
        self._set_running(False)
        self.status_label.setText(self.tr("Failed."))
        QtWidgets.QMessageBox.critical(self, self.tr("Batch failed"), message)

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setValue(0)

    # ----- settings persistence -------------------------------------------

    def _key(self, name: str) -> str:
        return f"BatchProcessDialog/{name}"

    def _load_settings(self) -> None:
        s = self._settings
        self.scale_value.setValue(s.value(self._key("scale_value"), 0.0, type=float))
        self._select_data(
            self.scale_unit,
            s.value(self._key("scale_unit"), "mm", type=str),
            by_text=True,
        )
        self.resize_scale.setValue(s.value(self._key("resize_scale"), 1.0, type=float))
        self.remove_bg.setChecked(s.value(self._key("remove_bg"), False, type=bool))
        self.pith_auto.setChecked(
            s.value(self._key("pith_auto"), self.pith_auto.isChecked(), type=bool)
        )
        self._select_data(
            self.pith_method,
            s.value(self._key("pith_method"), self.pith_method.currentText(), type=str),
            by_text=True,
        )
        self._select_data(
            self.ring_method,
            s.value(self._key("ring_method"), self.ring_method.currentData(), type=str),
        )
        self.sampling_nr.setValue(
            s.value(self._key("sampling_nr"), self.sampling_nr.value(), type=int)
        )
        self._on_method_changed()

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue(self._key("scale_value"), float(self.scale_value.value()))
        s.setValue(self._key("scale_unit"), self.scale_unit.currentText())
        s.setValue(self._key("resize_scale"), float(self.resize_scale.value()))
        s.setValue(self._key("remove_bg"), self.remove_bg.isChecked())
        s.setValue(self._key("pith_auto"), self.pith_auto.isChecked())
        s.setValue(self._key("pith_method"), self.pith_method.currentText())
        s.setValue(self._key("ring_method"), self.ring_method.currentData())
        s.setValue(self._key("sampling_nr"), int(self.sampling_nr.value()))

    def closeEvent(self, event: Any) -> None:
        self._teardown_thread()
        super().closeEvent(event)
