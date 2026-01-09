from __future__ import annotations

import importlib
import io
import platform
import shutil
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication

from tras.utils.apd_helper import detect_pith_apd
from tras.utils.cstrd_helper import detect_rings_cstrd
from tras.utils.deepcstrd_helper import detect_rings_deepcstrd
from tras.utils.inbd_helper import detect_rings_inbd
from tras.utils.ring_sampling import resample_rings_by_rays


def _remove_background(
    image: np.ndarray,
    method_name: str,
    debug_dir: Path | None = None,
    debug_filename: str | None = None,
) -> tuple[np.ndarray, str | None]:
    """
    Remove background from image using U2Net.
    
    Args:
        image: Input image (RGB, numpy array)
        method_name: Name of the detection method for logging (e.g., "CS-TRD")
        debug_dir: Optional directory to save debug image
        debug_filename: Optional filename for debug image (e.g., "cstrd_bg_removed.png")
    
    Returns:
        Tuple of (processed_image, error_message).
        If successful, error_message is None.
        If failed, returns original image and error message.
    """
    logger.info(f"{method_name}: Removing background before detection...")
    
    try:
        remove_salient_object = importlib.import_module(
            "tras.tree_ring_methods.urudendro.remove_salient_object"
        ).remove_salient_object

        # Ensure image is uint8 before saving
        image_to_save = image
        if image_to_save.dtype != np.uint8:
            image_to_save = (255 * (image_to_save.astype(np.float32) / image_to_save.max())).astype(np.uint8)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.png"
            output_path = Path(temp_dir) / "output.png"
            
            # Save current image (convert RGB to BGR for cv2.imwrite)
            cv2.imwrite(str(input_path), cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))
            
            # Run U2Net background removal
            remove_salient_object(str(input_path), str(output_path))
            
            # Load result using PIL (matches how remove_salient_object saves it)
            result_pil = Image.open(str(output_path)).convert("RGB")
            result = np.array(result_pil, dtype=np.uint8)
            
            if result is None or result.size == 0:
                raise Exception("U2Net did not produce output")
            
            # Ensure same dimensions as input
            if result.shape[:2] != image_to_save.shape[:2]:
                logger.warning(f"{method_name}: Output shape {result.shape[:2]} != input shape {image_to_save.shape[:2]}, resizing...")
                result = cv2.resize(result, (image_to_save.shape[1], image_to_save.shape[0]))
            
            result = np.ascontiguousarray(result, dtype=np.uint8)
            
            # Save debug image if requested
            if debug_dir is not None and debug_filename is not None:
                bg_removed_debug_path = debug_dir / debug_filename
                cv2.imwrite(str(bg_removed_debug_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                logger.info(f"{method_name}: Saved background-removed image to {bg_removed_debug_path}")
            
            logger.info(f"{method_name}: Background removal completed")
            return result, None
            
    except Exception as bg_error:
        logger.warning(f"{method_name}: Background removal failed: {bg_error}, continuing with original image")
        return image, str(bg_error)


class TreeRingDialog(QtWidgets.QDialog):
    def __init__(self, image_width: int, image_height: int, parent=None, image_np=None, initial_cx=None, initial_cy=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Tree Ring Detection"))
        self.setModal(True)

        # Store references
        self.parent_window = parent

        # Store image first
        self.image_np = image_np
        self.detected_pith_xy: tuple[float, float] | None = None
        self.cstrd_rings: list[object] | None = None
        self.deepcstrd_rings: list[object] | None = None
        self.inbd_rings: list[object] | None = None

        self._initial_pith_provided = initial_cx is not None and initial_cy is not None
        
        # Use actual image dimensions if image_np is provided (may differ from QImage dimensions after preprocessing)
        if image_np is not None:
            actual_width = image_np.shape[1]
            actual_height = image_np.shape[0]
        else:
            actual_width = image_width
            actual_height = image_height

        # Use provided coordinates or default to center
        if initial_cx is not None and initial_cy is not None:
            cx_default = float(initial_cx)
            cy_default = float(initial_cy)
        else:
            cx_default = float(actual_width) / 2.0
            cy_default = float(actual_height) / 2.0

        self._settings = QtCore.QSettings("TRAS", "TRAS")

        self.main_tabs = QtWidgets.QTabWidget()

        pith_tab = self._build_pith_tab(
            actual_width=actual_width,
            actual_height=actual_height,
            cx_default=cx_default,
            cy_default=cy_default,
        )
        rings_tab = self._build_ring_tab()

        self.main_tabs.addTab(pith_tab, self.tr("Pith detection"))
        self.main_tabs.addTab(rings_tab, self.tr("Tree ring detection"))

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        btns.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.main_tabs)
        layout.addWidget(btns)
        self.setLayout(layout)

        self.setMinimumWidth(650)
        self.setMinimumHeight(520)

        self._set_default_method_tab()
        self._load_settings()

    def accept(self) -> None:
        self._save_settings()
        super().accept()

    def reject(self) -> None:
        self._save_settings()
        super().reject()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _wrap_scroll(self, content: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_pith_tab(
        self,
        *,
        actual_width: int,
        actual_height: int,
        cx_default: float,
        cy_default: float,
    ) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(content)

        self.cx = QtWidgets.QDoubleSpinBox()
        self.cx.setRange(0.0, float(actual_width - 1))
        self.cx.setDecimals(2)
        self.cx.setValue(cx_default)
        form.addRow(self.tr("Center X"), self.cx)

        self.cy = QtWidgets.QDoubleSpinBox()
        self.cy.setRange(0.0, float(actual_height - 1))
        self.cy.setDecimals(2)
        self.cy.setValue(cy_default)
        form.addRow(self.tr("Center Y"), self.cy)

        label = QtWidgets.QLabel(self.tr("<b>Set Pith Location</b>"))
        form.addRow(label)

        self.btn_click_pith = QtWidgets.QPushButton(self.tr("Click on image to set pith"))
        self.btn_click_pith.clicked.connect(self._on_click_pith)
        form.addRow(self.btn_click_pith)

        self.btn_auto_pith = QtWidgets.QPushButton(self.tr("Auto-detect pith (APD)"))
        self.btn_auto_pith.clicked.connect(self._on_auto_pith)
        form.addRow(self.btn_auto_pith)

        self.apd_advanced_group = QtWidgets.QGroupBox(self.tr("APD Advanced Parameters"))
        self.apd_advanced_group.setCheckable(True)
        self.apd_advanced_group.setChecked(False)
        apd_layout = QtWidgets.QFormLayout()

        self.apd_st_sigma = QtWidgets.QDoubleSpinBox()
        self.apd_st_sigma.setRange(0.1, 10.0)
        self.apd_st_sigma.setValue(1.2)
        self.apd_st_sigma.setSingleStep(0.1)
        apd_layout.addRow(self.tr("Structure Tensor Sigma:"), self.apd_st_sigma)

        self.apd_method = QtWidgets.QComboBox()
        self.apd_method.addItems(["apd", "apd_pcl", "apd_dl"])
        self.apd_method.setCurrentText("apd_dl")  # Set apd_dl as default
        apd_layout.addRow(self.tr("Method:"), self.apd_method)

        self.apd_advanced_group.setLayout(apd_layout)
        form.addRow(self.apd_advanced_group)

        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.addWidget(self._wrap_scroll(content))
        return tab

    def _build_ring_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Shared options
        self.remove_bg_checkbox = QtWidgets.QCheckBox(
            self.tr("Remove background before detection")
        )
        self.remove_bg_checkbox.setChecked(True)
        self.remove_bg_checkbox.setToolTip(
            self.tr(
                "Remove background using U2Net to improve detection performance.\n"
                "This can help reduce false detections from background artifacts."
            )
        )
        layout.addWidget(self.remove_bg_checkbox)

        # Sampling NR control (shared across all methods)
        sampling_row = QtWidgets.QHBoxLayout()
        sampling_label = QtWidgets.QLabel(self.tr("Sampling NR (postprocess):"))
        self.sampling_nr = QtWidgets.QSpinBox()
        self.sampling_nr.setRange(36, 720)
        self.sampling_nr.setValue(360)
        self.sampling_nr.setSingleStep(36)
        self.sampling_nr.setToolTip(
            self.tr(
                "Number of radial samples for post-processing ring resampling.\n"
                "This resamples detected rings to a fixed number of points.\n"
                "Higher values = more points per ring (smoother, larger files)."
            )
        )
        sampling_row.addWidget(sampling_label)
        sampling_row.addWidget(self.sampling_nr)
        sampling_row.addStretch()
        layout.addLayout(sampling_row)

        self.method_tabs = QtWidgets.QTabWidget()
        self.method_tabs.addTab(self._build_cstrd_tab(), self.tr("CS-TRD"))
        self.method_tabs.addTab(self._build_deepcstrd_tab(), self.tr("DeepCS-TRD"))
        self.method_tabs.addTab(self._build_inbd_tab(), self.tr("INBD"))
        layout.addWidget(self.method_tabs)

        return tab

    def _build_cstrd_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)

        self.btn_cstrd = QtWidgets.QPushButton(self.tr("Detect with CS-TRD (CPU)"))
        self.btn_cstrd.clicked.connect(self._on_cstrd)

        is_windows = platform.system() == "Windows"
        if is_windows:
            self.btn_cstrd.setEnabled(False)
            self.btn_cstrd.setToolTip(
                self.tr(
                    "CS-TRD is not available on Windows due to compilation requirements.\n"
                    "Please use DeepCS-TRD instead, or run TRAS on Linux/macOS for CS-TRD."
                )
            )
        else:
            self.btn_cstrd.setToolTip(
                self.tr("Classical edge-based tree ring detection (~73s, CPU-only)")
            )
        v.addWidget(self.btn_cstrd)

        self.cstrd_advanced_group = QtWidgets.QGroupBox(
            self.tr("CS-TRD Advanced Parameters")
        )
        self.cstrd_advanced_group.setCheckable(True)
        self.cstrd_advanced_group.setChecked(False)
        cstrd_layout = QtWidgets.QFormLayout()

        self.cstrd_sigma = QtWidgets.QDoubleSpinBox()
        self.cstrd_sigma.setRange(0.5, 10.0)
        self.cstrd_sigma.setValue(3.0)
        self.cstrd_sigma.setSingleStep(0.5)
        cstrd_layout.addRow(self.tr("Gaussian Sigma:"), self.cstrd_sigma)

        self.cstrd_th_low = QtWidgets.QDoubleSpinBox()
        self.cstrd_th_low.setRange(0.0, 50.0)
        self.cstrd_th_low.setValue(5.0)
        self.cstrd_th_low.setSingleStep(1.0)
        cstrd_layout.addRow(self.tr("Low Threshold:"), self.cstrd_th_low)

        self.cstrd_th_high = QtWidgets.QDoubleSpinBox()
        self.cstrd_th_high.setRange(0.0, 100.0)
        self.cstrd_th_high.setValue(20.0)
        self.cstrd_th_high.setSingleStep(1.0)
        cstrd_layout.addRow(self.tr("High Threshold:"), self.cstrd_th_high)

        self.cstrd_alpha = QtWidgets.QSpinBox()
        self.cstrd_alpha.setRange(1, 180)
        self.cstrd_alpha.setValue(30)
        cstrd_layout.addRow(self.tr("Alpha (deg):"), self.cstrd_alpha)

        self.cstrd_nr = QtWidgets.QSpinBox()
        self.cstrd_nr.setRange(36, 720)
        self.cstrd_nr.setValue(360)
        self.cstrd_nr.setSingleStep(36)
        cstrd_layout.addRow(self.tr("Radial Samples:"), self.cstrd_nr)

        self.cstrd_width = QtWidgets.QSpinBox()
        self.cstrd_width.setRange(0, 10000)
        self.cstrd_width.setValue(0)
        self.cstrd_width.setSingleStep(100)
        self.cstrd_width.setToolTip(
            self.tr(
                "Resize image width (0 = no resize).\n"
                "Smaller images process faster but may reduce accuracy.\n"
                "Recommended: 1000-2000 for faster processing."
            )
        )
        cstrd_layout.addRow(self.tr("Resize Width (px):"), self.cstrd_width)

        self.cstrd_height = QtWidgets.QSpinBox()
        self.cstrd_height.setRange(0, 10000)
        self.cstrd_height.setValue(0)
        self.cstrd_height.setSingleStep(100)
        self.cstrd_height.setToolTip(
            self.tr(
                "Resize image height (0 = no resize).\n"
                "Leave both at 0 to use original size."
            )
        )
        cstrd_layout.addRow(self.tr("Resize Height (px):"), self.cstrd_height)

        self.cstrd_advanced_group.setLayout(cstrd_layout)
        if is_windows:
            self.cstrd_advanced_group.setEnabled(False)
        v.addWidget(self.cstrd_advanced_group)
        v.addStretch(1)

        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.addWidget(self._wrap_scroll(content))
        return tab

    def _build_deepcstrd_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)

        self.btn_deepcstrd = QtWidgets.QPushButton(
            self.tr("Detect with DeepCSTRD (GPU)")
        )
        self.btn_deepcstrd.clicked.connect(self._on_deepcstrd)
        self.btn_deepcstrd.setToolTip(
            self.tr("Deep learning-based tree ring detection (~101s, GPU-accelerated)")
        )
        v.addWidget(self.btn_deepcstrd)

        self.deepcstrd_advanced_group = QtWidgets.QGroupBox(
            self.tr("DeepCS-TRD Model & Parameters")
        )
        self.deepcstrd_advanced_group.setCheckable(True)
        self.deepcstrd_advanced_group.setChecked(False)
        deepcstrd_layout = QtWidgets.QFormLayout()

        model_row = QtWidgets.QHBoxLayout()
        self.deepcstrd_model = QtWidgets.QComboBox()
        self._populate_model_list()
        self.deepcstrd_model.setToolTip(
            self.tr(
                "Select the model trained on your target species.\n"
                "Generic works well for most species.\n"
                "Species-specific models may provide better results."
            )
        )
        model_row.addWidget(self.deepcstrd_model, stretch=3)

        self.btn_upload_model = QtWidgets.QPushButton(self.tr("📁 Upload"))
        self.btn_upload_model.setToolTip(
            self.tr(
                "Upload a new DeepCS-TRD model (.pth file).\n"
                "You'll be asked for a model name and tile size."
            )
        )
        self.btn_upload_model.clicked.connect(self._on_upload_model)
        model_row.addWidget(self.btn_upload_model, stretch=1)

        deepcstrd_layout.addRow(self.tr("Model:"), model_row)

        self.deepcstrd_tile_size = QtWidgets.QComboBox()
        self.deepcstrd_tile_size.addItems(["0 (Full image)", "256 (Tiled)"])
        self.deepcstrd_tile_size.setToolTip(
            self.tr(
                "Tiled processing (256) uses less memory but may be slower.\n"
                "Full image (0) is faster but requires more GPU memory."
            )
        )
        deepcstrd_layout.addRow(self.tr("Processing Mode:"), self.deepcstrd_tile_size)

        self.deepcstrd_alpha = QtWidgets.QSpinBox()
        self.deepcstrd_alpha.setRange(1, 180)
        self.deepcstrd_alpha.setValue(45)
        deepcstrd_layout.addRow(self.tr("Alpha (deg):"), self.deepcstrd_alpha)

        self.deepcstrd_nr = QtWidgets.QSpinBox()
        self.deepcstrd_nr.setRange(36, 720)
        self.deepcstrd_nr.setValue(360)
        self.deepcstrd_nr.setSingleStep(36)
        deepcstrd_layout.addRow(self.tr("Radial Samples:"), self.deepcstrd_nr)

        self.deepcstrd_width = QtWidgets.QSpinBox()
        self.deepcstrd_width.setRange(0, 10000)
        self.deepcstrd_width.setValue(1504)
        self.deepcstrd_width.setSingleStep(100)
        self.deepcstrd_width.setToolTip(
            self.tr(
                "Resize image width (0 = no resize).\n"
                "Smaller images use less GPU memory and process faster.\n"
                "Recommended: 1000-2000 for faster processing."
            )
        )
        deepcstrd_layout.addRow(self.tr("Resize Width (px):"), self.deepcstrd_width)

        self.deepcstrd_height = QtWidgets.QSpinBox()
        self.deepcstrd_height.setRange(0, 10000)
        self.deepcstrd_height.setValue(1504)
        self.deepcstrd_height.setSingleStep(100)
        self.deepcstrd_height.setToolTip(
            self.tr(
                "Resize image height (0 = no resize).\n"
                "Leave both at 0 to use original size."
            )
        )
        deepcstrd_layout.addRow(self.tr("Resize Height (px):"), self.deepcstrd_height)

        self.deepcstrd_rotations = QtWidgets.QSpinBox()
        self.deepcstrd_rotations.setRange(1, 10)
        self.deepcstrd_rotations.setValue(5)
        self.deepcstrd_rotations.setToolTip(
            self.tr("Test-time augmentation rotations.\nHigher = more accurate but slower.")
        )
        deepcstrd_layout.addRow(self.tr("Rotations (TTA):"), self.deepcstrd_rotations)

        self.deepcstrd_threshold = QtWidgets.QDoubleSpinBox()
        self.deepcstrd_threshold.setRange(0.0, 1.0)
        self.deepcstrd_threshold.setValue(0.5)
        self.deepcstrd_threshold.setSingleStep(0.05)
        self.deepcstrd_threshold.setToolTip(
            self.tr(
                "Prediction confidence threshold.\n"
                "Lower = more rings (may include false positives)\n"
                "Higher = fewer rings (may miss some)."
            )
        )
        deepcstrd_layout.addRow(
            self.tr("Prediction Threshold:"), self.deepcstrd_threshold
        )

        self.deepcstrd_advanced_group.setLayout(deepcstrd_layout)
        v.addWidget(self.deepcstrd_advanced_group)
        v.addStretch(1)

        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.addWidget(self._wrap_scroll(content))
        return tab

    def _build_inbd_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)

        self.btn_inbd = QtWidgets.QPushButton(self.tr("Detect with INBD (GPU)"))
        self.btn_inbd.clicked.connect(self._on_inbd)
        self.btn_inbd.setToolTip(
            self.tr("INBD (CVPR 2023): Iterative boundary detection for tree rings")
        )
        v.addWidget(self.btn_inbd)

        self.inbd_advanced_group = QtWidgets.QGroupBox(
            self.tr("INBD Model & Parameters")
        )
        self.inbd_advanced_group.setCheckable(True)
        self.inbd_advanced_group.setChecked(False)
        inbd_layout = QtWidgets.QFormLayout()

        model_row = QtWidgets.QHBoxLayout()
        self.inbd_model = QtWidgets.QComboBox()
        self.inbd_model.addItems(
            [
                "INBD_EH (Empetrum hermaphroditum)",
                "INBD_DO (Dryas octopetala)",
                "INBD_VM (Vaccinium myrtillus)",
                "INBD_UruDendro1 (Pinus taeda)",
            ]
        )
        self.inbd_model.setToolTip(
            self.tr(
                "Select the INBD model trained on your target species.\n"
                "EH/DO/VM: Shrub species\n"
                "UruDendro1: Tree species (Pinus taeda)"
            )
        )
        model_row.addWidget(self.inbd_model, stretch=3)

        self.btn_upload_inbd_model = QtWidgets.QPushButton(self.tr("📁 Upload"))
        self.btn_upload_inbd_model.setToolTip(
            self.tr(
                "Upload a custom INBD model (.pt.zip file).\n"
                "You'll be asked for a model name."
            )
        )
        self.btn_upload_inbd_model.clicked.connect(self._on_upload_inbd_model)
        model_row.addWidget(self.btn_upload_inbd_model, stretch=1)
        inbd_layout.addRow(self.tr("Model:"), model_row)

        self.inbd_auto_pith = QtWidgets.QCheckBox(
            self.tr("Use auto-detection (recommended)")
        )
        self.inbd_auto_pith.setToolTip(
            self.tr(
                "Let INBD detect the pith automatically from the innermost ring.\n"
                "This is the recommended mode for INBD.\n"
                "If unchecked, uses the pith coordinates from Step 1."
            )
        )
        self.inbd_auto_pith.setChecked(True)
        inbd_layout.addRow(self.inbd_auto_pith)

        self.inbd_advanced_group.setLayout(inbd_layout)
        v.addWidget(self.inbd_advanced_group)
        v.addStretch(1)

        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.addWidget(self._wrap_scroll(content))
        return tab

    def _set_default_method_tab(self) -> None:
        is_windows = platform.system() == "Windows"
        default_index = 1 if is_windows else 0  # DeepCS-TRD or CS-TRD
        if hasattr(self, "method_tabs"):
            self.method_tabs.setCurrentIndex(default_index)

    def _settings_key(self, name: str) -> str:
        return f"TreeRingDialog/{name}"

    def _load_settings(self) -> None:
        # Shared
        self.remove_bg_checkbox.setChecked(
            self._settings.value(
                self._settings_key("shared/remove_bg"),
                self.remove_bg_checkbox.isChecked(),
                type=bool,
            )
        )
        self.sampling_nr.setValue(
            int(self._settings.value(
                self._settings_key("shared/sampling_nr"),
                self.sampling_nr.value(),
                type=int,
            ))
        )

        # Pith / APD
        self.apd_advanced_group.setChecked(
            self._settings.value(
                self._settings_key("apd/advanced_open"),
                self.apd_advanced_group.isChecked(),
                type=bool,
            )
        )
        self.apd_st_sigma.setValue(
            float(
                self._settings.value(
                    self._settings_key("apd/st_sigma"),
                    self.apd_st_sigma.value(),
                    type=float,
                )
            )
        )
        apd_method = self._settings.value(
            self._settings_key("apd/method"), self.apd_method.currentText(), type=str
        )
        idx = self.apd_method.findText(apd_method)
        if idx >= 0:
            self.apd_method.setCurrentIndex(idx)

        if not self._initial_pith_provided:
            cx = float(
                self._settings.value(
                    self._settings_key("pith/cx"), self.cx.value(), type=float
                )
            )
            cy = float(
                self._settings.value(
                    self._settings_key("pith/cy"), self.cy.value(), type=float
                )
            )
            self.cx.setValue(min(max(cx, self.cx.minimum()), self.cx.maximum()))
            self.cy.setValue(min(max(cy, self.cy.minimum()), self.cy.maximum()))

        # CS-TRD
        self.cstrd_advanced_group.setChecked(
            self._settings.value(
                self._settings_key("cstrd/advanced_open"),
                self.cstrd_advanced_group.isChecked(),
                type=bool,
            )
        )
        self.cstrd_sigma.setValue(
            float(self._settings.value(self._settings_key("cstrd/sigma"), self.cstrd_sigma.value(), type=float))
        )
        self.cstrd_th_low.setValue(
            float(self._settings.value(self._settings_key("cstrd/th_low"), self.cstrd_th_low.value(), type=float))
        )
        self.cstrd_th_high.setValue(
            float(self._settings.value(self._settings_key("cstrd/th_high"), self.cstrd_th_high.value(), type=float))
        )
        self.cstrd_alpha.setValue(
            int(self._settings.value(self._settings_key("cstrd/alpha"), self.cstrd_alpha.value(), type=int))
        )
        self.cstrd_nr.setValue(
            int(self._settings.value(self._settings_key("cstrd/nr"), self.cstrd_nr.value(), type=int))
        )
        self.cstrd_width.setValue(
            int(self._settings.value(self._settings_key("cstrd/width"), self.cstrd_width.value(), type=int))
        )
        self.cstrd_height.setValue(
            int(self._settings.value(self._settings_key("cstrd/height"), self.cstrd_height.value(), type=int))
        )

        # DeepCS-TRD
        self.deepcstrd_advanced_group.setChecked(
            self._settings.value(
                self._settings_key("deepcstrd/advanced_open"),
                self.deepcstrd_advanced_group.isChecked(),
                type=bool,
            )
        )
        deep_model = self._settings.value(
            self._settings_key("deepcstrd/model"),
            self.deepcstrd_model.currentText(),
            type=str,
        )
        idx = self.deepcstrd_model.findText(deep_model)
        if idx >= 0:
            self.deepcstrd_model.setCurrentIndex(idx)
        self.deepcstrd_tile_size.setCurrentIndex(
            int(
                self._settings.value(
                    self._settings_key("deepcstrd/tile_idx"),
                    self.deepcstrd_tile_size.currentIndex(),
                    type=int,
                )
            )
        )
        self.deepcstrd_alpha.setValue(
            int(self._settings.value(self._settings_key("deepcstrd/alpha"), self.deepcstrd_alpha.value(), type=int))
        )
        self.deepcstrd_nr.setValue(
            int(self._settings.value(self._settings_key("deepcstrd/nr"), self.deepcstrd_nr.value(), type=int))
        )
        self.deepcstrd_rotations.setValue(
            int(
                self._settings.value(
                    self._settings_key("deepcstrd/rotations"),
                    self.deepcstrd_rotations.value(),
                    type=int,
                )
            )
        )
        self.deepcstrd_threshold.setValue(
            float(
                self._settings.value(
                    self._settings_key("deepcstrd/threshold"),
                    self.deepcstrd_threshold.value(),
                    type=float,
                )
            )
        )
        self.deepcstrd_width.setValue(
            int(self._settings.value(self._settings_key("deepcstrd/width"), self.deepcstrd_width.value(), type=int))
        )
        self.deepcstrd_height.setValue(
            int(self._settings.value(self._settings_key("deepcstrd/height"), self.deepcstrd_height.value(), type=int))
        )

        # INBD
        self.inbd_advanced_group.setChecked(
            self._settings.value(
                self._settings_key("inbd/advanced_open"),
                self.inbd_advanced_group.isChecked(),
                type=bool,
            )
        )
        inbd_model = self._settings.value(
            self._settings_key("inbd/model"),
            self.inbd_model.currentText(),
            type=str,
        )
        idx = self.inbd_model.findText(inbd_model)
        if idx >= 0:
            self.inbd_model.setCurrentIndex(idx)
        self.inbd_auto_pith.setChecked(
            self._settings.value(
                self._settings_key("inbd/auto_pith"),
                self.inbd_auto_pith.isChecked(),
                type=bool,
            )
        )

    def _save_settings(self) -> None:
        if not hasattr(self, "_settings"):
            return

        self._settings.setValue(
            self._settings_key("shared/remove_bg"), self.remove_bg_checkbox.isChecked()
        )
        self._settings.setValue(
            self._settings_key("shared/sampling_nr"), self.sampling_nr.value()
        )

        self._settings.setValue(
            self._settings_key("apd/advanced_open"), self.apd_advanced_group.isChecked()
        )
        self._settings.setValue(
            self._settings_key("apd/st_sigma"), float(self.apd_st_sigma.value())
        )
        self._settings.setValue(
            self._settings_key("apd/method"), self.apd_method.currentText()
        )

        self._settings.setValue(self._settings_key("pith/cx"), float(self.cx.value()))
        self._settings.setValue(self._settings_key("pith/cy"), float(self.cy.value()))

        self._settings.setValue(
            self._settings_key("cstrd/advanced_open"),
            self.cstrd_advanced_group.isChecked(),
        )
        self._settings.setValue(self._settings_key("cstrd/sigma"), float(self.cstrd_sigma.value()))
        self._settings.setValue(self._settings_key("cstrd/th_low"), float(self.cstrd_th_low.value()))
        self._settings.setValue(self._settings_key("cstrd/th_high"), float(self.cstrd_th_high.value()))
        self._settings.setValue(self._settings_key("cstrd/alpha"), int(self.cstrd_alpha.value()))
        self._settings.setValue(self._settings_key("cstrd/nr"), int(self.cstrd_nr.value()))
        self._settings.setValue(self._settings_key("cstrd/width"), int(self.cstrd_width.value()))
        self._settings.setValue(self._settings_key("cstrd/height"), int(self.cstrd_height.value()))

        self._settings.setValue(
            self._settings_key("deepcstrd/advanced_open"),
            self.deepcstrd_advanced_group.isChecked(),
        )
        self._settings.setValue(
            self._settings_key("deepcstrd/model"), self.deepcstrd_model.currentText()
        )
        self._settings.setValue(
            self._settings_key("deepcstrd/tile_idx"),
            int(self.deepcstrd_tile_size.currentIndex()),
        )
        self._settings.setValue(
            self._settings_key("deepcstrd/alpha"), int(self.deepcstrd_alpha.value())
        )
        self._settings.setValue(
            self._settings_key("deepcstrd/nr"), int(self.deepcstrd_nr.value())
        )
        self._settings.setValue(
            self._settings_key("deepcstrd/rotations"),
            int(self.deepcstrd_rotations.value()),
        )
        self._settings.setValue(
            self._settings_key("deepcstrd/threshold"),
            float(self.deepcstrd_threshold.value()),
        )
        self._settings.setValue(
            self._settings_key("deepcstrd/width"), int(self.deepcstrd_width.value())
        )
        self._settings.setValue(
            self._settings_key("deepcstrd/height"), int(self.deepcstrd_height.value())
        )

        self._settings.setValue(
            self._settings_key("inbd/advanced_open"),
            self.inbd_advanced_group.isChecked(),
        )
        self._settings.setValue(
            self._settings_key("inbd/model"), self.inbd_model.currentText()
        )
        self._settings.setValue(
            self._settings_key("inbd/auto_pith"), self.inbd_auto_pith.isChecked()
        )

        self._settings.sync()

    def _on_cstrd(self):
        if self.image_np is None:
            QtWidgets.QMessageBox.warning(self, self.tr("No image"), self.tr("No image data available for CS-TRD."))
            return
        try:
            # Get current center coordinates
            cx = float(self.cx.value())
            cy = float(self.cy.value())
            
            # DEBUG: Check image properties
            logger.info(f"CS-TRD: Image check - shape={self.image_np.shape}, dtype={self.image_np.dtype}")
            logger.info(f"CS-TRD: Image stats - min={self.image_np.min()}, max={self.image_np.max()}, mean={self.image_np.mean():.2f}")
            logger.info(f"CS-TRD: Image memory - contiguous={self.image_np.flags['C_CONTIGUOUS']}, writable={self.image_np.flags['WRITEABLE']}")
            
            # Show progress message
            QtWidgets.QMessageBox.information(
                self, 
                self.tr("CS-TRD Running"), 
                self.tr("CS-TRD is running. This may take 1-2 minutes. Click OK and wait...")
            )
            
            # Set wait cursor and log
            logger.info(f"CS-TRD: Starting ring detection on image {self.image_np.shape}, center=({cx:.1f}, {cy:.1f})")
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Get parameters from UI
            sigma = self.cstrd_sigma.value()
            th_low = self.cstrd_th_low.value()
            th_high = self.cstrd_th_high.value()
            alpha = self.cstrd_alpha.value()
            nr = self.cstrd_nr.value()
            width = self.cstrd_width.value()
            height = self.cstrd_height.value()
            
            logger.info(f"CS-TRD: Parameters - sigma={sigma}, th_low={th_low}, th_high={th_high}, alpha={alpha}, nr={nr}, resize=({width}, {height})")
            
            # Apply background removal if enabled
            image_to_process = self.image_np.copy()
            
            # Ensure image is in RGB format and contiguous (exactly like preprocessing dialog)
            # Ensure image is in RGB format (OpenCV-safe copy)
            image_to_process = np.ascontiguousarray(image_to_process, dtype=np.uint8)
            
            # Save original image for debugging
            debug_dir = Path.home() / ".tras_debug"
            debug_dir.mkdir(exist_ok=True)
            original_debug_path = debug_dir / "cstrd_original.png"
            cv2.imwrite(str(original_debug_path), cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR))
            logger.info(f"CS-TRD: Saved original image to {original_debug_path}")
            
            if self.remove_bg_checkbox.isChecked():
                image_to_process, bg_error = _remove_background(
                    image_to_process, "CS-TRD", debug_dir, "cstrd_bg_removed.png"
                )
                if bg_error:
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("Background Removal Failed"),
                        self.tr(f"Background removal failed: {bg_error}\n\nContinuing with original image.")
                    )
            
            # Run CS-TRD with current parameters
            rings = detect_rings_cstrd(
                image_to_process, 
                center_xy=(cx, cy),
                sigma=sigma,
                th_low=th_low,
                th_high=th_high,
                alpha=alpha,
                nr=nr,
                width=width,
                height=height
            )
            
            QApplication.restoreOverrideCursor()
            
            if not rings:
                logger.warning("CS-TRD: No rings detected")
                QtWidgets.QMessageBox.information(self, self.tr("No rings found"), self.tr("CS-TRD did not detect any rings."))
                return
            
            logger.info(f"CS-TRD: Detected {len(rings)} rings")
            
            # Apply postprocess resampling
            sampling_nr = self.sampling_nr.value()
            logger.info(f"CS-TRD: Resampling rings to {sampling_nr} points")
            rings = resample_rings_by_rays(rings, (cx, cy), sampling_nr)
            logger.info(f"CS-TRD: Resampled to {len(rings)} rings")
            
            self.cstrd_rings = rings
            self.detected_pith_xy = (cx, cy)  # Store pith coordinates
            QtWidgets.QMessageBox.information(
                self, 
                self.tr("CS-TRD Success"), 
                self.tr(f"Detected {len(rings)} rings. Click OK to insert them.")
            )
            self.accept()  # Close dialog and signal to use these rings
        except Exception as e:
            QApplication.restoreOverrideCursor()
            error_details = traceback.format_exc()
            logger.error(f"CS-TRD Error: {error_details}")
            QtWidgets.QMessageBox.critical(
                self, 
                self.tr("CS-TRD Error"), 
                self.tr(f"Failed to detect rings:\n{str(e)}\n\nCheck console for details.")
            )
    
    def _on_deepcstrd(self):
        if self.image_np is None:
            QtWidgets.QMessageBox.warning(self, self.tr("No image"), self.tr("No image data available for DeepCS-TRD."))
            return
        try:
            # Get current center coordinates
            cx = float(self.cx.value())
            cy = float(self.cy.value())
            
            # DEBUG: Check image properties
            logger.info(f"DeepCS-TRD: Image check - shape={self.image_np.shape}, dtype={self.image_np.dtype}")
            logger.info(f"DeepCS-TRD: Image stats - min={self.image_np.min()}, max={self.image_np.max()}, mean={self.image_np.mean():.2f}")
            logger.info(f"DeepCS-TRD: Image memory - contiguous={self.image_np.flags['C_CONTIGUOUS']}, writable={self.image_np.flags['WRITEABLE']}")
            
            # Show progress message
            QtWidgets.QMessageBox.information(
                self, 
                self.tr("DeepCS-TRD Running"), 
                self.tr("DeepCS-TRD is running. This may take 1-2 minutes. Click OK and wait...")
            )
            
            # Set wait cursor and log
            logger.info(f"DeepCS-TRD: Starting ring detection on image {self.image_np.shape}, center=({cx:.1f}, {cy:.1f})")
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Get model selection and parameters from UI
            model_text = self.deepcstrd_model.currentText()
            model_id = model_text.split(" ")[0]  # Extract "generic", "pinus_v1", etc.
            print(f"[DEBUG] UI model_text: '{model_text}'")
            print(f"[DEBUG] Extracted model_id: '{model_id}'")
            
            tile_text = self.deepcstrd_tile_size.currentText()
            tile_size = 256 if "256" in tile_text else 0
            
            alpha = self.deepcstrd_alpha.value()
            nr = self.deepcstrd_nr.value()
            total_rotations = self.deepcstrd_rotations.value()
            prediction_map_threshold = self.deepcstrd_threshold.value()
            width = self.deepcstrd_width.value()
            height = self.deepcstrd_height.value()
            
            logger.info(f"DeepCS-TRD: Model={model_id}, tile_size={tile_size}, alpha={alpha}, nr={nr}, rotations={total_rotations}, threshold={prediction_map_threshold}, resize=({width}, {height})")
            
            # Apply background removal if enabled
            image_to_process = self.image_np.copy()
            
            # Ensure image is in RGB format and contiguous (exactly like preprocessing dialog)
            # Ensure image is in RGB format (OpenCV-safe copy)
            image_to_process = np.ascontiguousarray(image_to_process, dtype=np.uint8)
            
            # Save original image for debugging
            debug_dir = Path.home() / ".tras_debug"
            debug_dir.mkdir(exist_ok=True)
            original_debug_path = debug_dir / "deepcstrd_original.png"
            cv2.imwrite(str(original_debug_path), cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR))
            logger.info(f"DeepCS-TRD: Saved original image to {original_debug_path}")
            
            if self.remove_bg_checkbox.isChecked():
                image_to_process, bg_error = _remove_background(
                    image_to_process, "DeepCS-TRD", debug_dir, "deepcstrd_bg_removed.png"
                )
                if bg_error:
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("Background Removal Failed"),
                        self.tr(f"Background removal failed: {bg_error}\n\nContinuing with original image.")
                    )
            
            # Run DeepCS-TRD with current parameters
            rings = detect_rings_deepcstrd(
                image_to_process, 
                center_xy=(cx, cy),
                model_id=model_id,
                tile_size=tile_size,
                alpha=alpha,
                nr=nr,
                total_rotations=total_rotations,
                prediction_map_threshold=prediction_map_threshold,
                width=width,
                height=height
            )
            
            QApplication.restoreOverrideCursor()
            
            if not rings:
                logger.warning("DeepCS-TRD: No rings detected")
                QtWidgets.QMessageBox.information(self, self.tr("No rings found"), self.tr("DeepCS-TRD did not detect any rings."))
                return
            
            logger.info(f"DeepCS-TRD: Detected {len(rings)} rings")
            
            # Apply postprocess resampling
            sampling_nr = self.sampling_nr.value()
            logger.info(f"DeepCS-TRD: Resampling rings to {sampling_nr} points")
            rings = resample_rings_by_rays(rings, (cx, cy), sampling_nr)
            logger.info(f"DeepCS-TRD: Resampled to {len(rings)} rings")
            
            self.deepcstrd_rings = rings
            self.detected_pith_xy = (cx, cy)  # Store pith coordinates
            QtWidgets.QMessageBox.information(
                self, 
                self.tr("DeepCS-TRD Success"), 
                self.tr(f"Detected {len(rings)} rings. Click OK to insert them.")
            )
            self.accept()  # Close dialog and signal to use these rings
        except Exception as e:
            QApplication.restoreOverrideCursor()
            error_details = traceback.format_exc()
            logger.error(f"DeepCS-TRD Error: {error_details}")
            QtWidgets.QMessageBox.critical(
                self, 
                self.tr("DeepCS-TRD Error"), 
                self.tr(f"Failed to detect rings:\n{str(e)}\n\nCheck console for details.")
            )
    
    def _on_inbd(self):
        """Handle INBD tree ring detection."""
        if self.image_np is None:
            QtWidgets.QMessageBox.warning(self, self.tr("No image"), self.tr("No image data available for INBD."))
            return
        try:
            # Get current center coordinates
            cx = float(self.cx.value())
            cy = float(self.cy.value())
            
            # DEBUG: Check image properties
            logger.info(f"INBD: Image check - shape={self.image_np.shape}, dtype={self.image_np.dtype}")
            logger.info(f"INBD: Image stats - min={self.image_np.min()}, max={self.image_np.max()}, mean={self.image_np.mean():.2f}")
            
            # Show progress message
            QtWidgets.QMessageBox.information(
                self, 
                self.tr("INBD Running"), 
                self.tr("INBD is running. This may take 1-2 minutes. Click OK and wait...")
            )
            
            # Set wait cursor and log
            logger.info(f"INBD: Starting ring detection on image {self.image_np.shape}, center=({cx:.1f}, {cy:.1f})")
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Get model selection from UI
            model_text = self.inbd_model.currentText()
            # Extract model ID (e.g., "INBD_EH" from "INBD_EH (Empetrum hermaphroditum)")
            model_id = model_text.split(" ")[0]
            
            logger.info(f"INBD: Model={model_id}")
            
            # Apply background removal if enabled
            image_to_process = self.image_np.copy()
            image_to_process = np.ascontiguousarray(image_to_process, dtype=np.uint8)
            
            if self.remove_bg_checkbox.isChecked():
                image_to_process, bg_error = _remove_background(image_to_process, "INBD")
                if bg_error:
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("Background Removal Failed"),
                        self.tr(f"Background removal failed: {bg_error}\n\nContinuing with original image.")
                    )
            
            # Run INBD with current parameters
            # Check if user wants to use auto-detection
            pith_for_sampling = None
            if self.inbd_auto_pith.isChecked():
                logger.info("INBD: Using auto-detection mode (no pith coordinates)")
                rings, pith_for_sampling = detect_rings_inbd(
                    image_to_process, 
                    center_xy=None,  # Let INBD auto-detect
                    model_id=model_id,
                    return_pith=True
                )
                logger.info(f"INBD: Computed pith for sampling: ({pith_for_sampling[0]:.1f}, {pith_for_sampling[1]:.1f})")
            else:
                logger.info(f"INBD: Using manual pith coordinates: ({cx:.1f}, {cy:.1f})")
                rings, pith_for_sampling = detect_rings_inbd(
                    image_to_process, 
                    center_xy=(cx, cy),
                    model_id=model_id,
                    return_pith=True
                )
                # In manual mode, pith_for_sampling should match (cx, cy) after padding adjustment
                logger.info(f"INBD: Using provided pith for sampling: ({pith_for_sampling[0]:.1f}, {pith_for_sampling[1]:.1f})")
            
            QApplication.restoreOverrideCursor()
            
            if not rings:
                logger.warning("INBD: No rings detected")
                QtWidgets.QMessageBox.information(self, self.tr("No rings found"), self.tr("INBD did not detect any rings."))
                return
            
            logger.info(f"INBD: Detected {len(rings)} rings")
            
            # Apply postprocess resampling using the appropriate pith
            sampling_nr = self.sampling_nr.value()
            logger.info(f"INBD: Resampling rings to {sampling_nr} points using pith ({pith_for_sampling[0]:.1f}, {pith_for_sampling[1]:.1f})")
            rings = resample_rings_by_rays(rings, pith_for_sampling, sampling_nr)
            logger.info(f"INBD: Resampled to {len(rings)} rings")
            
            self.inbd_rings = rings
            self.detected_pith_xy = pith_for_sampling  # Store pith coordinates (computed or provided)
            QtWidgets.QMessageBox.information(
                self, 
                self.tr("INBD Success"), 
                self.tr(f"Detected {len(rings)} rings. Click OK to insert them.")
            )
            self.accept()  # Close dialog and signal to use these rings
        except FileNotFoundError as e:
            QApplication.restoreOverrideCursor()
            # Handle missing INBD setup with helpful instructions
            error_msg = str(e)
            if "INBD model" in error_msg or "not found" in error_msg:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("INBD Not Set Up"),
                    self.tr(
                        "INBD models are not installed. Please follow these steps:\n\n"
                        "1. Clone INBD repository:\n"
                        "   cd tras/tree_ring_methods/inbd\n"
                        "   git clone https://github.com/hmarichal93/INBD.git src\n\n"
                        "2. Download models:\n"
                        "   cd src\n"
                        "   python fetch_pretrained_models.py\n\n"
                        "Or use the download script:\n"
                        "   cd tras/tree_ring_methods/inbd\n"
                        "   ./download_models.sh\n\n"
                        "After setup, models will be available at:\n"
                        "   tras/tree_ring_methods/inbd/src/checkpoints/\n\n"
                        "See tras/tree_ring_methods/inbd/README.md for details."
                    )
                )
            else:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("File Not Found"),
                    self.tr(f"File not found:\n{str(e)}")
                )
        except ImportError as e:
            QApplication.restoreOverrideCursor()
            # Handle missing INBD source code
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("INBD Not Installed"),
                self.tr(
                    "INBD source code is not installed.\n\n"
                    "Please clone the INBD repository:\n"
                    "   cd tras/tree_ring_methods/inbd\n"
                    "   git clone https://github.com/hmarichal93/INBD.git src\n\n"
                    "Then download the models:\n"
                    "   cd src\n"
                    "   python fetch_pretrained_models.py\n\n"
                    f"Error details: {str(e)}"
                )
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            error_details = traceback.format_exc()
            logger.error(f"INBD Error: {error_details}")
            QtWidgets.QMessageBox.critical(
                self, 
                self.tr("INBD Error"), 
                self.tr(f"Failed to detect rings:\n{str(e)}\n\nCheck console for details.")
            )

    def _on_click_pith(self):
        """Request to click on image to set pith - closes dialog and signals parent"""
        logger.info("User requested click-to-set-pith mode")
        # Close dialog with special return code
        self.done(2)  # 2 = click pith mode requested
    
    def _on_auto_pith(self):
        if self.image_np is None:
            QtWidgets.QMessageBox.warning(self, self.tr("No image"), self.tr("No image data available for pith detection."))
            return
        try:
            # Get APD parameters from UI
            method = self.apd_method.currentText()
            
            logger.info(f"APD: Starting pith detection on image {self.image_np.shape}, method={method}")
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            x, y = detect_pith_apd(self.image_np, method=method)
            
            QApplication.restoreOverrideCursor()
            logger.info(f"APD: Detected pith at ({x:.1f}, {y:.1f})")
            
            self.cx.setValue(x)
            self.cy.setValue(y)
            
            self._show_pith_preview(x, y)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            error_details = traceback.format_exc()
            logger.error(f"APD Error: {error_details}")
            QtWidgets.QMessageBox.critical(
                self, 
                self.tr("APD Error"), 
                self.tr(f"Failed to detect pith:\n{str(e)}\n\nCheck console for details.")
            )

    def get_cstrd_rings(self):
        # Returns rings if CS-TRD was used, else None
        return getattr(self, 'cstrd_rings', None)
    
    def get_deepcstrd_rings(self):
        # Returns rings if DeepCSTRD was used, else None
        return getattr(self, 'deepcstrd_rings', None)
    
    def get_inbd_rings(self):
        # Returns rings if INBD was used, else None
        return getattr(self, 'inbd_rings', None)
    
    def get_pith_xy(self):
        """Return pith coordinates used for detection, or None"""
        return getattr(self, "detected_pith_xy", None)

    def _show_pith_preview(self, x: float, y: float) -> None:
        if self.image_np is None:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("APD Success"),
                self.tr(f"Pith detected at ({x:.1f}, {y:.1f})"),
            )
            return

        image_arr = self.image_np
        if image_arr.ndim == 2:
            image_arr = np.stack([image_arr] * 3, axis=-1)
        elif image_arr.shape[2] == 4:
            image_arr = image_arr[:, :, :3]
        image_arr = np.clip(image_arr, 0, 255).astype(np.uint8)

        image_pil = Image.fromarray(image_arr)
        draw = ImageDraw.Draw(image_pil)
        radius = max(5, int(min(image_pil.size) * 0.02))
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, outline="red", width=max(2, radius // 3))
        draw.line((x - radius * 1.5, y, x + radius * 1.5, y), fill="red", width=2)
        draw.line((x, y - radius * 1.5, x, y + radius * 1.5), fill="red", width=2)

        buffer = io.BytesIO()
        image_pil.save(buffer, format="PNG")
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(buffer.getvalue())

        preview = QtWidgets.QDialog(self)
        preview.setWindowTitle(self.tr("APD Preview"))
        layout = QtWidgets.QVBoxLayout(preview)
        layout.addWidget(
            QtWidgets.QLabel(
                self.tr(f"Pith detected at ({x:.1f}, {y:.1f}). Review and press OK.")
            )
        )
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setPixmap(
            pixmap.scaled(
                600,
                600,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )
        layout.addWidget(label)
        button = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        button.accepted.connect(preview.accept)
        layout.addWidget(button)
        preview.exec_()

    def _populate_model_list(self):
        """Populate the model dropdown with available models."""
        # Get models directory
        models_dir = Path(__file__).parent.parent / "tree_ring_methods" / "deepcstrd" / "models" / "deep_cstrd"
        
        # Predefined models with friendly names
        model_names = {
            "generic": "generic (all species)",
            "pinus_v1": "pinus_v1 (Pinus taeda v1)",
            "pinus_v2": "pinus_v2 (Pinus taeda v2)",
            "gleditsia": "gleditsia (Gleditsia triacanthos)",
            "salix": "salix (Salix glauca)"
        }
        
        # Scan for additional custom models
        if models_dir.exists():
            for model_file in models_dir.glob("0_*.pth"):
                model_id = model_file.stem.replace("0_", "").replace("_1504", "")
                if model_id not in model_names:
                    # Custom model - add with generic name
                    model_names[model_id] = f"{model_id} (custom)"
        
        # Add models to dropdown
        self.deepcstrd_model.clear()
        for model_id, display_name in model_names.items():
            self.deepcstrd_model.addItem(display_name)
    
    def _on_upload_inbd_model(self):
        """Handle INBD model upload button click."""
        # File dialog to select .pt.zip file
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select INBD Model File"),
            "",
            self.tr("INBD Model Files (*.pt.zip);;All Files (*)")
        )
        
        if not file_path:
            return  # User cancelled
        
        # Ask for model name
        model_name, ok = QtWidgets.QInputDialog.getText(
            self,
            self.tr("Model Name"),
            self.tr("Enter a name for this model (e.g., 'eucalyptus', 'oak'):\n"
                   "(Use only letters, numbers, and underscores)\n"
                   "Will be saved as: INBD_<name>")
        )
        
        if not ok or not model_name:
            return  # User cancelled
        
        # Validate model name
        model_name = model_name.strip().replace(" ", "_")
        if not model_name.replace("_", "").isalnum():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Invalid Name"),
                self.tr("Model name can only contain letters, numbers, and underscores.")
            )
            return
        
        # Create model directory name
        model_dir_name = f"INBD_{model_name}"
        
        # Get destination path - store in INBD checkpoints directory
        checkpoints_dir = Path(__file__).parent.parent / "tree_ring_methods" / "inbd" / "src" / "checkpoints"
        dest_dir = checkpoints_dir / model_dir_name
        dest_file = dest_dir / "model.pt.zip"
        
        if dest_file.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                self.tr("Model Exists"),
                self.tr(f"Model '{model_dir_name}' already exists. Overwrite?"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        
        try:
            # Create directory if it doesn't exist
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model file
            shutil.copy2(file_path, dest_file)
            
            # Add to model list if not already there
            new_model_text = f"{model_dir_name} (custom)"
            if self.inbd_model.findText(new_model_text) < 0:
                self.inbd_model.addItem(new_model_text)
            
            # Select the newly uploaded model
            index = self.inbd_model.findText(new_model_text)
            if index >= 0:
                self.inbd_model.setCurrentIndex(index)
            
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Upload Successful"),
                self.tr(f"Model uploaded successfully!\n\n"
                       f"Model: {model_dir_name}\n"
                       f"Location: {dest_file}")
            )
            
            logger.info(f"Uploaded custom INBD model: {dest_file}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Upload Failed"),
                self.tr(f"Failed to upload model:\n{str(e)}")
            )
            logger.error(f"Failed to upload INBD model: {e}", exc_info=True)
    
    def _on_upload_model(self):
        """Handle DeepCS-TRDmodel upload button click."""
        # File dialog to select .pth file
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select DeepCS-TRD Model File"),
            "",
            self.tr("PyTorch Model Files (*.pth);;All Files (*)")
        )
        
        if not file_path:
            return  # User cancelled
        
        # Ask for model name
        model_name, ok = QtWidgets.QInputDialog.getText(
            self,
            self.tr("Model Name"),
            self.tr("Enter a name for this model (e.g., 'eucalyptus', 'oak'):\n"
                   "(Use only letters, numbers, and underscores)")
        )
        
        if not ok or not model_name:
            return  # User cancelled
        
        # Validate model name
        model_name = model_name.strip().lower().replace(" ", "_")
        if not model_name.replace("_", "").isalnum():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Invalid Name"),
                self.tr("Model name can only contain letters, numbers, and underscores.")
            )
            return
        
        # Ask for tile size
        tile_size, ok = QtWidgets.QInputDialog.getItem(
            self,
            self.tr("Tile Size"),
            self.tr("What tile size was this model trained with?"),
            ["0 (Full image)", "256 (Tiled)"],
            0,
            False
        )
        
        if not ok:
            return  # User cancelled
        
        tile_prefix = "256" if "256" in tile_size else "0"
        
        # Get destination path
        models_dir = Path(__file__).parent.parent / "tree_ring_methods" / "deepcstrd" / "models" / "deep_cstrd"
        dest_file = models_dir / f"{tile_prefix}_{model_name}_1504.pth"
        
        if dest_file.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                self.tr("File Exists"),
                self.tr(f"Model '{model_name}' already exists. Overwrite?"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        
        try:
            # Copy model file
            shutil.copy2(file_path, dest_file)
            
            # Refresh model list
            current_selection = self.deepcstrd_model.currentText()
            self._populate_model_list()
            
            # Try to select the newly uploaded model
            new_model_text = f"{model_name} (custom)"
            index = self.deepcstrd_model.findText(new_model_text)
            if index >= 0:
                self.deepcstrd_model.setCurrentIndex(index)
            else:
                # Restore previous selection if the uploaded model isn't found in the list
                prev_index = self.deepcstrd_model.findText(current_selection)
                if prev_index >= 0:
                    self.deepcstrd_model.setCurrentIndex(prev_index)
            
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Upload Success"),
                self.tr(f"Model '{model_name}' uploaded successfully!\n\n"
                       f"File: {dest_file.name}\n"
                       f"Location: {models_dir}")
            )
            
            logger.info(f"Uploaded new DeepCS-TRD model: {dest_file}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Upload Failed"),
                self.tr(f"Failed to upload model:\n{str(e)}")
            )
            logger.error(f"Failed to upload model: {e}")
