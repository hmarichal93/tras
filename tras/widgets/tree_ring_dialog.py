from __future__ import annotations

import platform
from typing import Optional

from loguru import logger
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

from tras.utils.apd_helper import detect_pith_apd
from tras.utils.cstrd_helper import detect_rings_cstrd
from tras.utils.deepcstrd_helper import detect_rings_deepcstrd

class TreeRingDialog(QtWidgets.QDialog):
    def __init__(self, image_width: int, image_height: int, parent=None, image_np=None, initial_cx=None, initial_cy=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Tree Ring Detection"))
        self.setModal(True)

        # Use provided coordinates or default to center
        if initial_cx is not None and initial_cy is not None:
            cx_default = float(initial_cx)
            cy_default = float(initial_cy)
        else:
            cx_default = float(image_width) / 2.0
            cy_default = float(image_height) / 2.0

        self.form = QtWidgets.QFormLayout()

        self.cx = QtWidgets.QDoubleSpinBox()
        self.cx.setRange(0.0, float(image_width - 1))
        self.cx.setDecimals(2)
        self.cx.setValue(cx_default)
        self.form.addRow(self.tr("Center X"), self.cx)

        self.cy = QtWidgets.QDoubleSpinBox()
        self.cy.setRange(0.0, float(image_height - 1))
        self.cy.setDecimals(2)
        self.cy.setValue(cy_default)
        self.form.addRow(self.tr("Center Y"), self.cy)

        # Store references
        self.image_np = image_np
        self.parent_window = parent
        self.detected_pith_xy = None  # Store pith coordinates when detection succeeds
        
        # STEP 1: Pith Selection
        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.HLine)
        separator1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.form.addRow(separator1)
        
        pith_label = QtWidgets.QLabel(self.tr("<b>Step 1: Set Pith Location</b>"))
        self.form.addRow(pith_label)
        
        # Click to set pith button
        self.btn_click_pith = QtWidgets.QPushButton(self.tr("Click on image to set pith"))
        self.btn_click_pith.clicked.connect(self._on_click_pith)
        self.form.addRow(self.btn_click_pith)
        
        # Auto-detect pith button
        self.btn_auto_pith = QtWidgets.QPushButton(self.tr("Auto-detect pith (APD)"))
        self.btn_auto_pith.clicked.connect(self._on_auto_pith)
        self.form.addRow(self.btn_auto_pith)
        
        # APD Advanced Parameters (collapsible)
        self.apd_advanced_group = QtWidgets.QGroupBox(self.tr("APD Advanced Parameters"))
        self.apd_advanced_group.setCheckable(True)
        self.apd_advanced_group.setChecked(False)  # Collapsed by default
        apd_layout = QtWidgets.QFormLayout()
        
        self.apd_st_sigma = QtWidgets.QDoubleSpinBox()
        self.apd_st_sigma.setRange(0.1, 10.0)
        self.apd_st_sigma.setValue(1.2)
        self.apd_st_sigma.setSingleStep(0.1)
        apd_layout.addRow(self.tr("Structure Tensor Sigma:"), self.apd_st_sigma)
        
        self.apd_method = QtWidgets.QComboBox()
        self.apd_method.addItems(["apd", "apd_pcl"])
        apd_layout.addRow(self.tr("Method:"), self.apd_method)
        
        self.apd_advanced_group.setLayout(apd_layout)
        self.form.addRow(self.apd_advanced_group)

        # STEP 2: Ring Detection
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.HLine)
        separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.form.addRow(separator2)
        
        detection_label = QtWidgets.QLabel(self.tr("<b>Step 2: Detect Tree Rings</b>"))
        self.form.addRow(detection_label)
        
        # Add CS-TRD button (disabled on Windows)
        self.btn_cstrd = QtWidgets.QPushButton(self.tr("Detect with CS-TRD (CPU)"))
        self.btn_cstrd.clicked.connect(self._on_cstrd)
        
        # Disable CS-TRD on Windows
        is_windows = platform.system() == "Windows"
        if is_windows:
            self.btn_cstrd.setEnabled(False)
            self.btn_cstrd.setToolTip(
                self.tr("CS-TRD is not available on Windows due to compilation requirements.\n"
                       "Please use DeepCS-TRD instead, or run TRAS on Linux/macOS for CS-TRD.")
            )
        else:
            self.btn_cstrd.setToolTip(self.tr("Classical edge-based tree ring detection (~73s, CPU-only)"))
        
        self.form.addRow(self.btn_cstrd)
        
        # CS-TRD Advanced Parameters (collapsible)
        self.cstrd_advanced_group = QtWidgets.QGroupBox(self.tr("CS-TRD Advanced Parameters"))
        self.cstrd_advanced_group.setCheckable(True)
        self.cstrd_advanced_group.setChecked(False)  # Collapsed by default
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
        
        self.cstrd_advanced_group.setLayout(cstrd_layout)
        if is_windows:
            self.cstrd_advanced_group.setEnabled(False)
        self.form.addRow(self.cstrd_advanced_group)

        # Add DeepCSTRD button
        self.btn_deepcstrd = QtWidgets.QPushButton(self.tr("Detect with DeepCSTRD (GPU)"))
        self.btn_deepcstrd.clicked.connect(self._on_deepcstrd)
        self.btn_deepcstrd.setToolTip(self.tr("Deep learning-based tree ring detection (~101s, GPU-accelerated)"))
        self.form.addRow(self.btn_deepcstrd)
        
        # DeepCS-TRD Model Selection & Advanced Parameters
        self.deepcstrd_advanced_group = QtWidgets.QGroupBox(self.tr("DeepCS-TRD Model & Parameters"))
        self.deepcstrd_advanced_group.setCheckable(True)
        self.deepcstrd_advanced_group.setChecked(False)  # Collapsed by default
        deepcstrd_layout = QtWidgets.QFormLayout()
        
        # Model Selection
        self.deepcstrd_model = QtWidgets.QComboBox()
        self.deepcstrd_model.addItems([
            "generic (all species)",
            "pinus_v1 (Pinus taeda v1)",
            "pinus_v2 (Pinus taeda v2)",
            "gleditsia (Gleditsia triacanthos)",
            "salix (Salix humboldtiana)"
        ])
        self.deepcstrd_model.setToolTip(
            self.tr("Select the model trained on your target species.\n"
                   "Generic works well for most species.\n"
                   "Species-specific models may provide better results.")
        )
        deepcstrd_layout.addRow(self.tr("Model:"), self.deepcstrd_model)
        
        # Tile Size
        self.deepcstrd_tile_size = QtWidgets.QComboBox()
        self.deepcstrd_tile_size.addItems(["0 (Full image)", "256 (Tiled)"])
        self.deepcstrd_tile_size.setToolTip(
            self.tr("Tiled processing (256) uses less memory but may be slower.\n"
                   "Full image (0) is faster but requires more GPU memory.")
        )
        deepcstrd_layout.addRow(self.tr("Processing Mode:"), self.deepcstrd_tile_size)
        
        # Advanced parameters
        self.deepcstrd_alpha = QtWidgets.QSpinBox()
        self.deepcstrd_alpha.setRange(1, 180)
        self.deepcstrd_alpha.setValue(45)
        deepcstrd_layout.addRow(self.tr("Alpha (deg):"), self.deepcstrd_alpha)
        
        self.deepcstrd_nr = QtWidgets.QSpinBox()
        self.deepcstrd_nr.setRange(36, 720)
        self.deepcstrd_nr.setValue(360)
        self.deepcstrd_nr.setSingleStep(36)
        deepcstrd_layout.addRow(self.tr("Radial Samples:"), self.deepcstrd_nr)
        
        self.deepcstrd_rotations = QtWidgets.QSpinBox()
        self.deepcstrd_rotations.setRange(1, 10)
        self.deepcstrd_rotations.setValue(5)
        self.deepcstrd_rotations.setToolTip(
            self.tr("Test-time augmentation rotations.\n"
                   "Higher = more accurate but slower.")
        )
        deepcstrd_layout.addRow(self.tr("Rotations (TTA):"), self.deepcstrd_rotations)
        
        self.deepcstrd_threshold = QtWidgets.QDoubleSpinBox()
        self.deepcstrd_threshold.setRange(0.0, 1.0)
        self.deepcstrd_threshold.setValue(0.5)
        self.deepcstrd_threshold.setSingleStep(0.05)
        self.deepcstrd_threshold.setToolTip(
            self.tr("Prediction confidence threshold.\n"
                   "Lower = more rings (may include false positives)\n"
                   "Higher = fewer rings (may miss some).")
        )
        deepcstrd_layout.addRow(self.tr("Prediction Threshold:"), self.deepcstrd_threshold)
        
        self.deepcstrd_advanced_group.setLayout(deepcstrd_layout)
        self.form.addRow(self.deepcstrd_advanced_group)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
        )
        btns.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.form)
        layout.addWidget(btns)
        self.setLayout(layout)

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
            
            logger.info(f"CS-TRD: Parameters - sigma={sigma}, th_low={th_low}, th_high={th_high}, alpha={alpha}, nr={nr}")
            
            # Run CS-TRD with current parameters
            rings = detect_rings_cstrd(
                self.image_np, 
                center_xy=(cx, cy),
                sigma=sigma,
                th_low=th_low,
                th_high=th_high,
                alpha=alpha,
                nr=nr
            )
            
            QApplication.restoreOverrideCursor()
            
            if not rings:
                logger.warning("CS-TRD: No rings detected")
                QtWidgets.QMessageBox.information(self, self.tr("No rings found"), self.tr("CS-TRD did not detect any rings."))
                return
            
            logger.info(f"CS-TRD: Detected {len(rings)} rings")
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
            import traceback
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
            
            tile_text = self.deepcstrd_tile_size.currentText()
            tile_size = 256 if "256" in tile_text else 0
            
            alpha = self.deepcstrd_alpha.value()
            nr = self.deepcstrd_nr.value()
            total_rotations = self.deepcstrd_rotations.value()
            prediction_map_threshold = self.deepcstrd_threshold.value()
            
            logger.info(f"DeepCS-TRD: Model={model_id}, tile_size={tile_size}, alpha={alpha}, nr={nr}, rotations={total_rotations}, threshold={prediction_map_threshold}")
            
            # Run DeepCS-TRD with current parameters
            rings = detect_rings_deepcstrd(
                self.image_np, 
                center_xy=(cx, cy),
                model_id=model_id,
                tile_size=tile_size,
                alpha=alpha,
                nr=nr,
                total_rotations=total_rotations,
                prediction_map_threshold=prediction_map_threshold
            )
            
            QApplication.restoreOverrideCursor()
            
            if not rings:
                logger.warning("DeepCS-TRD: No rings detected")
                QtWidgets.QMessageBox.information(self, self.tr("No rings found"), self.tr("DeepCS-TRD did not detect any rings."))
                return
            
            logger.info(f"DeepCS-TRD: Detected {len(rings)} rings")
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
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"DeepCS-TRD Error: {error_details}")
            QtWidgets.QMessageBox.critical(
                self, 
                self.tr("DeepCS-TRD Error"), 
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
            
            QtWidgets.QMessageBox.information(
                self,
                self.tr("APD Success"),
                self.tr(f"Pith detected at ({x:.1f}, {y:.1f})")
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
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
    
    def get_pith_xy(self):
        """Return pith coordinates used for detection, or None"""
        return getattr(self, "detected_pith_xy", None)
