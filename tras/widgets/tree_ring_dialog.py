from __future__ import annotations

import platform
from typing import Optional

import io
import tempfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw
from loguru import logger
from PyQt5 import QtCore, QtGui, QtWidgets
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
        
        # Background removal option
        self.remove_bg_checkbox = QtWidgets.QCheckBox(self.tr("Remove background before detection"))
        self.remove_bg_checkbox.setChecked(True)  # Checked by default
        self.remove_bg_checkbox.setToolTip(
            self.tr("Remove background using U2Net to improve detection performance.\n"
                   "This can help reduce false detections from background artifacts.")
        )
        self.form.addRow(self.remove_bg_checkbox)
        
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
        
        # Resize parameters
        self.cstrd_width = QtWidgets.QSpinBox()
        self.cstrd_width.setRange(0, 10000)
        self.cstrd_width.setValue(0)
        self.cstrd_width.setSingleStep(100)
        self.cstrd_width.setToolTip(
            self.tr("Resize image width (0 = no resize).\n"
                   "Smaller images process faster but may reduce accuracy.\n"
                   "Recommended: 1000-2000 for faster processing.")
        )
        cstrd_layout.addRow(self.tr("Resize Width (px):"), self.cstrd_width)
        
        self.cstrd_height = QtWidgets.QSpinBox()
        self.cstrd_height.setRange(0, 10000)
        self.cstrd_height.setValue(0)
        self.cstrd_height.setSingleStep(100)
        self.cstrd_height.setToolTip(
            self.tr("Resize image height (0 = no resize).\n"
                   "Leave both at 0 to use original size.")
        )
        cstrd_layout.addRow(self.tr("Resize Height (px):"), self.cstrd_height)
        
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
        model_row = QtWidgets.QHBoxLayout()
        self.deepcstrd_model = QtWidgets.QComboBox()
        self._populate_model_list()
        self.deepcstrd_model.setToolTip(
            self.tr("Select the model trained on your target species.\n"
                   "Generic works well for most species.\n"
                   "Species-specific models may provide better results.")
        )
        model_row.addWidget(self.deepcstrd_model, stretch=3)
        
        # Upload Model button
        self.btn_upload_model = QtWidgets.QPushButton(self.tr("📁 Upload"))
        self.btn_upload_model.setToolTip(
            self.tr("Upload a new DeepCS-TRD model (.pth file).\n"
                   "You'll be asked for a model name and tile size.")
        )
        self.btn_upload_model.clicked.connect(self._on_upload_model)
        model_row.addWidget(self.btn_upload_model, stretch=1)
        
        deepcstrd_layout.addRow(self.tr("Model:"), model_row)
        
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
        
        # Resize parameters
        self.deepcstrd_width = QtWidgets.QSpinBox()
        self.deepcstrd_width.setRange(0, 10000)
        self.deepcstrd_width.setValue(1504)
        self.deepcstrd_width.setSingleStep(100)
        self.deepcstrd_width.setToolTip(
            self.tr("Resize image width (0 = no resize).\n"
                   "Smaller images use less GPU memory and process faster.\n"
                   "Recommended: 1000-2000 for faster processing.")
        )
        deepcstrd_layout.addRow(self.tr("Resize Width (px):"), self.deepcstrd_width)
        
        self.deepcstrd_height = QtWidgets.QSpinBox()
        self.deepcstrd_height.setRange(0, 10000)
        self.deepcstrd_height.setValue(1504)
        self.deepcstrd_height.setSingleStep(100)
        self.deepcstrd_height.setToolTip(
            self.tr("Resize image height (0 = no resize).\n"
                   "Leave both at 0 to use original size.")
        )
        deepcstrd_layout.addRow(self.tr("Resize Height (px):"), self.deepcstrd_height)
        
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
            width = self.cstrd_width.value()
            height = self.cstrd_height.value()
            
            logger.info(f"CS-TRD: Parameters - sigma={sigma}, th_low={th_low}, th_high={th_high}, alpha={alpha}, nr={nr}, resize=({width}, {height})")
            
            # Apply background removal if enabled
            image_to_process = self.image_np.copy()
            
            # Ensure image is in RGB format and contiguous (exactly like preprocessing dialog)
            # Ensure image is in RGB format (OpenCV-safe copy)
            image_to_process = np.ascontiguousarray(image_to_process, dtype=np.uint8)
            
            if self.remove_bg_checkbox.isChecked():
                logger.info("CS-TRD: Removing background before detection...")
                logger.info(f"CS-TRD: Image before BG removal - shape={image_to_process.shape}, dtype={image_to_process.dtype}, min={image_to_process.min()}, max={image_to_process.max()}")
                try:
                    from tras.tree_ring_methods.urudendro.remove_salient_object import (
                        remove_salient_object,
                    )
                    
                        # Create temporary files for U2Net
                    with tempfile.TemporaryDirectory() as temp_dir:
                        input_path = Path(temp_dir) / "input.png"
                        output_path = Path(temp_dir) / "output.png"
                        
                        # Ensure image is uint8 before saving (fix for depth warning)
                        if image_to_process.dtype != np.uint8:
                            image_to_process = (255 * (image_to_process.astype(np.float32) / image_to_process.max())).astype(np.uint8)
                        
                        # Save current image (convert RGB to BGR for cv2.imwrite)
                        cv2.imwrite(str(input_path), cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR))
                        
                        # Run U2Net background removal
                        remove_salient_object(str(input_path), str(output_path))
                        
                        # Load result (convert BGR back to RGB)
                        result = cv2.imread(str(output_path))
                        if result is not None:
                            image_to_process = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            image_to_process = np.ascontiguousarray(image_to_process, dtype=np.uint8)
                            logger.info(f"CS-TRD: Image after BG removal - shape={image_to_process.shape}, dtype={image_to_process.dtype}, min={image_to_process.min()}, max={image_to_process.max()}")
                            logger.info("CS-TRD: Background removal completed")
                        else:
                            raise Exception("U2Net did not produce output")
                except Exception as bg_error:
                    logger.warning(f"CS-TRD: Background removal failed: {bg_error}, continuing with original image")
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
            
            if self.remove_bg_checkbox.isChecked():
                logger.info("DeepCS-TRD: Removing background before detection...")
                logger.info(f"DeepCS-TRD: Image before BG removal - shape={image_to_process.shape}, dtype={image_to_process.dtype}, min={image_to_process.min()}, max={image_to_process.max()}")
                try:
                    from tras.tree_ring_methods.urudendro.remove_salient_object import (
                        remove_salient_object,
                    )
                    
                    # Create temporary files for U2Net
                    with tempfile.TemporaryDirectory() as temp_dir:
                        input_path = Path(temp_dir) / "input.png"
                        output_path = Path(temp_dir) / "output.png"
                        
                        # Ensure image is uint8 before saving (fix for depth warning)
                        if image_to_process.dtype != np.uint8:
                            image_to_process = (255 * (image_to_process.astype(np.float32) / image_to_process.max())).astype(np.uint8)
                        
                        # Save current image (convert RGB to BGR for cv2.imwrite)
                        cv2.imwrite(str(input_path), cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR))
                        
                        # Run U2Net background removal
                        remove_salient_object(str(input_path), str(output_path))
                        
                        # Load result (convert BGR back to RGB)
                        result = cv2.imread(str(output_path))
                        if result is not None:
                            image_to_process = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            image_to_process = np.ascontiguousarray(image_to_process, dtype=np.uint8)
                            logger.info(f"DeepCS-TRD: Image after BG removal - shape={image_to_process.shape}, dtype={image_to_process.dtype}, min={image_to_process.min()}, max={image_to_process.max()}")
                            logger.info("DeepCS-TRD: Background removal completed")
                        else:
                            raise Exception("U2Net did not produce output")
                except Exception as bg_error:
                    logger.warning(f"DeepCS-TRD: Background removal failed: {bg_error}, continuing with original image")
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
            
            self._show_pith_preview(x, y)
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
        from pathlib import Path
        
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
    
    def _on_upload_model(self):
        """Handle model upload button click."""
        from pathlib import Path
        import shutil
        
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
