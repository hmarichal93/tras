from __future__ import annotations

from typing import Optional

from loguru import logger
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

from labelme.utils.apd_helper import detect_pith_apd
from labelme.utils.cstrd_helper import detect_rings_cstrd
from labelme.utils.deepcstrd_helper import detect_rings_deepcstrd

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

        # STEP 2: Ring Detection
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.HLine)
        separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.form.addRow(separator2)
        
        detection_label = QtWidgets.QLabel(self.tr("<b>Step 2: Detect Tree Rings</b>"))
        self.form.addRow(detection_label)
        
        # Add CS-TRD button
        self.btn_cstrd = QtWidgets.QPushButton(self.tr("Detect with CS-TRD (CPU)"))
        self.btn_cstrd.clicked.connect(self._on_cstrd)
        self.form.addRow(self.btn_cstrd)

        # Add DeepCSTRD button
        self.btn_deepcstrd = QtWidgets.QPushButton(self.tr("Detect with DeepCSTRD (GPU)"))
        self.btn_deepcstrd.clicked.connect(self._on_deepcstrd)
        self.form.addRow(self.btn_deepcstrd)

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
            
            # Run CS-TRD with current parameters
            rings = detect_rings_cstrd(
                self.image_np, 
                center_xy=(cx, cy),
                sigma=3.0,
                th_low=5.0,
                th_high=20.0,
                alpha=30,
                nr=360
            )
            
            QApplication.restoreOverrideCursor()
            
            if not rings:
                logger.warning("CS-TRD: No rings detected")
                QtWidgets.QMessageBox.information(self, self.tr("No rings found"), self.tr("CS-TRD did not detect any rings."))
                return
            
            logger.info(f"CS-TRD: Detected {len(rings)} rings")
            self.cstrd_rings = rings
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
            
            # Run DeepCS-TRD with current parameters
            rings = detect_rings_deepcstrd(
                self.image_np, 
                center_xy=(cx, cy),
                model_id="generic",
                tile_size=0,
                alpha=45,
                nr=360,
                total_rotations=5,
                prediction_map_threshold=0.5
            )
            
            QApplication.restoreOverrideCursor()
            
            if not rings:
                logger.warning("DeepCS-TRD: No rings detected")
                QtWidgets.QMessageBox.information(self, self.tr("No rings found"), self.tr("DeepCS-TRD did not detect any rings."))
                return
            
            logger.info(f"DeepCS-TRD: Detected {len(rings)} rings")
            self.deepcstrd_rings = rings
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
            logger.info(f"APD: Starting pith detection on image {self.image_np.shape}")
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            x, y = detect_pith_apd(self.image_np)
            
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
