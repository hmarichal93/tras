from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets




from labelme.utils.apd_helper import detect_pith_apd
from labelme.utils.deepcstrd_helper import detect_rings_deepcstrd

class TreeRingDialog(QtWidgets.QDialog):
    def __init__(self, image_width: int, image_height: int, parent=None, image_np=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Tree Ring Detection"))
        self.setModal(True)

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

        self.angular_steps = QtWidgets.QSpinBox()
        self.angular_steps.setRange(90, 4096)
        self.angular_steps.setValue(720)
        self.form.addRow(self.tr("Angular steps"), self.angular_steps)

        self.min_radius = QtWidgets.QDoubleSpinBox()
        self.min_radius.setRange(0.0, 1e6)
        self.min_radius.setDecimals(2)
        self.min_radius.setValue(5.0)
        self.form.addRow(self.tr("Min radius"), self.min_radius)

        self.rel_thr = QtWidgets.QDoubleSpinBox()
        self.rel_thr.setRange(0.0, 1.0)
        self.rel_thr.setSingleStep(0.05)
        self.rel_thr.setValue(0.3)
        self.form.addRow(self.tr("Relative threshold"), self.rel_thr)

        self.min_peak_dist = QtWidgets.QSpinBox()
        self.min_peak_dist.setRange(1, 9999)
        self.min_peak_dist.setValue(3)
        self.form.addRow(self.tr("Min peak distance"), self.min_peak_dist)

        self.min_coverage = QtWidgets.QDoubleSpinBox()
        self.min_coverage.setRange(0.0, 1.0)
        self.min_coverage.setSingleStep(0.05)
        self.min_coverage.setValue(0.6)
        self.form.addRow(self.tr("Min coverage"), self.min_coverage)

        self.max_rings = QtWidgets.QSpinBox()
        self.max_rings.setRange(0, 4096)
        self.max_rings.setToolTip(self.tr("0 = no limit"))
        self.max_rings.setValue(0)
        self.form.addRow(self.tr("Max rings"), self.max_rings)

        # Add auto-detect pith button
        self.image_np = image_np
        self.btn_auto_pith = QtWidgets.QPushButton(self.tr("Auto-detect pith"))
        self.btn_auto_pith.clicked.connect(self._on_auto_pith)
        self.form.addRow(self.btn_auto_pith)

        # Add DeepCSTRD button
        self.btn_deepcstrd = QtWidgets.QPushButton(self.tr("Detect with DeepCSTRD (AI)"))
        self.btn_deepcstrd.clicked.connect(self._on_deepcstrd)
        self.form.addRow(self.btn_deepcstrd)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.form)
        layout.addWidget(btns)
        self.setLayout(layout)

    def _on_deepcstrd(self):
        if self.image_np is None:
            QtWidgets.QMessageBox.warning(self, self.tr("No image"), self.tr("No image data available for DeepCS-TRD."))
            return
        try:
            # Get current center coordinates
            cx = float(self.cx.value())
            cy = float(self.cy.value())
            
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
            
            if not rings:
                QtWidgets.QMessageBox.information(self, self.tr("No rings found"), self.tr("DeepCS-TRD did not detect any rings."))
                return
            
            self.deepcstrd_rings = rings
            QtWidgets.QMessageBox.information(
                self, 
                self.tr("DeepCS-TRD Success"), 
                self.tr(f"Detected {len(rings)} rings. Click OK to insert them.")
            )
            self.accept()  # Close dialog and signal to use these rings
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, self.tr("DeepCS-TRD Error"), str(e))

    def _on_auto_pith(self):
        if self.image_np is None:
            QtWidgets.QMessageBox.warning(self, self.tr("No image"), self.tr("No image data available for pith detection."))
            return
        try:
            x, y = detect_pith_apd(self.image_np)
            self.cx.setValue(x)
            self.cy.setValue(y)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, self.tr("APD Error"), str(e))

    def get_params(self) -> dict:
        params = dict(
            center_x=float(self.cx.value()),
            center_y=float(self.cy.value()),
            angular_steps=int(self.angular_steps.value()),
            min_radius=float(self.min_radius.value()),
            relative_threshold=float(self.rel_thr.value()),
            min_peak_distance=int(self.min_peak_dist.value()),
            min_coverage=float(self.min_coverage.value()),
            max_rings=(
                None if int(self.max_rings.value()) == 0 else int(self.max_rings.value())
            ),
        )
        return params

    def get_deepcstrd_rings(self):
        # Returns rings if DeepCSTRD was used, else None
        return getattr(self, 'deepcstrd_rings', None)
