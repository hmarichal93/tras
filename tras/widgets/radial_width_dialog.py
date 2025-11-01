from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

class RadialWidthDialog(QtWidgets.QDialog):
    """Dialog for managing radial ring width measurements"""
    
    def __init__(self, parent=None, pith_xy=None, has_measurement=False, measurement_data=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Radial Ring Width Measurement"))
        self.setModal(True)
        self.parent_window = parent
        self.pith_xy = pith_xy
        self.has_measurement = has_measurement
        self.measurement_data = measurement_data
        self.action_result = None  # 'set_direction', 'clear', 'export', None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Info section
        info_group = QtWidgets.QGroupBox(self.tr("Measurement Information"))
        info_layout = QtWidgets.QVBoxLayout()
        
        if self.pith_xy:
            pith_label = QtWidgets.QLabel(
                self.tr(f"Pith location: ({self.pith_xy[0]:.1f}, {self.pith_xy[1]:.1f})")
            )
            info_layout.addWidget(pith_label)
        
        if self.has_measurement and self.measurement_data:
            n_rings = len(self.measurement_data.get('measurements', {}))
            status_label = QtWidgets.QLabel(
                self.tr(f"✓ Measured {n_rings} rings along radial line")
            )
            status_label.setStyleSheet("color: green; font-weight: bold;")
            info_layout.addWidget(status_label)
        else:
            status_label = QtWidgets.QLabel(
                self.tr("No radial measurement set yet")
            )
            status_label.setStyleSheet("color: gray;")
            info_layout.addWidget(status_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Actions section
        actions_group = QtWidgets.QGroupBox(self.tr("Actions"))
        actions_layout = QtWidgets.QVBoxLayout()
        
        # Set direction button
        self.btn_set_direction = QtWidgets.QPushButton(
            self.tr("📐 Set Measurement Direction")
        )
        self.btn_set_direction.setToolTip(
            self.tr("Click to define the radial line direction from the pith.\n"
                   "Ring widths will be measured along this line.")
        )
        self.btn_set_direction.clicked.connect(self._on_set_direction)
        actions_layout.addWidget(self.btn_set_direction)
        
        # Clear line button
        self.btn_clear = QtWidgets.QPushButton(
            self.tr("🗑️ Clear Measurement Line")
        )
        self.btn_clear.setToolTip(
            self.tr("Remove the current radial measurement line")
        )
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_clear.setEnabled(self.has_measurement)
        actions_layout.addWidget(self.btn_clear)
        
        actions_layout.addSpacing(10)
        
        # Export button
        self.btn_export = QtWidgets.QPushButton(
            self.tr("💾 Export to .POS (CooRecorder)")
        )
        self.btn_export.setToolTip(
            self.tr("Export radial width measurements in .POS format\n"
                   "Compatible with CooRecorder dendrochronology software")
        )
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(self.has_measurement)
        actions_layout.addWidget(self.btn_export)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Help text
        help_text = QtWidgets.QLabel(
            self.tr("ℹ️ Radial width measures ring boundaries along a specific line,\n"
                   "which is the standard method in dendrochronology research.")
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #666; font-size: 10px; padding: 10px;")
        layout.addWidget(help_text)
        
        # Close button
        close_btn = QtWidgets.QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def _on_set_direction(self):
        """User wants to set/change measurement direction"""
        self.action_result = 'set_direction'
        self.accept()
    
    def _on_clear(self):
        """User wants to clear measurement line"""
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("Clear Measurement"),
            self.tr("Clear the current radial measurement line?\n\n"
                   "You can set a new direction afterwards."),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.action_result = 'clear'
            self.accept()
    
    def _on_export(self):
        """User wants to export to .POS format"""
        self.action_result = 'export'
        self.accept()
    
    def get_action(self):
        """Return the action the user selected"""
        return self.action_result

