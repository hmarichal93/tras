"""
Scale Calibration Dialog - Set physical scale for measurements
"""
from PyQt5 import QtCore, QtWidgets


class ScaleDialog(QtWidgets.QDialog):
    """Dialog for setting image scale (pixels to physical units)"""
    
    def __init__(self, parent=None, current_scale=None, current_unit='mm'):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Set Image Scale"))
        self.setModal(True)
        self.resize(500, 300)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Info
        info = QtWidgets.QLabel(
            self.tr("Set the relationship between pixels and physical units.\n"
                   "This enables real-world measurements in ring properties.")
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)
        
        # Method selection
        method_group = QtWidgets.QGroupBox(self.tr("Calibration Method"))
        method_layout = QtWidgets.QVBoxLayout()

        self.method_buttons = QtWidgets.QButtonGroup()
        
        # Method 1: Draw line
        self.draw_line_radio = QtWidgets.QRadioButton(
            self.tr("📏 Draw a line segment on the image")
        )
        self.draw_line_radio.setChecked(True)
        self.method_buttons.addButton(self.draw_line_radio, 1)
        method_layout.addWidget(self.draw_line_radio)
        
        draw_hint = QtWidgets.QLabel(
            self.tr("   → Draw a line of known length, then specify its physical size")
        )
        draw_hint.setStyleSheet("color: gray; font-size: 10px;")
        method_layout.addWidget(draw_hint)
        
        # Method 2: Direct input
        self.direct_input_radio = QtWidgets.QRadioButton(
            self.tr("⌨️  Enter scale directly (if known)")
        )
        self.method_buttons.addButton(self.direct_input_radio, 2)
        method_layout.addWidget(self.direct_input_radio)
        
        direct_hint = QtWidgets.QLabel(
            self.tr("   → Example: 0.02 mm/pixel, 20 μm/pixel")
        )
        direct_hint.setStyleSheet("color: gray; font-size: 10px;")
        method_layout.addWidget(direct_hint)

        # Method 3: Enter DPI
        self.dpi_radio = QtWidgets.QRadioButton(
            self.tr("🖨️  Enter image DPI (pixels per inch)")
        )
        self.method_buttons.addButton(self.dpi_radio, 3)
        method_layout.addWidget(self.dpi_radio)

        dpi_hint = QtWidgets.QLabel(
            self.tr("   → Converts DPI to physical units (mm/cm/μm per pixel)")
        )
        dpi_hint.setStyleSheet("color: gray; font-size: 10px;")
        method_layout.addWidget(dpi_hint)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Direct input fields (only shown when method 2 is selected)
        self.direct_input_widget = QtWidgets.QWidget()
        direct_layout = QtWidgets.QFormLayout()
        
        self.scale_input = QtWidgets.QDoubleSpinBox()
        self.scale_input.setDecimals(6)
        self.scale_input.setRange(0.000001, 1000.0)
        self.scale_input.setValue(current_scale if current_scale else 0.02)
        self.scale_input.setSingleStep(0.01)
        
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(['mm', 'cm', 'μm'])
        self.unit_combo.setCurrentText(current_unit)
        
        unit_layout = QtWidgets.QHBoxLayout()
        unit_layout.addWidget(self.scale_input)
        unit_layout.addWidget(QtWidgets.QLabel(self.tr("/pixel")))
        unit_layout.addStretch()
        
        direct_layout.addRow(self.tr("Scale value:"), unit_layout)
        direct_layout.addRow(self.tr("Unit:"), self.unit_combo)

        self.direct_input_widget.setLayout(direct_layout)
        layout.addWidget(self.direct_input_widget)

        # DPI input fields
        self.dpi_input_widget = QtWidgets.QWidget()
        dpi_layout = QtWidgets.QFormLayout()

        self.dpi_input = QtWidgets.QDoubleSpinBox()
        self.dpi_input.setDecimals(2)
        self.dpi_input.setRange(1.0, 100000.0)
        self.dpi_input.setValue(300.0)
        self.dpi_input.setSingleStep(10.0)

        self.dpi_unit_combo = QtWidgets.QComboBox()
        self.dpi_unit_combo.addItems(["mm", "cm", "μm"])
        self.dpi_unit_combo.setCurrentText(current_unit)

        dpi_layout.addRow(self.tr("DPI (pixels/inch):"), self.dpi_input)
        dpi_layout.addRow(self.tr("Convert to unit:"), self.dpi_unit_combo)

        self.dpi_input_widget.setLayout(dpi_layout)
        layout.addWidget(self.dpi_input_widget)
        
        # Current scale display
        if current_scale:
            current_label = QtWidgets.QLabel(
                self.tr(f"Current scale: {current_scale:.6f} {current_unit}/pixel")
            )
            current_label.setStyleSheet("color: blue; font-weight: bold;")
            layout.addWidget(current_label)
        
        layout.addStretch()
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox()
        
        if self.draw_line_radio.isChecked():
            draw_btn = button_box.addButton(
                self.tr("Draw Line"), 
                QtWidgets.QDialogButtonBox.ActionRole
            )
            draw_btn.clicked.connect(lambda: self.done(2))  # Custom code for draw mode
        
        button_box.addButton(QtWidgets.QDialogButtonBox.Ok)
        button_box.addButton(QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Connect radio buttons to update UI
        self.draw_line_radio.toggled.connect(self._update_ui)
        self.direct_input_radio.toggled.connect(self._update_ui)
        self.dpi_radio.toggled.connect(self._update_ui)

        self._update_ui()

    def _update_ui(self):
        """Update UI based on selected method"""
        is_direct = self.direct_input_radio.isChecked()
        is_dpi = self.dpi_radio.isChecked()
        self.direct_input_widget.setVisible(is_direct)
        self.dpi_input_widget.setVisible(is_dpi)

    def get_method(self):
        """Get selected calibration method"""
        if self.draw_line_radio.isChecked():
            return "draw"
        if self.direct_input_radio.isChecked():
            return "direct"
        return "dpi"

    def get_scale_value(self):
        """Get the scale value (only valid for direct input)"""
        return self.scale_input.value()
    
    def get_unit(self):
        """Get the selected unit"""
        return self.unit_combo.currentText()

    def get_dpi_value(self):
        return self.dpi_input.value()

    def get_dpi_unit(self):
        return self.dpi_unit_combo.currentText()


class LineCalibrationDialog(QtWidgets.QDialog):
    """Dialog for entering physical length of drawn line"""
    
    def __init__(self, line_length_pixels, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Calibrate Scale"))
        self.setModal(True)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Show line length
        info = QtWidgets.QLabel(
            self.tr(f"Line drawn: {line_length_pixels:.2f} pixels\n\n"
                   f"Enter the physical length of this line:")
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Input fields
        form = QtWidgets.QFormLayout()
        
        self.length_input = QtWidgets.QDoubleSpinBox()
        self.length_input.setDecimals(3)
        self.length_input.setRange(0.001, 10000.0)
        self.length_input.setValue(10.0)
        self.length_input.setSingleStep(1.0)
        
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(['mm', 'cm', 'μm'])
        
        form.addRow(self.tr("Physical length:"), self.length_input)
        form.addRow(self.tr("Unit:"), self.unit_combo)
        
        layout.addLayout(form)
        
        # Calculated scale preview
        self.scale_label = QtWidgets.QLabel()
        self.scale_label.setStyleSheet("color: blue; font-weight: bold; padding: 10px;")
        layout.addWidget(self.scale_label)
        
        # Update scale preview when values change
        self.length_input.valueChanged.connect(lambda: self._update_scale_preview(line_length_pixels))
        self.unit_combo.currentTextChanged.connect(lambda: self._update_scale_preview(line_length_pixels))
        
        # Initial preview
        self._update_scale_preview(line_length_pixels)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def _update_scale_preview(self, line_length_pixels):
        """Update the calculated scale preview"""
        physical_length = self.length_input.value()
        unit = self.unit_combo.currentText()
        
        if line_length_pixels > 0 and physical_length > 0:
            scale = physical_length / line_length_pixels
            self.scale_label.setText(
                self.tr(f"Calculated scale: {scale:.6f} {unit}/pixel")
            )
    
    def get_physical_length(self):
        """Get the entered physical length"""
        return self.length_input.value()
    
    def get_unit(self):
        """Get the selected unit"""
        return self.unit_combo.currentText()
    
    def get_scale(self, line_length_pixels):
        """Calculate and return the scale"""
        if line_length_pixels > 0:
            return self.get_physical_length() / line_length_pixels
        return None
