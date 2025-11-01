from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
import numpy as np
import cv2


class PreprocessDialog(QtWidgets.QDialog):
    """Dialog for preprocessing wood cross-section images"""
    
    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Preprocess Image"))
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.original_image = image.copy()
        self.processed_image = image.copy()
        self.crop_rect = None  # (x, y, w, h)
        self.scale_factor = 1.0
        self.background_removed = False
        
        layout = QtWidgets.QVBoxLayout()
        
        # Instructions
        instructions = QtWidgets.QLabel(
            self.tr("Preprocess image before tree ring detection.\n"
                   "Changes will be applied when you click Apply.")
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)
        
        # Image info
        info_layout = QtWidgets.QFormLayout()
        self.original_size_label = QtWidgets.QLabel(
            f"{image.shape[1]} x {image.shape[0]} pixels"
        )
        info_layout.addRow(self.tr("Original size:"), self.original_size_label)
        
        self.current_size_label = QtWidgets.QLabel(
            f"{image.shape[1]} x {image.shape[0]} pixels"
        )
        info_layout.addRow(self.tr("Current size:"), self.current_size_label)
        layout.addLayout(info_layout)
        
        # Separator
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.HLine)
        line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line2)
        
        # Resize section
        resize_group = QtWidgets.QGroupBox(self.tr("Resize Image"))
        resize_layout = QtWidgets.QFormLayout()
        
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(10, 100)  # 10% to 100%
        self.scale_slider.setValue(100)
        self.scale_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.scale_slider.setTickInterval(10)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        
        self.scale_value_label = QtWidgets.QLabel("100%")
        
        scale_h_layout = QtWidgets.QHBoxLayout()
        scale_h_layout.addWidget(self.scale_slider)
        scale_h_layout.addWidget(self.scale_value_label)
        
        resize_layout.addRow(self.tr("Scale:"), scale_h_layout)
        resize_group.setLayout(resize_layout)
        layout.addWidget(resize_group)
        
        # Background removal section
        bg_group = QtWidgets.QGroupBox(self.tr("Background Removal"))
        bg_layout = QtWidgets.QVBoxLayout()
        
        self.bg_checkbox = QtWidgets.QCheckBox(
            self.tr("Remove background (simple thresholding)")
        )
        self.bg_checkbox.stateChanged.connect(self._on_bg_changed)
        bg_layout.addWidget(self.bg_checkbox)
        
        self.bg_threshold = QtWidgets.QSpinBox()
        self.bg_threshold.setRange(0, 255)
        self.bg_threshold.setValue(240)
        self.bg_threshold.setPrefix(self.tr("Threshold: "))
        self.bg_threshold.setEnabled(False)
        self.bg_threshold.valueChanged.connect(self._on_bg_threshold_changed)
        bg_layout.addWidget(self.bg_threshold)
        
        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)
        
        # Crop section (placeholder)
        crop_group = QtWidgets.QGroupBox(self.tr("Crop Image"))
        crop_layout = QtWidgets.QVBoxLayout()
        crop_info = QtWidgets.QLabel(
            self.tr("Use the rectangle tool in the main window to select crop region,\n"
                   "then come back to this dialog and click Apply.")
        )
        crop_info.setWordWrap(True)
        crop_layout.addWidget(crop_info)
        crop_group.setLayout(crop_layout)
        # Disable for now - would need canvas integration
        crop_group.setEnabled(False)
        layout.addWidget(crop_group)
        
        # Preview button
        self.preview_btn = QtWidgets.QPushButton(self.tr("Preview Changes"))
        self.preview_btn.clicked.connect(self._on_preview)
        layout.addWidget(self.preview_btn)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | 
            QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def _on_scale_changed(self, value):
        """Update scale factor"""
        self.scale_factor = value / 100.0
        self.scale_value_label.setText(f"{value}%")
        self._update_size_label()
    
    def _on_bg_changed(self, state):
        """Toggle background removal"""
        self.background_removed = state == QtCore.Qt.Checked
        self.bg_threshold.setEnabled(self.background_removed)
    
    def _on_bg_threshold_changed(self, value):
        """Background threshold changed"""
        pass  # Will be applied on preview/apply
    
    def _update_size_label(self):
        """Update current size label"""
        h, w = self.original_image.shape[:2]
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        self.current_size_label.setText(f"{new_w} x {new_h} pixels")
    
    def _on_preview(self):
        """Preview preprocessing"""
        self._apply_preprocessing()
        
        # Show message with new size
        h, w = self.processed_image.shape[:2]
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Preview"),
            self.tr(f"Processed image size: {w} x {h} pixels\n\n"
                   f"Original: {self.original_image.shape[1]} x {self.original_image.shape[0]}\n"
                   f"Scale factor: {self.scale_factor:.2f}\n"
                   f"Background removed: {self.background_removed}")
        )
    
    def _apply_preprocessing(self):
        """Apply all preprocessing steps"""
        img = self.original_image.copy()
        
        # Ensure image is in RGB format (OpenCV-safe copy)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # 1. Resize if needed
        if self.scale_factor != 1.0:
            h, w = img.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            # cv2.resize preserves channel order (doesn't convert RGB to BGR)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Ensure output is contiguous and maintains RGB
            img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # 2. Remove background if enabled
        if self.background_removed:
            threshold = self.bg_threshold.value()
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img.copy()
            
            # Create mask (background = white/bright)
            mask = gray < threshold
            
            # Apply mask
            if len(img.shape) == 3:
                img = img.copy()
                img[~mask] = [255, 255, 255]  # Set background to white
            else:
                img = img.copy()
                img[~mask] = 255
        
        self.processed_image = img
    
    def get_processed_image(self) -> np.ndarray:
        """Get the processed image"""
        self._apply_preprocessing()
        return self.processed_image
    
    def get_preprocessing_info(self) -> dict:
        """Get preprocessing metadata"""
        return {
            "scale_factor": self.scale_factor,
            "background_removed": self.background_removed,
            "background_threshold": self.bg_threshold.value() if self.background_removed else None,
            "original_size": [
                int(self.original_image.shape[1]),
                int(self.original_image.shape[0])
            ],
            "processed_size": [
                int(self.processed_image.shape[1]),
                int(self.processed_image.shape[0])
            ]
        }

