from __future__ import annotations

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication
import numpy as np
import cv2
import tempfile
from pathlib import Path

from tras.tree_ring_methods.urudendro._model_weights import get_model_path


class PreprocessDialog(QtWidgets.QDialog):
    """Dialog for preprocessing wood cross-section images"""
    
    def __init__(self, image: np.ndarray, parent=None, crop_rect=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Preprocess Image"))
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.original_image = image.copy()
        self.processed_image = image.copy()
        self.crop_rect = crop_rect  # (x, y, w, h) if provided
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
        
        # Crop section
        crop_group = QtWidgets.QGroupBox(self.tr("1. Crop Image (Optional)"))
        crop_layout = QtWidgets.QVBoxLayout()
        
        # Warning about tight cropping
        crop_warning = QtWidgets.QLabel(
            self.tr("⚠️ Important: Leave adequate margin (>100px) around the wood section.\n"
                   "CS-TRD and DeepCS-TRD may fail on tightly cropped images.")
        )
        crop_warning.setWordWrap(True)
        crop_warning.setStyleSheet("color: #ff6b00; font-size: 10px; padding: 5px; "
                                   "background-color: #fff3cd; border-radius: 3px;")
        crop_layout.addWidget(crop_warning)
        
        # Crop button
        self.crop_btn = QtWidgets.QPushButton(self.tr("📐 Draw Crop Rectangle"))
        self.crop_btn.clicked.connect(self._on_crop_button)
        crop_layout.addWidget(self.crop_btn)
        
        # Crop status
        if self.crop_rect:
            x, y, w, h = self.crop_rect
            crop_status = QtWidgets.QLabel(
                self.tr(f"✓ Crop region selected: {w}x{h} pixels at ({x}, {y})")
            )
            crop_status.setStyleSheet("color: green;")
        else:
            crop_status = QtWidgets.QLabel(
                self.tr("No crop region selected")
            )
            crop_status.setStyleSheet("color: gray;")
        crop_layout.addWidget(crop_status)
        
        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)
        
        # Resize section
        resize_group = QtWidgets.QGroupBox(self.tr("2. Resize Image (Optional)"))
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
        
        # Background removal section (U2Net)
        bg_group = QtWidgets.QGroupBox(self.tr("3. Background Removal (Optional)"))
        bg_layout = QtWidgets.QVBoxLayout()
        
        self.bg_checkbox = QtWidgets.QCheckBox(
            self.tr("Remove background using U2Net model")
        )
        self.bg_checkbox.stateChanged.connect(self._on_bg_changed)
        bg_layout.addWidget(self.bg_checkbox)
        
        bg_info = QtWidgets.QLabel(
            self.tr("Uses urudendro U2Net model for salient object removal.\n"
                   "⚠️ This may take 10-30 seconds depending on image size and GPU availability.")
        )
        bg_info.setWordWrap(True)
        bg_info.setStyleSheet("color: gray; font-size: 10px;")
        bg_layout.addWidget(bg_info)
        
        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)
        
        # Preview button
        self.preview_btn = QtWidgets.QPushButton(self.tr("Preview Changes"))
        self.preview_btn.clicked.connect(self._on_preview)
        layout.addWidget(self.preview_btn)
        
        # Preview area
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        layout.addWidget(self.preview_label)
        
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
        if state == QtCore.Qt.Checked:
            model_path = get_model_path()
            if not model_path.exists():
                model_path_str = str(model_path)
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Model Not Found"),
                    self.tr(
                        "U2Net background removal requires the model file at:\n"
                        f"{model_path_str}\n\n"
                        "Run `python tools/download_release_assets.py --url "
                        "https://github.com/hmarichal93/tras/releases/tag/v2.0.2_models "
                        "--dest ./downloaded_assets` to download it."
                    ),
                )
                self.bg_checkbox.blockSignals(True)
                self.bg_checkbox.setChecked(False)
                self.bg_checkbox.blockSignals(False)
                self.background_removed = False
                return
            self.background_removed = True
        else:
            self.background_removed = False
    
    def _on_crop_button(self):
        """Handle crop button click - close dialog to let user draw rectangle"""
        self.done(2)  # Custom return code for crop action
    
    def _update_size_label(self):
        """Update current size label"""
        # Calculate size after crop (if any)
        if self.crop_rect:
            _, _, w, h = self.crop_rect
        else:
            h, w = self.original_image.shape[:2]
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        self.current_size_label.setText(f"{new_w} x {new_h} pixels")
    
    def _on_preview(self):
        """Preview preprocessing"""
        try:
            # Set wait cursor during processing
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            self._apply_preprocessing()
            
            # Restore cursor
            QApplication.restoreOverrideCursor()
            
            # Display preview image
            display_img = self.processed_image.copy()
            h, w = display_img.shape[:2]
            
            # Scale to fit preview area (max 400px)
            max_size = 400
            if h > max_size or w > max_size:
                scale = min(max_size / h, max_size / w)
                new_h, new_w = int(h * scale), int(w * scale)
                display_img = cv2.resize(display_img, (new_w, new_h))
            
            # Convert to QPixmap
            h, w = display_img.shape[:2]
            if len(display_img.shape) == 3:
                bytes_per_line = 3 * w
                q_img = QtGui.QImage(display_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            else:
                bytes_per_line = w
                q_img = QtGui.QImage(display_img.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            
            pixmap = QtGui.QPixmap.fromImage(q_img)
            self.preview_label.setPixmap(pixmap)
            
            # Show info
            h, w = self.processed_image.shape[:2]
            info_text = f"✓ Preview generated\nFinal size: {w} x {h} pixels"
            if self.crop_rect:
                info_text += f"\n✓ Cropped"
            if self.scale_factor != 1.0:
                info_text += f"\n✓ Scaled to {int(self.scale_factor * 100)}%"
            if self.background_removed:
                info_text += f"\n✓ Background removed"
            
            QtWidgets.QMessageBox.information(self, self.tr("Preview"), info_text)
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, 
                self.tr("Preview Error"),
                f"Failed to generate preview:\n{str(e)}"
            )
    
    def _apply_preprocessing(self):
        """Apply all preprocessing steps"""
        img = self.original_image.copy()
        
        # Ensure image is in RGB format (OpenCV-safe copy)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # 1. Crop if region is selected
        if self.crop_rect:
            x, y, w, h = self.crop_rect
            img = img[y:y+h, x:x+w]
            img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # 2. Resize if needed
        if self.scale_factor != 1.0:
            h, w = img.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            
            # Use PIL for resize to avoid orientation issues
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img, mode='RGB')
            pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.BICUBIC)
            img = np.array(pil_img, dtype=np.uint8)
            
            # Ensure output is contiguous
            img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # 3. Remove background using U2Net if enabled
        if self.background_removed:
            try:
                from tras.tree_ring_methods.urudendro.remove_salient_object import (
                    remove_salient_object,
                )
                
                # Create temporary files
                with tempfile.TemporaryDirectory() as temp_dir:
                    input_path = Path(temp_dir) / "input.png"
                    output_path = Path(temp_dir) / "output.png"
                    
                    # Save current image (convert RGB to BGR for cv2.imwrite)
                    cv2.imwrite(str(input_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    # Run U2Net background removal
                    remove_salient_object(str(input_path), str(output_path))
                    
                    # Load result (convert BGR back to RGB)
                    result = cv2.imread(str(output_path))
                    if result is not None:
                        img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        img = np.ascontiguousarray(img, dtype=np.uint8)
                    else:
                        raise Exception("U2Net did not produce output")
                        
            except Exception as e:
                raise Exception(f"U2Net background removal failed: {str(e)}")
        
        self.processed_image = img
    
    def get_processed_image(self) -> np.ndarray:
        """Get the processed image"""
        try:
            # Set wait cursor during processing
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self._apply_preprocessing()
            QApplication.restoreOverrideCursor()
            return self.processed_image
        except Exception as e:
            QApplication.restoreOverrideCursor()
            raise e
    
    def get_preprocessing_info(self) -> dict:
        """Get preprocessing metadata"""
        return {
            "crop_rect": self.crop_rect,
            "scale_factor": self.scale_factor,
            "background_removed": self.background_removed,
            "background_method": "u2net" if self.background_removed else None,
            "original_size": [
                int(self.original_image.shape[1]),
                int(self.original_image.shape[0])
            ],
            "processed_size": [
                int(self.processed_image.shape[1]),
                int(self.processed_image.shape[0])
            ]
        }
