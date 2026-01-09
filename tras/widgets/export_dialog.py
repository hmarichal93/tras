from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from loguru import logger
from pathlib import Path

from tras._label_file import LabelFile


class ExportDialog(QtWidgets.QDialog):
    """Dialog for exporting annotations, measurements, and PDF reports."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export Data"))
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.parent_window = parent
        
        layout = QtWidgets.QVBoxLayout()
        
        # Instructions
        instructions = QtWidgets.QLabel(
            self.tr("Select what you want to export. You can export multiple formats at once.")
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)
        
        # Export options
        options_group = QtWidgets.QGroupBox(self.tr("Export Options"))
        options_layout = QtWidgets.QVBoxLayout()
        
        # JSON annotations checkbox
        self.export_json_checkbox = QtWidgets.QCheckBox(
            self.tr("Export annotations as JSON")
        )
        self.export_json_checkbox.setToolTip(
            self.tr("Export all ring annotations, labels, metadata, and preprocessing info.\n"
                   "This is the same format used by File > Save.")
        )
        self.export_json_checkbox.setChecked(True)
        options_layout.addWidget(self.export_json_checkbox)
        
        # Save image checkbox
        self.export_image_checkbox = QtWidgets.QCheckBox(
            self.tr("Save image as separate file")
        )
        self.export_image_checkbox.setToolTip(
            self.tr("Save the current image (with annotations if visible) as a separate image file.\n"
                   "The image will be saved in PNG format.")
        )
        self.export_image_checkbox.setChecked(False)
        options_layout.addWidget(self.export_image_checkbox)
        
        # CSV measurements checkbox
        self.export_csv_checkbox = QtWidgets.QCheckBox(
            self.tr("Export measurements as CSV")
        )
        self.export_csv_checkbox.setToolTip(
            self.tr("Export ring properties (area, perimeter, radial width) as CSV.\n"
                   "Only available if rings have been detected.")
        )
        options_layout.addWidget(self.export_csv_checkbox)
        
        # POS measurements checkbox
        self.export_pos_checkbox = QtWidgets.QCheckBox(
            self.tr("Export radial measurements as .POS")
        )
        self.export_pos_checkbox.setToolTip(
            self.tr("Export radial width measurements in CooRecorder .POS format.\n"
                   "Only available if radial measurements have been performed.")
        )
        options_layout.addWidget(self.export_pos_checkbox)
        
        # PDF report checkbox
        self.export_pdf_checkbox = QtWidgets.QCheckBox(
            self.tr("Export PDF report")
        )
        self.export_pdf_checkbox.setToolTip(
            self.tr("Generate a comprehensive PDF report with ring overlays and analysis plots.\n"
                   "Includes metadata, ring visualization, and measurement charts.")
        )
        self.export_pdf_checkbox.setChecked(True)
        options_layout.addWidget(self.export_pdf_checkbox)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Output directory selection
        dir_group = QtWidgets.QGroupBox(self.tr("Output Directory"))
        dir_layout = QtWidgets.QHBoxLayout()
        
        self.dir_path_edit = QtWidgets.QLineEdit()
        self.dir_path_edit.setPlaceholderText(self.tr("Select output directory..."))
        dir_layout.addWidget(self.dir_path_edit)
        
        browse_dir_btn = QtWidgets.QPushButton(self.tr("Browse..."))
        browse_dir_btn.clicked.connect(self._browse_directory)
        dir_layout.addWidget(browse_dir_btn)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Status label
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | 
            QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.button(QtWidgets.QDialogButtonBox.Ok).setText(self.tr("Export"))
        button_box.accepted.connect(self._export)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Update status
        self._update_status()
        
        # Connect checkboxes to update status
        self.export_json_checkbox.toggled.connect(self._update_status)
        self.export_image_checkbox.toggled.connect(self._update_status)
        self.export_csv_checkbox.toggled.connect(self._update_status)
        self.export_pos_checkbox.toggled.connect(self._update_status)
        self.export_pdf_checkbox.toggled.connect(self._update_status)
    
    def _update_status(self):
        """Update status label based on selected options and available data."""
        status_parts = []
        
        # Check what's available
        has_rings = False
        has_radial_measurements = False
        
        if self.parent_window:
            has_rings = len(self.parent_window.labelList) > 0
            has_radial_measurements = (
                hasattr(self.parent_window, 'radial_line_measurements') and
                self.parent_window.radial_line_measurements is not None
            )
        
        # Update checkbox states based on availability
        if not has_rings:
            self.export_csv_checkbox.setEnabled(False)
            self.export_csv_checkbox.setToolTip(
                self.tr("No rings detected. Please detect rings first.")
            )
        else:
            self.export_csv_checkbox.setEnabled(True)
        
        if not has_radial_measurements:
            self.export_pos_checkbox.setEnabled(False)
            self.export_pos_checkbox.setToolTip(
                self.tr("No radial measurements available. Please measure ring widths first.")
            )
        else:
            self.export_pos_checkbox.setEnabled(True)
        
        # Build status message
        selected = []
        if self.export_json_checkbox.isChecked():
            selected.append(self.tr("JSON"))
        if self.export_image_checkbox.isChecked():
            selected.append(self.tr("Image"))
        if self.export_csv_checkbox.isChecked() and self.export_csv_checkbox.isEnabled():
            selected.append(self.tr("CSV"))
        if self.export_pos_checkbox.isChecked() and self.export_pos_checkbox.isEnabled():
            selected.append(self.tr(".POS"))
        if self.export_pdf_checkbox.isChecked():
            selected.append(self.tr("PDF"))
        
        if selected:
            self.status_label.setText(
                self.tr("Will export: {}").format(", ".join(selected))
            )
        else:
            self.status_label.setText(
                self.tr("Please select at least one export option.")
            )
    
    def _browse_directory(self):
        """Open directory selection dialog."""
        current_dir = self.dir_path_edit.text() or str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Output Directory"),
            current_dir
        )
        if directory:
            self.dir_path_edit.setText(directory)
    
    def _export(self):
        """Perform the export operations."""
        output_dir = self.dir_path_edit.text().strip()
        
        if not output_dir:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("No Directory Selected"),
                self.tr("Please select an output directory.")
            )
            return
        
        # Check if at least one option is selected
        if not any([
            self.export_json_checkbox.isChecked(),
            self.export_image_checkbox.isChecked(),
            self.export_csv_checkbox.isChecked() and self.export_csv_checkbox.isEnabled(),
            self.export_pos_checkbox.isChecked() and self.export_pos_checkbox.isEnabled(),
            self.export_pdf_checkbox.isChecked(),
        ]):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("No Export Option Selected"),
                self.tr("Please select at least one export option.")
            )
            return
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Directory Error"),
                self.tr(f"Failed to create output directory:\n{str(e)}")
            )
            return
        
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
        exported_files = []
        errors = []
        
        try:
            # Export JSON annotations
            if self.export_json_checkbox.isChecked():
                try:
                    json_file = self._export_json(output_path)
                    if json_file:
                        exported_files.append(json_file)
                except Exception as e:
                    errors.append(f"JSON: {str(e)}")
                    logger.error(f"Failed to export JSON: {e}", exc_info=True)
            
            # Export image as separate file
            if self.export_image_checkbox.isChecked():
                try:
                    image_file = self._export_image(output_path)
                    if image_file:
                        exported_files.append(image_file)
                except Exception as e:
                    errors.append(f"Image: {str(e)}")
                    logger.error(f"Failed to export image: {e}", exc_info=True)
            
            # Export CSV measurements
            if self.export_csv_checkbox.isChecked() and self.export_csv_checkbox.isEnabled():
                try:
                    csv_file = self._export_csv(output_path)
                    if csv_file:
                        exported_files.append(csv_file)
                except Exception as e:
                    errors.append(f"CSV: {str(e)}")
                    logger.error(f"Failed to export CSV: {e}", exc_info=True)
            
            # Export POS measurements
            if self.export_pos_checkbox.isChecked() and self.export_pos_checkbox.isEnabled():
                try:
                    pos_file = self._export_pos(output_path)
                    if pos_file:
                        exported_files.append(pos_file)
                except Exception as e:
                    errors.append(f".POS: {str(e)}")
                    logger.error(f"Failed to export .POS: {e}", exc_info=True)
            
            # Export PDF report
            if self.export_pdf_checkbox.isChecked():
                try:
                    pdf_file = self._export_pdf(output_path)
                    if pdf_file:
                        exported_files.append(pdf_file)
                except Exception as e:
                    errors.append(f"PDF: {str(e)}")
                    logger.error(f"Failed to export PDF: {e}", exc_info=True)
            
        finally:
            QApplication.restoreOverrideCursor()
        
        # Show results
        if exported_files:
            message = self.tr("Successfully exported:\n\n")
            message += "\n".join(f"  • {f}" for f in exported_files)
            if errors:
                message += "\n\n" + self.tr("Errors:\n")
                message += "\n".join(f"  • {e}" for e in errors)
            
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Export Complete"),
                message
            )
            self.accept()
        else:
            error_msg = self.tr("Export failed:\n\n")
            error_msg += "\n".join(f"  • {e}" for e in errors) if errors else self.tr("Unknown error")
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Export Failed"),
                error_msg
            )
    
    def _export_json(self, output_dir: Path) -> str | None:
        """Export annotations as JSON."""
        if not self.parent_window:
            logger.error("Export JSON: No parent window")
            return None
        
        try:
            # Use default filename based on image or sample code
            metadata = getattr(self.parent_window, 'sample_metadata', None) or {}
            if self.parent_window.filename:
                # self.filename is the image file path, convert to JSON filename
                image_path = Path(self.parent_window.filename)
                json_file = output_dir / f"{image_path.stem}.json"
            elif metadata.get('sample_code'):
                base_name = metadata['sample_code']
                json_file = output_dir / f"{base_name}.json"
            else:
                base_name = "annotations"
                json_file = output_dir / f"{base_name}.json"
            
            logger.info(f"Export JSON: Target file is {json_file}")
            
            # Convert shapes to dictionaries using the same format as saveLabels
            def format_shape(s):
                from tras import utils
                import numpy as np
                
                data = s.other_data.copy() if hasattr(s, 'other_data') and s.other_data else {}
                data.update(
                    dict(
                        label=s.label,
                        points=[(p.x(), p.y()) for p in s.points],
                        group_id=s.group_id if hasattr(s, 'group_id') else None,
                        description=s.description if hasattr(s, 'description') else "",
                        shape_type=s.shape_type,
                        flags=s.flags if hasattr(s, 'flags') else {},
                        mask=None
                        if not hasattr(s, 'mask') or s.mask is None
                        else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                    )
                )
                return data
            
            shapes = [format_shape(shape) for shape in self.parent_window.canvas.shapes]
            
            if not shapes:
                logger.warning("Export JSON: No shapes to export")
                return None
            
            logger.info(f"Export JSON: Found {len(shapes)} shapes to export")
            
            # Save using LabelFile
            # Match the logic from MainWindow.saveLabels()
            import os.path as osp
            
            # Make imagePath relative to the JSON file location
            imagePath = ""
            if self.parent_window.imagePath:
                try:
                    imagePath = osp.relpath(self.parent_window.imagePath, osp.dirname(str(json_file)))
                except ValueError:
                    # relpath can fail if paths are on different drives (Windows)
                    # Fall back to absolute path or just the filename
                    logger.warning(f"Export JSON: Could not make imagePath relative, using basename")
                    imagePath = osp.basename(self.parent_window.imagePath)
            
            # Respect store_data config (same as saveLabels)
            imageData = None
            if hasattr(self.parent_window, '_config') and self.parent_window._config.get("store_data", True):
                imageData = self.parent_window.imageData
            elif not hasattr(self.parent_window, '_config'):
                # If config doesn't exist, default to storing data
                imageData = self.parent_window.imageData
            
            # Prepare otherData (same as saveLabels)
            otherData = {}
            if self.parent_window.otherData:
                otherData = self.parent_window.otherData.copy()
            
            # Ensure output directory exists
            json_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Export JSON: Saving to {json_file}")
            logger.info(f"Export JSON: imagePath={imagePath}, imageHeight={self.parent_window.image.height()}, imageWidth={self.parent_window.image.width()}")
            
            label_file = LabelFile()
            label_file.save(
                filename=str(json_file),
                shapes=shapes,
                imagePath=imagePath,
                imageHeight=self.parent_window.image.height(),
                imageWidth=self.parent_window.image.width(),
                imageData=imageData,
                otherData=otherData,
                flags={}
            )
            
            # Verify file was created
            if json_file.exists():
                logger.info(f"Export JSON: Successfully exported to {json_file} ({json_file.stat().st_size} bytes)")
                return str(json_file)
            else:
                logger.error(f"Export JSON: File was not created at {json_file}")
                return None
                
        except Exception as e:
            logger.error(f"Export JSON: Exception during export: {e}", exc_info=True)
            raise
    
    def _export_image(self, output_dir: Path) -> str | None:
        """Export image as separate file."""
        if not self.parent_window:
            logger.error("Export Image: No parent window")
            return None
        
        try:
            # Determine output filename based on image or sample code
            metadata = getattr(self.parent_window, 'sample_metadata', None) or {}
            if self.parent_window.filename:
                # Use image filename but change extension to PNG
                image_path = Path(self.parent_window.filename)
                image_file = output_dir / f"{image_path.stem}.png"
            elif metadata.get('sample_code'):
                base_name = metadata['sample_code']
                image_file = output_dir / f"{base_name}.png"
            else:
                base_name = "image"
                image_file = output_dir / f"{base_name}.png"
            
            logger.info(f"Export Image: Target file is {image_file}")
            
            # Get the image from parent window
            # Use imageArray if available (preprocessed), otherwise convert QImage
            if hasattr(self.parent_window, 'imageArray') and self.parent_window.imageArray is not None:
                # Use the numpy array directly (preprocessed image)
                import numpy as np
                from PIL import Image
                img_array = self.parent_window.imageArray
                logger.info(f"Export Image: Using imageArray, shape={img_array.shape}, dtype={img_array.dtype}")
                
                # Ensure it's uint8
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                
                # Convert to PIL Image and save
                pil_image = Image.fromarray(img_array)
                pil_image.save(str(image_file), format='PNG')
            else:
                # Fall back to QImage
                qimage = self.parent_window.image
                if qimage.isNull():
                    logger.error("Export Image: QImage is null")
                    return None
                
                logger.info(f"Export Image: Using QImage, size={qimage.width()}x{qimage.height()}")
                success = qimage.save(str(image_file), "PNG")
                if not success:
                    logger.error(f"Export Image: QImage.save() failed for {image_file}")
                    return None
            
            # Verify file was created
            if image_file.exists():
                logger.info(f"Export Image: Successfully exported to {image_file} ({image_file.stat().st_size} bytes)")
                return str(image_file)
            else:
                logger.error(f"Export Image: File was not created at {image_file}")
                return None
                
        except Exception as e:
            logger.error(f"Export Image: Exception during export: {e}", exc_info=True)
            raise
    
    def _export_csv(self, output_dir: Path) -> str | None:
        """Export measurements as CSV."""
        # Use the same logic as _action_ring_properties
        if not self.parent_window:
            return None
        
        import csv
        from tras.shape import Shape
        
        # Get ring shapes (polygons, not linestrips)
        ring_shapes = [
            s for s in self.parent_window.canvas.shapes 
            if getattr(s, "shape_type", "") == "polygon"
        ]
        
        if not ring_shapes:
            return None
        
        # Calculate properties using the same method as _action_ring_properties
        def _polygon_area(shape: Shape) -> float:
            points = [(p.x(), p.y()) for p in shape.points]
            if len(points) < 3:
                return 0.0
            area = 0.0
            for j in range(len(points)):
                x1, y1 = points[j]
                x2, y2 = points[(j + 1) % len(points)]
                area += x1 * y2 - x2 * y1
            return abs(area) / 2.0
        
        # Sort by area (smallest to largest)
        ring_shapes.sort(key=_polygon_area)
        
        measurements_dict = {}
        if hasattr(self.parent_window, 'radial_line_measurements') and self.parent_window.radial_line_measurements:
            measurements_dict = self.parent_window.radial_line_measurements.get("measurements", {})
        
        ring_properties = []
        prev_outer_area = 0.0
        
        for shape in ring_shapes:
            points = [(p.x(), p.y()) for p in shape.points]
            if len(points) < 3:
                continue
            
            # Calculate area
            area = 0.0
            n = len(points)
            for j in range(n):
                x1, y1 = points[j]
                x2, y2 = points[(j + 1) % n]
                area += x1 * y2 - x2 * y1
            area = abs(area) / 2.0
            
            # Calculate perimeter
            perimeter = 0.0
            for j in range(n):
                x1, y1 = points[j]
                x2, y2 = points[(j + 1) % n]
                perimeter += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            outer_area = area
            ring_area = max(outer_area - prev_outer_area, 0.0)
            
            props = {
                "label": shape.label,
                "area": ring_area,
                "cumulative_area": outer_area,
                "perimeter": perimeter,
            }
            prev_outer_area = outer_area
            
            if shape.label in measurements_dict:
                props["radial_width_px"] = measurements_dict[shape.label]["radial_width"]
            else:
                props["radial_width_px"] = None
            
            ring_properties.append(props)
        
        if not ring_properties:
            return None
        
        # Determine filename
        metadata = getattr(self.parent_window, 'sample_metadata', None) or {}
        if metadata.get('sample_code'):
            base_name = metadata['sample_code']
        else:
            base_name = "measurements"
        csv_file = output_dir / f"{base_name}.csv"
        
        # Get radial measurements if available
        radial_measurements = getattr(self.parent_window, 'radial_line_measurements', None)
        measurements_dict = {}
        if radial_measurements and isinstance(radial_measurements, dict):
            measurements_dict = radial_measurements.get("measurements", {})
        
        # Write CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            has_scale = self.parent_window.image_scale is not None
            scale_value = self.parent_window.image_scale.get('value') if has_scale else None
            unit = self.parent_window.image_scale.get('unit', 'px') if has_scale else 'px'
            
            if has_scale and scale_value is not None:
                writer.writerow([
                    "Ring", f"Area ({unit}²)", f"Perimeter ({unit})",
                    f"Radial Width ({unit})", f"Distance from Pith ({unit})"
                ])
            else:
                writer.writerow([
                    "Ring", "Area (px²)", "Perimeter (px)",
                    "Radial Width (px)", "Distance from Pith (px)"
                ])
            
            # Data rows
            for prop in ring_properties:
                label = prop.get('label', '')
                area = prop.get('area', 0)
                perimeter = prop.get('perimeter', 0)
                
                # Get radial width if available
                radial_width = None
                distance = None
                if measurements_dict and label in measurements_dict:
                    radial_width = measurements_dict[label].get('radial_width')
                    distance = measurements_dict[label].get('distance_from_pith')
                
                if has_scale and scale_value is not None:
                    area_scaled = area * (scale_value ** 2)
                    perimeter_scaled = perimeter * scale_value
                    radial_scaled = radial_width * scale_value if radial_width else None
                    distance_scaled = distance * scale_value if distance else None
                    
                    row = [
                        label,
                        f"{area_scaled:.4f}",
                        f"{perimeter_scaled:.4f}",
                        f"{radial_scaled:.4f}" if radial_scaled is not None else "N/A",
                        f"{distance_scaled:.4f}" if distance_scaled is not None else "—"
                    ]
                else:
                    row = [
                        label,
                        f"{area:.2f}",
                        f"{perimeter:.2f}",
                        f"{radial_width:.2f}" if radial_width is not None else "N/A",
                        f"{distance:.2f}" if distance is not None else "—"
                    ]
                writer.writerow(row)
        
        logger.info(f"Exported CSV measurements to {csv_file}")
        return str(csv_file)
    
    def _export_pos(self, output_dir: Path) -> str | None:
        """Export radial measurements as .POS."""
        from tras.utils.pos_exporter import export_to_pos
        
        radial_measurements = getattr(self.parent_window, 'radial_line_measurements', None)
        if not radial_measurements:
            return None
        
        pith_xy = getattr(self.parent_window, 'pith_xy', None)
        if not pith_xy:
            return None
        
        # Get direction point (needed for POS format)
        # Try to get from radial measurement dialog or use default
        direction_xy = getattr(self.parent_window, '_radial_direction_xy', None)
        if not direction_xy:
            # Use a default direction (e.g., right from pith)
            direction_xy = (pith_xy[0] + 100, pith_xy[1])
        
        # Determine filename
        metadata = getattr(self.parent_window, 'sample_metadata', None) or {}
        if metadata.get('sample_code'):
            base_name = metadata['sample_code']
        else:
            base_name = "measurements"
        pos_file = output_dir / f"{base_name}.pos"
        
        # Get measurements dict
        measurements_dict = radial_measurements.get("measurements", {}) if isinstance(radial_measurements, dict) else {}
        
        # Export
        success = export_to_pos(
            str(pos_file),
            measurements_dict,
            pith_xy,
            direction_xy,
            self.parent_window.image_scale,
            metadata
        )
        
        if success:
            logger.info(f"Exported .POS measurements to {pos_file}")
            return str(pos_file)
        return None
    
    def _export_pdf(self, output_dir: Path) -> str | None:
        """Export PDF report using RingPropertiesDialog logic."""
        # Reuse the parent window's method to prepare data (no duplication)
        data = self.parent_window._prepare_ring_properties_data()
        if data is None:
            return None
        
        ring_properties, radial_measurements, metadata_dict = data
        
        # Determine filename
        if metadata_dict.get('sample_code'):
            base_name = metadata_dict['sample_code']
        else:
            base_name = "report"
        pdf_file = output_dir / f"{base_name}_report.pdf"
        
        # Create RingPropertiesDialog with the data (same as _action_ring_properties)
        from tras.widgets.ring_properties_dialog import RingPropertiesDialog
        
        dialog = RingPropertiesDialog(
            ring_properties,
            radial_measurements,
            parent=self.parent_window,
            metadata=metadata_dict if metadata_dict else None,
        )
        
        # Use RingPropertiesDialog's PDF generation methods directly
        # This is the same code that runs when clicking "Generate PDF Report" in the dialog
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.backends.backend_pdf import PdfPages
            from datetime import datetime
            
            # Use the exact same PDF generation logic as RingPropertiesDialog._export_pdf
            with PdfPages(str(pdf_file)) as pdf:
                dialog._create_cover_page(pdf)
                if dialog.has_polygon_data or dialog.has_radial_data:
                    dialog._create_ring_overlay_page(pdf)
                    dialog._create_ring_overlay_page(pdf, show_labels=False)
                if dialog.has_polygon_data:
                    dialog._create_analysis_plots(pdf)
                if dialog.has_radial_data:
                    dialog._create_radial_plot_page(pdf)
                
                pdf_metadata = pdf.infodict()
                pdf_metadata["Title"] = "Tree Ring Analysis Report"
                pdf_metadata["Author"] = "TRAS - Tree Ring Analyzer Suite"
                pdf_metadata["Subject"] = f"Sample: {metadata_dict.get('sample_code', 'Unknown')}"
                pdf_metadata["Keywords"] = "Dendrochronology, Tree Rings, Wood Analysis"
                pdf_metadata["CreationDate"] = datetime.now()
            
            logger.info(f"Exported PDF report to {pdf_file}")
            return str(pdf_file)
        except Exception as e:
            logger.error(f"Failed to export PDF: {e}", exc_info=True)
            raise

