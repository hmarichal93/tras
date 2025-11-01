"""
Ring Properties Dialog - Display and export tree ring measurements
Only supports radial width (transect-based) - no centroid width
"""
import csv
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt


class RingPropertiesDialog(QtWidgets.QDialog):
    """Dialog to display ring properties (area, perimeter, radial width)"""
    
    def __init__(self, ring_properties, parent=None, metadata=None):
        super().__init__(parent)
        self.ring_properties = ring_properties
        self.metadata = metadata or {}
        self.setWindowTitle(self.tr("Tree Ring Properties"))
        self.setModal(True)
        self.resize(700, 500)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Info label
        info_label = QtWidgets.QLabel(
            self.tr(f"Analyzed {len(ring_properties)} rings")
        )
        info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(info_label)
        
        # Check if scale information is available
        has_scale = self.metadata and 'scale' in self.metadata
        
        # Check if radial width measurements are available
        has_radial = any(p.get('radial_width_px') is not None for p in ring_properties)
        
        # Table widget
        self.table = QtWidgets.QTableWidget()
        
        # Set columns based on what's available
        if has_scale:
            unit = self.metadata['scale']['unit']
            if has_radial:
                # Scale + radial width
                self.table.setColumnCount(8)
                self.table.setHorizontalHeaderLabels([
                    self.tr("Ring"),
                    self.tr(f"Area ({unit}²)"),
                    self.tr(f"Cumul. Area ({unit}²)"),
                    self.tr(f"Perimeter ({unit})"),
                    self.tr(f"Radial Width ({unit})"),
                    self.tr("Area (px²)"),
                    self.tr("Cumul. Area (px²)"),
                    self.tr("Perimeter (px)")
                ])
            else:
                # Scale only
                self.table.setColumnCount(7)
                self.table.setHorizontalHeaderLabels([
                    self.tr("Ring"),
                    self.tr(f"Area ({unit}²)"),
                    self.tr(f"Cumul. Area ({unit}²)"),
                    self.tr(f"Perimeter ({unit})"),
                    self.tr("Area (px²)"),
                    self.tr("Cumul. Area (px²)"),
                    self.tr("Perimeter (px)")
                ])
        else:
            if has_radial:
                # Pixels + radial width
                self.table.setColumnCount(5)
                self.table.setHorizontalHeaderLabels([
                    self.tr("Ring"),
                    self.tr("Area (px²)"),
                    self.tr("Cumul. Area (px²)"),
                    self.tr("Perimeter (px)"),
                    self.tr("Radial Width (px)")
                ])
            else:
                # Pixels only
                self.table.setColumnCount(4)
                self.table.setHorizontalHeaderLabels([
                    self.tr("Ring"),
                    self.tr("Area (px²)"),
                    self.tr("Cumul. Area (px²)"),
                    self.tr("Perimeter (px)")
                ])
        
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        
        # Populate table
        self.table.setRowCount(len(ring_properties))
        for row, props in enumerate(ring_properties):
            col = 0
            
            # Ring label
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(props['label']))
            col += 1
            
            if has_scale:
                scale_value = self.metadata['scale']['value']
                
                # Physical measurements
                area_physical = props['area'] * (scale_value ** 2)
                cumul_physical = props['cumulative_area'] * (scale_value ** 2)
                perim_physical = props['perimeter'] * scale_value
                
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{area_physical:.4f}"))
                col += 1
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{cumul_physical:.4f}"))
                col += 1
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{perim_physical:.4f}"))
                col += 1
                
                # Radial width (physical) if available
                if has_radial:
                    radial_px = props.get('radial_width_px')
                    if radial_px is not None:
                        radial_physical = radial_px * scale_value
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{radial_physical:.4f}"))
                    else:
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem("N/A"))
                    col += 1
            
            # Pixel measurements
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{props['area']:.2f}"))
            col += 1
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{props['cumulative_area']:.2f}"))
            col += 1
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{props['perimeter']:.2f}"))
            col += 1
            
            # Radial width (pixels) if available and no scale
            if has_radial and not has_scale:
                radial_px = props.get('radial_width_px')
                if radial_px is not None:
                    self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{radial_px:.2f}"))
                else:
                    self.table.setItem(row, col, QtWidgets.QTableWidgetItem("N/A"))
        
        layout.addWidget(self.table)
        
        # Summary statistics
        total_area = sum(p['area'] for p in ring_properties)
        avg_area = total_area / len(ring_properties)
        total_perim = sum(p['perimeter'] for p in ring_properties)
        avg_perim = total_perim / len(ring_properties)
        
        if has_scale:
            scale_value = self.metadata['scale']['value']
            unit = self.metadata['scale']['unit']
            total_area_physical = total_area * (scale_value ** 2)
            avg_area_physical = avg_area * (scale_value ** 2)
            total_perim_physical = total_perim * scale_value
            avg_perim_physical = avg_perim * scale_value
            
            summary_text = (
                f"<b>Summary Statistics:</b><br>"
                f"Total Rings: {len(ring_properties)}<br>"
                f"Total Area: {total_area_physical:.4f} {unit}² ({total_area:.2f} px²)<br>"
                f"Average Area: {avg_area_physical:.4f} {unit}² ({avg_area:.2f} px²)<br>"
                f"Total Perimeter: {total_perim_physical:.4f} {unit} ({total_perim:.2f} px)<br>"
                f"Average Perimeter: {avg_perim_physical:.4f} {unit} ({avg_perim:.2f} px)"
            )
        else:
            summary_text = (
                f"<b>Summary Statistics:</b><br>"
                f"Total Rings: {len(ring_properties)}<br>"
                f"Total Area: {total_area:.2f} px²<br>"
                f"Average Area: {avg_area:.2f} px²<br>"
                f"Total Perimeter: {total_perim:.2f} px<br>"
                f"Average Perimeter: {avg_perim:.2f} px"
            )
        
        summary_label = QtWidgets.QLabel(summary_text)
        summary_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(summary_label)
        
        # Export button
        export_btn = QtWidgets.QPushButton(self.tr("Export to CSV"))
        export_btn.clicked.connect(self._export_csv)
        layout.addWidget(export_btn)
        
        # Close button
        close_btn = QtWidgets.QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def _export_csv(self):
        """Export ring properties to CSV file"""
        # Get default filename from sample code
        default_filename = "ring_properties.csv"
        if self.metadata.get('sample_code'):
            sample_code = self.metadata['sample_code']
            safe_code = "".join(c for c in sample_code if c.isalnum() or c in ('-', '_'))
            default_filename = f"{safe_code}.csv"
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.tr("Export Ring Properties"), default_filename, 
            self.tr("CSV Files (*.csv)")
        )
        
        if not filename:
            return
        
        try:
            has_scale = self.metadata and 'scale' in self.metadata
            has_radial = any(p.get('radial_width_px') is not None for p in self.ring_properties)
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write metadata header
                if self.metadata:
                    writer.writerow(['# Metadata'])
                    if 'harvested_year' in self.metadata:
                        writer.writerow(['# Harvested Year', self.metadata['harvested_year']])
                    if 'sample_code' in self.metadata:
                        writer.writerow(['# Sample Code', self.metadata['sample_code']])
                    if 'observation' in self.metadata:
                        writer.writerow(['# Observation', self.metadata['observation']])
                    writer.writerow([])
                
                # Write scale information
                if has_scale:
                    scale_value = self.metadata['scale']['value']
                    unit = self.metadata['scale']['unit']
                    writer.writerow(['# Scale', f'{scale_value:.6f} {unit}/pixel'])
                    writer.writerow([])
                
                # Write column headers
                if has_scale:
                    unit = self.metadata['scale']['unit']
                    if has_radial:
                        writer.writerow(['Ring', f'Area ({unit}²)', f'Cumulative Area ({unit}²)', 
                                       f'Perimeter ({unit})', f'Radial Width ({unit})',
                                       'Area (px²)', 'Cumulative Area (px²)', 'Perimeter (px)'])
                    else:
                        writer.writerow(['Ring', f'Area ({unit}²)', f'Cumulative Area ({unit}²)', 
                                       f'Perimeter ({unit})', 'Area (px²)', 'Cumulative Area (px²)', 
                                       'Perimeter (px)'])
                else:
                    if has_radial:
                        writer.writerow(['Ring', 'Area (px²)', 'Cumulative Area (px²)', 
                                       'Perimeter (px)', 'Radial Width (px)'])
                    else:
                        writer.writerow(['Ring', 'Area (px²)', 'Cumulative Area (px²)', 
                                       'Perimeter (px)'])
                
                # Write data rows
                for props in self.ring_properties:
                    if has_scale:
                        scale_value = self.metadata['scale']['value']
                        area_physical = props['area'] * (scale_value ** 2)
                        cumul_physical = props['cumulative_area'] * (scale_value ** 2)
                        perim_physical = props['perimeter'] * scale_value
                        
                        if has_radial:
                            radial_px = props.get('radial_width_px')
                            radial_physical = radial_px * scale_value if radial_px is not None else None
                            writer.writerow([
                                props['label'],
                                f"{area_physical:.4f}", f"{cumul_physical:.4f}",
                                f"{perim_physical:.4f}",
                                f"{radial_physical:.4f}" if radial_physical is not None else 'N/A',
                                f"{props['area']:.2f}", f"{props['cumulative_area']:.2f}",
                                f"{props['perimeter']:.2f}"
                            ])
                        else:
                            writer.writerow([
                                props['label'],
                                f"{area_physical:.4f}", f"{cumul_physical:.4f}",
                                f"{perim_physical:.4f}",
                                f"{props['area']:.2f}", f"{props['cumulative_area']:.2f}",
                                f"{props['perimeter']:.2f}"
                            ])
                    else:
                        if has_radial:
                            radial_px = props.get('radial_width_px')
                            writer.writerow([
                                props['label'],
                                f"{props['area']:.2f}",
                                f"{props['cumulative_area']:.2f}",
                                f"{props['perimeter']:.2f}",
                                f"{radial_px:.2f}" if radial_px is not None else 'N/A'
                            ])
                        else:
                            writer.writerow([
                                props['label'],
                                f"{props['area']:.2f}",
                                f"{props['cumulative_area']:.2f}",
                                f"{props['perimeter']:.2f}"
                            ])
            
            QtWidgets.QMessageBox.information(
                self, self.tr("Export Successful"),
                self.tr(f"Ring properties exported to:\n{filename}")
            )
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, self.tr("Export Failed"),
                self.tr(f"Failed to export CSV:\n{str(e)}")
            )
