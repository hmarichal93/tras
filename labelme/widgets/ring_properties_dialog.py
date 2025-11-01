"""
Ring Properties Dialog - Display and export tree ring measurements
"""
import csv
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt


class RingPropertiesDialog(QtWidgets.QDialog):
    """Dialog to display ring properties (area, perimeter, etc.)"""
    
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
        
        # Check if physical measurements are available
        has_scale = ring_properties[0].get('scale') is not None if ring_properties else False
        unit = ring_properties[0].get('unit', 'mm') if has_scale else None
        
        # Table widget
        self.table = QtWidgets.QTableWidget()
        
        # Check if radial width measurements are available
        has_radial = any(p.get('radial_width_px') is not None for p in ring_properties)
        
        # Set columns based on whether we have scale and radial measurements
        if has_scale and has_radial:
            self.table.setColumnCount(11)
            self.table.setHorizontalHeaderLabels([
                self.tr("Ring"),
                self.tr(f"Area ({unit}²)"),
                self.tr(f"Cumul. Area ({unit}²)"),
                self.tr(f"Perimeter ({unit})"),
                self.tr(f"Width ({unit})"),
                self.tr(f"Radial Width ({unit})"),
                self.tr("Area (px²)"),
                self.tr("Cumul. Area (px²)"),
                self.tr("Perimeter (px)"),
                self.tr("Width (px)"),
                self.tr("Radial Width (px)")
            ])
        elif has_scale:
            self.table.setColumnCount(9)
            self.table.setHorizontalHeaderLabels([
                self.tr("Ring"),
                self.tr(f"Area ({unit}²)"),
                self.tr(f"Cumul. Area ({unit}²)"),
                self.tr(f"Perimeter ({unit})"),
                self.tr(f"Width ({unit})"),
                self.tr("Area (px²)"),
                self.tr("Cumul. Area (px²)"),
                self.tr("Perimeter (px)"),
                self.tr("Width (px)")
            ])
        elif has_radial:
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels([
                self.tr("Ring"),
                self.tr("Area (px²)"),
                self.tr("Cumulative Area (px²)"),
                self.tr("Perimeter (px)"),
                self.tr("Ring Width (px)"),
                self.tr("Radial Width (px)")
            ])
        else:
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels([
                self.tr("Ring"),
                self.tr("Area (px²)"),
                self.tr("Cumulative Area (px²)"),
                self.tr("Perimeter (px)"),
                self.tr("Ring Width (px)")
            ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        
        # Populate table
        self.table.setRowCount(len(ring_properties))
        for i, props in enumerate(ring_properties):
            col = 0
            
            # Ring name
            self.table.setItem(i, col, QtWidgets.QTableWidgetItem(props['label']))
            col += 1
            
            if has_scale:
                # Physical measurements first
                # Area (physical)
                area_item = QtWidgets.QTableWidgetItem(f"{props['area_physical']:.4f}")
                area_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, col, area_item)
                col += 1
                
                # Cumulative area (physical)
                cum_area_item = QtWidgets.QTableWidgetItem(f"{props['cumulative_area_physical']:.4f}")
                cum_area_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, col, cum_area_item)
                col += 1
                
                # Perimeter (physical)
                perim_item = QtWidgets.QTableWidgetItem(f"{props['perimeter_physical']:.4f}")
                perim_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, col, perim_item)
                col += 1
                
                # Ring width (physical)
                if props['ring_width_physical'] is not None:
                    width_item = QtWidgets.QTableWidgetItem(f"{props['ring_width_physical']:.4f}")
                else:
                    width_item = QtWidgets.QTableWidgetItem("N/A")
                width_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, col, width_item)
                col += 1
                
                # Radial width (physical) if available
                if has_radial:
                    if props.get('radial_width_physical') is not None:
                        radial_width_item = QtWidgets.QTableWidgetItem(f"{props['radial_width_physical']:.4f}")
                    else:
                        radial_width_item = QtWidgets.QTableWidgetItem("N/A")
                    radial_width_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.table.setItem(i, col, radial_width_item)
                    col += 1
            
            # Pixel measurements
            # Area (pixels)
            area_px_item = QtWidgets.QTableWidgetItem(f"{props['area_px']:.2f}")
            area_px_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, col, area_px_item)
            col += 1
            
            # Cumulative area (pixels)
            cum_area_px_item = QtWidgets.QTableWidgetItem(f"{props['cumulative_area_px']:.2f}")
            cum_area_px_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, col, cum_area_px_item)
            col += 1
            
            # Perimeter (pixels)
            perim_px_item = QtWidgets.QTableWidgetItem(f"{props['perimeter_px']:.2f}")
            perim_px_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, col, perim_px_item)
            col += 1
            
            # Ring width (pixels)
            if props['ring_width_px'] is not None:
                width_px_item = QtWidgets.QTableWidgetItem(f"{props['ring_width_px']:.2f}")
            else:
                width_px_item = QtWidgets.QTableWidgetItem("N/A")
            width_px_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, col, width_px_item)
            col += 1
            
            # Radial width (pixels) if available
            if has_radial:
                if props.get('radial_width_px') is not None:
                    radial_width_px_item = QtWidgets.QTableWidgetItem(f"{props['radial_width_px']:.2f}")
                else:
                    radial_width_px_item = QtWidgets.QTableWidgetItem("N/A")
                radial_width_px_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, col, radial_width_px_item)
        
        # Auto-resize columns to content
        self.table.resizeColumnsToContents()
        
        layout.addWidget(self.table)
        
        # Summary statistics
        summary_group = QtWidgets.QGroupBox(self.tr("Summary"))
        summary_layout = QtWidgets.QFormLayout()
        
        summary_layout.addRow(self.tr("Total Rings:"), QtWidgets.QLabel(str(len(ring_properties))))
        
        if has_scale:
            # Physical measurements summary
            total_area_phys = sum(p['area_physical'] for p in ring_properties)
            avg_area_phys = total_area_phys / len(ring_properties) if ring_properties else 0
            avg_perim_phys = sum(p['perimeter_physical'] for p in ring_properties) / len(ring_properties) if ring_properties else 0
            
            ring_widths_phys = [p['ring_width_physical'] for p in ring_properties if p['ring_width_physical'] is not None]
            avg_width_phys = sum(ring_widths_phys) / len(ring_widths_phys) if ring_widths_phys else 0
            
            summary_layout.addRow(self.tr(f"Total Area ({unit}²):"), QtWidgets.QLabel(f"{total_area_phys:.4f}"))
            summary_layout.addRow(self.tr(f"Avg Area ({unit}²):"), QtWidgets.QLabel(f"{avg_area_phys:.4f}"))
            summary_layout.addRow(self.tr(f"Avg Perimeter ({unit}):"), QtWidgets.QLabel(f"{avg_perim_phys:.4f}"))
            summary_layout.addRow(self.tr(f"Avg Width ({unit}):"), QtWidgets.QLabel(f"{avg_width_phys:.4f}"))
            
            # Add scale info
            scale_val = ring_properties[0]['scale']
            summary_layout.addRow(self.tr("Scale:"), QtWidgets.QLabel(f"{scale_val:.6f} {unit}/pixel"))
        else:
            # Pixel measurements only
            total_area = sum(p['area_px'] for p in ring_properties)
            avg_area = total_area / len(ring_properties) if ring_properties else 0
            avg_perimeter = sum(p['perimeter_px'] for p in ring_properties) / len(ring_properties) if ring_properties else 0
            
            ring_widths = [p['ring_width_px'] for p in ring_properties if p['ring_width_px'] is not None]
            avg_width = sum(ring_widths) / len(ring_widths) if ring_widths else 0
            
            summary_layout.addRow(self.tr("Total Area:"), QtWidgets.QLabel(f"{total_area:.2f} px²"))
            summary_layout.addRow(self.tr("Average Area:"), QtWidgets.QLabel(f"{avg_area:.2f} px²"))
            summary_layout.addRow(self.tr("Average Perimeter:"), QtWidgets.QLabel(f"{avg_perimeter:.2f} px"))
            summary_layout.addRow(self.tr("Average Ring Width:"), QtWidgets.QLabel(f"{avg_width:.2f} px"))
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        export_btn = QtWidgets.QPushButton(self.tr("Export to CSV"))
        export_btn.clicked.connect(self._export_csv)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        close_btn = QtWidgets.QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _export_csv(self):
        """Export ring properties to CSV file"""
        # Use sample code as default filename if available
        default_filename = "ring_properties.csv"
        if self.metadata.get('sample_code'):
            # Sanitize filename
            sample_code = self.metadata['sample_code']
            safe_code = "".join(c for c in sample_code if c.isalnum() or c in ('-', '_'))
            default_filename = f"{safe_code}.csv"
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export Ring Properties"),
            default_filename,
            self.tr("CSV Files (*.csv)")
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Check if we have physical measurements
                has_scale = self.ring_properties[0].get('scale') is not None if self.ring_properties else False
                unit = self.ring_properties[0].get('unit', 'mm') if has_scale else None
                
                # Write metadata section if available
                if self.metadata:
                    writer.writerow(['# Metadata'])
                    if 'harvested_year' in self.metadata:
                        writer.writerow(['# Harvested Year', self.metadata['harvested_year']])
                    if 'sample_code' in self.metadata:
                        writer.writerow(['# Sample Code', self.metadata['sample_code']])
                    if 'observation' in self.metadata:
                        writer.writerow(['# Observation', self.metadata['observation']])
                
                # Write scale info if available
                if has_scale:
                    writer.writerow(['# Scale', f"{self.ring_properties[0]['scale']:.6f} {unit}/pixel"])
                
                writer.writerow([])  # Empty line separator
                
                # Check if radial measurements are available
                has_radial = any(p.get('radial_width_px') is not None for p in self.ring_properties)
                
                # Header and data based on whether we have scale and radial measurements
                if has_scale and has_radial:
                    writer.writerow([
                        'Ring',
                        f'Area ({unit}²)', f'Cumulative Area ({unit}²)', f'Perimeter ({unit})', f'Width ({unit})', f'Radial Width ({unit})',
                        'Area (px²)', 'Cumulative Area (px²)', 'Perimeter (px)', 'Width (px)', 'Radial Width (px)'
                    ])
                    for props in self.ring_properties:
                        writer.writerow([
                            props['label'],
                            f"{props['area_physical']:.4f}",
                            f"{props['cumulative_area_physical']:.4f}",
                            f"{props['perimeter_physical']:.4f}",
                            f"{props['ring_width_physical']:.4f}" if props['ring_width_physical'] is not None else 'N/A',
                            f"{props['radial_width_physical']:.4f}" if props.get('radial_width_physical') is not None else 'N/A',
                            f"{props['area_px']:.2f}",
                            f"{props['cumulative_area_px']:.2f}",
                            f"{props['perimeter_px']:.2f}",
                            f"{props['ring_width_px']:.2f}" if props['ring_width_px'] is not None else 'N/A',
                            f"{props['radial_width_px']:.2f}" if props.get('radial_width_px') is not None else 'N/A'
                        ])
                elif has_scale:
                    writer.writerow([
                        'Ring',
                        f'Area ({unit}²)', f'Cumulative Area ({unit}²)', f'Perimeter ({unit})', f'Width ({unit})',
                        'Area (px²)', 'Cumulative Area (px²)', 'Perimeter (px)', 'Width (px)'
                    ])
                    for props in self.ring_properties:
                        writer.writerow([
                            props['label'],
                            f"{props['area_physical']:.4f}",
                            f"{props['cumulative_area_physical']:.4f}",
                            f"{props['perimeter_physical']:.4f}",
                            f"{props['ring_width_physical']:.4f}" if props['ring_width_physical'] is not None else 'N/A',
                            f"{props['area_px']:.2f}",
                            f"{props['cumulative_area_px']:.2f}",
                            f"{props['perimeter_px']:.2f}",
                            f"{props['ring_width_px']:.2f}" if props['ring_width_px'] is not None else 'N/A'
                        ])
                elif has_radial:
                    writer.writerow([
                        'Ring',
                        'Area (px²)',
                        'Cumulative Area (px²)',
                        'Perimeter (px)',
                        'Ring Width (px)',
                        'Radial Width (px)'
                    ])
                    for props in self.ring_properties:
                        writer.writerow([
                            props['label'],
                            f"{props['area_px']:.2f}",
                            f"{props['cumulative_area_px']:.2f}",
                            f"{props['perimeter_px']:.2f}",
                            f"{props['ring_width_px']:.2f}" if props['ring_width_px'] is not None else 'N/A',
                            f"{props['radial_width_px']:.2f}" if props.get('radial_width_px') is not None else 'N/A'
                        ])
                else:
                    writer.writerow([
                        'Ring',
                        'Area (px²)',
                        'Cumulative Area (px²)',
                        'Perimeter (px)',
                        'Ring Width (px)'
                    ])
                    for props in self.ring_properties:
                        writer.writerow([
                            props['label'],
                            f"{props['area_px']:.2f}",
                            f"{props['cumulative_area_px']:.2f}",
                            f"{props['perimeter_px']:.2f}",
                            f"{props['ring_width_px']:.2f}" if props['ring_width_px'] is not None else 'N/A'
                        ])
            
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Export Successful"),
                self.tr(f"Ring properties exported to:\n{filename}")
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Export Failed"),
                self.tr(f"Failed to export CSV:\n{str(e)}")
            )

