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
        
        # Table widget
        self.table = QtWidgets.QTableWidget()
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
            # Ring name
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(props['label']))
            
            # Area
            area_item = QtWidgets.QTableWidgetItem(f"{props['area']:.2f}")
            area_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 1, area_item)
            
            # Cumulative area
            cum_area_item = QtWidgets.QTableWidgetItem(f"{props['cumulative_area']:.2f}")
            cum_area_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 2, cum_area_item)
            
            # Perimeter
            perim_item = QtWidgets.QTableWidgetItem(f"{props['perimeter']:.2f}")
            perim_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 3, perim_item)
            
            # Ring width
            if props['ring_width'] is not None:
                width_item = QtWidgets.QTableWidgetItem(f"{props['ring_width']:.2f}")
            else:
                width_item = QtWidgets.QTableWidgetItem("N/A")
            width_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 4, width_item)
        
        # Auto-resize columns to content
        self.table.resizeColumnsToContents()
        
        layout.addWidget(self.table)
        
        # Summary statistics
        summary_group = QtWidgets.QGroupBox(self.tr("Summary"))
        summary_layout = QtWidgets.QFormLayout()
        
        total_area = sum(p['area'] for p in ring_properties)
        avg_area = total_area / len(ring_properties) if ring_properties else 0
        avg_perimeter = sum(p['perimeter'] for p in ring_properties) / len(ring_properties) if ring_properties else 0
        
        ring_widths = [p['ring_width'] for p in ring_properties if p['ring_width'] is not None]
        avg_width = sum(ring_widths) / len(ring_widths) if ring_widths else 0
        
        summary_layout.addRow(self.tr("Total Rings:"), QtWidgets.QLabel(str(len(ring_properties))))
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
                
                # Write metadata section if available
                if self.metadata:
                    writer.writerow(['# Metadata'])
                    if 'harvested_year' in self.metadata:
                        writer.writerow(['# Harvested Year', self.metadata['harvested_year']])
                    if 'sample_code' in self.metadata:
                        writer.writerow(['# Sample Code', self.metadata['sample_code']])
                    if 'observation' in self.metadata:
                        writer.writerow(['# Observation', self.metadata['observation']])
                    writer.writerow([])  # Empty line separator
                
                # Header
                writer.writerow([
                    'Ring',
                    'Area (px²)',
                    'Cumulative Area (px²)',
                    'Perimeter (px)',
                    'Ring Width (px)'
                ])
                # Data
                for props in self.ring_properties:
                    writer.writerow([
                        props['label'],
                        f"{props['area']:.2f}",
                        f"{props['cumulative_area']:.2f}",
                        f"{props['perimeter']:.2f}",
                        f"{props['ring_width']:.2f}" if props['ring_width'] is not None else 'N/A'
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

