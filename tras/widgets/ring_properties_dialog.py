"""
Ring Properties Dialog - Display and export tree ring measurements
Only supports radial width (transect-based) - no centroid width
"""
import csv
from pathlib import Path
from datetime import datetime

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
import numpy as np


class RingPropertiesDialog(QtWidgets.QDialog):
    """Dialog to display ring properties (area, perimeter, radial width)"""
    
    def __init__(self, ring_properties, parent=None, metadata=None):
        super().__init__(parent)
        self.ring_properties = ring_properties
        self.metadata = metadata or {}
        self.parent_window = parent  # Store reference to main window
        self.setWindowTitle(self.tr("Tree Ring Properties"))
        self.setModal(True)
        self.resize(700, 500)
        
        # Compute correct cumulative areas from pith outward
        # Rings are ordered outermost to innermost, so cumsum in reverse
        areas = [p['area'] for p in ring_properties]
        cumulative_areas = np.cumsum(areas[::-1])[::-1].tolist()
        self.cumulative_areas = cumulative_areas
        
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
            
            # Get correct cumulative area for this row
            cumul_area_px = self.cumulative_areas[row]
            
            # Ring label
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(props['label']))
            col += 1
            
            if has_scale:
                scale_value = self.metadata['scale']['value']
                
                # Physical measurements
                area_physical = props['area'] * (scale_value ** 2)
                cumul_physical = cumul_area_px * (scale_value ** 2)
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
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{cumul_area_px:.2f}"))
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
        
        # Export buttons row
        export_layout = QtWidgets.QHBoxLayout()
        
        export_csv_btn = QtWidgets.QPushButton(self.tr("📊 Export to CSV"))
        export_csv_btn.clicked.connect(self._export_csv)
        export_layout.addWidget(export_csv_btn)
        
        export_pdf_btn = QtWidgets.QPushButton(self.tr("📄 Generate PDF Report"))
        export_pdf_btn.clicked.connect(self._export_pdf)
        export_layout.addWidget(export_pdf_btn)
        
        layout.addLayout(export_layout)
        
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
                for idx, props in enumerate(self.ring_properties):
                    # Get correct cumulative area for this row
                    cumul_area_px = self.cumulative_areas[idx]
                    
                    if has_scale:
                        scale_value = self.metadata['scale']['value']
                        area_physical = props['area'] * (scale_value ** 2)
                        cumul_physical = cumul_area_px * (scale_value ** 2)
                        perim_physical = props['perimeter'] * scale_value
                        
                        if has_radial:
                            radial_px = props.get('radial_width_px')
                            radial_physical = radial_px * scale_value if radial_px is not None else None
                            writer.writerow([
                                props['label'],
                                f"{area_physical:.4f}", f"{cumul_physical:.4f}",
                                f"{perim_physical:.4f}",
                                f"{radial_physical:.4f}" if radial_physical is not None else 'N/A',
                                f"{props['area']:.2f}", f"{cumul_area_px:.2f}",
                                f"{props['perimeter']:.2f}"
                            ])
                        else:
                            writer.writerow([
                                props['label'],
                                f"{area_physical:.4f}", f"{cumul_physical:.4f}",
                                f"{perim_physical:.4f}",
                                f"{props['area']:.2f}", f"{cumul_area_px:.2f}",
                                f"{props['perimeter']:.2f}"
                            ])
                    else:
                        if has_radial:
                            radial_px = props.get('radial_width_px')
                            writer.writerow([
                                props['label'],
                                f"{props['area']:.2f}",
                                f"{cumul_area_px:.2f}",
                                f"{props['perimeter']:.2f}",
                                f"{radial_px:.2f}" if radial_px is not None else 'N/A'
                            ])
                        else:
                            writer.writerow([
                                props['label'],
                                f"{props['area']:.2f}",
                                f"{cumul_area_px:.2f}",
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
    
    def _export_pdf(self):
        """Generate PDF report with ring overlays and analysis plots"""
        # Get default filename from sample code
        default_filename = "tree_ring_report.pdf"
        if self.metadata.get('sample_code'):
            sample_code = self.metadata['sample_code']
            safe_code = "".join(c for c in sample_code if c.isalnum() or c in ('-', '_'))
            default_filename = f"{safe_code}_report.pdf"
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.tr("Save PDF Report"), default_filename, 
            self.tr("PDF Files (*.pdf)")
        )
        
        if not filename:
            return
        
        try:
            # Set wait cursor
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # Import matplotlib here (lazy import)
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.patches as patches
            from matplotlib.gridspec import GridSpec
            
            # Create PDF
            with PdfPages(filename) as pdf:
                # Page 1: Cover page with metadata and summary
                self._create_cover_page(pdf)
                
                # Page 2: Image with ring overlays
                self._create_ring_overlay_page(pdf)
                
                # Page 3: Analysis plots
                self._create_analysis_plots(pdf)
                
                # Add metadata to PDF
                d = pdf.infodict()
                d['Title'] = 'Tree Ring Analysis Report'
                d['Author'] = 'TRAS - Tree Ring Analyzer Suite'
                d['Subject'] = f"Sample: {self.metadata.get('sample_code', 'Unknown')}"
                d['Keywords'] = 'Dendrochronology, Tree Rings, Wood Analysis'
                d['CreationDate'] = datetime.now()
            
            QApplication.restoreOverrideCursor()
            
            QtWidgets.QMessageBox.information(
                self, self.tr("PDF Generated"),
                self.tr(f"PDF report successfully generated:\n{filename}")
            )
        
        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
            error_details = traceback.format_exc()
            print(f"PDF Export Error: {error_details}")
            QtWidgets.QMessageBox.critical(
                self, self.tr("PDF Generation Error"),
                self.tr(f"Failed to generate PDF:\n{str(e)}\n\nMake sure matplotlib is installed.")
            )
    
    def _create_cover_page(self, pdf):
        """Create cover page with metadata and summary statistics"""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        y_pos = 0.95
        ax.text(0.5, y_pos, 'Tree Ring Analysis Report', 
                ha='center', va='top', fontsize=24, fontweight='bold', 
                color='#2d5016')
        y_pos -= 0.08
        
        ax.text(0.5, y_pos, 'TRAS - Tree Ring Analyzer Suite', 
                ha='center', va='top', fontsize=12, color='#666666')
        y_pos -= 0.05
        
        # Horizontal line
        ax.plot([0.1, 0.9], [y_pos, y_pos], 'k-', linewidth=2, color='#8b4513')
        y_pos -= 0.08
        
        # Metadata section
        if self.metadata:
            ax.text(0.1, y_pos, 'Sample Information', 
                    fontsize=16, fontweight='bold', color='#2d5016')
            y_pos -= 0.05
            
            if 'sample_code' in self.metadata:
                ax.text(0.15, y_pos, f"Sample Code: {self.metadata['sample_code']}", 
                        fontsize=12)
                y_pos -= 0.04
            
            if 'harvested_year' in self.metadata:
                ax.text(0.15, y_pos, f"Harvested Year: {self.metadata['harvested_year']}", 
                        fontsize=12)
                y_pos -= 0.04
            
            if 'observation' in self.metadata:
                obs_text = self.metadata['observation']
                # Wrap long observations using textwrap for better formatting
                import textwrap
                if len(obs_text) > 60:
                    # Limit to first 300 characters and wrap
                    obs_truncated = obs_text[:300] + ('...' if len(obs_text) > 300 else '')
                    wrapped_lines = textwrap.wrap(obs_truncated, width=70)
                    ax.text(0.15, y_pos, f"Observations:", fontsize=12, fontweight='bold')
                    y_pos -= 0.035
                    # Limit to max 4 lines to prevent overflow
                    for line in wrapped_lines[:4]:
                        ax.text(0.18, y_pos, line, fontsize=10, style='italic')
                        y_pos -= 0.025
                    y_pos -= 0.01  # Extra spacing after observations
                else:
                    ax.text(0.15, y_pos, f"Observations: {obs_text}", 
                            fontsize=11, style='italic')
                    y_pos -= 0.04
            
            if 'scale' in self.metadata:
                scale_value = self.metadata['scale']['value']
                unit = self.metadata['scale']['unit']
                ax.text(0.15, y_pos, f"Scale: {scale_value:.6f} {unit}/pixel", 
                        fontsize=12)
                y_pos -= 0.04
            
            y_pos -= 0.03
        
        # Summary statistics
        ax.text(0.1, y_pos, 'Summary Statistics', 
                fontsize=16, fontweight='bold', color='#2d5016')
        y_pos -= 0.05
        
        total_rings = len(self.ring_properties)
        total_area = sum(p['area'] for p in self.ring_properties)
        avg_area = total_area / total_rings if total_rings > 0 else 0
        total_perim = sum(p['perimeter'] for p in self.ring_properties)
        avg_perim = total_perim / total_rings if total_rings > 0 else 0
        
        has_scale = self.metadata and 'scale' in self.metadata
        has_radial = any(p.get('radial_width_px') is not None for p in self.ring_properties)
        
        if has_scale:
            unit = self.metadata['scale']['unit']
            scale_factor = self.metadata['scale']['value']
            scale_factor_sq = scale_factor ** 2
            
            ax.text(0.15, y_pos, f"Total Rings Detected: {total_rings}", fontsize=12)
            y_pos -= 0.04
            ax.text(0.15, y_pos, f"Total Area: {total_area*scale_factor_sq:.2f} {unit}² ({total_area:.2f} px²)", 
                    fontsize=12)
            y_pos -= 0.04
            ax.text(0.15, y_pos, f"Average Ring Area: {avg_area*scale_factor_sq:.2f} {unit}² ({avg_area:.2f} px²)", 
                    fontsize=12)
            y_pos -= 0.04
            ax.text(0.15, y_pos, f"Total Perimeter: {total_perim*scale_factor:.2f} {unit} ({total_perim:.2f} px)", 
                    fontsize=12)
            y_pos -= 0.04
            
            if has_radial:
                radial_widths = [p.get('radial_width_px', 0) for p in self.ring_properties if p.get('radial_width_px') is not None]
                if radial_widths:
                    avg_width = sum(radial_widths) / len(radial_widths)
                    ax.text(0.15, y_pos, f"Average Ring Width: {avg_width*scale_factor:.4f} {unit} ({avg_width:.2f} px)", 
                            fontsize=12)
                    y_pos -= 0.04
        else:
            ax.text(0.15, y_pos, f"Total Rings Detected: {total_rings}", fontsize=12)
            y_pos -= 0.04
            ax.text(0.15, y_pos, f"Total Area: {total_area:.2f} px²", fontsize=12)
            y_pos -= 0.04
            ax.text(0.15, y_pos, f"Average Ring Area: {avg_area:.2f} px²", fontsize=12)
            y_pos -= 0.04
            ax.text(0.15, y_pos, f"Total Perimeter: {total_perim:.2f} px", fontsize=12)
            y_pos -= 0.04
        
        # Footer
        ax.text(0.5, 0.05, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ha='center', fontsize=9, color='#666666')
        ax.text(0.5, 0.02, 'TRAS v2.0.0 | github.com/hmarichal93/tras', 
                ha='center', fontsize=8, color='#999999')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_ring_overlay_page(self, pdf):
        """Create page with image and ring overlays"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if not self.parent_window:
            print("Warning: No parent window available for ring overlay")
            return
        
        try:
            # Get image from parent - try multiple sources
            image = None
            
            # Try 1: Get from QImage
            if hasattr(self.parent_window, 'image') and self.parent_window.image:
                from tras.utils import img_qt_to_arr
                image = img_qt_to_arr(self.parent_window.image)
                print(f"Got image from QImage: {image.shape}")
            
            # Try 2: Get from imageData (numpy array)
            if image is None and hasattr(self.parent_window, 'imageData') and self.parent_window.imageData is not None:
                from tras.utils import img_b64_to_arr
                image = img_b64_to_arr(self.parent_window.imageData)
                print(f"Got image from imageData: {image.shape}")
            
            # Try 3: Load from filename
            if image is None and hasattr(self.parent_window, 'filename') and self.parent_window.filename:
                from PIL import Image as PILImage
                pil_img = PILImage.open(self.parent_window.filename)
                image = np.array(pil_img)
                print(f"Got image from file: {image.shape}")
            
            if image is None:
                raise ValueError("Could not load image from any source")
            
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(image)
            ax.set_title('Tree Rings with Detected Boundaries', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            
            # Get ring shapes from parent
            if hasattr(self.parent_window, 'labelList'):
                # Get all ring shapes (filter out radial measurement lines)
                for item_idx in range(len(self.parent_window.labelList)):
                    item = self.parent_window.labelList[item_idx]
                    shape = item.shape()
                    
                    # Only draw polygon shapes (rings), not lines (radial measurement)
                    if shape and hasattr(shape, 'points') and shape.points:
                        # Skip non-polygon shapes (like radial measurement lines)
                        if hasattr(shape, 'shape_type') and shape.shape_type != 'polygon':
                            continue
                        # Also skip shapes that don't look like rings
                        if not (hasattr(shape, 'label') and 'ring' in shape.label.lower()):
                            continue
                        # Draw ring boundary
                        # Convert QPointF objects to numpy array of coordinates
                        points = np.array([[p.x(), p.y()] for p in shape.points])
                        # Close the polygon by adding the first point at the end
                        points_closed = np.vstack([points, points[0:1]])
                        ax.plot(points_closed[:, 0], points_closed[:, 1], 
                               'g-', linewidth=2, alpha=0.7)
                        
                        # Add ring label
                        center_x = np.mean(points[:, 0])
                        center_y = np.mean(points[:, 1])
                        ax.text(center_x, center_y, shape.label, 
                               ha='center', va='center', 
                               fontsize=8, fontweight='bold',
                               color='white',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='green', alpha=0.7))
            
            # Add scale bar if available
            if self.metadata and 'scale' in self.metadata:
                scale_value = self.metadata['scale']['value']
                unit = self.metadata['scale']['unit']
                
                # Add 1 cm scale bar (or appropriate size)
                img_height, img_width = image.shape[:2]
                scale_length_physical = 1.0  # 1 unit
                scale_length_pixels = scale_length_physical / scale_value
                
                # Position in bottom-right corner
                margin = 50
                bar_y = img_height - margin
                bar_x_start = img_width - margin - scale_length_pixels
                bar_x_end = img_width - margin
                
                ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 
                       'k-', linewidth=3)
                ax.text((bar_x_start + bar_x_end) / 2, bar_y - 15, 
                       f'1 {unit}',
                       ha='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close()
        
        except Exception as e:
            import traceback
            print(f"Error creating ring overlay: {e}")
            print(traceback.format_exc())
            # Create a placeholder page
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.text(0.5, 0.5, f'Ring overlay image not available\nError: {str(e)}', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close()
    
    def _create_analysis_plots(self, pdf):
        """Create analysis plots page"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        has_scale = self.metadata and 'scale' in self.metadata
        has_radial = any(p.get('radial_width_px') is not None for p in self.ring_properties)
        has_years = 'harvested_year' in self.metadata and all(
            p['label'].replace('ring_', '').isdigit() or p['label'].isdigit() 
            for p in self.ring_properties
        )
        
        # Create figure with subplots
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Prepare data (rings are stored outermost to innermost)
        ring_nums = list(range(1, len(self.ring_properties) + 1))
        areas = [p['area'] for p in self.ring_properties]
        perimeters = [p['perimeter'] for p in self.ring_properties]
        
        # Compute cumulative area from pith (innermost) outward
        # Rings are ordered outermost to innermost, so we cumsum in reverse
        import numpy as np
        cumulative_areas = np.cumsum(areas[::-1])[::-1].tolist()
        
        # Reverse all data to plot from innermost (pith) to outermost (bark)
        areas = areas[::-1]
        cumulative_areas = cumulative_areas[::-1]
        
        if has_scale:
            scale_factor = self.metadata['scale']['value']
            scale_factor_sq = scale_factor ** 2
            unit = self.metadata['scale']['unit']
            areas_scaled = [a * scale_factor_sq for a in areas]
            cumulative_areas_scaled = [ca * scale_factor_sq for ca in cumulative_areas]
        
        # Determine x-axis (ring number or year)
        if has_years:
            try:
                harvested_year = int(self.metadata['harvested_year'])
                # Extract year from label or calculate (reversed for innermost to outermost)
                x_values = []
                for i, p in enumerate(reversed(self.ring_properties)):
                    label = p['label'].replace('ring_', '')
                    if label.isdigit():
                        x_values.append(int(label))
                    else:
                        # Calculate year based on position
                        x_values.append(harvested_year - (len(self.ring_properties) - i - 1))
                x_label = 'Year'
            except:
                x_values = list(reversed(ring_nums))
                x_label = 'Ring Number'
        else:
            x_values = list(reversed(ring_nums))
            x_label = 'Ring Number (Innermost to Outermost)'
        
        # Plot 1: Area vs Ring/Year
        ax1 = fig.add_subplot(gs[0, 0])
        if has_scale:
            ax1.plot(x_values, areas_scaled, 'o-', color='#8b4513', linewidth=2, markersize=4)
            ax1.set_ylabel(f'Ring Area ({unit}²)', fontsize=10, fontweight='bold')
            ax1.set_title('Ring Area Over Time', fontsize=12, fontweight='bold')
        else:
            ax1.plot(x_values, areas, 'o-', color='#8b4513', linewidth=2, markersize=4)
            ax1.set_ylabel('Ring Area (px²)', fontsize=10, fontweight='bold')
            ax1.set_title('Ring Area', fontsize=12, fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Area
        ax2 = fig.add_subplot(gs[0, 1])
        if has_scale:
            ax2.plot(x_values, cumulative_areas_scaled, 'o-', color='#2d5016', linewidth=2, markersize=4)
            ax2.fill_between(x_values, cumulative_areas_scaled, alpha=0.3, color='#2d5016')
            ax2.set_ylabel(f'Cumulative Area ({unit}²)', fontsize=10, fontweight='bold')
            ax2.set_title('Cumulative Ring Area', fontsize=12, fontweight='bold')
        else:
            ax2.plot(x_values, cumulative_areas, 'o-', color='#2d5016', linewidth=2, markersize=4)
            ax2.fill_between(x_values, cumulative_areas, alpha=0.3, color='#2d5016')
            ax2.set_ylabel('Cumulative Area (px²)', fontsize=10, fontweight='bold')
            ax2.set_title('Cumulative Ring Area', fontsize=12, fontweight='bold')
        ax2.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Individual Ring Width (if available) or Area Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        if has_radial:
            radial_widths = []
            radial_x_values = []
            # Iterate in reversed order to match the plot direction (innermost to outermost)
            for i, p in enumerate(reversed(self.ring_properties)):
                if p.get('radial_width_px') is not None:
                    if has_scale:
                        radial_widths.append(p['radial_width_px'] * scale_factor)
                    else:
                        radial_widths.append(p['radial_width_px'])
                    radial_x_values.append(x_values[i])
            
            if radial_widths:
                # Plot individual ring widths (varies based on growth conditions)
                ax3.plot(radial_x_values, radial_widths, 'o-', color='#ff8c00', linewidth=2, markersize=4)
                if has_scale:
                    ax3.set_ylabel(f'Ring Width ({unit})', fontsize=10, fontweight='bold')
                    ax3.set_title('Individual Ring Width Over Time', fontsize=12, fontweight='bold')
                else:
                    ax3.set_ylabel('Ring Width (px)', fontsize=10, fontweight='bold')
                    ax3.set_title('Individual Ring Width', fontsize=12, fontweight='bold')
                ax3.set_xlabel(x_label, fontsize=10, fontweight='bold')
                ax3.grid(True, alpha=0.3)
        else:
            # Show area distribution histogram
            ax3.hist(areas_scaled if has_scale else areas, bins=min(20, len(areas)), 
                    color='#8b4513', alpha=0.7, edgecolor='black')
            if has_scale:
                ax3.set_xlabel(f'Ring Area ({unit}²)', fontsize=10, fontweight='bold')
            else:
                ax3.set_xlabel('Ring Area (px²)', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax3.set_title('Ring Area Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Growth Rate (year-over-year area change) or Perimeter
        ax4 = fig.add_subplot(gs[1, 1])
        if len(areas) > 1:
            # Calculate growth rate (area difference between consecutive rings)
            growth_rates = [areas[i] - areas[i-1] for i in range(1, len(areas))]
            growth_x = x_values[1:]  # Skip first ring
            
            if has_scale:
                growth_rates_scaled = [gr * scale_factor_sq for gr in growth_rates]
                ax4.bar(growth_x, growth_rates_scaled, color='#228b22', alpha=0.7, edgecolor='black')
                ax4.set_ylabel(f'Area Change ({unit}²)', fontsize=10, fontweight='bold')
                ax4.set_title('Ring-to-Ring Area Change', fontsize=12, fontweight='bold')
            else:
                ax4.bar(growth_x, growth_rates, color='#228b22', alpha=0.7, edgecolor='black')
                ax4.set_ylabel('Area Change (px²)', fontsize=10, fontweight='bold')
                ax4.set_title('Ring-to-Ring Area Change', fontsize=12, fontweight='bold')
            ax4.set_xlabel(x_label, fontsize=10, fontweight='bold')
            ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor growth rate analysis', 
                    ha='center', va='center', fontsize=12)
            ax4.axis('off')
        
        plt.suptitle('Tree Ring Analysis - Quantitative Measurements', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
