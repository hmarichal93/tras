"""
Ring Properties Dialog - Display and export tree ring measurements.
Supports closed rings (polygons) and open rings (radial-width measurements).
"""
import csv
from pathlib import Path
from datetime import datetime

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
import numpy as np


class RingPropertiesDialog(QtWidgets.QDialog):
    """Dialog to display ring properties (area, perimeter, radial width)."""

    def __init__(
        self,
        ring_properties,
        radial_measurements,
        parent=None,
        metadata=None,
    ):
        super().__init__(parent)
        self.ring_properties = ring_properties or []
        self.radial_measurements = radial_measurements or []
        self.metadata = metadata or {}
        self.parent_window = parent  # Store reference to main window

        self.setWindowTitle(self.tr("Tree Ring Properties"))
        self.setModal(True)
        self.resize(760, 540)

        self.has_polygon_data = bool(self.ring_properties)
        self.has_radial_data = bool(self.radial_measurements)

        self.cumulative_areas, self.annual_growth_areas = self._compute_polygon_metrics()

        layout = QtWidgets.QVBoxLayout()

        info_parts = []
        if self.has_polygon_data:
            info_parts.append(self.tr(f"Closed rings: {len(self.ring_properties)}"))
        if self.has_radial_data:
            info_parts.append(self.tr(f"Open rings: {len(self.radial_measurements)}"))
        if not info_parts:
            info_parts.append(self.tr("No ring data available"))
        info_label = QtWidgets.QLabel(" | ".join(info_parts))
        info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(info_label)

        sections: list[tuple[str, QtWidgets.QWidget]] = []
        if self.has_polygon_data:
            sections.append((self.tr("Closed Rings"), self._build_polygon_section()))
        if self.has_radial_data:
            sections.append((self.tr("Open Rings"), self._build_radial_section()))

        if len(sections) > 1:
            tabs = QtWidgets.QTabWidget()
            for title, widget in sections:
                tabs.addTab(widget, title)
            layout.addWidget(tabs)
        elif sections:
            layout.addWidget(sections[0][1])
        else:
            placeholder = QtWidgets.QLabel(
                self.tr("No ring polygons or radial measurements available.")
            )
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)

        export_layout = QtWidgets.QHBoxLayout()
        export_layout.addStretch(1)

        export_csv_btn = QtWidgets.QPushButton(self.tr("📄 Export CSV"))
        export_csv_btn.clicked.connect(self._export_csv)
        export_layout.addWidget(export_csv_btn)

        export_pdf_btn = QtWidgets.QPushButton(self.tr("📘 Generate PDF Report"))
        export_pdf_btn.clicked.connect(self._export_pdf)
        export_layout.addWidget(export_pdf_btn)

        layout.addLayout(export_layout)

        close_btn = QtWidgets.QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        self.setLayout(layout)

    def _compute_polygon_metrics(self):
        if not self.has_polygon_data:
            return [], []

        cumulative_areas = [p.get("cumulative_area", 0.0) for p in self.ring_properties]
        annual_growth_areas = []
        for i, cumulative in enumerate(cumulative_areas):
            if i == 0:
                growth = cumulative
            else:
                growth = cumulative - cumulative_areas[i - 1]
            annual_growth_areas.append(max(growth, 0.0))

        return cumulative_areas, annual_growth_areas

    def _build_polygon_section(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        has_scale = self.metadata and "scale" in self.metadata
        unit = self.metadata["scale"]["unit"] if has_scale else None
        scale_value = self.metadata["scale"]["value"] if has_scale else None
        has_radial_column = any(
            props.get("radial_width_px") is not None for props in self.ring_properties
        )

        table = QtWidgets.QTableWidget()
        if has_scale:
            headers = [
                self.tr("Ring"),
                self.tr(f"Area ({unit}²)"),
                self.tr(f"Cumul. Area ({unit}²)"),
                self.tr(f"Perimeter ({unit})"),
            ]
            if has_radial_column:
                headers.append(self.tr(f"Radial Width ({unit})"))
            headers.extend(
                [
                    self.tr("Area (px²)"),
                    self.tr("Cumul. Area (px²)"),
                    self.tr("Perimeter (px)"),
                ]
            )
        else:
            headers = [
                self.tr("Ring"),
                self.tr("Area (px²)"),
                self.tr("Cumul. Area (px²)"),
                self.tr("Perimeter (px)"),
            ]
            if has_radial_column:
                headers.append(self.tr("Radial Width (px)"))

        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setRowCount(len(self.ring_properties))

        for row, props in enumerate(self.ring_properties):
            col = 0
            table.setItem(row, col, QtWidgets.QTableWidgetItem(props["label"]))
            col += 1

            cumul_area_px = self.cumulative_areas[row]
            if has_scale and scale_value is not None:
                area_physical = props["area"] * (scale_value ** 2)
                cumul_physical = cumul_area_px * (scale_value ** 2)
                perim_physical = props["perimeter"] * scale_value

                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{area_physical:.4f}"))
                col += 1
                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{cumul_physical:.4f}"))
                col += 1
                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{perim_physical:.4f}"))
                col += 1

                if has_radial_column:
                    radial_px = props.get("radial_width_px")
                    if radial_px is not None:
                        radial_physical = radial_px * scale_value
                        table.setItem(
                            row, col, QtWidgets.QTableWidgetItem(f"{radial_physical:.4f}")
                        )
                    else:
                        table.setItem(row, col, QtWidgets.QTableWidgetItem("N/A"))
                    col += 1

                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{props['area']:.2f}"))
                col += 1
                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{cumul_area_px:.2f}"))
                col += 1
                table.setItem(
                    row,
                    col,
                    QtWidgets.QTableWidgetItem(f"{props['perimeter']:.2f}"),
                )
                col += 1
            else:
                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{props['area']:.2f}"))
                col += 1
                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{cumul_area_px:.2f}"))
                col += 1
                table.setItem(
                    row,
                    col,
                    QtWidgets.QTableWidgetItem(f"{props['perimeter']:.2f}"),
                )
                col += 1

                if has_radial_column:
                    radial_px = props.get("radial_width_px")
                    if radial_px is not None:
                        table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{radial_px:.2f}"))
                    else:
                        table.setItem(row, col, QtWidgets.QTableWidgetItem("N/A"))

        layout.addWidget(table)

        total_area = sum(p["area"] for p in self.ring_properties)
        avg_area = total_area / len(self.ring_properties) if self.ring_properties else 0.0
        total_perim = sum(p["perimeter"] for p in self.ring_properties)
        avg_perim = total_perim / len(self.ring_properties) if self.ring_properties else 0.0

        if has_scale and scale_value is not None:
            total_area_physical = total_area * (scale_value ** 2)
            avg_area_physical = avg_area * (scale_value ** 2)
            total_perim_physical = total_perim * scale_value
            avg_perim_physical = avg_perim * scale_value
            summary_text = (
                f"<b>{self.tr('Closed-ring Summary')}:</b><br>"
                f"{self.tr('Total Area')}: {total_area_physical:.4f} {unit}² "
                f"({total_area:.2f} px²)<br>"
                f"{self.tr('Average Area')}: {avg_area_physical:.4f} {unit}² "
                f"({avg_area:.2f} px²)<br>"
                f"{self.tr('Total Perimeter')}: {total_perim_physical:.4f} {unit} "
                f"({total_perim:.2f} px)<br>"
                f"{self.tr('Average Perimeter')}: {avg_perim_physical:.4f} {unit} "
                f"({avg_perim:.2f} px)"
            )
        else:
            summary_text = (
                f"<b>{self.tr('Closed-ring Summary')}:</b><br>"
                f"{self.tr('Total Area')}: {total_area:.2f} px²<br>"
                f"{self.tr('Average Area')}: {avg_area:.2f} px²<br>"
                f"{self.tr('Total Perimeter')}: {total_perim:.2f} px<br>"
                f"{self.tr('Average Perimeter')}: {avg_perim:.2f} px"
            )

        summary_label = QtWidgets.QLabel(summary_text)
        summary_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(summary_label)
        return widget

    def _build_radial_section(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        has_scale = self.metadata and "scale" in self.metadata
        unit = self.metadata["scale"]["unit"] if has_scale else None
        scale_value = self.metadata["scale"]["value"] if has_scale else None

        headers = [self.tr("Ring")]
        if has_scale:
            headers.append(self.tr(f"Radial Width ({unit})"))
        headers.append(self.tr("Radial Width (px)"))
        headers.append(self.tr("Distance from Pith (px)"))

        table = QtWidgets.QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setRowCount(len(self.radial_measurements))

        for row, entry in enumerate(self.radial_measurements):
            col = 0
            table.setItem(row, col, QtWidgets.QTableWidgetItem(entry["label"]))
            col += 1

            radial_px = entry.get("radial_width_px")
            if has_scale and scale_value is not None:
                if radial_px is not None:
                    table.setItem(
                        row,
                        col,
                        QtWidgets.QTableWidgetItem(f"{radial_px * scale_value:.4f}"),
                    )
                else:
                    table.setItem(row, col, QtWidgets.QTableWidgetItem("N/A"))
                col += 1

            if radial_px is not None:
                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{radial_px:.2f}"))
            else:
                table.setItem(row, col, QtWidgets.QTableWidgetItem("N/A"))
            col += 1

            dist_px = entry.get("distance_from_pith")
            if dist_px is not None:
                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{dist_px:.2f}"))
            else:
                table.setItem(row, col, QtWidgets.QTableWidgetItem("—"))

        layout.addWidget(table)

        radial_values = [
            entry["radial_width_px"]
            for entry in self.radial_measurements
            if entry.get("radial_width_px") is not None
        ]
        avg_radial_px = sum(radial_values) / len(radial_values) if radial_values else 0.0

        if has_scale and scale_value is not None:
            summary_text = (
                f"<b>{self.tr('Open-ring Summary')}:</b><br>"
                f"{self.tr('Measured Rings')}: {len(self.radial_measurements)}<br>"
                f"{self.tr('Average Radial Width')}: {avg_radial_px * scale_value:.4f} {unit} "
                f"({avg_radial_px:.2f} px)"
            )
        else:
            summary_text = (
                f"<b>{self.tr('Open-ring Summary')}:</b><br>"
                f"{self.tr('Measured Rings')}: {len(self.radial_measurements)}<br>"
                f"{self.tr('Average Radial Width')}: {avg_radial_px:.2f} px"
            )

        summary_label = QtWidgets.QLabel(summary_text)
        summary_label.setStyleSheet("padding: 10px; background-color: #f6f6f6; border-radius: 5px;")
        layout.addWidget(summary_label)
        return widget

    def _export_csv(self):
        """Export ring properties to CSV file."""
        default_filename = "ring_properties.csv"
        if self.metadata.get("sample_code"):
            sample_code = self.metadata["sample_code"]
            safe_code = "".join(c for c in sample_code if c.isalnum() or c in ("-", "_"))
            default_filename = f"{safe_code}.csv"

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export Ring Properties"),
            default_filename,
            self.tr("CSV Files (*.csv)"),
        )
        if not filename:
            return

        has_scale = self.metadata and "scale" in self.metadata
        scale_value = self.metadata["scale"]["value"] if has_scale else None
        unit = self.metadata["scale"]["unit"] if has_scale else None

        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if self.metadata:
                    writer.writerow(["# Metadata"])
                    if "harvested_year" in self.metadata:
                        writer.writerow(["# Harvested Year", self.metadata["harvested_year"]])
                    if "sample_code" in self.metadata:
                        writer.writerow(["# Sample Code", self.metadata["sample_code"]])
                    if "observation" in self.metadata:
                        writer.writerow(["# Observation", self.metadata["observation"]])
                    writer.writerow([])

                if has_scale and scale_value is not None:
                    writer.writerow(["# Scale", f"{scale_value:.6f} {unit}/pixel"])
                    writer.writerow([])

                if self.has_polygon_data:
                    writer.writerow(["# Closed Rings"])
                    polygon_has_radial = any(
                        props.get("radial_width_px") is not None for props in self.ring_properties
                    )
                    if has_scale and scale_value is not None:
                        headers = [
                            "Ring",
                            f"Area ({unit}²)",
                            f"Cumulative Area ({unit}²)",
                            f"Perimeter ({unit})",
                        ]
                        if polygon_has_radial:
                            headers.append(f"Radial Width ({unit})")
                        headers.extend(
                            ["Area (px²)", "Cumulative Area (px²)", "Perimeter (px)"]
                        )
                    else:
                        headers = ["Ring", "Area (px²)", "Cumulative Area (px²)", "Perimeter (px)"]
                        if polygon_has_radial:
                            headers.append("Radial Width (px)")
                    writer.writerow(headers)

                    for idx, props in enumerate(self.ring_properties):
                        cumul_area_px = self.cumulative_areas[idx]
                        row = [props["label"]]
                        if has_scale and scale_value is not None:
                            area_physical = props["area"] * (scale_value ** 2)
                            cumul_physical = cumul_area_px * (scale_value ** 2)
                            perim_physical = props["perimeter"] * scale_value
                            row.extend(
                                [
                                    f"{area_physical:.4f}",
                                    f"{cumul_physical:.4f}",
                                    f"{perim_physical:.4f}",
                                ]
                            )
                            if polygon_has_radial:
                                radial_px = props.get("radial_width_px")
                                row.append(
                                    f"{radial_px * scale_value:.4f}" if radial_px is not None else "N/A"
                                )
                            row.extend(
                                [
                                    f"{props['area']:.2f}",
                                    f"{cumul_area_px:.2f}",
                                    f"{props['perimeter']:.2f}",
                                ]
                            )
                        else:
                            row.extend(
                                [
                                    f"{props['area']:.2f}",
                                    f"{cumul_area_px:.2f}",
                                    f"{props['perimeter']:.2f}",
                                ]
                            )
                            if polygon_has_radial:
                                radial_px = props.get("radial_width_px")
                                row.append(f"{radial_px:.2f}" if radial_px is not None else "N/A")
                        writer.writerow(row)
                    writer.writerow([])

                if self.has_radial_data:
                    writer.writerow(["# Open Rings (Radial Widths)"])
                    if has_scale and scale_value is not None:
                        writer.writerow(
                            ["Ring", f"Radial Width ({unit})", "Radial Width (px)", "Distance (px)"]
                        )
                    else:
                        writer.writerow(["Ring", "Radial Width (px)", "Distance (px)"])
                    for entry in self.radial_measurements:
                        radial_px = entry.get("radial_width_px")
                        distance = entry.get("distance_from_pith")
                        if has_scale and scale_value is not None:
                            radial_unit = (
                                f"{radial_px * scale_value:.4f}" if radial_px is not None else "N/A"
                            )
                            row = [
                                entry["label"],
                                radial_unit,
                                f"{radial_px:.2f}" if radial_px is not None else "N/A",
                                f"{distance:.2f}" if distance is not None else "—",
                            ]
                        else:
                            row = [
                                entry["label"],
                                f"{radial_px:.2f}" if radial_px is not None else "N/A",
                                f"{distance:.2f}" if distance is not None else "—",
                            ]
                        writer.writerow(row)

            QtWidgets.QMessageBox.information(
                self,
                self.tr("Export Successful"),
                self.tr(f"Ring properties exported to:\n{filename}"),
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Export Failed"),
                self.tr(f"Failed to export CSV:\n{str(e)}"),
            )

    def _export_pdf(self):
        """Generate PDF report with ring overlays and analysis plots."""
        default_filename = "tree_ring_report.pdf"
        if self.metadata.get("sample_code"):
            sample_code = self.metadata["sample_code"]
            safe_code = "".join(c for c in sample_code if c.isalnum() or c in ("-", "_"))
            default_filename = f"{safe_code}_report.pdf"

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Save PDF Report"),
            default_filename,
            self.tr("PDF Files (*.pdf)"),
        )
        if not filename:
            return

        try:
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(filename) as pdf:
                self._create_cover_page(pdf)
                if self.has_polygon_data or self.has_radial_data:
                    self._create_ring_overlay_page(pdf)
                    self._create_ring_overlay_page(pdf, show_labels=False)
                if self.has_polygon_data:
                    self._create_analysis_plots(pdf)
                if self.has_radial_data:
                    self._create_radial_plot_page(pdf)

                pdf_metadata = pdf.infodict()
                pdf_metadata["Title"] = "Tree Ring Analysis Report"
                pdf_metadata["Author"] = "TRAS - Tree Ring Analyzer Suite"
                pdf_metadata["Subject"] = f"Sample: {self.metadata.get('sample_code', 'Unknown')}"
                pdf_metadata["Keywords"] = "Dendrochronology, Tree Rings, Wood Analysis"
                pdf_metadata["CreationDate"] = datetime.now()

            QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.information(
                self,
                self.tr("PDF Generated"),
                self.tr(f"PDF report successfully generated:\n{filename}"),
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("PDF Generation Error"),
                self.tr(f"Failed to generate PDF:\n{str(e)}\n\nMake sure matplotlib is installed."),
            )

    def _create_cover_page(self, pdf):
        """Create cover page with metadata and summary statistics."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 11))
        self._add_header(fig)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        from tras import __version__

        ax.text(
            0.5,
            0.92,
            "Tree Ring Analysis Report",
            ha="center",
            fontsize=20,
            fontweight="bold",
            color="#2d5016",
        )
        ax.text(
            0.5,
            0.88,
            f"TRAS - Tree Ring Analyzer Suite v{__version__}",
            ha="center",
            fontsize=11,
            color="#666666",
        )
        ax.plot([0.1, 0.9], [0.85, 0.85], "k-", linewidth=1.5, color="#8b4513")

        y = 0.80
        if self.metadata:
            ax.text(0.1, y, "Sample", fontsize=11, fontweight="bold", color="#2d5016")
            y -= 0.03
            items = []
            if "sample_code" in self.metadata:
                items.append(f"{self.metadata['sample_code']}")
            if "harvested_year" in self.metadata:
                items.append(f"{self.metadata['harvested_year']}")
            if "scale" in self.metadata:
                scale_value = self.metadata["scale"]["value"]
                unit = self.metadata["scale"]["unit"]
                items.append(f"{scale_value:.4f} {unit}/px")
            if items:
                ax.text(0.1, y, " • ".join(items), fontsize=9)
                y -= 0.025
            if "observation" in self.metadata:
                import textwrap

                obs_text = self.metadata["observation"]
                wrapped_lines = textwrap.wrap(obs_text, width=80)
                for line in wrapped_lines[:5]:
                    ax.text(0.1, y, line, fontsize=8, style="italic", color="#555")
                    y -= 0.02
                if len(wrapped_lines) > 5:
                    ax.text(0.1, y, "...", fontsize=8, style="italic", color="#555")
                    y -= 0.02

        y -= 0.02
        ax.text(0.1, y, "Statistics", fontsize=11, fontweight="bold", color="#2d5016")
        y -= 0.03
        ax.text(0.1, y, f"Closed rings: {len(self.ring_properties)}", fontsize=9)
        y -= 0.025
        ax.text(0.1, y, f"Open rings: {len(self.radial_measurements)}", fontsize=9)
        y -= 0.025

        if self.has_polygon_data:
            has_scale = self.metadata and "scale" in self.metadata
            if has_scale:
                scale_value = self.metadata["scale"]["value"]
                unit = self.metadata["scale"]["unit"]
                total_area = sum(p["area"] for p in self.ring_properties) * (scale_value ** 2)
                ax.text(0.1, y, f"Total area: {total_area:.1f} {unit}²", fontsize=9)
            else:
                total_area = sum(p["area"] for p in self.ring_properties)
                ax.text(0.1, y, f"Total area: {total_area:.0f} px²", fontsize=9)
            y -= 0.02
        elif self.has_radial_data:
            radial_values = [
                entry["radial_width_px"]
                for entry in self.radial_measurements
                if entry.get("radial_width_px") is not None
            ]
            avg_radial = sum(radial_values) / len(radial_values) if radial_values else 0.0
            if self.metadata and "scale" in self.metadata:
                unit = self.metadata["scale"]["unit"]
                scale_value = self.metadata["scale"]["value"]
                ax.text(
                    0.1,
                    y,
                    f"Avg. radial width: {avg_radial * scale_value:.3f} {unit} ({avg_radial:.2f} px)",
                    fontsize=9,
                )
            else:
                ax.text(0.1, y, f"Avg. radial width: {avg_radial:.2f} px", fontsize=9)
            y -= 0.02

        ax.text(
            0.5,
            0.05,
            f"{datetime.now().strftime('%Y-%m-%d')}",
            ha="center",
            fontsize=8,
            color="#999",
        )
        ax.text(
            0.5,
            0.02,
            "github.com/hmarichal93/tras",
            ha="center",
            fontsize=7,
            color="#ccc",
        )
        pdf.savefig(fig)
        plt.close()

    def _create_ring_overlay_page(self, pdf, show_labels: bool = True):
        """Create page with image and ring overlays."""
        import matplotlib.pyplot as plt
        from PyQt5.QtCore import QBuffer, QIODevice
        from PIL import Image as PILImage
        import io

        if not self.parent_window:
            return

        image = None
        try:
            if getattr(self.parent_window, "imageArray", None) is not None:
                image = self.parent_window.imageArray.copy()
            elif getattr(self.parent_window, "image", None):
                buffer = QBuffer()
                buffer.open(QIODevice.WriteOnly)
                self.parent_window.image.save(buffer, "PNG")
                pil_img = PILImage.open(io.BytesIO(buffer.data())).convert("RGB")
                image = np.array(pil_img)
            elif getattr(self.parent_window, "imageData", None) is not None:
                from tras.utils import img_b64_to_arr

                image = img_b64_to_arr(self.parent_window.imageData)
        except Exception as e:
            print(f"Failed to load base image for overlay: {e}")

        if image is None:
            return

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        fig, ax = plt.subplots(figsize=(8.5, 11))
        title = "Tree Rings with Detected Boundaries" if show_labels else "Tree Rings (Overlay)"
        ax.imshow(image)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        img_height, img_width = image.shape[:2]
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
        ax.axis("off")

        if hasattr(self.parent_window, "labelList"):
            ring_count = 0
            for idx in range(len(self.parent_window.labelList)):
                item = self.parent_window.labelList[idx]
                shape = item.shape()
                if (
                    shape
                    and getattr(shape, "points", None)
                    and getattr(shape, "shape_type", "") == "polygon"
                ):
                    points = np.array([[p.x(), p.y()] for p in shape.points])
                    points_closed = np.vstack([points, points[0:1]])
                    ax.plot(points_closed[:, 0], points_closed[:, 1], "g-", linewidth=2, alpha=0.7)
                    if show_labels:
                        angle_deg = (ring_count * 25) % 360
                        angle_idx = int(len(points) * angle_deg / 360)
                        label_x = points[angle_idx, 0]
                        label_y = points[angle_idx, 1]
                        ax.text(
                            label_x,
                            label_y,
                            shape.label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="white",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="green", alpha=0.8),
                        )
                    ring_count += 1

            for idx in range(len(self.parent_window.labelList)):
                item = self.parent_window.labelList[idx]
                shape = item.shape()
                if (
                    shape
                    and getattr(shape, "points", None)
                    and getattr(shape, "shape_type", "") == "linestrip"
                ):
                    points = np.array([[p.x(), p.y()] for p in shape.points])
                    ax.plot(
                        points[:, 0],
                        points[:, 1],
                        color="#ff8c00",
                        linewidth=2,
                        alpha=0.9,
                        linestyle="-",
                        label="Open ring",
                    )
                    if show_labels:
                        midpoint_idx = len(points) // 2
                        label_x = points[midpoint_idx, 0]
                        label_y = points[midpoint_idx, 1]
                        ax.text(
                            label_x,
                            label_y,
                            shape.label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="black",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffcc80", alpha=0.9),
                        )

            for idx in range(len(self.parent_window.labelList)):
                item = self.parent_window.labelList[idx]
                shape = item.shape()
                if (
                    shape
                    and getattr(shape, "points", None)
                    and getattr(shape, "shape_type", "") == "line"
                    and getattr(shape, "label", "") == "radial_measurement_line"
                ):
                    points = np.array([[p.x(), p.y()] for p in shape.points])
                    ax.plot(
                        points[:, 0],
                        points[:, 1],
                        "r--",
                        linewidth=3,
                        alpha=0.8,
                        label="Radial measurement",
                    )
                    if len(points) >= 2:
                        dx = points[-1, 0] - points[-2, 0]
                        dy = points[-1, 1] - points[-2, 1]
                        ax.arrow(
                            points[-2, 0],
                            points[-2, 1],
                            dx,
                            dy,
                            head_width=20,
                            head_length=30,
                            fc="red",
                            ec="red",
                            alpha=0.8,
                            linewidth=2,
                        )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_analysis_plots(self, pdf):
        """Create analysis plots page for closed rings."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        if not self.has_polygon_data:
            return

        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        ring_count = len(self.ring_properties)
        if ring_count == 0:
            return

        has_scale = self.metadata and "scale" in self.metadata
        scale_value = self.metadata["scale"]["value"] if has_scale else None
        unit = self.metadata["scale"]["unit"] if has_scale else None

        annual_growth = self.annual_growth_areas
        cumulative_areas = self.cumulative_areas

        if self.metadata and "harvested_year" in self.metadata:
            harvested_year = int(self.metadata["harvested_year"])
            x_values = list(range(harvested_year - ring_count + 1, harvested_year + 1))
            x_label = "Year"
        else:
            x_values = list(range(1, ring_count + 1))
            x_label = "Ring Index (inner → outer)"

        ax1 = fig.add_subplot(gs[0, 0])
        if has_scale and scale_value is not None:
            annual_growth_scaled = [a * (scale_value ** 2) for a in annual_growth]
            ax1.plot(x_values, annual_growth_scaled, "o-", color="#8b4513", linewidth=2, markersize=4)
            ax1.set_ylabel(f"Annual Growth Area ({unit}²)", fontsize=10, fontweight="bold")
        else:
            ax1.plot(x_values, annual_growth, "o-", color="#8b4513", linewidth=2, markersize=4)
            ax1.set_ylabel("Annual Growth Area (px²)", fontsize=10, fontweight="bold")
        ax1.set_xlabel(x_label, fontsize=10, fontweight="bold")
        ax1.set_title("Annual Growth Area", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        if has_scale and scale_value is not None:
            cumulative_scaled = [c * (scale_value ** 2) for c in cumulative_areas]
            ax2.plot(x_values, cumulative_scaled, "o-", color="#2d5016", linewidth=2, markersize=4)
            ax2.set_ylabel(f"Cumulative Area ({unit}²)", fontsize=10, fontweight="bold")
            fill_values = cumulative_scaled
        else:
            ax2.plot(x_values, cumulative_areas, "o-", color="#2d5016", linewidth=2, markersize=4)
            ax2.set_ylabel("Cumulative Area (px²)", fontsize=10, fontweight="bold")
            fill_values = cumulative_areas
        ax2.fill_between(x_values, fill_values, alpha=0.3, color="#2d5016")
        ax2.set_xlabel(x_label, fontsize=10, fontweight="bold")
        ax2.set_title("Cumulative Ring Area", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        polygon_has_radial = any(
            props.get("radial_width_px") is not None for props in self.ring_properties
        )
        if polygon_has_radial:
            radial_widths = []
            radial_x = []
            for idx, props in enumerate(self.ring_properties):
                value = props.get("radial_width_px")
                if value is not None:
                    radial_x.append(x_values[idx])
                    if has_scale and scale_value is not None:
                        radial_widths.append(value * scale_value)
                    else:
                        radial_widths.append(value)
            if radial_widths:
                ax3.plot(radial_x, radial_widths, "o-", color="#ff8c00", linewidth=2, markersize=4)
                label = f"Ring Width ({unit})" if has_scale and scale_value is not None else "Ring Width (px)"
                ax3.set_ylabel(label, fontsize=10, fontweight="bold")
                ax3.set_title("Ring Width Over Time", fontsize=12, fontweight="bold")
                ax3.set_xlabel(x_label, fontsize=10, fontweight="bold")
                ax3.grid(True, alpha=0.3)
        else:
            ax3.hist(
                annual_growth if not has_scale else [a * (scale_value ** 2) for a in annual_growth],
                bins=min(20, len(annual_growth)),
                color="#8b4513",
                alpha=0.7,
                edgecolor="black",
            )
            label = f"Annual Growth Area ({unit}²)" if has_scale and scale_value is not None else "Annual Growth Area (px²)"
            ax3.set_xlabel(label, fontsize=10, fontweight="bold")
            ax3.set_ylabel("Frequency", fontsize=10, fontweight="bold")
            ax3.set_title("Annual Growth Area Distribution", fontsize=12, fontweight="bold")
            ax3.grid(True, alpha=0.3, axis="y")

        ax4 = fig.add_subplot(gs[1, 1])
        if len(annual_growth) > 1:
            growth_rates = [annual_growth[i] - annual_growth[i - 1] for i in range(1, len(annual_growth))]
            growth_x = x_values[1:]
            if has_scale and scale_value is not None:
                growth_scaled = [g * (scale_value ** 2) for g in growth_rates]
                ax4.bar(growth_x, growth_scaled, color="#228b22", alpha=0.7, edgecolor="black")
                ax4.set_ylabel(f"Growth Change ({unit}²)", fontsize=10, fontweight="bold")
            else:
                ax4.bar(growth_x, growth_rates, color="#228b22", alpha=0.7, edgecolor="black")
                ax4.set_ylabel("Growth Change (px²)", fontsize=10, fontweight="bold")
            ax4.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
            ax4.set_xlabel(x_label, fontsize=10, fontweight="bold")
            ax4.set_title("Year-to-Year Growth Change", fontsize=12, fontweight="bold")
            ax4.grid(True, alpha=0.3, axis="y")
        else:
            ax4.text(
                0.5,
                0.5,
                "Insufficient data\nfor growth rate analysis",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax4.axis("off")

        plt.suptitle("Tree Ring Analysis - Quantitative Measurements", fontsize=14, fontweight="bold", y=0.98)
        pdf.savefig(fig)
        plt.close()

    def _create_radial_plot_page(self, pdf):
        """Create page summarizing radial width measurements."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_title("Radial Width Measurements", fontsize=16, fontweight="bold", pad=20)

        radial_entries = [
            entry for entry in self.radial_measurements if entry.get("radial_width_px") is not None
        ]
        if not radial_entries:
            ax.text(0.5, 0.5, "No radial width measurements available", ha="center", va="center", fontsize=14)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close()
            return

        x_vals = list(range(1, len(radial_entries) + 1))
        labels = [entry["label"] for entry in radial_entries]
        widths_px = [entry["radial_width_px"] for entry in radial_entries]
        has_scale = self.metadata and "scale" in self.metadata
        scale_value = self.metadata["scale"]["value"] if has_scale else None
        unit = self.metadata["scale"]["unit"] if has_scale else None

        if has_scale and scale_value is not None:
            widths_plot = [w * scale_value for w in widths_px]
            ax.set_ylabel(f"Radial Width ({unit})", fontsize=12, fontweight="bold")
        else:
            widths_plot = widths_px
            ax.set_ylabel("Radial Width (px)", fontsize=12, fontweight="bold")

        ax.plot(x_vals, widths_plot, "o-", color="#ff8c00", linewidth=2, markersize=5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Measurement order (pith → bark)", fontsize=12, fontweight="bold")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        if has_scale and scale_value is not None:
            ax2 = ax.twinx()
            ax2.plot(x_vals, widths_px, "s--", color="#888", linewidth=1, markersize=4, alpha=0.6)
            ax2.set_ylabel("Radial Width (px)", fontsize=10)

        avg_width_px = sum(widths_px) / len(widths_px)
        if has_scale and scale_value is not None:
            avg_text = f"Average width: {avg_width_px * scale_value:.4f} {unit} ({avg_width_px:.2f} px)"
        else:
            avg_text = f"Average width: {avg_width_px:.2f} px"
        ax.text(0.02, 0.95, avg_text, transform=ax.transAxes, fontsize=11, bbox=dict(facecolor="white", alpha=0.6))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

    def _add_header(self, fig):
        import matplotlib.pyplot as plt

        ax_header = fig.add_axes([0, 0.9, 1, 0.1])
        ax_header.axis("off")
        ax_header.text(
            0.02,
            0.5,
            "TRAS - Tree Ring Analyzer Suite",
            fontsize=14,
            fontweight="bold",
            color="#2d5016",
            va="center",
        )
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "tras-logo.png"
        if logo_path.exists():
            import matplotlib.image as mpimg

            logo = mpimg.imread(str(logo_path))
            ax_logo = fig.add_axes([0.8, 0.9, 0.18, 0.08])
            ax_logo.imshow(logo)
            ax_logo.axis("off")
