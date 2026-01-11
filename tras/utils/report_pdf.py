"""Headless PDF report generator for CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
import numpy as np
import numpy.typing as npt
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_pdf_report(
    output_path: Path,
    image: npt.NDArray[np.uint8],
    pith_xy: tuple[float, float],
    rings: list[npt.NDArray[np.float32]],
    ring_properties: list[dict[str, Any]],
    scale_value: float,
    scale_unit: str,
    sample_code: Optional[str] = None,
    harvested_year: Optional[int] = None,
) -> None:
    """Generate PDF report with ring overlays and analysis plots.

    Args:
        output_path: Path to output PDF file
        image: Base image array (H x W x 3, uint8)
        pith_xy: Pith coordinates (x, y)
        rings: List of ring polygons (sorted inner to outer)
        ring_properties: List of ring property dictionaries
        scale_value: Physical scale (unit per pixel)
        scale_unit: Physical unit name (e.g., "mm", "cm")
        sample_code: Optional sample code for metadata
        harvested_year: Optional harvested year for time series plots
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        # Cover page
        _create_cover_page(
            pdf, ring_properties, scale_value, scale_unit, sample_code, harvested_year
        )

        # Ring overlay pages
        if rings:
            _create_ring_overlay_page(pdf, image, rings, pith_xy, show_labels=True)
            _create_ring_overlay_page(pdf, image, rings, pith_xy, show_labels=False)

        # Analysis plots
        if ring_properties:
            _create_analysis_plots(
                pdf, ring_properties, scale_value, scale_unit, harvested_year
            )

        # PDF metadata
        pdf_metadata = pdf.infodict()
        pdf_metadata["Title"] = "Tree Ring Analysis Report"
        pdf_metadata["Author"] = "TRAS - Tree Ring Analyzer Suite"
        pdf_metadata["Subject"] = f"Sample: {sample_code or 'Unknown'}"
        pdf_metadata["Keywords"] = "Dendrochronology, Tree Rings, Wood Analysis"
        pdf_metadata["CreationDate"] = datetime.now()

    logger.info(f"Generated PDF report: {output_path}")


def _create_cover_page(
    pdf: PdfPages,
    ring_properties: list[dict[str, Any]],
    scale_value: float,
    scale_unit: str,
    sample_code: Optional[str],
    harvested_year: Optional[int],
) -> None:
    """Create cover page with metadata and summary statistics."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header logo (if available)
    header_path = Path(__file__).parent.parent.parent / "assets" / "header.png"
    if header_path.exists():
        try:
            header_img = plt.imread(str(header_path))
            ax.imshow(header_img, extent=[0.05, 0.95, 0.88, 0.99], aspect="auto", zorder=0)
        except Exception:
            pass

    from tras import __version__

    title_y = 0.84
    subtitle_y = title_y - 0.035
    divider_y = subtitle_y - 0.03

    ax.text(
        0.5,
        title_y,
        "Tree Ring Analysis Report",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color="#2d5016",
    )
    ax.text(
        0.5,
        subtitle_y,
        f"TRAS - Tree Ring Analyzer Suite v{__version__}",
        ha="center",
        fontsize=11,
        color="#666666",
    )
    ax.plot([0.1, 0.9], [divider_y, divider_y], "k-", linewidth=1.5, color="#8b4513")

    y = divider_y - 0.05
    ax.text(0.1, y, "Sample", fontsize=11, fontweight="bold", color="#2d5016")
    y -= 0.03

    items = []
    if sample_code:
        items.append(sample_code)
    if harvested_year:
        items.append(str(harvested_year))
    items.append(f"{scale_value:.4f} {scale_unit}/px")
    if items:
        ax.text(0.1, y, " • ".join(items), fontsize=9)
        y -= 0.025

    y -= 0.02
    ax.text(0.1, y, "Statistics", fontsize=11, fontweight="bold", color="#2d5016")
    y -= 0.03
    ax.text(0.1, y, f"Rings detected: {len(ring_properties)}", fontsize=9)
    y -= 0.025

    if ring_properties:
        total_area_px = sum(p["area"] for p in ring_properties)
        total_area_phys = total_area_px * (scale_value ** 2)
        ax.text(
            0.1, y, f"Total area: {total_area_phys:.1f} {scale_unit}²", fontsize=9
        )
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


def _create_ring_overlay_page(
    pdf: PdfPages,
    image: npt.NDArray[np.uint8],
    rings: list[npt.NDArray[np.float32]],
    pith_xy: tuple[float, float],
    show_labels: bool = True,
) -> None:
    """Create page with image and ring overlays."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    title = "Tree Rings with Detected Boundaries" if show_labels else "Tree Rings (Overlay)"
    ax.imshow(image)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    img_height, img_width = image.shape[:2]
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.axis("off")

    # Draw rings
    for ring_idx, ring in enumerate(rings):
        points_closed = np.vstack([ring, ring[0:1]])
        ax.plot(
            points_closed[:, 0],
            points_closed[:, 1],
            "g-",
            linewidth=2,
            alpha=0.7,
        )
        if show_labels:
            angle_deg = (ring_idx * 25) % 360
            angle_idx = int(len(ring) * angle_deg / 360)
            label_x = ring[angle_idx, 0]
            label_y = ring[angle_idx, 1]
            ax.text(
                label_x,
                label_y,
                f"ring_{ring_idx + 1}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="green", alpha=0.8),
            )

    # Draw pith marker
    ax.plot(pith_xy[0], pith_xy[1], "r+", markersize=15, markeredgewidth=3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


def _create_analysis_plots(
    pdf: PdfPages,
    ring_properties: list[dict[str, Any]],
    scale_value: float,
    scale_unit: str,
    harvested_year: Optional[int],
) -> None:
    """Create analysis plots page."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(8.5, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ring_count = len(ring_properties)
    if ring_count == 0:
        return

    annual_growth = [p["area"] for p in ring_properties]
    cumulative_areas = [p["cumulative_area"] for p in ring_properties]

    if harvested_year:
        x_values = list(range(harvested_year - ring_count + 1, harvested_year + 1))
        x_label = "Year"
    else:
        x_values = list(range(1, ring_count + 1))
        x_label = "Ring Index (inner → outer)"

    # Annual growth area
    ax1 = fig.add_subplot(gs[0, 0])
    annual_growth_scaled = [a * (scale_value ** 2) for a in annual_growth]
    ax1.plot(
        x_values, annual_growth_scaled, "o-", color="#8b4513", linewidth=2, markersize=4
    )
    ax1.set_ylabel(f"Annual Growth Area ({scale_unit}²)", fontsize=10, fontweight="bold")
    ax1.set_xlabel(x_label, fontsize=10, fontweight="bold")
    ax1.set_title("Annual Growth Area", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Cumulative area
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative_scaled = [c * (scale_value ** 2) for c in cumulative_areas]
    ax2.plot(
        x_values, cumulative_scaled, "o-", color="#2d5016", linewidth=2, markersize=4
    )
    ax2.set_ylabel(f"Cumulative Area ({scale_unit}²)", fontsize=10, fontweight="bold")
    ax2.fill_between(x_values, cumulative_scaled, alpha=0.3, color="#2d5016")
    ax2.set_xlabel(x_label, fontsize=10, fontweight="bold")
    ax2.set_title("Cumulative Ring Area", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Histogram of annual growth
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(
        annual_growth_scaled,
        bins=min(20, len(annual_growth)),
        color="#8b4513",
        alpha=0.7,
        edgecolor="black",
    )
    ax3.set_xlabel(f"Annual Growth Area ({scale_unit}²)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax3.set_title("Annual Growth Area Distribution", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # Year-to-year growth change
    ax4 = fig.add_subplot(gs[1, 1])
    if len(annual_growth) > 1:
        growth_rates = [
            annual_growth[i] - annual_growth[i - 1] for i in range(1, len(annual_growth))
        ]
        growth_x = x_values[1:]
        growth_scaled = [g * (scale_value ** 2) for g in growth_rates]
        ax4.bar(growth_x, growth_scaled, color="#228b22", alpha=0.7, edgecolor="black")
        ax4.set_ylabel(f"Growth Change ({scale_unit}²)", fontsize=10, fontweight="bold")
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

    plt.suptitle(
        "Tree Ring Analysis - Quantitative Measurements",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    pdf.savefig(fig)
    plt.close()

