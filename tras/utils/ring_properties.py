"""Headless ring properties computation and CSV export for CLI."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger


def polygon_area(points: npt.NDArray[np.float32]) -> float:
    """Compute polygon area using shoelace formula.

    Args:
        points: Nx2 array of (x, y) coordinates

    Returns:
        Area in square pixels
    """
    if len(points) < 3:
        return 0.0

    n = len(points)
    area = 0.0
    for j in range(n):
        x1, y1 = points[j]
        x2, y2 = points[(j + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_perimeter(points: npt.NDArray[np.float32]) -> float:
    """Compute polygon perimeter.

    Args:
        points: Nx2 array of (x, y) coordinates

    Returns:
        Perimeter in pixels
    """
    if len(points) < 2:
        return 0.0

    n = len(points)
    perimeter = 0.0
    for j in range(n):
        x1, y1 = points[j]
        x2, y2 = points[(j + 1) % n]
        perimeter += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return perimeter


def compute_ring_properties(
    rings: list[npt.NDArray[np.float32]], pith_xy: tuple[float, float]
) -> list[dict[str, Any]]:
    """Compute ring properties (area, perimeter, cumulative area) from ring polygons.

    Rings are sorted by area (inner to outer) and properties are computed.

    Args:
        rings: List of ring polygons, each as Nx2 array
        pith_xy: Pith coordinates (x, y)

    Returns:
        List of property dictionaries, each with:
        - label: ring label (ring_1, ring_2, ...)
        - area: annual growth area (px²)
        - cumulative_area: cumulative area from pith (px²)
        - perimeter: ring perimeter (px)
        - distance_from_pith: average distance from pith (px)
    """
    if not rings:
        return []

    # Compute areas and sort rings by area (inner to outer)
    ring_areas = [(i, polygon_area(ring)) for i, ring in enumerate(rings)]
    ring_areas.sort(key=lambda x: x[1])

    properties = []
    prev_cumulative_area = 0.0

    for ring_idx, (original_idx, outer_area) in enumerate(ring_areas):
        ring = rings[original_idx]

        # Annual area is the difference from previous cumulative area
        annual_area = max(outer_area - prev_cumulative_area, 0.0)

        # Compute perimeter
        perimeter = polygon_perimeter(ring)

        # Compute average distance from pith
        distances = np.sqrt(
            (ring[:, 0] - pith_xy[0]) ** 2 + (ring[:, 1] - pith_xy[1]) ** 2
        )
        avg_distance = float(np.mean(distances))

        properties.append(
            {
                "label": f"ring_{ring_idx + 1}",
                "area": annual_area,
                "cumulative_area": outer_area,
                "perimeter": perimeter,
                "distance_from_pith": avg_distance,
            }
        )

        prev_cumulative_area = outer_area

    return properties


def write_ring_properties_csv(
    output_path: Path,
    ring_properties: list[dict[str, Any]],
    scale_value: float,
    scale_unit: str,
) -> None:
    """Write ring properties to CSV file with physical scaling.

    Args:
        output_path: Path to output CSV file
        ring_properties: List of ring property dictionaries
        scale_value: Physical scale (unit per pixel)
        scale_unit: Physical unit name (e.g., "mm", "cm")
    """
    if not ring_properties:
        logger.warning(f"No ring properties to write to {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header with physical units
        writer.writerow(
            [
                "Ring",
                f"Area ({scale_unit}²)",
                f"Cumulative Area ({scale_unit}²)",
                f"Perimeter ({scale_unit})",
                f"Distance from Pith ({scale_unit})",
                "Area (px²)",
                "Cumulative Area (px²)",
                "Perimeter (px)",
                "Distance from Pith (px)",
            ]
        )

        # Data rows
        for prop in ring_properties:
            area_px = prop["area"]
            cumul_px = prop["cumulative_area"]
            perim_px = prop["perimeter"]
            dist_px = prop["distance_from_pith"]

            # Convert to physical units
            area_phys = area_px * (scale_value ** 2)
            cumul_phys = cumul_px * (scale_value ** 2)
            perim_phys = perim_px * scale_value
            dist_phys = dist_px * scale_value

            writer.writerow(
                [
                    prop["label"],
                    f"{area_phys:.4f}",
                    f"{cumul_phys:.4f}",
                    f"{perim_phys:.4f}",
                    f"{dist_phys:.4f}",
                    f"{area_px:.2f}",
                    f"{cumul_px:.2f}",
                    f"{perim_px:.2f}",
                    f"{dist_px:.2f}",
                ]
            )

    logger.info(f"Wrote ring properties CSV to {output_path}")

