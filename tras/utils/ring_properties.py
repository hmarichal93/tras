"""Headless ring properties and CSV export (thin wrapper over ``tras.analysis.rings``)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from tras.analysis.geometry import polygon_area as _polygon_area
from tras.analysis.geometry import polygon_perimeter as _polygon_perimeter
from tras.analysis.rings import measure_rings


def polygon_area(points: npt.NDArray[np.float32]) -> float:
    """Polygon area in px² (shoelace formula)."""
    return _polygon_area(points)


def polygon_perimeter(points: npt.NDArray[np.float32]) -> float:
    """Closed polygon perimeter in px."""
    return _polygon_perimeter(points)


def compute_ring_properties(
    rings: list[npt.NDArray[np.float32]],
    pith_xy: tuple[float, float],
    outer_year: int | None = None,
) -> list[dict[str, Any]]:
    """Ring properties in pixels as dicts, innermost first.

    Keys: ``label``, ``year``, ``area``, ``cumulative_area``, ``perimeter``,
    ``distance_from_pith`` (mean radius), ``mean_width``, ``median_width``, ``min_width``,
    ``max_width``, ``eccentricity``. See ``tras.analysis.rings.RingMeasurement``.
    """
    rows = []
    for m in measure_rings(rings, pith_xy, outer_year=outer_year):
        row = m.scaled(1.0)
        row["distance_from_pith"] = m.mean_radius
        rows.append(row)
    return rows


# (column title, dict key, physical dimension: 1 = length, 2 = area)
_CSV_COLUMNS = [
    ("Area", "area", 2),
    ("Cumulative Area", "cumulative_area", 2),
    ("Perimeter", "perimeter", 1),
    ("Distance from Pith", "distance_from_pith", 1),
    ("Mean Width", "mean_width", 1),
    ("Median Width", "median_width", 1),
    ("Min Width", "min_width", 1),
    ("Max Width", "max_width", 1),
    ("Eccentricity", "eccentricity", 1),
]


def write_ring_properties_csv(
    output_path: Path,
    ring_properties: list[dict[str, Any]],
    scale_value: float,
    scale_unit: str,
) -> None:
    """Write ring properties to CSV in physical units, followed by the pixel values.

    Args:
        output_path: Destination CSV path.
        ring_properties: Rows from ``compute_ring_properties``.
        scale_value: Physical size of one pixel (``scale_unit`` per px).
        scale_unit: Physical unit name, e.g. ``mm``.
    """
    if not ring_properties:
        logger.warning(f"No ring properties to write to {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [c for c in _CSV_COLUMNS if c[1] in ring_properties[0]]
    header = ["Ring", "Year"]
    header += [f"{title} ({scale_unit}{'²' if dim == 2 else ''})" for title, _, dim in columns]
    header += [f"{title} (px{'²' if dim == 2 else ''})" for title, _, dim in columns]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for prop in ring_properties:
            year = prop.get("year")
            row = [prop["label"], "" if year is None else year]
            row += [f"{prop[key] * scale_value**dim:.4f}" for _, key, dim in columns]
            row += [f"{prop[key]:.2f}" for _, key, _ in columns]
            writer.writerow(row)

    logger.info(f"Wrote ring properties CSV to {output_path}")
