"""Bridge between GUI canvas shapes and ``tras.analysis`` ring measurements.

Canvas shapes hold ``QPointF`` points; analysis functions take ``(N, 2)`` arrays. This module
converts between them (by duck typing, so it needs no Qt import) and is the only place the
GUI turns shapes into ring measurements. Ring polygons and exclusion zones are told apart by
the ``exclusion`` label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from tras.analysis.geometry import polygon_centroid
from tras.analysis.rings import measure_rings
from tras.analysis.rings import order_rings_by_area

EXCLUSION_LABEL = "exclusion"
MIN_POLYGON_POINTS = 3


def shape_points(shape: Any) -> npt.NDArray[np.float64]:
    """``(N, 2)`` x/y array of a canvas shape's points."""
    return np.array([(p.x(), p.y()) for p in shape.points], dtype=np.float64).reshape(-1, 2)


def split_ring_shapes(shapes: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """``(ring polygons, exclusion polygons)`` among canvas shapes."""
    polygons = [
        s
        for s in shapes
        if getattr(s, "shape_type", "") == "polygon" and len(s.points) >= MIN_POLYGON_POINTS
    ]
    rings = [s for s in polygons if s.label != EXCLUSION_LABEL]
    exclusions = [s for s in polygons if s.label == EXCLUSION_LABEL]
    return rings, exclusions


def rings_inner_to_outer(ring_shapes: Sequence[Any]) -> list[Any]:
    """Ring shapes sorted by enclosed area, innermost first."""
    arrays = [shape_points(s) for s in ring_shapes]
    return [ring_shapes[i] for i in order_rings_by_area(arrays)]


def ring_property_rows(
    ring_shapes: Sequence[Any],
    exclusion_shapes: Sequence[Any] = (),
    pith_xy: tuple[float, float] | None = None,
    radial_measurements: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per-ring property dicts (px units) for the GUI dialogs and exports, innermost first.

    Keys: ``label``, ``area``, ``cumulative_area``, ``perimeter``, ``mean_width``,
    ``median_width``, ``min_width``, ``max_width``, ``eccentricity``, ``mean_radius``,
    ``radial_width_px`` (from a radial-line measurement, else None) and, when exclusion zones
    exist, ``area_no_exclusion`` / ``cumulative_area_no_exclusion``.
    """
    if not ring_shapes:
        return []
    rings = [shape_points(s) for s in ring_shapes]
    labels = [s.label for s in ring_shapes]
    pith = pith_xy if pith_xy is not None else _innermost_centroid(rings)
    exclusions = [shape_points(s) for s in exclusion_shapes]

    measured = measure_rings(rings, pith, labels=labels, exclusions=exclusions)
    unexcluded = measure_rings(rings, pith, labels=labels) if exclusions else measured
    radial = radial_measurements or {}

    rows = []
    for m, plain in zip(measured, unexcluded):
        row = m.scaled(1.0)
        row["radial_width_px"] = radial.get(m.label, {}).get("radial_width")
        if exclusions:
            row["area_no_exclusion"] = plain.area
            row["cumulative_area_no_exclusion"] = plain.cumulative_area
        rows.append(row)
    return rows


def _innermost_centroid(rings: Sequence[npt.NDArray[np.float64]]) -> tuple[float, float]:
    """Pith fallback: centroid of the smallest ring."""
    return polygon_centroid(rings[order_rings_by_area(rings)[0]])
