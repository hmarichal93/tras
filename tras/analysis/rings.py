"""Ring measurements for cross-sections, shared by the GUI, CLI, batch runner and reports.

A *ring boundary* is a closed polygon around the pith. Boundaries are ordered by enclosed
area (innermost first); ring ``k`` is the band between boundary ``k-1`` (or the pith) and
boundary ``k``.

Widths are measured on ``n_rays`` rays from the pith: on each ray the width of ring ``k``
is ``r_k(θ) - r_{k-1}(θ)``. Their mean, median, min and max describe how even the growth
was around the stem. Optional *exclusion zones* (cracks, knots, rot) are subtracted from
areas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tras.analysis.geometry import fill_missing_by_angle
from tras.analysis.geometry import polygon_area
from tras.analysis.geometry import polygon_centroid
from tras.analysis.geometry import polygon_perimeter
from tras.analysis.geometry import ray_angles
from tras.analysis.geometry import ray_polygon_distances
from tras.io.tucson import RingWidthSeries

DEFAULT_N_RAYS = 360


@dataclass(frozen=True)
class RingMeasurement:
    """Measurements of one annual ring, in pixels (use ``scaled`` for physical units)."""

    index: int  # 1 = innermost ring
    label: str
    year: int | None
    area: float  # area of this ring's band (growth of the year)
    cumulative_area: float  # area enclosed by the outer boundary
    perimeter: float  # outer boundary length
    mean_radius: float  # mean pith-to-boundary distance over all rays
    mean_width: float
    median_width: float
    min_width: float
    max_width: float
    eccentricity: float  # distance from pith to the outer boundary's centroid
    excluded_area: float = 0.0  # cumulative area removed by exclusion zones

    def scaled(self, unit_per_px: float) -> dict[str, Any]:
        """Return a dict with lengths × scale and areas × scale²."""
        row = asdict(self)
        for key in _LENGTH_FIELDS:
            row[key] = row[key] * unit_per_px
        for key in _AREA_FIELDS:
            row[key] = row[key] * unit_per_px**2
        return row


_LENGTH_FIELDS = (
    "perimeter",
    "mean_radius",
    "mean_width",
    "median_width",
    "min_width",
    "max_width",
    "eccentricity",
)
_AREA_FIELDS = ("area", "cumulative_area", "excluded_area")


def order_rings_by_area(rings: Sequence[npt.ArrayLike]) -> list[int]:
    """Indices of ``rings`` sorted innermost (smallest area) to outermost."""
    return sorted(range(len(rings)), key=lambda i: polygon_area(rings[i]))


def radial_profiles(
    rings: Sequence[npt.ArrayLike], pith_xy: tuple[float, float], n_rays: int
) -> npt.NDArray[np.float64]:
    """``(n_rings, n_rays)`` pith-to-boundary distances, in the order given."""
    angles = ray_angles(n_rays)
    profiles = [
        fill_missing_by_angle(ray_polygon_distances(ring, pith_xy, angles), ring, pith_xy, angles)
        for ring in rings
    ]
    return np.array(profiles, dtype=np.float64).reshape(len(profiles), n_rays)


def radial_widths(profiles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-ray widths from ordered radial profiles; the first ring starts at the pith.

    Negative values mean two boundaries cross on that ray (see ``tras.analysis.qc``).
    """
    inner = np.vstack([np.zeros((1, profiles.shape[1])), profiles[:-1]])
    return profiles - inner


def measure_rings(
    rings: Sequence[npt.ArrayLike],
    pith_xy: tuple[float, float],
    *,
    labels: Sequence[str] | None = None,
    outer_year: int | None = None,
    exclusions: Sequence[npt.ArrayLike] = (),
    n_rays: int = DEFAULT_N_RAYS,
) -> list[RingMeasurement]:
    """Measure every ring of a cross-section.

    Args:
        rings: Ring boundaries as ``(N, 2)`` x/y arrays, in any order.
        pith_xy: Pith position.
        labels: Optional label per input ring (default ``ring_<k>`` after ordering).
        outer_year: Calendar year of the outermost ring; years count back inward.
        exclusions: Polygons to subtract from areas (cracks, knots, rot).
        n_rays: Number of rays used for radial widths.

    Returns:
        One ``RingMeasurement`` per ring with ≥ 3 points, innermost first.
    """
    valid = [i for i, ring in enumerate(rings) if len(np.asarray(ring)) >= 3]
    order = [valid[i] for i in order_rings_by_area([rings[j] for j in valid])]
    ordered = [np.asarray(rings[i], dtype=np.float64) for i in order]
    if not ordered:
        return []

    profiles = radial_profiles(ordered, pith_xy, n_rays)
    widths = radial_widths(profiles)
    excluded = _excluded_areas(ordered, exclusions)
    n_rings = len(ordered)

    measurements = []
    previous_area = 0.0
    for k, ring in enumerate(ordered):
        cumulative = polygon_area(ring) - excluded[k]
        centroid = polygon_centroid(ring)
        measurements.append(
            RingMeasurement(
                index=k + 1,
                label=labels[order[k]] if labels is not None else f"ring_{k + 1}",
                year=None if outer_year is None else outer_year - (n_rings - 1 - k),
                area=max(cumulative - previous_area, 0.0),
                cumulative_area=cumulative,
                perimeter=polygon_perimeter(ring),
                mean_radius=float(profiles[k].mean()),
                mean_width=float(widths[k].mean()),
                median_width=float(np.median(widths[k])),
                min_width=float(widths[k].min()),
                max_width=float(widths[k].max()),
                eccentricity=float(np.hypot(centroid[0] - pith_xy[0], centroid[1] - pith_xy[1])),
                excluded_area=excluded[k],
            )
        )
        previous_area = cumulative
    return measurements


def _excluded_areas(
    ordered_rings: Sequence[npt.NDArray[np.float64]], exclusions: Sequence[npt.ArrayLike]
) -> list[float]:
    """Area of the exclusion zones inside each ring boundary."""
    if not exclusions:
        return [0.0] * len(ordered_rings)

    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    zones = [_valid_polygon(Polygon(z)) for z in exclusions if len(np.asarray(z)) >= 3]
    union = unary_union(zones)
    return [float(_valid_polygon(Polygon(r)).intersection(union).area) for r in ordered_rings]


def _valid_polygon(polygon: Any) -> Any:
    """Repair self-intersecting polygons (``buffer(0)`` is shapely's standard fix)."""
    return polygon if polygon.is_valid else polygon.buffer(0)


def radial_width_series(
    rings: Sequence[npt.ArrayLike],
    pith_xy: tuple[float, float],
    directions_deg: Sequence[float],
    outer_year: int,
    unit_per_px: float = 1.0,
    name: str = "R",
) -> list[RingWidthSeries]:
    """One ring-width series per direction (degrees, image coordinates: 0 = +x, 90 = +y).

    The first ring is measured from the pith. Lets a cross-section be crossdated against
    itself, like several cores from one tree.
    """
    ordered = [np.asarray(rings[i], dtype=np.float64) for i in order_rings_by_area(rings)]
    angles = np.deg2rad(np.asarray(directions_deg, dtype=np.float64)) % (2 * np.pi)
    profiles = np.array(
        [
            fill_missing_by_angle(ray_polygon_distances(r, pith_xy, angles), r, pith_xy, angles)
            for r in ordered
        ]
    )
    widths = radial_widths(profiles) * unit_per_px  # (n_rings, n_directions)
    first_year = outer_year - len(ordered) + 1
    return [
        RingWidthSeries(
            f"{name}{int(round(d)):03d}", first_year, tuple(float(w) for w in widths[:, k])
        )
        for k, d in enumerate(directions_deg)
    ]


# Compass directions in image coordinates (x right, y down; image top = north).
COMPASS_DEGREES = {"N": 270.0, "E": 0.0, "S": 90.0, "W": 180.0}


def compass_series(
    rings: Sequence[npt.ArrayLike],
    pith_xy: tuple[float, float],
    outer_year: int,
    mm_per_px: float,
    sample_name: str,
    directions: str = "NESW",
) -> list[RingWidthSeries]:
    """Ring-width series (mm) along compass radii, IDs ``<sample[:7]><letter>``.

    The image top is taken as north, as when discs are photographed with the north mark up.
    """
    from tras.io.tucson import sanitize_series_id

    degrees = [COMPASS_DEGREES[d] for d in directions]
    series = radial_width_series(rings, pith_xy, degrees, outer_year, unit_per_px=mm_per_px)
    stem = sanitize_series_id(sample_name)[:7]
    return [
        RingWidthSeries(f"{stem}{letter}", s.first_year, s.widths_mm)
        for letter, s in zip(directions, series)
    ]
