"""Ray-based resampling of ring polygons to a fixed number of points.

Every ring is resampled at the same ``sampling_nr`` angles around the pith, so point ``k``
of every ring lies on the same ray. This makes per-direction radial widths a simple
difference between consecutive rings (see ``tras.analysis.rings``).
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from tras.analysis.geometry import fill_missing_by_angle
from tras.analysis.geometry import points_at_distances
from tras.analysis.geometry import ray_angles
from tras.analysis.geometry import ray_polygon_distances


def build_rays_xy(
    sampling_nr: int, center_xy: tuple[float, float], max_radius: float
) -> list[LineString]:
    """Rays from ``center_xy`` at evenly spaced angles, each ``max_radius`` long."""
    ends = points_at_distances(
        center_xy, ray_angles(sampling_nr), np.full(sampling_nr, max_radius)
    )
    return [LineString([center_xy, (float(x), float(y))]) for x, y in ends]


def ray_radii(
    ring_xy: np.ndarray, center_xy: tuple[float, float], sampling_nr: int
) -> np.ndarray:
    """Distance from the center to the ring along each of ``sampling_nr`` rays.

    Takes the farthest crossing per ray; rays that miss the ring (center outside it) use
    the ring vertex closest in angle.
    """
    angles = ray_angles(sampling_nr)
    distances = ray_polygon_distances(ring_xy, center_xy, angles)
    return fill_missing_by_angle(distances, ring_xy, center_xy, angles)


def resample_ring_by_rays(
    ring_xy: np.ndarray, center_xy: tuple[float, float], sampling_nr: int
) -> np.ndarray:
    """Resample one ring polygon to exactly ``sampling_nr`` points ordered by angle.

    Rings with fewer than 3 points are returned unchanged.
    """
    ring_xy = np.asarray(ring_xy)
    if len(ring_xy) < 3:
        return ring_xy
    radii = ray_radii(ring_xy, center_xy, sampling_nr)
    points = points_at_distances(center_xy, ray_angles(sampling_nr), radii)
    return points.astype(np.float32)


def resample_rings_by_rays(
    rings: list[np.ndarray], center_xy: tuple[float, float], sampling_nr: int
) -> list[np.ndarray]:
    """Resample every ring to ``sampling_nr`` points (see ``resample_ring_by_rays``)."""
    return [resample_ring_by_rays(ring, center_xy, sampling_nr) for ring in rings]
