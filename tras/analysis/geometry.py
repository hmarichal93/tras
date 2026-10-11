"""Vectorized polygon and ray geometry for ring analysis.

All functions take ``(N, 2)`` float arrays of ``(x, y)`` points (image coordinates, y down)
and treat a polygon as implicitly closed (last point connects to the first).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Determinants below this are treated as parallel ray/edge pairs.
_PARALLEL_EPS = 1e-12


def polygon_area(points: npt.ArrayLike) -> float:
    """Unsigned polygon area (shoelace formula). Fewer than 3 points → 0."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_perimeter(points: npt.ArrayLike) -> float:
    """Closed polygon perimeter. Fewer than 2 points → 0."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum())


def polygon_centroid(points: npt.ArrayLike) -> tuple[float, float]:
    """Area centroid of a simple polygon (vertex mean if degenerate)."""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    signed_area = cross.sum() / 2.0
    if abs(signed_area) < _PARALLEL_EPS:
        return float(x.mean()), float(y.mean())
    cx = ((x + x1) * cross).sum() / (6.0 * signed_area)
    cy = ((y + y1) * cross).sum() / (6.0 * signed_area)
    return float(cx), float(cy)


def ray_angles(n_rays: int) -> FloatArray:
    """``n_rays`` evenly spaced angles in radians, starting at 0 (pointing +x)."""
    return np.arange(n_rays, dtype=np.float64) * (2.0 * np.pi / n_rays)


def ray_polygon_distances(
    polygon: npt.ArrayLike, center_xy: tuple[float, float], angles: FloatArray
) -> FloatArray:
    """Distance from ``center_xy`` to the polygon boundary along each ray.

    For every angle, intersect the ray ``center + t·(cos θ, sin θ), t ≥ 0`` with all polygon
    edges and keep the **farthest** hit, i.e. the outer envelope of the ring seen from the
    pith. Rays that miss the polygon (pith outside a ring) get ``NaN``.

    Returns:
        ``(len(angles),)`` array of distances in pixels.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    starts = pts - center  # edge start relative to the center, (N, 2)
    edges = np.roll(pts, -1, axis=0) - pts  # (N, 2)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (K, 2)

    # Solve t·d = start + s·edge  ⇒  t = cross(start, edge)/cross(d, edge),
    #                                s = cross(start, d)/cross(d, edge).
    denom = _cross(dirs[:, None, :], edges[None, :, :])  # (K, N)
    parallel = np.abs(denom) < _PARALLEL_EPS
    safe = np.where(parallel, 1.0, denom)
    t = _cross(starts[None, :, :], edges[None, :, :]) / safe
    s = _cross(starts[None, :, :], dirs[:, None, :]) / safe
    valid = ~parallel & (t >= 0.0) & (s >= 0.0) & (s <= 1.0)

    t = np.where(valid, t, -np.inf)
    farthest = t.max(axis=1)
    return np.where(np.isfinite(farthest), farthest, np.nan)


def points_at_distances(
    center_xy: tuple[float, float], angles: FloatArray, distances: FloatArray
) -> FloatArray:
    """Convert polar samples around ``center_xy`` back to ``(K, 2)`` x/y points."""
    cx, cy = center_xy
    return np.stack(
        [cx + distances * np.cos(angles), cy + distances * np.sin(angles)], axis=1
    )


def fill_missing_by_angle(
    distances: FloatArray, polygon: npt.ArrayLike, center_xy: tuple[float, float],
    angles: FloatArray,
) -> FloatArray:
    """Replace ``NaN`` distances with the polygon vertex closest in angle.

    Used when the pith lies outside (or on) a ring, so some rays miss it.
    """
    missing = np.isnan(distances)
    if not missing.any():
        return distances
    pts = np.asarray(polygon, dtype=np.float64)
    rel = pts - np.asarray(center_xy, dtype=np.float64)
    vertex_angles = np.arctan2(rel[:, 1], rel[:, 0]) % (2.0 * np.pi)
    vertex_dist = np.hypot(rel[:, 0], rel[:, 1])
    gap = np.abs(angles[missing, None] - vertex_angles[None, :])
    gap = np.minimum(gap, 2.0 * np.pi - gap)
    filled = distances.copy()
    filled[missing] = vertex_dist[gap.argmin(axis=1)]
    return filled


def _cross(a: FloatArray, b: FloatArray) -> FloatArray:
    """2-D cross product (z component), broadcasting over leading axes."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
