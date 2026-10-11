"""Straighten an image along a measurement path into a *ribbon*.

A ribbon is the image resampled on a curvilinear grid: column ``j`` is the point at arc
length ``s_j`` along the path, row ``i`` is offset ``o_i`` along the path normal.
Ring boundaries crossing the path become (roughly) vertical lines in the ribbon, so
boundary detection reduces to 1-D signal processing along ``s``.

Coordinates: ``xy = path(s) + o · normal(s)``, with the normal the tangent rotated
+90° in image coordinates (x right, y down).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

FloatArray = npt.NDArray[np.float64]

# Tangent smoothing as a fraction of the ribbon half-width: corners in a hand-drawn
# polyline must not make the sampling grid fold over itself.
_TANGENT_SMOOTHING = 0.5


@dataclass(frozen=True)
class SampledPath:
    """A path resampled at a constant arc-length step (default 1 px)."""

    xy: FloatArray  # (M, 2) points
    tangent: FloatArray  # (M, 2) unit tangents
    normal: FloatArray  # (M, 2) unit normals
    step: float

    @property
    def length(self) -> float:
        return self.step * (len(self.xy) - 1)

    def point_at(self, s: float | FloatArray, offset: float | FloatArray = 0.0) -> FloatArray:
        """Image xy of arc length ``s`` (px) and normal ``offset`` (px)."""
        idx = np.asarray(s, dtype=np.float64) / self.step
        x = np.interp(idx, np.arange(len(self.xy)), self.xy[:, 0])
        y = np.interp(idx, np.arange(len(self.xy)), self.xy[:, 1])
        nx = np.interp(idx, np.arange(len(self.xy)), self.normal[:, 0])
        ny = np.interp(idx, np.arange(len(self.xy)), self.normal[:, 1])
        return np.stack([x + offset * nx, y + offset * ny], axis=-1)

    def reversed(self) -> SampledPath:
        return SampledPath(
            xy=self.xy[::-1].copy(),
            tangent=-self.tangent[::-1],
            normal=-self.normal[::-1],
            step=self.step,
        )


@dataclass(frozen=True)
class Ribbon:
    """Straightened image strip; ``values[i, j]`` at offset ``offsets[i]``, arc ``s[j]``."""

    values: FloatArray  # (n_offsets, n_s) or (n_offsets, n_s, channels)
    valid: npt.NDArray[np.bool_]  # (n_offsets, n_s) inside the image
    offsets: FloatArray
    path: SampledPath

    @property
    def s(self) -> FloatArray:
        return np.arange(self.values.shape[1]) * self.path.step


def sample_path(polyline: npt.ArrayLike, step: float = 1.0, smoothing_px: float = 0.0) -> SampledPath:
    """Resample a polyline at constant arc-length ``step`` and compute its frame.

    Args:
        polyline: ``(N, 2)`` points, N ≥ 2.
        step: Arc-length spacing of the output points (px).
        smoothing_px: Gaussian sigma (px of arc length) applied to the tangents.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]  # drop repeats
    if len(pts) < 2:
        raise ValueError("A path needs at least two distinct points.")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.arange(0.0, arc[-1] + 1e-9, step)
    xy = np.stack([np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])], axis=1)

    tangent = np.gradient(xy, axis=0) if len(xy) > 1 else np.array([[1.0, 0.0]])
    if smoothing_px > 0 and len(xy) > 2:
        tangent = gaussian_filter1d(tangent, sigma=smoothing_px / step, axis=0, mode="nearest")
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return SampledPath(xy=xy, tangent=tangent, normal=normal, step=step)


def extract_ribbon(
    image: npt.NDArray, polyline: npt.ArrayLike, half_width: int, step: float = 1.0
) -> Ribbon:
    """Sample ``image`` on a ``(2·half_width+1) × path_length`` grid along ``polyline``.

    Works on grayscale or multi-channel images; values are bilinear-interpolated floats.
    """
    path = sample_path(polyline, step=step, smoothing_px=_TANGENT_SMOOTHING * half_width)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    grid = path.xy[None, :, :] + offsets[:, None, None] * path.normal[None, :, :]
    map_x = grid[..., 0].astype(np.float32)
    map_y = grid[..., 1].astype(np.float32)

    source = np.asarray(image, dtype=np.float32)
    values = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    h, w = source.shape[:2]
    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    return Ribbon(values=values.astype(np.float64), valid=valid, offsets=offsets, path=path)
