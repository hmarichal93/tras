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
    offsets: FloatArray  # across-path positions in samples (× path.step = px)
    path: SampledPath

    @property
    def s(self) -> FloatArray:
        return np.arange(self.values.shape[1]) * self.path.step


def sample_path(
    polyline: npt.ArrayLike, step: float = 1.0, smoothing_px: float = 0.0
) -> SampledPath:
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
    """Sample ``image`` on a ``(2·half_width+1) × n_samples`` grid along ``polyline``.

    ``step`` is the grid spacing in pixels, both along and across the path; ``half_width``
    and the ribbon ``offsets`` are in samples. With ``step > 1`` the image is low-pass
    filtered first so the coarser grid does not alias fine texture.
    Works on grayscale or multi-channel images; values are bilinear-interpolated floats.
    """
    path = sample_path(polyline, step=step, smoothing_px=_TANGENT_SMOOTHING * half_width * step)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    grid = path.xy[None, :, :] + (offsets * step)[:, None, None] * path.normal[None, :, :]
    source, origin = _source_for_step(np.asarray(image, dtype=np.float32), grid, step)
    map_x = (grid[..., 0] - origin[0]).astype(np.float32)
    map_y = (grid[..., 1] - origin[1]).astype(np.float32)

    values = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    h, w = image.shape[:2]
    gx, gy = grid[..., 0], grid[..., 1]
    valid = (gx >= 0) & (gx <= w - 1) & (gy >= 0) & (gy <= h - 1)
    return Ribbon(values=values.astype(np.float64), valid=valid, offsets=offsets, path=path)


# Anti-aliasing blur (sigma, in grid steps) applied before sampling on a coarse grid.
_ANTI_ALIAS_SIGMA = 0.4


def _source_for_step(
    image: npt.NDArray[np.float32], grid: FloatArray, step: float
) -> tuple[npt.NDArray[np.float32], tuple[int, int]]:
    """The image (cropped to the grid and low-passed when ``step > 1``) and its origin."""
    if step <= 1.0:
        return image, (0, 0)
    sigma = _ANTI_ALIAS_SIGMA * step
    margin = int(np.ceil(3 * sigma)) + 2
    h, w = image.shape[:2]
    x0 = max(int(np.floor(grid[..., 0].min())) - margin, 0)
    y0 = max(int(np.floor(grid[..., 1].min())) - margin, 0)
    x1 = min(int(np.ceil(grid[..., 0].max())) + margin + 1, w)
    y1 = min(int(np.ceil(grid[..., 1].max())) + margin + 1, h)
    if x1 <= x0 or y1 <= y0:
        return image, (0, 0)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    return cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma), (x0, y0)
