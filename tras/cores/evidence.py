"""Boundary evidence along a ribbon: how likely each arc position is a ring boundary.

Classical evidence (no model needed):

1. Intensity per lane (a lane is one ribbon row, parallel to the path).
2. Flatten slow illumination and wood-colour changes: subtract a wide moving average and
   divide by a moving standard deviation along the path.
3. Signed edge response: Gaussian derivative along the path.
4. **Tilt search.** Boundaries rarely cross the path at exactly 90°. For a tilt ``φ`` a
   boundary through ``s`` meets lane offset ``o`` at ``s + o·tan φ``. For each candidate
   tilt the lanes are sheared accordingly and combined with a *median*, so a crack or
   resin streak in a few lanes cannot create or hide a boundary. The best tilt per
   position is kept: it also gives the boundary orientation used for perpendicular widths.

Polarity: in conifers the year boundary is the sharp step from dark latewood to light
earlywood, walking from pith to bark. The derivative sign of that step tells which path
end is the pith (see ``choose_polarity``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.ndimage import uniform_filter1d

from tras.cores.ribbon import Ribbon

FloatArray = npt.NDArray[np.float64]

# Luminance weights (ITU-R BT.601) for RGB ribbons.
_LUMA = np.array([0.299, 0.587, 0.114])
_EPS = 1e-6


@dataclass(frozen=True)
class Evidence:
    """Per-position boundary score and best tilt (radians) for one polarity."""

    score: FloatArray  # (n_s,)
    tilt: FloatArray  # (n_s,) angle between boundary and path normal


def ribbon_intensity(ribbon: Ribbon, channel: str = "gray") -> FloatArray:
    """Single-channel ribbon. ``channel``: gray, red, green, blue or min."""
    values = ribbon.values
    if values.ndim == 2:
        return values
    if channel == "gray":
        return values[..., :3] @ _LUMA
    if channel == "min":
        return values[..., :3].min(axis=-1)
    index = {"red": 0, "green": 1, "blue": 2}[channel]
    return values[..., index]


def normalize_lanes(
    intensity: FloatArray, background_px: float, contrast_floor: float = 0.5
) -> FloatArray:
    """Remove slow trends along the path and equalize local contrast, lane by lane.

    The local standard deviation is floored at ``contrast_floor`` × its median over the
    ribbon, so faint, featureless wood (e.g. juvenile wood near the pith) is not
    amplified into noise boundaries.
    """
    size = max(3, int(round(background_px)) | 1)
    mean = uniform_filter1d(intensity, size=size, axis=1, mode="reflect")
    centered = intensity - mean
    std = np.sqrt(np.maximum(uniform_filter1d(centered**2, size=size, axis=1, mode="reflect"), 0.0))
    floor = contrast_floor * float(np.median(std))
    return centered / (np.maximum(std, floor) + _EPS)


def edge_response(normalized: FloatArray, sigma_px: float) -> FloatArray:
    """Gaussian derivative along the path (positive = getting brighter with ``s``)."""
    return gaussian_filter1d(normalized, sigma=sigma_px, axis=1, order=1, mode="reflect") * sigma_px


def tilt_search(
    response: FloatArray,
    offsets: FloatArray,
    valid: npt.NDArray[np.bool_],
    max_tilt_deg: float,
    n_tilts: int,
) -> Evidence:
    """Best lane-median response over candidate boundary tilts, per position.

    The tilt is refined between grid steps by fitting a parabola through the best score
    and its two neighbours.
    """
    tilts = np.deg2rad(np.linspace(-max_tilt_deg, max_tilt_deg, n_tilts)) if n_tilts > 1 else np.zeros(1)
    scores = np.stack([_sheared_median(response, offsets, valid, t) for t in tilts])  # (T, S)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    best = scores.argmax(axis=0)
    columns = np.arange(scores.shape[1])
    score = scores[best, columns]
    tilt = tilts[best] + _parabolic_offset(scores, best) * (tilts[1] - tilts[0] if n_tilts > 1 else 0.0)
    score[~np.isfinite(score)] = 0.0
    return Evidence(score=score, tilt=tilt)


def _sheared_median(
    response: FloatArray, offsets: FloatArray, valid: npt.NDArray[np.bool_], tilt: float
) -> FloatArray:
    """Median over lanes after shifting lane ``o`` by ``o·tan(tilt)`` along the path."""
    n_lanes, n_s = response.shape
    masked = np.where(valid, response, np.nan)
    rows = np.repeat(np.arange(n_lanes, dtype=np.float64)[:, None], n_s, axis=1)
    cols = np.arange(n_s, dtype=np.float64)[None, :] + offsets[:, None] * np.tan(tilt)
    sheared = map_coordinates(masked, [rows, cols], order=1, mode="nearest")
    return np.nanmedian(sheared, axis=0)


def _parabolic_offset(scores: FloatArray, best: npt.NDArray[np.intp]) -> FloatArray:
    """Sub-step position of each column's maximum, in grid steps (−0.5 … 0.5)."""
    n_tilts = scores.shape[0]
    columns = np.arange(scores.shape[1])
    inner = (best > 0) & (best < n_tilts - 1)
    left = scores[np.clip(best - 1, 0, n_tilts - 1), columns]
    centre = scores[best, columns]
    right = scores[np.clip(best + 1, 0, n_tilts - 1), columns]
    curvature = left - 2 * centre + right
    ok = inner & np.isfinite(left) & np.isfinite(right) & (curvature < 0)
    offset = np.zeros(scores.shape[1])
    offset[ok] = 0.5 * (left[ok] - right[ok]) / curvature[ok]
    return np.clip(offset, -0.5, 0.5)


def peak_strength(score: FloatArray, min_distance_px: int, top: int = 20) -> float:
    """Mean of the ``top`` strongest separated peaks: how 'boundary-like' a signal is."""
    from scipy.signal import find_peaks

    peaks, props = find_peaks(score, distance=max(1, min_distance_px), height=0)
    if len(peaks) == 0:
        return 0.0
    heights = np.sort(props["peak_heights"])[::-1][:top]
    return float(heights.mean())


def choose_polarity(positive: Evidence, negative: Evidence, min_distance_px: int) -> int:
    """+1 if bright-going steps (along +s) are the sharper boundaries, else −1."""
    strength_pos = peak_strength(positive.score, min_distance_px)
    strength_neg = peak_strength(negative.score, min_distance_px)
    return 1 if strength_pos >= strength_neg else -1
