"""Synthetic conifer-like core images with known boundaries (tests, demos, docs)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

DEFAULT_BOUNDARIES = (40.0, 95.0, 140.0, 200.0, 245.0, 280.0, 330.0, 370.0, 400.0, 445.0)
# Earlywood brightness and how much latewood darkens it (0–255 grey levels).
EARLYWOOD_LEVEL = 210
LATEWOOD_DROP = 120


def synthetic_core(
    length: int = 500,
    height: int = 40,
    boundaries: npt.ArrayLike = DEFAULT_BOUNDARIES,
    tilt_deg: float = 0.0,
    flip: bool = False,
    noise_sigma: float = 3.0,
    seed: int = 0,
) -> npt.NDArray[np.uint8]:
    """RGB core along +x: light earlywood darkening to latewood, then a sharp boundary.

    The pith is on the left (x = 0) unless ``flip``. Boundaries are tilted ``tilt_deg``
    from vertical.
    """
    x = np.arange(length, dtype=np.float64)[None, :]
    y = np.arange(height, dtype=np.float64)[:, None] - height / 2
    shifted = x - y * np.tan(np.deg2rad(tilt_deg))
    edges = np.r_[0.0, np.asarray(boundaries, dtype=np.float64), length * 2]
    idx = np.clip(np.searchsorted(edges, shifted, side="right") - 1, 0, len(edges) - 2)
    start, end = edges[idx], edges[idx + 1]
    phase = np.clip((shifted - start) / (end - start), 0, 1)
    intensity = EARLYWOOD_LEVEL - LATEWOOD_DROP * phase**3
    intensity = intensity + np.random.default_rng(seed).normal(0, noise_sigma, intensity.shape)
    image = np.repeat(np.clip(intensity, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    return image[:, ::-1] if flip else image


def centre_path(image: npt.NDArray, margin: int = 0) -> npt.NDArray[np.float64]:
    """Horizontal path along the middle row of ``image``."""
    h, w = image.shape[:2]
    return np.array([[margin, h / 2], [w - 1 - margin, h / 2]], dtype=np.float64)
