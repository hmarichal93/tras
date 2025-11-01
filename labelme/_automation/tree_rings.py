from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger
from skimage import color, filters, transform


@dataclass
class RingDetectParams:
    # Number of angular samples [columns] in the polar image
    angular_steps: int = 720
    # Minimum radius to consider (ignore the pith neighborhood)
    min_radius: float = 5.0
    # Maximum radius to consider; if None, computed from image and center
    max_radius: float | None = None
    # Smoothing in the radial and angular directions before peak search
    smooth_sigma_radial: float = 1.0
    smooth_sigma_theta: float = 1.0
    # Relative threshold for peak picking on the radial gradient (0-1)
    relative_threshold: float = 0.3
    # Minimum distance in pixels between adjacent peaks per angle
    min_peak_distance: int = 3
    # Minimum fraction of valid angles required to keep a ring
    min_coverage: float = 0.6
    # Optional maximum number of rings to keep (closest first)
    max_rings: int | None = None


def _to_gray_float01(image: npt.NDArray[np.uint8 | np.float32 | np.float64]) -> npt.NDArray[np.float32]:
    if image.ndim == 3:
        # Assume RGB or RGBA
        if image.shape[2] == 4:
            image = image[..., :3]
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32)
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    # Normalize to [0,1]
    vmin, vmax = float(gray.min()), float(gray.max())
    if vmax > vmin:
        gray = (gray - vmin) / (vmax - vmin)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)
    return gray


def _compute_polar(
    gray: npt.NDArray[np.float32],
    center_xy: Tuple[float, float],
    angular_steps: int,
    max_radius: float | None,
) -> Tuple[npt.NDArray[np.float32], float]:
    h, w = gray.shape[:2]
    cx, cy = center_xy
    # skimage uses (row=y, col=x) for center
    if max_radius is None:
        max_radius = float(
            np.clip([cx, w - 1 - cx, cy, h - 1 - cy], 0, None).min()
        )
    radius = int(max(1, np.floor(max_radius)))
    polar = transform.warp_polar(
        gray,
        center=(cy, cx),
        radius=radius,
        output_shape=(radius, int(angular_steps)),
        scaling="linear",
        order=1,
        mode="edge",
        preserve_range=True,
    ).astype(np.float32)
    return polar, float(radius)


def _local_maxima_1d(
    signal: npt.NDArray[np.float32],
    threshold: float,
    min_dist: int,
    r_min: int,
    r_max: int,
) -> list[int]:
    # Simple 1D peak detector: strictly larger than neighbors and above threshold
    n = signal.shape[0]
    peaks: list[int] = []
    for i in range(max(1, r_min), min(n - 1, r_max)):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
    if not peaks:
        return []
    # Enforce minimum distance: keep strongest peaks
    peaks_sorted = sorted(peaks, key=lambda i: signal[i], reverse=True)
    selected: list[int] = []
    for idx in peaks_sorted:
        if all(abs(idx - s) >= min_dist for s in selected):
            selected.append(idx)
    return sorted(selected)


def detect_tree_rings(
    image: npt.NDArray[np.floating | np.integer],
    center_xy: Tuple[float, float],
    params: RingDetectParams | None = None,
) -> list[npt.NDArray[np.float32]]:
    """
    Detect tree rings in a wood cross-section image.

    Args:
        image: HxW or HxWx{3,4} array; any dtype. Will be converted to grayscale [0,1].
        center_xy: (x, y) pith center in pixel coordinates (image frame).
        params: Tweaks for detection. See RingDetectParams.

    Returns:
        A list of rings, each as Nx2 array of (x, y) points forming a closed polyline.
        Rings are ordered from inner to outer.
    """
    if params is None:
        params = RingDetectParams()

    gray = _to_gray_float01(image)
    polar, radius = _compute_polar(
        gray=gray,
        center_xy=center_xy,
        angular_steps=params.angular_steps,
        max_radius=params.max_radius,
    )

    # Compute radial gradient magnitude and smooth
    grad_r = np.abs(np.gradient(polar, axis=0))
    if params.smooth_sigma_radial > 0 or params.smooth_sigma_theta > 0:
        grad_r = filters.gaussian(
            grad_r,
            sigma=(params.smooth_sigma_radial, params.smooth_sigma_theta),
            preserve_range=True,
        ).astype(np.float32)

    r_min = int(max(0, np.floor(params.min_radius)))
    r_max = int(radius) - 1

    # Global threshold from robust percentile across all angles within [r_min:r_max]
    roi = grad_r[r_min:r_max] if r_min < r_max else grad_r
    if roi.size == 0:
        logger.warning("ROI for ring detection is empty; returning no rings.")
        return []
    ref = np.percentile(roi, 95.0).item()
    thr = float(params.relative_threshold) * float(ref)

    # Peak list per angle
    H, W = grad_r.shape
    peak_lists: list[list[int]] = []
    for j in range(W):
        prof = grad_r[:, j]
        peaks = _local_maxima_1d(
            prof, threshold=thr, min_dist=int(params.min_peak_distance), r_min=r_min, r_max=r_max
        )
        peak_lists.append(peaks)

    # Determine max number of rings over all angles
    max_peaks = max((len(p) for p in peak_lists), default=0)
    if max_peaks == 0:
        logger.info("No peaks found in polar gradient; returning no rings.")
        return []

    # Align by index (inner to outer): i-th peak across angles ~ i-th ring
    rings_r = np.full((max_peaks, W), np.nan, dtype=np.float32)
    for j, peaks in enumerate(peak_lists):
        # sort increasing radius
        peaks_sorted = sorted(peaks)
        for i, r in enumerate(peaks_sorted[:max_peaks]):
            rings_r[i, j] = float(r)

    # Filter by coverage
    coverage = np.mean(~np.isnan(rings_r), axis=1)
    keep_idx = [i for i, c in enumerate(coverage) if c >= params.min_coverage]
    rings_r = rings_r[keep_idx]
    if rings_r.size == 0:
        logger.info("All candidate rings failed coverage threshold; returning no rings.")
        return []

    # Optionally limit number of rings (prefer inner rings)
    if params.max_rings is not None and rings_r.shape[0] > params.max_rings:
        rings_r = rings_r[: params.max_rings]

    # Interpolate missing r over angle and smooth
    thetas = np.linspace(0.0, 2 * np.pi, W, endpoint=False, dtype=np.float32)
    rings_xy: list[npt.NDArray[np.float32]] = []
    cx, cy = center_xy
    for rr in rings_r:
        mask = ~np.isnan(rr)
        if not np.any(mask):
            continue
        # Circular interpolation: duplicate endpoints to enforce periodicity
        idx = np.flatnonzero(mask)
        vals = rr[mask]
        # Build an interpolation over indices; wrap by adding endpoints +/- W
        xi = np.r_[idx - W, idx, idx + W]
        xv = np.r_[vals, vals, vals]
        full = np.interp(np.arange(W), xi, xv).astype(np.float32)
        # Optional angular smoothing
        if params.smooth_sigma_theta > 0:
            # convert sigma in samples
            sigma_theta = max(0.0, float(params.smooth_sigma_theta))
            # Using gaussian on 1D by treating as 2D with one row
            full = filters.gaussian(full[None, :], sigma=(0.0, sigma_theta), preserve_range=True)[0].astype(np.float32)

        # Convert to XY
        x = cx + full * np.cos(thetas)
        y = cy + full * np.sin(thetas)
        ring_xy = np.stack([x, y], axis=1).astype(np.float32)
        rings_xy.append(ring_xy)

    logger.info("Detected %d rings", len(rings_xy))
    return rings_xy
