"""Find increment cores in an image and estimate a centreline path for each.

A core is a long, narrow strip of wood. On a scan or photo it is an *elongated* region
that differs in brightness from the background (paper, cloth, mount). The steps:

1. Downscale for speed and blur.
2. Split pixels into two classes with Otsu's threshold. Either class can be the wood,
   so both are tried.
3. Keep connected regions that are large and elongated (length/width ≥ ``min_elongation``
   from the principal axes). Use the class whose candidates cover more area.
4. For each region, slice it perpendicular to its principal axis. The centre of each
   slice gives the centreline; the slice widths give the core diameter.

Several cores in one image (a mounted batch) are returned top-to-bottom / left-to-right.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter

# Images are analysed at this maximum side length (px).
ANALYSIS_SIDE_PX = 1024
# A region is a core candidate if it covers this share of the image...
MIN_AREA_FRACTION = 0.002
# ...and is at least this many times longer than wide.
MIN_ELONGATION = 4.0
# Minimum share of its principal-axis bounding box a core fills (a straight strip ≈ 0.9;
# background wedges between a diagonal core and the image corners ≈ 0.5).
MIN_RECTANGULARITY = 0.65
# Centreline samples per core.
N_SLICES = 60
# Share of the core length trimmed at each end (ragged or broken tips).
END_TRIM = 0.02
# Ribbon half-width as a share of the core diameter (stay clear of the core edges).
HALF_WIDTH_FRACTION = 0.3


@dataclass(frozen=True)
class DetectedCore:
    path_xy: npt.NDArray[np.float64]  # (N, 2) centreline in full-resolution px
    diameter_px: float
    length_px: float

    @property
    def half_width(self) -> int:
        return max(2, int(round(HALF_WIDTH_FRACTION * self.diameter_px)))


def find_cores(image: npt.NDArray, max_cores: int = 20) -> list[DetectedCore]:
    """Detect elongated wood strips and return their centrelines."""
    gray = _to_gray(image)
    factor = min(1.0, ANALYSIS_SIDE_PX / max(gray.shape))
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)

    cores: list[DetectedCore] = []
    for region in _best_candidate_regions(small)[:max_cores]:
        core = _centreline(region, factor)
        if core is not None:
            cores.append(core)
    return sorted(cores, key=lambda c: (round(c.path_xy[:, 1].mean() / 50), c.path_xy[:, 0].mean()))


def _to_gray(image: npt.NDArray) -> npt.NDArray[np.uint8]:
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return array.astype(np.uint8)


def _best_candidate_regions(small: npt.NDArray[np.uint8]) -> list[npt.NDArray[np.float64]]:
    """Pixel coordinates (x, y) of each elongated region in the better Otsu class."""
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    options = [_elongated_regions(mask), _elongated_regions(cv2.bitwise_not(mask))]
    return max(options, key=lambda regions: sum(len(r) for r in regions))


def _elongated_regions(mask: npt.NDArray[np.uint8]) -> list[npt.NDArray[np.float64]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats_mat, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = np.asarray(stats_mat)
    min_area = MIN_AREA_FRACTION * mask.size

    regions = []
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero(labels == label)
        points = np.stack([xs, ys], axis=1).astype(np.float64)
        if _elongation(points) >= MIN_ELONGATION and _rectangularity(points) >= MIN_RECTANGULARITY:
            regions.append(points)
    return sorted(regions, key=len, reverse=True)


def _elongation(points: npt.NDArray[np.float64]) -> float:
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
    return float(np.sqrt(eigenvalues[1] / max(eigenvalues[0], 1e-9)))


def _rectangularity(points: npt.NDArray[np.float64]) -> float:
    """Pixel count over the area of the bounding box aligned with the principal axes."""
    _, vectors = np.linalg.eigh(np.cov(points.T))
    projected = (points - points.mean(axis=0)) @ vectors
    extent = np.ptp(projected, axis=0) + 1.0
    return float(len(points) / (extent[0] * extent[1]))


def _centreline(points: npt.NDArray[np.float64], factor: float) -> DetectedCore | None:
    """Centreline from slices perpendicular to the region's principal axis."""
    mean = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(points.T))
    major, minor = vectors[:, 1], vectors[:, 0]
    u = (points - mean) @ major
    v = (points - mean) @ minor

    lo, hi = np.quantile(u, [END_TRIM, 1 - END_TRIM])
    edges = np.linspace(lo, hi, N_SLICES + 1)
    centres, widths, positions = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        in_slice = (u >= a) & (u < b)
        if in_slice.sum() < 3:
            continue
        v_lo, v_hi = np.quantile(v[in_slice], [0.05, 0.95])
        centres.append((v_lo + v_hi) / 2)
        widths.append(v_hi - v_lo)
        positions.append((a + b) / 2)
    if len(positions) < 3:
        return None

    centres_arr = median_filter(np.asarray(centres), size=5, mode="nearest")
    path = mean + np.outer(positions, major) + np.outer(centres_arr, minor)
    path /= factor
    return DetectedCore(
        path_xy=path,
        diameter_px=float(np.median(widths)) / factor,
        length_px=float(positions[-1] - positions[0]) / factor,
    )
