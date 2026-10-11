"""Score detected cross-section rings against ground-truth annotations.

Matching follows the UruDendro protocol (Marichal et al., CS-TRD, 2023):

1. Sample every ring on ``n_rays`` rays from the ground-truth pith.
2. On each ray, ground-truth ring ``k`` owns an *influence zone*: from the midpoint with
   ring ``k-1`` (or the pith) to the midpoint with ring ``k+1`` (or ``+∞``).
3. A detected ring votes, ray by ray, for the ground-truth ring whose zone contains it.
   It is assigned to the ring with most votes if that ring wins at least
   ``min_ray_fraction`` of the rays.
4. Each ground-truth ring keeps at most one detection (the one with most votes). Matches
   are true positives (TP); other detections are false positives (FP), unmatched ground
   truth rings are false negatives (FN).

Reported: precision, recall, F1 and the radial RMSE (px) of matched rings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from tras.analysis.rings import order_rings_by_area
from tras.analysis.rings import radial_profiles

DEFAULT_MIN_RAY_FRACTION = 0.6
DEFAULT_N_RAYS = 360


@dataclass(frozen=True)
class RingMatchScore:
    tp: int
    fp: int
    fn: int
    rmse_px: float  # radial RMSE over matched rings (NaN if no match)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_rings(
    detected: Sequence[npt.ArrayLike],
    ground_truth: Sequence[npt.ArrayLike],
    pith_xy: tuple[float, float],
    *,
    n_rays: int = DEFAULT_N_RAYS,
    min_ray_fraction: float = DEFAULT_MIN_RAY_FRACTION,
) -> RingMatchScore:
    """Match detected rings to ground truth and count TP/FP/FN."""
    gt = _ordered_profiles(ground_truth, pith_xy, n_rays)
    det = _ordered_profiles(detected, pith_xy, n_rays)
    if len(gt) == 0 or len(det) == 0:
        return RingMatchScore(tp=0, fp=len(det), fn=len(gt), rmse_px=float("nan"))

    lower, upper = influence_zones(gt)
    best_gt, fractions = _vote(det, lower, upper)
    matches = _one_detection_per_ring(best_gt, fractions, min_ray_fraction)

    tp = len(matches)
    errors = [det[d] - gt[g] for g, d in matches.items()]
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
    return RingMatchScore(tp=tp, fp=len(det) - tp, fn=len(gt) - tp, rmse_px=rmse)


def influence_zones(
    gt_profiles: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-ray ``[lower, upper)`` radius bounds of each ground-truth ring's zone."""
    padded = np.vstack([np.zeros((1, gt_profiles.shape[1])), gt_profiles])
    mid = (padded[1:] + padded[:-1]) / 2.0  # midpoint with the inner neighbour
    lower = mid
    upper = np.vstack([mid[1:], np.full((1, gt_profiles.shape[1]), np.inf)])
    return lower, upper


def _vote(
    det: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """For each detection: the ground-truth ring with most rays, and that ray fraction."""
    inside = (det[:, None, :] >= lower[None]) & (det[:, None, :] < upper[None])  # (D, G, R)
    votes = inside.mean(axis=2)  # (D, G)
    best = votes.argmax(axis=1)
    return best, votes[np.arange(len(det)), best]


def _one_detection_per_ring(
    best_gt: npt.NDArray[np.int64], fractions: npt.NDArray[np.float64], min_fraction: float
) -> dict[int, int]:
    """``{gt_index: det_index}`` keeping the strongest detection per ground-truth ring."""
    matches: dict[int, int] = {}
    for d in np.argsort(-fractions):
        g = int(best_gt[d])
        if fractions[d] >= min_fraction and g not in matches:
            matches[g] = int(d)
    return matches


def _ordered_profiles(
    rings: Sequence[npt.ArrayLike], pith_xy: tuple[float, float], n_rays: int
) -> npt.NDArray[np.float64]:
    valid = [np.asarray(r, dtype=np.float64) for r in rings if len(np.asarray(r)) >= 3]
    ordered = [valid[i] for i in order_rings_by_area(valid)]
    if not ordered:
        return np.empty((0, n_rays))
    return radial_profiles(ordered, pith_xy, n_rays)
