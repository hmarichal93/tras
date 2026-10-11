"""Score boundary positions detected along a 1-D path (increment cores, radii).

A boundary is a position (distance along the path, in px or mm). A detection matches a
ground-truth boundary when they are within ``tolerance``; matching is one-to-one and
minimizes total distance (Hungarian algorithm on the 1-D distances).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

# Cost assigned to pairs farther apart than the tolerance (never selected).
_FORBIDDEN = 1e9


@dataclass(frozen=True)
class BoundaryScore:
    tp: int
    fp: int
    fn: int
    mae: float  # mean |detected - true| over matched boundaries (NaN if none)

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


def match_boundaries(
    detected: Sequence[float], ground_truth: Sequence[float], tolerance: float
) -> list[tuple[int, int]]:
    """One-to-one ``(det_index, gt_index)`` pairs within ``tolerance``."""
    det = np.asarray(detected, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if len(det) == 0 or len(gt) == 0:
        return []
    distance = np.abs(det[:, None] - gt[None, :])
    cost = np.where(distance <= tolerance, distance, _FORBIDDEN)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if distance[r, c] <= tolerance]


def score_boundaries(
    detected: Sequence[float], ground_truth: Sequence[float], tolerance: float
) -> BoundaryScore:
    """Precision/recall/F1 and mean position error of detected boundaries."""
    pairs = match_boundaries(detected, ground_truth, tolerance)
    errors = [abs(detected[d] - ground_truth[g]) for d, g in pairs]
    return BoundaryScore(
        tp=len(pairs),
        fp=len(detected) - len(pairs),
        fn=len(ground_truth) - len(pairs),
        mae=float(np.mean(errors)) if errors else float("nan"),
    )
