"""Quality-control (QC) flags for detected rings.

Automatic detectors make a few typical errors. QC points the user at the rings worth
reviewing instead of making them inspect every ring:

* ``crossing``: a boundary crosses its inner neighbour on some rays (impossible in wood);
* ``narrow``: much narrower than its neighbours. Often a false ring, e.g. a latewood
  band split in two, or a double detection;
* ``wide``: much wider than its neighbours, often a missed ring in between;
* ``partial``: radial width ~0 on many rays, so the boundary only covers part of the
  circumference.

Width outliers use a robust z-score of log mean width against a moving median of the
neighbouring rings, so a normal growth trend is not flagged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tras.analysis.rings import RingMeasurement

# Innermost rings skipped by the width check: the first boundary encloses the pith and
# juvenile rings widen fast, so neighbours are no reference. Measured on the 64 UruDendro
# ground-truth images: with this rule 2 of 1,221 true rings are flagged.
JUVENILE_RINGS = 2
# Robust z-score (in MAD units) beyond which a ring width is an outlier.
WIDTH_OUTLIER_Z = 3.5
# Neighbours on each side used for the local reference width.
NEIGHBOURHOOD = 3
# Share of rays where a boundary may cross its inner neighbour before flagging.
MAX_CROSSING_FRACTION = 0.02
# Share of rays with ~zero width before a ring is reported as partial.
MAX_COLLAPSED_FRACTION = 0.25
# Widths below this (px) count as collapsed onto the inner boundary.
COLLAPSED_WIDTH_PX = 0.5
# Floor on the log-width spread: year-to-year variation below ~15 % is normal, so a
# near-constant series (MAD ≈ 0) must not turn tiny differences into outliers.
MIN_LOG_WIDTH_SIGMA = 0.15
# Scale factor making the MAD a consistent estimator of the standard deviation.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class QCFlag:
    ring_index: int  # 1 = innermost, matches RingMeasurement.index
    code: str  # crossing | narrow | wide | partial
    message: str


def flag_rings(
    measurements: Sequence[RingMeasurement],
    widths: npt.NDArray[np.float64] | None = None,
) -> list[QCFlag]:
    """Return QC flags for the given rings (innermost first).

    Args:
        measurements: Output of ``measure_rings``.
        widths: Optional ``(n_rings, n_rays)`` per-ray widths (``radial_widths``) for the
            geometric checks. Without it only width outliers are checked.
    """
    flags = flag_width_outliers(measurements)
    if widths is not None:
        flags += flag_geometry(measurements, widths)
    return sorted(flags, key=lambda f: (f.ring_index, f.code))


def flag_width_outliers(measurements: Sequence[RingMeasurement]) -> list[QCFlag]:
    """Flag rings whose mean width is far from that of their neighbours."""
    if len(measurements) < 2 * NEIGHBOURHOOD:
        return []
    log_width = np.log(np.maximum([m.mean_width for m in measurements], 1e-6))
    reference = _moving_median_excluding_self(log_width, NEIGHBOURHOOD)
    residual = log_width - reference
    sigma = np.median(np.abs(residual - np.median(residual))) * _MAD_TO_SIGMA
    z = residual / max(sigma, MIN_LOG_WIDTH_SIGMA)

    flags = []
    for m, score in zip(measurements, z):
        if m.index <= JUVENILE_RINGS:
            continue
        if score < -WIDTH_OUTLIER_Z:
            flags.append(
                QCFlag(m.index, "narrow", f"{m.label} is unusually narrow (z={score:.1f})")
            )
        elif score > WIDTH_OUTLIER_Z:
            flags.append(QCFlag(m.index, "wide", f"{m.label} is unusually wide (z={score:.1f})"))
    return flags


def flag_geometry(
    measurements: Sequence[RingMeasurement], widths: npt.NDArray[np.float64]
) -> list[QCFlag]:
    """Flag boundaries that cross their inner neighbour or cover only part of the stem."""
    flags = []
    for m, ring_widths in zip(measurements, widths):
        crossing = float(np.mean(ring_widths < -COLLAPSED_WIDTH_PX))
        collapsed = float(np.mean(np.abs(ring_widths) <= COLLAPSED_WIDTH_PX))
        if crossing > MAX_CROSSING_FRACTION:
            flags.append(
                QCFlag(
                    m.index,
                    "crossing",
                    f"{m.label} crosses the inner ring on {crossing:.0%} of rays",
                )
            )
        if collapsed > MAX_COLLAPSED_FRACTION:
            flags.append(
                QCFlag(
                    m.index,
                    "partial",
                    f"{m.label} touches the inner ring on {collapsed:.0%} of rays",
                )
            )
    return flags


def _moving_median_excluding_self(
    values: npt.NDArray[np.float64], half: int
) -> npt.NDArray[np.float64]:
    out = np.empty_like(values)
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        neighbours = np.concatenate([values[lo:i], values[i + 1 : hi]])
        out[i] = np.median(neighbours)
    return out
