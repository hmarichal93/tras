"""Which end of a measurement path is the pith? Two independent cues.

* **Curvature.** Rings are roughly circles around the pith. Seen in the ribbon, a ring of
  radius ``R`` crossing the path at ``s₀`` follows ``s(o) ≈ s₀ ∓ o² / (2R)``: it bends
  back towards the pith. Fitting ``s(o) = c + a·o + b·o²`` per boundary, ``b < 0`` means
  the pith is at the path start. Strong near the pith; too small to measure on narrow
  cores far from it.
* **Polarity** (``evidence.choose_polarity``). The sharp side of a ring is the
  latewood → earlywood step, which brightens walking outward. Works anywhere, but
  depends on the wood showing that pattern.

``curvature_vote`` returns ``None`` when the curvature evidence is inconclusive, so the
caller falls back on polarity.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tras.cores.evidence import edge_response
from tras.cores.evidence import normalize_lanes
from tras.cores.evidence import ribbon_intensity
from tras.cores.ribbon import extract_ribbon

# Ribbon half-width used to measure curvature (px).
CURVATURE_HALF_WIDTH = 30
# Minimum number of boundaries with a significant curvature fit.
MIN_CURVED_BOUNDARIES = 4
# Required agreement among those boundaries.
MIN_AGREEMENT = 0.75
# Minimum sag at the ribbon edge (px) for a boundary to count as curved.
MIN_SAG_PX = 1.0
# Maximum RMS residual (px) of the quadratic fit for a trace to count.
MAX_FIT_RESIDUAL_PX = 1.0
# Search radius when tracing a boundary from one lane to the next (px).
TRACE_STEP_PX = 1


def curvature_vote(
    image: npt.NDArray,
    path: npt.ArrayLike,
    boundaries_s: npt.ArrayLike,
    tilts: npt.ArrayLike,
    channel: str = "gray",
    edge_sigma: float = 1.5,
    background: float = 81.0,
    half_width: int = CURVATURE_HALF_WIDTH,
    step: float = 1.0,
) -> bool | None:
    """True if curvature says the pith is at the path start, False if at the end.

    ``boundaries_s`` are arc lengths in px; ``half_width`` is in grid samples of ``step`` px.

    ``half_width`` must stay inside the wood (on a core: below half its diameter).
    """
    if half_width < 5:
        return None
    ribbon = extract_ribbon(image, path, half_width, step)
    response = np.abs(
        edge_response(normalize_lanes(ribbon_intensity(ribbon, channel), background), edge_sigma)
    )
    offsets = ribbon.offsets
    signs = []
    samples = np.asarray(boundaries_s, dtype=float) / step
    for s0, tilt in zip(samples, np.asarray(tilts, dtype=float)):
        b = _fit_curvature(response, offsets, s0, tilt)
        if b is not None and abs(b) * half_width**2 >= MIN_SAG_PX:
            signs.append(b < 0)
    if len(signs) < MIN_CURVED_BOUNDARIES:
        return None
    share_start = float(np.mean(signs))
    if share_start >= MIN_AGREEMENT:
        return True
    if share_start <= 1 - MIN_AGREEMENT:
        return False
    return None


def _fit_curvature(
    response: npt.NDArray[np.float64], offsets: npt.NDArray[np.float64], s0: float, tilt: float
) -> float | None:
    """Quadratic coefficient of the boundary trace ``s(o)``, or None if unreliable.

    The boundary is traced lane by lane from the centre lane outwards, each lane
    searching ``±TRACE_STEP_PX`` around the previous lane's position (plus the tilt).
    """
    centre_lane = int(np.argmin(np.abs(offsets)))
    positions = {centre_lane: float(s0)}
    for step in (1, -1):
        previous = float(s0)
        lane = centre_lane + step
        while 0 <= lane < len(offsets):
            predicted = previous + step * np.tan(tilt)
            found = _local_peak(response[lane], predicted, TRACE_STEP_PX)
            if found is None:
                break
            positions[lane] = previous = found
            lane += step
    if len(positions) < len(offsets) * 0.6:
        return None
    lanes = sorted(positions)
    x, y = offsets[lanes], np.array([positions[i] for i in lanes])
    coefficients = np.polyfit(x, y, 2)
    residual = np.sqrt(np.mean((np.polyval(coefficients, x) - y) ** 2))
    return float(coefficients[0]) if residual <= MAX_FIT_RESIDUAL_PX else None


def _local_peak(profile: npt.NDArray[np.float64], centre: float, radius: int) -> float | None:
    lo, hi = int(np.floor(centre)) - radius, int(np.ceil(centre)) + radius + 1
    if lo < 0 or hi > len(profile):
        return None
    return float(lo + np.argmax(profile[lo:hi]))
