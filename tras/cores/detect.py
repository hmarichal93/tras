"""Automatic ring-boundary detection along an increment core (or any measurement path).

Pipeline::

    path (given or auto-estimated) → ribbon → boundary evidence (both polarities)
        → orientation (which end is the pith) → peak picking → boundaries → widths

The result is oriented pith → bark: ``s = 0`` is the pith end of the path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from tras.cores.evidence import Evidence
from tras.cores.evidence import choose_polarity
from tras.cores.evidence import edge_response
from tras.cores.evidence import normalize_lanes
from tras.cores.evidence import ribbon_intensity
from tras.cores.evidence import tilt_search
from tras.cores.ribbon import Ribbon
from tras.cores.ribbon import SampledPath
from tras.cores.ribbon import extract_ribbon

FloatArray = npt.NDArray[np.float64]

# MAD → standard deviation for Gaussian noise.
_MAD_TO_SIGMA = 1.4826

PITH_SIDES = ("auto", "start", "end")
EVIDENCE_SOURCES = ("classical", "deep", "hybrid")
# Context the U-Net sees on each side of the path, in model pixels (after resizing).
DEEP_CONTEXT_PX = 96
BOUNDARY_STEPS = ("dark_to_light", "light_to_dark")


@dataclass(frozen=True)
class CoreParams:
    """Parameters of the core boundary detector. Lengths are in pixels.

    Attributes:
        half_width: Ribbon half-width across the path. Wider = more lanes for the median
            (robust to cracks) but more blur when boundaries curve.
        min_ring_width: Closest allowed boundary spacing (CooRecorder's "lowest width of
            earlywood"); suppresses false boundaries inside latewood bands.
        edge_sigma: Smoothing scale of the edge detector.
        background: Window of the illumination/contrast normalization (≫ ring width).
        sensitivity: Boundary threshold in robust noise units; lower finds more rings.
        max_tilt_deg / n_tilts: Range and number of boundary tilts tried.
        channel: Intensity channel: gray, red, green, blue or min.
        boundary_step: Intensity change across a year boundary walking pith → bark.
            Conifers and most diffuse-porous woods: ``dark_to_light`` (latewood →
            earlywood).
        pith_side: ``start`` / ``end`` of the path, or ``auto`` (from ``boundary_step``).
        evidence: ``classical`` (intensity edges), ``deep`` (DeepCS-TRD U-Net boundary
            probability) or ``hybrid`` (sum of both, each in robust noise units).
        model: DeepCS-TRD model id or weights path (deep/hybrid only).
        deep_scale: Resize factor bringing rings to the model's training scale; ``None``
            uses DeepCS-TRD's rule for a cross-section of the input image's size.
        deep_threshold: Minimum boundary probability of a deep peak.
    """

    half_width: int = 15
    min_ring_width: float = 5.0
    edge_sigma: float = 1.5
    background: float = 81.0
    sensitivity: float = 2.5
    max_tilt_deg: float = 45.0
    n_tilts: int = 13
    channel: str = "gray"
    boundary_step: str = "dark_to_light"
    pith_side: str = "auto"
    evidence: str = "classical"
    model: str = "generic"
    deep_scale: Optional[float] = None
    deep_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.evidence not in EVIDENCE_SOURCES:
            raise ValueError(f"evidence must be one of {EVIDENCE_SOURCES}")
        if self.pith_side not in PITH_SIDES:
            raise ValueError(f"pith_side must be one of {PITH_SIDES}")
        if self.boundary_step not in BOUNDARY_STEPS:
            raise ValueError(f"boundary_step must be one of {BOUNDARY_STEPS}")
        if self.half_width < 0 or self.min_ring_width <= 0 or self.edge_sigma <= 0:
            raise ValueError("half_width must be >= 0; min_ring_width, edge_sigma > 0")


@dataclass(frozen=True)
class CoreBoundary:
    s: float  # arc length from the pith end of the path (px)
    xy: tuple[float, float]  # image position on the path
    tilt: float  # radians between the boundary and the path normal
    strength: float  # evidence at the peak, in robust noise units


@dataclass(frozen=True)
class CoreResult:
    """Detected boundaries along a path oriented pith → bark."""

    path: SampledPath
    boundaries: tuple[CoreBoundary, ...]
    half_width: int
    mm_per_px: Optional[float] = None
    outer_year: Optional[int] = None
    evidence: Optional[FloatArray] = field(default=None, repr=False)

    @property
    def path_widths_px(self) -> FloatArray:
        """Distance along the path between consecutive boundaries."""
        return np.diff([b.s for b in self.boundaries])

    @property
    def widths_px(self) -> FloatArray:
        """Ring widths perpendicular to the boundaries (CooRecorder-style)."""
        s = np.array([b.s for b in self.boundaries])
        tilt = np.array([b.tilt for b in self.boundaries])
        if len(s) < 2:
            return np.empty(0)
        return np.diff(s) * np.cos((tilt[1:] + tilt[:-1]) / 2.0)

    @property
    def widths_mm(self) -> Optional[FloatArray]:
        return None if self.mm_per_px is None else self.widths_px * self.mm_per_px

    @property
    def years(self) -> Optional[list[int]]:
        """Calendar year of each measured ring (oldest first) if ``outer_year`` is set."""
        if self.outer_year is None:
            return None
        n = len(self.widths_px)
        return list(range(self.outer_year - n + 1, self.outer_year + 1))


def detect_core_rings(
    image: npt.NDArray,
    path: Sequence[Sequence[float]],
    params: CoreParams = CoreParams(),
    *,
    mm_per_px: Optional[float] = None,
    outer_year: Optional[int] = None,
) -> CoreResult:
    """Detect ring boundaries along ``path`` in ``image``.

    Args:
        image: Grayscale or RGB image.
        path: Polyline (``(N, 2)`` x/y px) along the core, crossing the rings.
        params: Detector parameters.
        mm_per_px: Physical pixel size, used for ``widths_mm``.
        outer_year: Calendar year of the outermost measured ring.
    """
    ribbon = extract_ribbon(image, path, params.half_width)
    positive, negative = boundary_evidence(ribbon, params)
    pith_at_start, evidence = orient(positive, negative, params)
    if params.evidence != "classical":
        evidence = combine_evidence(evidence, deep_evidence(image, path, params), params.evidence)
    peaks, strengths = pick_boundaries(
        evidence.score, params.min_ring_width, _threshold(params), params.deep_threshold
    )

    boundaries = [
        CoreBoundary(
            s=float(ribbon.s[p]),
            xy=tuple(float(v) for v in ribbon.path.xy[p]),  # type: ignore[arg-type]
            tilt=float(evidence.tilt[p]),
            strength=float(st),
        )
        for p, st in zip(peaks, strengths)
    ]
    path_obj, score = ribbon.path, evidence.score
    if not pith_at_start:
        path_obj, boundaries, score = _reverse(ribbon, boundaries, score)
    return CoreResult(
        path=path_obj,
        boundaries=tuple(boundaries),
        half_width=params.half_width,
        mm_per_px=mm_per_px,
        outer_year=outer_year,
        evidence=score,
    )


def boundary_evidence(ribbon: Ribbon, params: CoreParams) -> tuple[Evidence, Evidence]:
    """Classical evidence for brightening (+s) and darkening (+s) boundaries."""
    intensity = ribbon_intensity(ribbon, params.channel)
    response = edge_response(normalize_lanes(intensity, params.background), params.edge_sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
        args = (ribbon.offsets, ribbon.valid, params.max_tilt_deg, params.n_tilts)
        return tilt_search(response, *args), tilt_search(-response, *args)


def deep_evidence(image: npt.NDArray, path: Sequence[Sequence[float]], params: CoreParams) -> Evidence:
    """U-Net boundary probability along the path, combined over lanes with tilt search."""
    from tras.cores.deep import boundary_probability_along_path
    from tras.cores.deep import training_scale

    scale = params.deep_scale or training_scale(image.shape)
    ribbon = boundary_probability_along_path(
        image, path, params.half_width, params.model, scale, DEEP_CONTEXT_PX
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return tilt_search(
            ribbon.values, ribbon.offsets, ribbon.valid, params.max_tilt_deg, params.n_tilts
        )


def combine_evidence(classical: Evidence, deep: Evidence, mode: str) -> Evidence:
    """``deep``: probability only. ``hybrid``: sum of both in robust noise units.

    Hybrid scores are rescaled so a deep probability of 1 with no classical support
    stays comparable to the deep threshold.
    """
    if mode == "deep":
        return deep
    z_classical = _robust_z(classical.score)
    z_deep = _robust_z(deep.score)
    return Evidence(score=z_classical + z_deep, tilt=np.where(z_deep > z_classical, deep.tilt, classical.tilt))


def _robust_z(values: FloatArray) -> FloatArray:
    return (values - np.median(values)) / _robust_sigma(values)


def _threshold(params: CoreParams) -> Optional[float]:
    """Peak threshold; ``None`` means the score is already a probability."""
    return None if params.evidence == "deep" else params.sensitivity


def orient(positive: Evidence, negative: Evidence, params: CoreParams) -> tuple[bool, Evidence]:
    """Decide which path end is the pith and which evidence polarity marks boundaries.

    Walking pith → bark, a ``dark_to_light`` boundary is a brightening step. If the pith
    is at the path start, brightening along +s is the positive polarity.
    """
    brightening_outward = params.boundary_step == "dark_to_light"
    if params.pith_side == "auto":
        polarity = choose_polarity(positive, negative, int(params.min_ring_width))
        pith_at_start = (polarity > 0) == brightening_outward
    else:
        pith_at_start = params.pith_side == "start"
        polarity = 1 if pith_at_start == brightening_outward else -1
    return pith_at_start, positive if polarity > 0 else negative


def pick_boundaries(
    score: FloatArray,
    min_distance: float,
    sensitivity: Optional[float],
    probability_threshold: float = 0.3,
) -> tuple[npt.NDArray[np.intp], FloatArray]:
    """Peaks of ``score`` at least ``min_distance`` apart.

    With ``sensitivity`` the score is normalized to robust noise units and peaks must be
    ``sensitivity`` high and prominent. With ``sensitivity=None`` the score is a
    probability and peaks must exceed ``probability_threshold``.

    Returns peak indices and their strength (noise units or probability).
    """
    distance = max(1, int(round(min_distance)))
    if sensitivity is None:
        peaks, _ = find_peaks(
            score, distance=distance, height=probability_threshold,
            prominence=probability_threshold / 2,
        )
        return peaks, score[peaks]
    normalized = _robust_z(score)
    peaks, _ = find_peaks(normalized, distance=distance, height=sensitivity, prominence=sensitivity)
    return peaks, normalized[peaks]


def _robust_sigma(values: FloatArray) -> float:
    mad = np.median(np.abs(values - np.median(values)))
    return float(max(mad * _MAD_TO_SIGMA, 1e-9))


def _reverse(
    ribbon: Ribbon, boundaries: list[CoreBoundary], score: FloatArray
) -> tuple[SampledPath, list[CoreBoundary], FloatArray]:
    """Re-express results with ``s = 0`` at the path end (the pith side)."""
    length = ribbon.path.length
    flipped = [
        CoreBoundary(s=length - b.s, xy=b.xy, tilt=b.tilt, strength=b.strength)
        for b in reversed(boundaries)
    ]
    return ribbon.path.reversed(), flipped, score[::-1].copy()
