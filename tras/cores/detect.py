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

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from tras.cores.evidence import Evidence
from tras.cores.evidence import choose_polarity
from tras.cores.evidence import edge_response
from tras.cores.evidence import normalize_lanes
from tras.cores.evidence import ribbon_intensity
from tras.cores.evidence import tilt_search
from tras.cores.orientation import curvature_vote
from tras.cores.ribbon import Ribbon
from tras.cores.ribbon import SampledPath
from tras.cores.ribbon import extract_ribbon
from tras.cores.ribbon import sample_path

FloatArray = npt.NDArray[np.float64]

# MAD → standard deviation for Gaussian noise.
_MAD_TO_SIGMA = 1.4826

PITH_SIDES = ("auto", "start", "end")
EVIDENCE_SOURCES = ("classical", "deep", "hybrid")
# Context the U-Net sees on each side of the path, in model pixels (after resizing).
DEEP_CONTEXT_PX = 96
# Effective resolution of the DeepCS-TRD training images (mm per model pixel): UruDendro
# and the ForestScience discs at 0.16–0.2 mm/px, resized to 1504 px, give 0.31–0.33.
DEEP_MODEL_MM_PER_PX = 0.3
# Boundaries up to this many minimum ring widths beyond the user's path ends are kept.
END_TOLERANCE_RINGS = 0.5
# Finest analysis grid (mm per sample) of the classical detector, ≈ 1270 dpi.
MIN_MM_PER_SAMPLE = 0.02
BOUNDARY_STEPS = ("dark_to_light", "light_to_dark")


@dataclass(frozen=True)
class CoreParams:
    """Parameters of the core boundary detector. Lengths are in grid samples: 1 sample =
    1 px for photos and scans up to ~1270 dpi; finer scans with a known scale are analysed
    at 0.02 mm per sample (see ``sampling_step``).

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
        curvature_half_width: Ribbon half-width for the curvature orientation cue; must
            stay inside the wood (0 disables the cue).
        snap_px: With deep/hybrid evidence, move each boundary to the sharpest classical
            edge within this distance (0 disables).
        sampling_step: Analysis grid spacing in px. ``None``: 1 px, or coarser for scans
            finer than ``MIN_MM_PER_SAMPLE`` mm/px when the scale is known. Lengths above
            (``half_width``, ``min_ring_width``, ...) are in grid samples.
        end_margin: The path is analysed this many minimum ring widths beyond both ends,
            so a boundary exactly at a path end is still found (0 disables).
        gap_fill_ratio: With deep evidence, gaps longer than this × the neighbouring ring
            widths are searched for a classical edge (0 disables; see ``fill_gaps``).
        gap_fill_sensitivity: Robust z-score a recovered classical edge must reach.
    """

    half_width: int = 25
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
    deep_scale: float | None = None
    deep_threshold: float = 0.3
    curvature_half_width: int = 30
    snap_px: int = 0
    sampling_step: float | None = None
    end_margin: float = 0.0
    gap_fill_ratio: float = 0.0
    gap_fill_sensitivity: float = 3.0

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
    mm_per_px: float | None = None
    outer_year: int | None = None
    evidence: FloatArray | None = field(default=None, repr=False)

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
    def widths_mm(self) -> FloatArray | None:
        return None if self.mm_per_px is None else self.widths_px * self.mm_per_px

    @property
    def years(self) -> list[int] | None:
        """Calendar year of each measured ring (oldest first) if ``outer_year`` is set."""
        if self.outer_year is None:
            return None
        n = len(self.widths_px)
        return list(range(self.outer_year - n + 1, self.outer_year + 1))


def detect_core_rings(
    image: npt.NDArray,
    path: npt.ArrayLike,
    params: CoreParams = CoreParams(),
    *,
    mm_per_px: float | None = None,
    outer_year: int | None = None,
) -> CoreResult:
    """Detect ring boundaries along ``path`` in ``image``.

    Args:
        image: Grayscale or RGB image.
        path: Polyline (``(N, 2)`` x/y px) along the core, crossing the rings.
        params: Detector parameters.
        mm_per_px: Physical pixel size, used for ``widths_mm``.
        outer_year: Calendar year of the outermost measured ring.
    """
    step = sampling_step(params, mm_per_px)
    # Analyse a slightly longer path: a boundary exactly at a path end (users often
    # start/stop on one) then becomes an interior peak instead of being lost.
    margin = params.end_margin * params.min_ring_width * step
    user_path = np.asarray(path, dtype=np.float64)
    path = extend_polyline(user_path, margin)
    ribbon = extract_ribbon(image, path, params.half_width, step)
    positive, negative = boundary_evidence(ribbon, params)
    deep = None
    if params.evidence != "classical":
        deep = deep_evidence(image, path, params, step, deep_scale(params, image.shape, mm_per_px))

    # Orientation from the user's path only: the extension may cover bark or background.
    inner = int(round(margin / step))
    pith_at_start = initial_orientation(_trim(positive, inner), _trim(negative, inner), params)
    evidence, peaks, strengths = _boundaries_for(pith_at_start, positive, negative, deep, params)
    if params.pith_side == "auto":
        # Tilts are not passed: under the wrong polarity they are meaningless, and the
        # lane-by-lane tracer follows boundaries up to 45° on its own.
        vote = curvature_vote(
            image,
            path,
            ribbon.s[np.round(peaks).astype(int)],
            np.zeros(len(peaks)),
            params.channel,
            params.edge_sigma,
            params.background,
            params.curvature_half_width,
            step,
        )
        if vote is not None and vote != pith_at_start:
            pith_at_start = vote
            evidence, peaks, strengths = _boundaries_for(
                pith_at_start, positive, negative, deep, params
            )

    boundaries = [
        CoreBoundary(
            s=float(p * ribbon.path.step),
            xy=tuple(float(v) for v in ribbon.path.point_at(p * ribbon.path.step)),  # type: ignore[arg-type]
            tilt=float(evidence.tilt[int(round(p))]),
            strength=float(st),
        )
        for p, st in zip(peaks, strengths)
    ]
    score = evidence.score
    if not pith_at_start:
        _, boundaries, score = _reverse(ribbon, boundaries, score, _arc_length(path))
    measured = sample_path(
        user_path, step=step, smoothing_px=ribbon.path.step * params.half_width / 2
    )
    if not pith_at_start:
        measured = measured.reversed()
    boundaries = _within_user_path(
        boundaries, margin, measured.length, END_TOLERANCE_RINGS * params.min_ring_width * step
    )
    return CoreResult(
        path=measured,
        boundaries=tuple(boundaries),
        half_width=int(round(params.half_width * step)),
        mm_per_px=mm_per_px,
        outer_year=outer_year,
        evidence=score,
    )


def _trim(evidence: Evidence, n: int) -> Evidence:
    """Evidence without ``n`` samples at each end."""
    if n <= 0 or 2 * n >= len(evidence.score):
        return evidence
    return Evidence(score=evidence.score[n:-n], tilt=evidence.tilt[n:-n])


def _arc_length(polyline: npt.ArrayLike) -> float:
    pts = np.asarray(polyline, dtype=np.float64)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def extend_polyline(polyline: npt.ArrayLike, margin: float) -> FloatArray:
    """Prolong a polyline by ``margin`` px beyond both ends along its end directions."""
    pts = np.asarray(polyline, dtype=np.float64)
    # Repeated points (e.g. a double-click ending a GUI linestrip) give no direction.
    pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]
    if margin <= 0 or len(pts) < 2:
        return pts
    start_dir = pts[0] - pts[1]
    end_dir = pts[-1] - pts[-2]
    start_dir /= max(np.linalg.norm(start_dir), 1e-12)
    end_dir /= max(np.linalg.norm(end_dir), 1e-12)
    return np.vstack([pts[0] + margin * start_dir, pts, pts[-1] + margin * end_dir])


def _within_user_path(
    boundaries: list[CoreBoundary], margin: float, length: float, tolerance: float
) -> list[CoreBoundary]:
    """Shift arc lengths back to the user's path and drop boundaries beyond its ends."""
    kept = []
    for b in boundaries:
        s = b.s - margin
        if -tolerance <= s <= length + tolerance:
            kept.append(CoreBoundary(s=s, xy=b.xy, tilt=b.tilt, strength=b.strength))
    return kept


def boundary_evidence(ribbon: Ribbon, params: CoreParams) -> tuple[Evidence, Evidence]:
    """Classical evidence for brightening (+s) and darkening (+s) boundaries."""
    intensity = ribbon_intensity(ribbon, params.channel)
    response = edge_response(normalize_lanes(intensity, params.background), params.edge_sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
        args = (ribbon.offsets, ribbon.valid, params.max_tilt_deg, params.n_tilts)
        return tilt_search(response, *args), tilt_search(-response, *args)


def sampling_step(params: CoreParams, mm_per_px: float | None) -> float:
    """Analysis grid spacing (px): 1 px, coarser only for very high-resolution scans.

    With a known pixel size finer than ``MIN_MM_PER_SAMPLE``, the grid is coarsened to that
    physical resolution, so pixel-valued parameters keep their meaning on 2400+ dpi scans.
    """
    if params.sampling_step is not None:
        return float(params.sampling_step)
    if mm_per_px is not None and mm_per_px < MIN_MM_PER_SAMPLE:
        return MIN_MM_PER_SAMPLE / mm_per_px
    return 1.0


def deep_scale(params: CoreParams, image_shape: tuple[int, ...], mm_per_px: float | None) -> float:
    """Resize factor bringing the image to the U-Net's training resolution.

    Physical when the pixel size is known (DeepCS-TRD saw ~``DEEP_MODEL_MM_PER_PX`` mm per
    model pixel); otherwise DeepCS-TRD's own rule for a cross-section of this image size.
    """
    from tras.cores.deep import training_scale

    if params.deep_scale is not None:
        return params.deep_scale
    if mm_per_px is not None:
        return mm_per_px / DEEP_MODEL_MM_PER_PX
    return training_scale(image_shape)


def deep_evidence(
    image: npt.NDArray, path: npt.ArrayLike, params: CoreParams, step: float, scale: float
) -> Evidence:
    """U-Net boundary probability along the path, combined over lanes with tilt search."""
    from tras.cores.deep import boundary_probability_along_path

    ribbon = boundary_probability_along_path(
        image, path, params.half_width, params.model, scale, DEEP_CONTEXT_PX, step
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
        # The probability ridge saturates over neighbouring tilts; the classical edge
        # response is sharp in tilt, so it supplies the boundary orientation.
        return Evidence(score=deep.score, tilt=classical.tilt)
    z_classical = _robust_z(classical.score)
    z_deep = _robust_z(deep.score)
    return Evidence(score=z_classical + z_deep, tilt=classical.tilt)


def _robust_z(values: FloatArray) -> FloatArray:
    return (values - np.median(values)) / _robust_sigma(values)


def _threshold(params: CoreParams) -> float | None:
    """Peak threshold; ``None`` means the score is already a probability."""
    return None if params.evidence == "deep" else params.sensitivity


def initial_orientation(positive: Evidence, negative: Evidence, params: CoreParams) -> bool:
    """Pith at the path start? From ``pith_side`` or, if auto, from edge polarity.

    Walking pith → bark, a ``dark_to_light`` boundary is a brightening step: if
    brightening (+s) edges are the sharper ones, the pith is at the start.
    """
    if params.pith_side != "auto":
        return params.pith_side == "start"
    polarity = choose_polarity(positive, negative, int(params.min_ring_width))
    return (polarity > 0) == (params.boundary_step == "dark_to_light")


def _boundaries_for(
    pith_at_start: bool,
    positive: Evidence,
    negative: Evidence,
    deep: Evidence | None,
    params: CoreParams,
) -> tuple[Evidence, FloatArray, FloatArray]:
    """Evidence for the given orientation and its picked (sub-pixel) peaks."""
    brightening_outward = params.boundary_step == "dark_to_light"
    classical = positive if pith_at_start == brightening_outward else negative
    evidence = classical if deep is None else combine_evidence(classical, deep, params.evidence)
    peaks, strengths = pick_boundaries(
        evidence.score, params.min_ring_width, _threshold(params), params.deep_threshold
    )
    if deep is not None and params.gap_fill_ratio > 0:
        peaks, strengths = fill_gaps(
            peaks,
            evidence.score,
            classical.score,
            params.min_ring_width,
            params.gap_fill_ratio,
            params.gap_fill_sensitivity,
        )
    if deep is not None and params.snap_px > 0:
        # The learned map decides *where* boundaries are; the sharp classical edge
        # localizes them more precisely.
        peaks = snap_to_local_maxima(classical.score, peaks, params.snap_px)
        return evidence, refine_peaks(classical.score, peaks), strengths
    return evidence, refine_peaks(evidence.score, peaks), strengths


def fill_gaps(
    peaks: npt.NDArray[np.intp],
    score: FloatArray,
    classical_score: FloatArray,
    min_distance: float,
    ratio: float,
    sensitivity: float,
    max_rounds: int = 3,
) -> tuple[npt.NDArray[np.intp], FloatArray]:
    """Recover boundaries the learned map missed, using ring-width continuity.

    A gap between consecutive boundaries (or a path end) longer than ``ratio`` × the median
    width of its neighbouring rings probably hides a ring. The strongest classical edge
    inside it, at least ``min_distance`` from both sides, is added if its robust z-score
    reaches ``sensitivity``. Repeats while boundaries are added (a gap may hide several).

    Returns all peaks (sorted) and their ``score`` values (low for recovered ones).
    """
    z = _robust_z(classical_score)
    found = sorted(int(p) for p in peaks)
    margin = max(1, int(round(min_distance)))
    for _ in range(max_rounds):
        anchors = [0, *found, len(score) - 1]
        widths = np.diff(anchors)
        added = []
        for k, (a, b) in enumerate(zip(anchors[:-1], anchors[1:])):
            if b - a <= ratio * _neighbour_median(widths, k):
                continue
            lo, hi = a + margin, b - margin
            if hi <= lo:
                continue
            candidate = lo + int(np.argmax(z[lo:hi]))
            if z[candidate] >= sensitivity:
                added.append(candidate)
        if not added:
            break
        found = sorted(found + added)
    result = np.array(found, dtype=np.intp)
    return result, score[result]


def _neighbour_median(widths: npt.NDArray[np.intp], k: int, half: int = 3) -> float:
    """Median width of up to ``half`` rings on each side of ring ``k`` (excluding it)."""
    neighbours = np.concatenate([widths[max(0, k - half) : k], widths[k + 1 : k + 1 + half]])
    return float(np.median(neighbours)) if len(neighbours) else float("inf")


def snap_to_local_maxima(
    score: FloatArray, peaks: npt.NDArray[np.intp], radius: int
) -> npt.NDArray[np.intp]:
    """Move each peak to the highest ``score`` within ``±radius`` samples."""
    snapped = peaks.copy()
    for i, p in enumerate(peaks):
        lo, hi = max(0, p - radius), min(len(score), p + radius + 1)
        snapped[i] = lo + int(np.argmax(score[lo:hi]))
    return snapped


def refine_peaks(score: FloatArray, peaks: npt.NDArray[np.intp]) -> FloatArray:
    """Sub-sample peak positions from a parabola through each peak and its neighbours."""
    refined = peaks.astype(np.float64)
    inner = (peaks > 0) & (peaks < len(score) - 1)
    p = peaks[inner]
    left, centre, right = score[p - 1], score[p], score[p + 1]
    curvature = left - 2 * centre + right
    offset = np.where(
        curvature < 0, 0.5 * (left - right) / np.where(curvature < 0, curvature, -1), 0.0
    )
    refined[inner] += np.clip(offset, -0.5, 0.5)
    return refined


def pick_boundaries(
    score: FloatArray,
    min_distance: float,
    sensitivity: float | None,
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
            score,
            distance=distance,
            height=probability_threshold,
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
    ribbon: Ribbon, boundaries: list[CoreBoundary], score: FloatArray, length: float
) -> tuple[SampledPath, list[CoreBoundary], FloatArray]:
    """Re-express results with ``s = 0`` at the path end (the pith side).

    ``length`` is the exact arc length of the analysed path; the sample grid may stop up
    to one step before its end.
    """
    flipped = [
        CoreBoundary(s=length - b.s, xy=b.xy, tilt=b.tilt, strength=b.strength)
        for b in reversed(boundaries)
    ]
    return ribbon.path.reversed(), flipped, score[::-1].copy()
