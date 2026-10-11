"""Rebuild core measurements from (possibly hand-edited) label shapes.

Core analysis puts its result on the canvas as editable shapes (``to_label_shapes``):
the path ``core_path_<k>`` and one ``line`` per boundary, ``core<k>_<year or b<i>>``.
After the user drags, deletes or adds boundaries, those shapes - not the detector output -
are the measurement. This module turns them back into ``CoreResult`` objects so every
export (.pos, .rwl, CSV) reflects the corrections.

A boundary is where its line crosses the path; its tilt is the line's angle to the path
normal, so widths stay perpendicular to the rings as in detection. Lines labelled
``core_boundary`` (drawn with *Draw Core Boundary*) join the core whose crossing is
nearest the line's midpoint. Two lines on the same boundary count once (the hand-drawn one
wins), and implausibly steep lines take the median tilt of their core.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from tras.cores.detect import CoreBoundary
from tras.cores.detect import CoreParams
from tras.cores.detect import CoreResult
from tras.cores.export import boundary_name
from tras.cores.ribbon import SampledPath
from tras.cores.ribbon import sample_path

FloatArray = npt.NDArray[np.float64]

CORE_BOUNDARY_LABEL = "core_boundary"
_PATH_LABEL = re.compile(r"^core_path_(\d+)$")
_BOUNDARY_LABEL = re.compile(r"^core(\d+)_(\d+|b\d+)$")

# A line counts as crossing the path if the crossing lies within this many half-lengths
# of the line's midpoint: hand-drawn lines often stop just short of the path.
REACH_HALF_LENGTHS = 1.5
# Boundaries closer than this along the path (px) are one boundary drawn twice; a real
# ring is at least min_ring_width (≥ 1 sample, default 5) wide.
DUPLICATE_TOLERANCE_PX = 2.0
# A hand-drawn line tilted more than this from the path normal (deg) is unreliable; the
# detector searches the same range (CoreParams.max_tilt_deg default).
MAX_HAND_TILT_DEG = CoreParams.max_tilt_deg
# Line half-length (px) used when a core has no boundary lines to take it from.
DEFAULT_HALF_WIDTH_PX = 25
_STRENGTH = re.compile(r"strength=([-+0-9.eE]+|nan)")


class ShapeDict(TypedDict, total=False):
    """Labelme shape as stored in TRAS label files."""

    label: str
    points: list[list[float]]
    shape_type: str
    description: str


@dataclass(frozen=True)
class AssignedBoundary:
    """A boundary line matched to a core path."""

    shape_index: int  # position of the line in the input shape list
    core: int
    boundary: CoreBoundary
    half_length: float


def cores_from_shapes(
    shapes: Sequence[ShapeDict],
    *,
    mm_per_px: float | None = None,
    outer_year: int | None = None,
) -> dict[int, CoreResult]:
    """Core index → measurement rebuilt from the path and boundary-line shapes."""
    paths = core_paths(shapes)
    assigned = assign_boundaries(shapes, paths)
    return {
        k: _result(path, core_boundaries(assigned, k), mm_per_px, outer_year)
        for k, path in paths.items()
    }


def boundary_labels(shapes: Sequence[ShapeDict], outer_year: int | None = None) -> dict[int, str]:
    """Shape index → label after re-numbering each core's boundaries pith → bark.

    Applying these labels keeps year labels correct after boundaries are added or deleted.
    Dropped duplicates and lines that cross no path get no new label.
    """
    assigned = assign_boundaries(shapes, core_paths(shapes))
    labels: dict[int, str] = {}
    for k in sorted({a.core for a in assigned}):
        ordered = core_boundaries(assigned, k)
        n_rings = len(ordered) - 1
        years = (
            None if outer_year is None else list(range(outer_year - n_rings + 1, outer_year + 1))
        )
        for i, a in enumerate(ordered):
            labels[a.shape_index] = f"core{k}_{boundary_name(i, years)}"
    return labels


def duplicate_boundaries(shapes: Sequence[ShapeDict]) -> list[int]:
    """Indices of boundary lines dropped because another line marks the same boundary."""
    assigned = assign_boundaries(shapes, core_paths(shapes))
    dropped: list[int] = []
    for k in sorted({a.core for a in assigned}):
        dropped += _drop_duplicates([a for a in assigned if a.core == k])[1]
    return sorted(dropped)


def unassigned_boundaries(shapes: Sequence[ShapeDict]) -> list[int]:
    """Indices of hand-drawn ``core_boundary`` lines that cross no core path (ignored)."""
    assigned = {a.shape_index for a in assign_boundaries(shapes, core_paths(shapes))}
    return [
        index
        for index, shape in enumerate(shapes)
        if shape.get("label") == CORE_BOUNDARY_LABEL and index not in assigned
    ]


def core_paths(shapes: Sequence[ShapeDict]) -> dict[int, SampledPath]:
    """Analysed core paths (``core_path_<k>``), oriented pith → bark as stored."""
    paths = {}
    for shape in shapes:
        match = _PATH_LABEL.match(shape.get("label") or "")
        if match and len(shape.get("points") or []) >= 2:
            paths[int(match.group(1))] = sample_path(shape["points"])
    return paths


def assign_boundaries(
    shapes: Sequence[ShapeDict], paths: dict[int, SampledPath]
) -> list[AssignedBoundary]:
    """Every boundary line that crosses its core's path (or, for ``core_boundary``, any)."""
    matches = (_assign_line(index, shape, paths) for index, shape in enumerate(shapes))
    return [match for match in matches if match is not None]


def core_boundaries(assigned: list[AssignedBoundary], core: int) -> list[AssignedBoundary]:
    """The boundaries of one core, pith → bark, without duplicates and with sane tilts."""
    kept, _ = _drop_duplicates([a for a in assigned if a.core == core])
    return _clamp_tilts(kept)


def _assign_line(
    index: int, shape: ShapeDict, paths: dict[int, SampledPath]
) -> AssignedBoundary | None:
    """Match one line to the candidate core whose crossing is nearest the line midpoint."""
    if shape.get("shape_type") != "line" or len(shape.get("points") or []) != 2:
        return None
    line = np.asarray(shape["points"], dtype=np.float64)
    crossings = {
        core: boundary
        for core in _candidate_cores(shape.get("label") or "", paths)
        if (boundary := boundary_on_path(line, paths[core])) is not None
    }
    if not crossings:
        return None
    midpoint = line.mean(axis=0)
    core = min(crossings, key=lambda k: _distance(crossings[k].xy, midpoint))
    boundary = replace(crossings[core], strength=_strength(shape.get("description") or ""))
    half_length = float(np.linalg.norm(line[1] - line[0])) / 2.0
    return AssignedBoundary(index, core, boundary, half_length)


def _candidate_cores(label: str, paths: dict[int, SampledPath]) -> list[int]:
    """Cores a line may belong to: its own (``core<k>_…``) or, for new lines, any."""
    if label == CORE_BOUNDARY_LABEL:
        return sorted(paths)
    match = _BOUNDARY_LABEL.match(label)
    return [int(match.group(1))] if match and int(match.group(1)) in paths else []


def boundary_on_path(line: FloatArray, path: SampledPath) -> CoreBoundary | None:
    """Where the (extended) line crosses the path, nearest to the line's midpoint.

    Returns None when the nearest crossing is farther than ``REACH_HALF_LENGTHS``
    half-lengths from the midpoint, i.e. the line is not on this path.
    """
    half_length = float(np.linalg.norm(line[1] - line[0])) / 2.0
    if half_length == 0:
        return None
    midpoint = line.mean(axis=0)
    direction = (line[1] - line[0]) / (2.0 * half_length)
    across = np.array([-direction[1], direction[0]])
    side = (path.xy - midpoint) @ across  # signed distance of path samples to the line
    crossings = np.flatnonzero(np.sign(side[:-1]) * np.sign(side[1:]) <= 0)
    if len(crossings) == 0:
        return None
    s_values = [_crossing_s(side, i, path.step) for i in crossings]
    points = path.point_at(np.array(s_values))
    distances = np.linalg.norm(points - midpoint, axis=1)
    best = int(np.argmin(distances))
    if distances[best] > REACH_HALF_LENGTHS * max(half_length, 1.0):
        return None
    s = s_values[best]
    return CoreBoundary(s, tuple(points[best].tolist()), _tilt(direction, path, s), np.nan)


def _distance(point: tuple[float, float], other: FloatArray) -> float:
    return float(np.linalg.norm(np.subtract(point, other)))


def _crossing_s(side: FloatArray, i: int, step: float) -> float:
    """Arc length of the zero crossing of ``side`` between samples ``i`` and ``i + 1``."""
    denominator = side[i] - side[i + 1]
    fraction = 0.0 if denominator == 0 else side[i] / denominator
    return (i + float(fraction)) * step


def _tilt(direction: FloatArray, path: SampledPath, s: float) -> float:
    """Angle (rad) of a boundary direction to the path normal, in (-π/2, π/2]."""
    idx = int(np.clip(round(s / path.step), 0, len(path.xy) - 1))
    along, across = direction @ path.tangent[idx], direction @ path.normal[idx]
    if across < 0:  # a line has no direction: use the one facing the normal
        along, across = -along, -across
    return float(np.arctan2(along, across))


def _result(
    path: SampledPath,
    ordered: list[AssignedBoundary],
    mm_per_px: float | None,
    outer_year: int | None,
) -> CoreResult:
    """Measurement of one core from its (de-duplicated, ordered) boundary lines."""
    half_lengths = [a.half_length for a in ordered]
    half_width = int(round(np.median(half_lengths))) if half_lengths else DEFAULT_HALF_WIDTH_PX
    return CoreResult(
        path=path,
        boundaries=tuple(a.boundary for a in ordered),
        half_width=half_width,
        mm_per_px=mm_per_px,
        outer_year=outer_year,
    )


def _drop_duplicates(
    assigned: list[AssignedBoundary],
) -> tuple[list[AssignedBoundary], list[int]]:
    """Sorted pith → bark, merging lines closer than ``DUPLICATE_TOLERANCE_PX``.

    Of two lines marking one boundary, the hand-drawn one wins (the user corrected the
    detector); otherwise the first one. Returns the kept lines and the dropped indices.
    """
    kept: list[AssignedBoundary] = []
    dropped: list[int] = []
    for a in sorted(assigned, key=lambda a: a.boundary.s):
        if not kept or a.boundary.s - kept[-1].boundary.s > DUPLICATE_TOLERANCE_PX:
            kept.append(a)
            continue
        if _is_hand_drawn(a) and not _is_hand_drawn(kept[-1]):
            kept[-1], a = a, kept[-1]
        dropped.append(a.shape_index)
    return kept, dropped


def _is_hand_drawn(a: AssignedBoundary) -> bool:
    """Detected lines carry their evidence strength; hand-drawn ones have none (NaN)."""
    return bool(np.isnan(a.boundary.strength))


def _clamp_tilts(ordered: list[AssignedBoundary]) -> list[AssignedBoundary]:
    """Replace tilts beyond ``MAX_HAND_TILT_DEG`` by the median of the core's other tilts.

    A steep tilt shrinks the perpendicular width by cos(tilt), so one sloppy hand-drawn
    line would distort two rings. With no reliable tilt left, assume 0 (across the path).
    """
    limit = np.deg2rad(MAX_HAND_TILT_DEG)
    reliable = [a.boundary.tilt for a in ordered if abs(a.boundary.tilt) <= limit]
    fallback = float(np.median(reliable)) if reliable else 0.0
    return [
        a
        if abs(a.boundary.tilt) <= limit
        else replace(a, boundary=replace(a.boundary, tilt=fallback))
        for a in ordered
    ]


def _strength(description: str) -> float:
    """Detector strength stored in the shape description; NaN for hand-drawn lines."""
    match = _STRENGTH.search(description)
    return float(match.group(1)) if match else float("nan")


def shape_to_dict(shape: Any) -> ShapeDict:
    """Duck-typed canvas ``Shape`` (QPointF points) → ``ShapeDict``; no Qt import needed."""
    return ShapeDict(
        label=shape.label or "",
        points=[[float(p.x()), float(p.y())] for p in shape.points],
        shape_type=shape.shape_type,
        description=getattr(shape, "description", "") or "",
    )


def _untranslated(text: str) -> str:
    return text


def export_warnings(
    cores: Sequence[CoreResult],
    *,
    n_unassigned: int = 0,
    n_duplicates: int = 0,
    tr: Callable[[str], str] = _untranslated,
) -> list[str]:
    """Why an export is incomplete, one message per problem (empty when all is well).

    ``tr`` translates the message templates (the GUI passes ``QObject.tr``).
    """
    warnings = []
    if any(core.mm_per_px is None for core in cores):
        warnings.append(
            tr(
                "No scale: .pos and .rwl skipped, widths in px. "
                "Set it with Tools → Increment Core → Set Image Scale."
            )
        )
    if any(core.outer_year is None for core in cores):
        warnings.append(
            tr(
                "Undated: .rwl skipped. Set the outermost ring year in Core Analysis "
                "or Sample Metadata."
            )
        )
    if n_unassigned:
        warnings.append(
            tr(
                "%d hand-drawn boundary line(s) do not cross a core path and were "
                "ignored (selected)."
            )
            % n_unassigned
        )
    if n_duplicates:
        warnings.append(
            tr("%d boundary line(s) duplicate another line on the same boundary and were ignored.")
            % n_duplicates
        )
    return warnings


def export_summary(
    cores: Sequence[CoreResult],
    written: Sequence[Path],
    warnings: Sequence[str] = (),
    tr: Callable[[str], str] = _untranslated,
) -> list[str]:
    """Lines of the "Export complete" message: totals, per-core rings, warnings, files."""
    n_rings = sum(len(core.widths_px) for core in cores)
    totals = tr("%d cores, %d rings") % (len(cores), n_rings)
    dated = [core.years for core in cores if core.years]
    if dated:
        totals += f", {min(y[0] for y in dated)}–{max(y[-1] for y in dated)}"
    lines = [totals]
    for k, core in enumerate(cores, start=1):
        span = f"{core.years[0]}–{core.years[-1]}" if core.years else tr("undated")
        lines.append(tr("Core %d: %d rings, %s") % (k, len(core.widths_px), span))
    lines += ["", tr("Year labels on the image updated.")]
    if warnings:
        lines += ["", *warnings]
    lines += ["", tr("Files written:"), *(str(path) for path in written)]
    return lines
