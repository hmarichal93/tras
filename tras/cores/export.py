"""Convert core detections to CooRecorder ``.pos``, Tucson ``.rwl``, CSV and label shapes.

``.pos`` geometry: CooRecorder sums path lengths between points. To make it reproduce
the *perpendicular* widths TRAS measures, every inner boundary is written as a double
point ``c  b``: ``b`` is where the path crosses the boundary, ``c`` is the foot of the
perpendicular dropped from the outer boundary's point onto this boundary. The ring then
measures ``|a → c|`` (perpendicular), and the next ring starts again from ``b``.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tras.cores.detect import CoreBoundary
from tras.cores.detect import CoreResult
from tras.io.pos import MM_PER_INCH
from tras.io.pos import PosFile
from tras.io.pos import PosPoint
from tras.io.pos import write_pos
from tras.io.tucson import RingWidthSeries
from tras.io.tucson import sanitize_series_id
from tras.io.tucson import write_rwl

CORE_PATH_LABEL = "core_path"


def boundary_direction(result: CoreResult, boundary: CoreBoundary) -> np.ndarray:
    """Unit vector along a boundary in image coordinates.

    The boundary makes angle ``tilt`` with the path normal: direction
    ``sin(tilt)·tangent + cos(tilt)·normal``.
    """
    idx = int(np.clip(round(boundary.s / result.path.step), 0, len(result.path.xy) - 1))
    tangent, normal = result.path.tangent[idx], result.path.normal[idx]
    return np.sin(boundary.tilt) * tangent + np.cos(boundary.tilt) * normal


def to_pos(
    result: CoreResult,
    image_file: str | None = None,
    comments: Sequence[str] = (),
) -> PosFile:
    """CooRecorder file, bark first, with perpendicular double points."""
    if result.mm_per_px is None:
        raise ValueError("A physical scale (mm_per_px) is required for .pos export.")
    mm = result.mm_per_px
    outward = list(result.boundaries)[::-1]  # bark → pith
    points = []
    for k, boundary in enumerate(outward):
        b = np.array(boundary.xy)
        if k == 0:
            points.append(PosPoint(b[0] * mm, b[1] * mm))
            continue
        a = np.array(outward[k - 1].xy)
        direction = boundary_direction(result, boundary)
        foot = b + np.dot(a - b, direction) * direction  # perpendicular foot of a
        points.append(PosPoint(foot[0] * mm, foot[1] * mm, end=(b[0] * mm, b[1] * mm)))
    return PosFile(
        points=tuple(points),
        dpi=MM_PER_INCH / mm,
        image_file=image_file,
        dated_year=result.outer_year,
        comments=tuple(comments) + ("TRAS automatic core detection",),
    )


def to_series(result: CoreResult, series_id: str) -> RingWidthSeries:
    """Tucson series (requires ``outer_year`` and ``mm_per_px``)."""
    if result.outer_year is None or result.widths_mm is None:
        raise ValueError("Tucson export needs outer_year and mm_per_px.")
    widths = tuple(float(w) for w in result.widths_mm)
    return RingWidthSeries(series_id, result.outer_year - len(widths) + 1, widths)


def write_csv(result: CoreResult, path: Path | str) -> None:
    """One row per ring (oldest first): year, width (px, mm), boundary position, QC data."""
    widths_px = result.widths_px
    widths_mm = result.widths_mm
    years = result.years
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ring",
                "year",
                "width_mm",
                "width_px",
                "path_width_px",
                "outer_boundary_s_px",
                "outer_boundary_x",
                "outer_boundary_y",
                "tilt_deg",
                "strength",
            ]
        )
        for k, width in enumerate(widths_px):
            outer = result.boundaries[k + 1]
            writer.writerow(
                [
                    k + 1,
                    "" if years is None else years[k],
                    "" if widths_mm is None else f"{widths_mm[k]:.4f}",
                    f"{width:.2f}",
                    f"{result.path_widths_px[k]:.2f}",
                    f"{outer.s:.2f}",
                    f"{outer.xy[0]:.2f}",
                    f"{outer.xy[1]:.2f}",
                    f"{np.rad2deg(outer.tilt):.1f}",
                    f"{outer.strength:.2f}",
                ]
            )


def to_label_shapes(result: CoreResult, core_index: int = 1) -> list[dict[str, Any]]:
    """Labelme shapes: the path (linestrip) and one line per boundary across the core.

    Boundary labels are ``core<k>_<year>`` when dated, else ``core<k>_b<index>``.
    """
    path_points = result.path.xy[:: max(1, len(result.path.xy) // 200)]
    shapes = [
        _shape(f"{CORE_PATH_LABEL}_{core_index}", path_points.tolist(), "linestrip", core_index)
    ]
    years = result.years
    for i, boundary in enumerate(result.boundaries):
        direction = boundary_direction(result, boundary) * result.half_width
        centre = np.array(boundary.xy)
        name = f"b{i}" if years is None or i == 0 else str(years[i - 1])
        shapes.append(
            _shape(
                f"core{core_index}_{name}",
                [(centre - direction).tolist(), (centre + direction).tolist()],
                "line",
                core_index,
                description=f"strength={boundary.strength:.2f}",
            )
        )
    return shapes


def _shape(
    label: str, points: list, shape_type: str, group: int, description: str = ""
) -> dict[str, Any]:
    return {
        "label": label,
        "points": [[float(x), float(y)] for x, y in points],
        "group_id": group,
        "shape_type": shape_type,
        "flags": {},
        "description": description,
    }


def export_results(
    results: Sequence[CoreResult], folder: Path | str, stem: str, image_file: str | None = None
) -> list[Path]:
    """Write per core ``<stem>_core<k>.csv`` (+ ``.pos`` when scaled) and, when dated and
    scaled, one ``<stem>.rwl`` with all cores. Returns the written paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    series = []
    for k, result in enumerate(results, start=1):
        name = f"{stem}_core{k}"
        write_csv(result, folder / f"{name}.csv")
        written.append(folder / f"{name}.csv")
        if result.mm_per_px is not None:
            write_pos(folder / f"{name}.pos", to_pos(result, image_file=image_file or stem))
            written.append(folder / f"{name}.pos")
        if result.outer_year is not None and result.mm_per_px is not None and len(result.widths_px):
            series.append(to_series(result, sanitize_series_id(f"{stem[:6]}{k}")))
    if series:
        write_rwl(folder / f"{stem}.rwl", series)
        written.append(folder / f"{stem}.rwl")
    return written
