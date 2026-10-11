"""CooRecorder / CDendro ``.pos`` coordinate files (read and write).

A ``.pos`` file stores the points a user (or TRAS) placed on ring boundaries along a
measurement path, in millimetres of image coordinates (``mm = px / DPI × 25.4``).
Spec: ``docs/research/core-detection-sota.md``.

Point lines:

* ``x,y``: a ring boundary;
* ``x1,y1  x2,y2``: a *double point*: one boundary seen at two places. The ring ends at
  the first coordinate and the next ring starts at the second; the jump is not measured;
* ``Dx,y #W#``: an earlywood/latewood border inside a ring (on the path, not a boundary);
* ``Dx,y #%gap``: crack marker. Two in a row: the segment between them is skipped. One
  alone: the segment from it to the next point is skipped.

Direction: CooRecorder convention (followed by modendro and TRAS's writer) is bark first,
so ``DATED <year>`` is the year of the first ring. Files measured from the pith outward
exist too. ``PosFile.pith_first()`` decides from ``PithCoordinates`` when present.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np

MM_PER_INCH = 25.4

RING = "ring"
SEASONWOOD = "seasonwood"
GAP = "gap"


@dataclass(frozen=True)
class PosPoint:
    """One point line. Coordinates in mm; ``end`` is the second half of a double point."""

    x: float
    y: float
    kind: str = RING  # ring | seasonwood | gap
    end: tuple[float, float] | None = None
    label: str = ""


@dataclass(frozen=True)
class PosFile:
    points: tuple[PosPoint, ...]
    dpi: float | None = None
    image_file: str | None = None
    dated_year: int | None = None  # year of the first ring in file order
    pith_xy_mm: tuple[float, float] | None = None
    distance_to_pith_mm: float | None = None
    years_to_pith: int | None = None
    comments: tuple[str, ...] = field(default_factory=tuple)

    def pith_first(self) -> bool | None:
        """True if points run pith→bark, False if bark→pith, None if unknown.

        Decided from ``PithCoordinates``: the end of the path closer to the pith.
        """
        rings = [p for p in self.points if p.kind == RING]
        if self.pith_xy_mm is None or len(rings) < 2:
            return None
        px, py = self.pith_xy_mm
        first = np.hypot(rings[0].x - px, rings[0].y - py)
        last = np.hypot(rings[-1].x - px, rings[-1].y - py)
        return bool(first < last)

    def ring_widths_mm(self) -> list[float]:
        """Ring widths in file order (one per pair of consecutive boundaries)."""
        return _ring_widths(self.points)

    def chronological_widths_mm(self, pith_first: bool | None = None) -> list[float]:
        """Ring widths ordered oldest (pith side) first.

        Args:
            pith_first: Override the direction; default ``self.pith_first()`` and, if
                that is unknown, the CooRecorder bark-first convention.
        """
        if pith_first is None:
            pith_first = self.pith_first() or False
        widths = self.ring_widths_mm()
        return widths if pith_first else widths[::-1]

    def to_pixels(self, xy_mm: tuple[float, float]) -> tuple[float, float]:
        if not self.dpi:
            raise ValueError("DPI unknown: cannot convert mm to pixels.")
        factor = self.dpi / MM_PER_INCH
        return xy_mm[0] * factor, xy_mm[1] * factor


# ---------------------------------------------------------------------------- reading

_FLOAT = r"[-+]?\d+(?:[.,]\d+)?"
_POINT_RE = re.compile(
    rf"^(?P<d>D)?(?P<x>{_FLOAT}),(?P<y>{_FLOAT})(?:\s+(?P<x2>{_FLOAT}),(?P<y2>{_FLOAT}))?"
    rf"\s*(?:#(?P<label>.*))?$"
)


def read_pos(path: Path | str) -> PosFile:
    """Parse a ``.pos`` file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    header: dict = {"comments": []}
    points: list[PosPoint] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _POINT_RE.match(line)
        if match:
            points.append(_parse_point(match))
        else:
            _parse_header_line(line, header)
    header["comments"] = tuple(header["comments"])
    return PosFile(points=tuple(points), **header)


def _parse_point(match: re.Match) -> PosPoint:
    label = (match["label"] or "").strip()
    if match["d"]:
        kind = GAP if label.startswith("%") else SEASONWOOD
    else:
        kind = RING
    end = None
    if match["x2"] is not None:
        end = (_num(match["x2"]), _num(match["y2"]))
    return PosPoint(_num(match["x"]), _num(match["y"]), kind=kind, end=end, label=label)


def _parse_header_line(line: str, header: dict) -> None:
    if line.startswith("#DPI"):
        header["dpi"] = _num(line.split()[1])
    elif line.startswith("#Imagefile"):
        header["image_file"] = line[len("#Imagefile") :].strip()
    elif line.startswith("#C"):
        _parse_comment(line[2:].strip(), header)


def _parse_comment(body: str, header: dict) -> None:
    if body.startswith("DATED"):
        year = body[len("DATED") :].strip().split(";")[0].strip()
        header["dated_year"] = int(year) if re.fullmatch(r"-?\d+", year) else None
        return
    if "=" not in body:
        header["comments"].append(body)
        return
    for item in filter(None, (part.strip() for part in body.split(";"))):
        key, _, value = item.partition("=")
        if key == "PithCoordinates":
            x, y = value.split(",")
            header["pith_xy_mm"] = (_num(x), _num(y))
        elif key == "DistanceToPith":
            header["distance_to_pith_mm"] = _num(value)
        elif key == "YearsToPith" and re.fullmatch(r"\d+", value.strip()):
            header["years_to_pith"] = int(value)


def _num(text: str) -> float:
    return float(text.replace(",", "."))


# ------------------------------------------------------------------- ring widths


def _ring_widths(points: tuple[PosPoint, ...]) -> list[float]:
    """Walk the path and sum segment lengths between consecutive ring boundaries."""
    widths: list[float] = []
    current: float | None = None  # None until the first boundary is reached
    previous: tuple[float, float] | None = None
    skip_next = False
    for i, point in enumerate(points):
        segment = (
            0.0
            if previous is None
            else float(np.hypot(point.x - previous[0], point.y - previous[1]))
        )
        if current is not None and not skip_next:
            current += segment
        skip_next = point.kind == GAP and not _is_second_of_gap_pair(points, i)
        if point.kind == RING:
            if current is not None:
                widths.append(current)
            current = 0.0
        previous = point.end or (point.x, point.y)
    return widths


def _is_second_of_gap_pair(points: tuple[PosPoint, ...], i: int) -> bool:
    """A gap marker closes a pair when an odd number of gap markers directly precede it."""
    run = 0
    j = i - 1
    while j >= 0 and points[j].kind == GAP:
        run += 1
        j -= 1
    return run % 2 == 1


# ---------------------------------------------------------------------------- writing


def write_pos(path: Path | str, pos: PosFile, written: _dt.datetime | None = None) -> None:
    """Write a ``.pos`` file CooRecorder/CDendro can open."""
    path = Path(path)
    stamp = (written or _dt.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"#DENDRO (Cybis Dendro program compatible format) Coordinate file written as "
        f"{path.name} {stamp}",
        f"#Imagefile {pos.image_file or ''}",
        f"#DPI {pos.dpi:.1f}" if pos.dpi else "#DPI 0.0",
        "#All coordinates in millimeters (mm)",
        "SCALE 1",
        f"#C DATED {pos.dated_year}" if pos.dated_year is not None else "#C DATED",
    ]
    lines += [f"#C {c}" for c in pos.comments]
    lines += _pith_comment(pos)
    lines.append(f"#C Written={stamp}; TRAS;")
    lines += [_format_point(p) for p in pos.points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pith_comment(pos: PosFile) -> list[str]:
    items = []
    if pos.pith_xy_mm is not None:
        items.append(f"PithCoordinates={pos.pith_xy_mm[0]:.3f},{pos.pith_xy_mm[1]:.3f}")
    if pos.distance_to_pith_mm is not None:
        items.append(f"DistanceToPith={pos.distance_to_pith_mm:.3f}")
    if pos.years_to_pith is not None:
        items.append(f"YearsToPith={pos.years_to_pith}")
    return [f"#C {'; '.join(items)};"] if items else []


def _format_point(point: PosPoint) -> str:
    prefix = "D" if point.kind in (SEASONWOOD, GAP) else ""
    text = f"{prefix}{point.x:.3f},{point.y:.3f}"
    if point.end is not None:
        text += f"  {point.end[0]:.3f},{point.end[1]:.3f}"
    if point.label:
        text += f" #{point.label}"
    return text
