"""Tucson ring-width files (``.rwl``), the exchange format of dplR, COFECHA and ITRDB.

Each data line holds an 8-character series ID, a 4-digit year (the decade start, except
for a series' first line) and up to ten 6-character integer widths. Units are 0.01 mm
(series ends with ``999``) or 0.001 mm (ends with ``-9999``); the end marker occupies the
slot after the last year. Spec: ``docs/research/core-detection-sota.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PRECISIONS = {0.01: 999, 0.001: -9999}
ID_WIDTH = 8
VALUES_PER_LINE = 10
_ID_ALLOWED = re.compile(r"[^A-Za-z0-9_\-.]")


@dataclass(frozen=True)
class RingWidthSeries:
    """Widths (mm) of consecutive years, oldest first."""

    series_id: str
    first_year: int
    widths_mm: tuple[float, ...]

    @property
    def last_year(self) -> int:
        return self.first_year + len(self.widths_mm) - 1

    @property
    def years(self) -> range:
        return range(self.first_year, self.last_year + 1)


def sanitize_series_id(name: str) -> str:
    """Make a Tucson-safe ID: allowed characters only, at most 8 chars."""
    cleaned = _ID_ALLOWED.sub("", name.replace(" ", "_"))
    return (cleaned or "TRAS")[:ID_WIDTH]


def write_rwl(
    path: Path | str, series: Sequence[RingWidthSeries], precision: float = 0.01
) -> None:
    """Write series to a Tucson ``.rwl`` file.

    Args:
        precision: 0.01 or 0.001 mm per unit. 0.001 is recommended for narrow rings.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    ids = [sanitize_series_id(s.series_id) for s in series]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Series IDs are not unique after sanitizing: {ids}")
    lines = []
    for sid, s in zip(ids, series):
        lines += _series_lines(sid, s, precision)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _series_lines(sid: str, series: RingWidthSeries, precision: float) -> list[str]:
    stop = PRECISIONS[precision]
    values = [_to_units(w, precision) for w in series.widths_mm] + [stop]
    years = list(range(series.first_year, series.first_year + len(values)))

    lines = []
    start = 0
    while start < len(values):
        year = years[start]
        # A line runs to the end of the decade (years ending in 9).
        count = min(VALUES_PER_LINE - (year % VALUES_PER_LINE), len(values) - start)
        chunk = values[start : start + count]
        lines.append(f"{sid:<{ID_WIDTH}}{year:>4}" + "".join(f"{v:>6d}" for v in chunk))
        start += count
    return lines


def _to_units(width_mm: float, precision: float) -> int:
    value = int(round(width_mm / precision))
    if precision == 0.01 and value == PRECISIONS[0.01]:
        value = 998  # avoid writing the stop marker as data (dplR does the same)
    return value


def read_rwl(path: Path | str) -> list[RingWidthSeries]:
    """Read a Tucson ``.rwl`` file (header lines are skipped)."""
    data: dict[str, dict[int, int]] = {}
    stops: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_data_line(line)
        if parsed is None:
            continue
        sid, year, values = parsed
        for offset, value in enumerate(values):
            if value in (999, -9999):
                stops[sid] = value
                break
            data.setdefault(sid, {})[year + offset] = value

    series = []
    for sid, by_year in data.items():
        precision = 0.001 if stops.get(sid) == -9999 else 0.01
        first, last = min(by_year), max(by_year)
        widths = tuple(by_year.get(y, 0) * precision for y in range(first, last + 1))
        series.append(RingWidthSeries(sid, first, widths))
    return series


def _parse_data_line(line: str) -> tuple[str, int, list[int]] | None:
    if len(line) < ID_WIDTH + 4:
        return None
    year_text = line[ID_WIDTH : ID_WIDTH + 4].strip()
    if not re.fullmatch(r"-?\d+", year_text):
        return None
    body = line[ID_WIDTH + 4 :].rstrip()
    chunks = [body[i : i + 6] for i in range(0, len(body), 6)]
    try:
        values = [int(c) for c in chunks if c.strip()]
    except ValueError:
        return None  # header line with letters in the value columns
    return line[:ID_WIDTH].strip(), int(year_text), values
