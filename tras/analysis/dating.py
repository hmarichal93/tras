"""Crossdating: check that ring-width series agree year by year.

Trees of one site respond to the same climate, so their ring-width *patterns* match when
the series are dated correctly. A missed or false ring shifts the pattern by a year from
that point on. This module provides the standard statistics:

* **Normalization** removes the age trend so only the year-to-year signal is compared:
  ``p2yrsl`` (CDendro's default, ``w_t / (w_t + w_{t-1})``) or ``log_diff``
  (``log w_t − log w_{t-1}``).
* **Pearson r** and **Student's t** (``t = r·√(n−2)/√(1−r²)``) between normalized series.
* **Gleichläufigkeit (GLK)**: share of years in which both series change in the same
  direction (0.5 = random). Ties count half (Buras & Wilmking 2015).
* **Lag scan**: the statistics at every shift; the correct dating has the highest t.
* **Segment check** (COFECHA-style): correlation of overlapping segments with a master;
  segments that correlate better at another lag point at a dating error.

All series are ``{year: width}`` mappings or ``RingWidthSeries``.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tras.io.tucson import RingWidthSeries

# Minimum overlapping years for a correlation (CDendro and dplR use ~15–50).
MIN_OVERLAP = 15
# Default lag range scanned (years).
MAX_LAG = 10
# COFECHA defaults: 50-year segments lagged by 25 years.
SEGMENT_LENGTH = 50
SEGMENT_STEP = 25
# COFECHA's 99 % critical correlation for 50-year segments.
CRITICAL_R = 0.3281
# Floor on widths before taking logs (mm); zero = locally absent ring.
_WIDTH_FLOOR = 1e-3

YearSeries = Mapping[int, float]


@dataclass(frozen=True)
class MatchStats:
    lag: int  # years added to the series' dates
    overlap: int
    r: float
    t: float
    glk: float


@dataclass(frozen=True)
class SegmentCheck:
    start: int
    end: int
    r: float  # correlation as dated
    best_lag: int
    best_r: float
    flag: str  # "" | "A" (low but best as dated) | "B" (better at another lag)


def as_year_series(series: RingWidthSeries | YearSeries) -> dict[int, float]:
    if isinstance(series, RingWidthSeries):
        return {year: float(w) for year, w in zip(series.years, series.widths_mm)}
    return {int(y): float(w) for y, w in series.items()}


def normalize(series: YearSeries, method: str = "p2yrsl") -> dict[int, float]:
    """Year-to-year index series (one shorter than the input)."""
    years = sorted(series)
    out = {}
    for prev, year in zip(years[:-1], years[1:]):
        if year != prev + 1:
            continue
        a, b = max(series[prev], _WIDTH_FLOOR), max(series[year], _WIDTH_FLOOR)
        if method == "p2yrsl":
            out[year] = b / (a + b)
        elif method == "log_diff":
            out[year] = float(np.log(b) - np.log(a))
        else:
            raise ValueError(f"Unknown normalization '{method}'")
    return out


def t_value(r: float, n: int) -> float:
    if n <= 2 or abs(r) >= 1:
        return float("inf") if r > 0 and n > 2 else 0.0
    return float(r * np.sqrt(n - 2) / np.sqrt(1 - r * r))


def glk(a: Sequence[float], b: Sequence[float]) -> float:
    """Gleichläufigkeit of two aligned *raw* series (sign agreement of year-to-year change)."""
    da, db = np.sign(np.diff(a)), np.sign(np.diff(b))
    if len(da) == 0:
        return float("nan")
    agreement = np.where((da == 0) | (db == 0), 0.5, (da == db).astype(float))
    return float(agreement.mean())


def compare(
    series: YearSeries, reference: YearSeries, lag: int = 0, method: str = "p2yrsl"
) -> MatchStats:
    """Statistics of ``series`` shifted by ``lag`` years against ``reference``."""
    shifted = {year + lag: w for year, w in series.items()}
    years = sorted(set(shifted) & set(reference))
    index_a, index_b = normalize(shifted, method), normalize(reference, method)
    common = sorted(set(index_a) & set(index_b))
    if len(common) < 3:
        return MatchStats(lag, len(common), float("nan"), float("nan"), float("nan"))
    a = np.array([index_a[y] for y in common])
    b = np.array([index_b[y] for y in common])
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    raw_a = [shifted[y] for y in years]
    raw_b = [reference[y] for y in years]
    return MatchStats(lag, len(common), r, t_value(r, len(common)), glk(raw_a, raw_b))


def scan_lags(
    series: YearSeries,
    reference: YearSeries,
    max_lag: int = MAX_LAG,
    min_overlap: int = MIN_OVERLAP,
    method: str = "p2yrsl",
) -> list[MatchStats]:
    """Statistics at every lag in ``[-max_lag, max_lag]`` with enough overlap, best t first."""
    results = [compare(series, reference, lag, method) for lag in range(-max_lag, max_lag + 1)]
    valid = [m for m in results if m.overlap >= min_overlap and np.isfinite(m.t)]
    return sorted(valid, key=lambda m: m.t, reverse=True)


def build_master(series_list: Sequence[YearSeries], method: str = "p2yrsl") -> dict[int, float]:
    """Mean normalized index per year over all series (a simple master chronology)."""
    indices = [normalize(s, method) for s in series_list]
    years = sorted(set().union(*indices)) if indices else []
    return {y: float(np.mean([i[y] for i in indices if y in i])) for y in years}


def leave_one_out_masters(
    series_list: Sequence[YearSeries], method: str = "p2yrsl"
) -> list[dict[int, float]]:
    """For each series, the master built from all *other* series (no self-correlation)."""
    return [
        build_master([s for j, s in enumerate(series_list) if j != i], method)
        for i in range(len(series_list))
    ]


def check_segments(
    series: YearSeries,
    master_index: Mapping[int, float],
    method: str = "p2yrsl",
    length: int = SEGMENT_LENGTH,
    step: int = SEGMENT_STEP,
    max_lag: int = MAX_LAG,
) -> list[SegmentCheck]:
    """COFECHA-style check of overlapping segments against a normalized master.

    Flag ``B``: the segment correlates better at another lag (likely dating error).
    Flag ``A``: correlation below ``CRITICAL_R`` even though the dating is the best one.
    """
    index = normalize(series, method)
    if not index:
        return []
    first, last = min(index), max(index)
    checks = []
    start = first - (first % step)
    while start + length - 1 <= last + step:
        segment = {y: v for y, v in index.items() if start <= y < start + length}
        if len(segment) >= MIN_OVERLAP:
            checks.append(_check_segment(segment, master_index, start, start + length - 1, max_lag))
        start += step
    return checks


def _check_segment(
    segment: Mapping[int, float], master: Mapping[int, float], start: int, end: int, max_lag: int
) -> SegmentCheck:
    def r_at(lag: int) -> float:
        common = [y for y in segment if y + lag in master]
        if len(common) < MIN_OVERLAP:
            return float("nan")
        a = np.array([segment[y] for y in common])
        b = np.array([master[y + lag] for y in common])
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

    by_lag = {lag: r_at(lag) for lag in range(-max_lag, max_lag + 1)}
    dated = by_lag[0]
    finite = {lag: r for lag, r in by_lag.items() if np.isfinite(r)}
    best_lag = max(finite, key=lambda lag: finite[lag]) if finite else 0
    best_r = finite.get(best_lag, float("nan"))
    flag = ""
    if best_lag != 0 and np.isfinite(dated) and best_r > dated:
        flag = "B"
    elif not np.isfinite(dated) or dated < CRITICAL_R:
        flag = "A"
    return SegmentCheck(start, end, dated, best_lag, best_r, flag)
