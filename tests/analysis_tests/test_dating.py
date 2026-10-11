"""Tests for crossdating statistics."""

import numpy as np
import pytest

from tras.analysis.dating import build_master
from tras.analysis.dating import check_segments
from tras.analysis.dating import compare
from tras.analysis.dating import glk
from tras.analysis.dating import leave_one_out_masters
from tras.analysis.dating import normalize
from tras.analysis.dating import scan_lags
from tras.analysis.dating import t_value
from tras.analysis.rings import radial_width_series

FIRST_YEAR, N_YEARS = 1900, 120


def site_series(seed, n=N_YEARS, first=FIRST_YEAR):
    """Common climate signal × individual noise × age trend (widths in mm)."""
    climate = np.random.default_rng(42).normal(0, 0.35, n)
    noise = np.random.default_rng(seed).normal(0, 0.12, n)
    trend = 2.0 * np.exp(-np.arange(n) / 60) + 0.5
    widths = trend * np.exp(climate + noise)
    return {first + i: float(w) for i, w in enumerate(widths)}


def drop_year(series, year):
    """Simulate a missed ring: the year vanishes and older years shift forward by one."""
    out = {}
    for y, w in series.items():
        if y < year:
            out[y + 1] = w
        elif y > year:
            out[y] = w
    return out


def test_t_value_and_glk_basics():
    assert t_value(0.5, 27) == pytest.approx(0.5 * 5 / np.sqrt(0.75))
    assert glk([1, 2, 3, 2], [1, 3, 4, 1]) == 1.0
    assert glk([1, 2, 3], [3, 2, 1]) == 0.0


def test_normalizations_remove_trend():
    series = {2000 + i: 10.0 * 0.9**i for i in range(20)}  # pure decay, no signal
    assert np.std(list(normalize(series, "p2yrsl").values())) < 1e-9
    assert np.std(list(normalize(series, "log_diff").values())) < 1e-9


def test_correct_dating_has_best_t():
    a, b = site_series(1), site_series(2)
    best = scan_lags(a, b)[0]
    assert best.lag == 0
    assert best.t > 8
    assert best.glk > 0.7


def test_shifted_series_is_redated():
    a = site_series(1)
    shifted = {y - 3: w for y, w in a.items()}
    assert scan_lags(shifted, site_series(2))[0].lag == 3


def test_segment_check_flags_missing_ring():
    series = [site_series(seed) for seed in range(2, 8)]
    faulty = drop_year(site_series(1), 1960)  # years before 1960 are now one year late
    master = build_master(series)
    checks = check_segments(faulty, master)
    early = [c for c in checks if c.end < 1960]
    late = [c for c in checks if c.start > 1960]
    assert all(c.flag == "B" and c.best_lag == -1 for c in early)
    assert all(c.flag == "" for c in late)


def test_leave_one_out_excludes_self():
    series = [site_series(seed) for seed in range(3)]
    masters = leave_one_out_masters(series)
    assert masters[0] == build_master(series[1:])


def test_compare_overlap_counts_index_years():
    stats = compare(site_series(1), site_series(2))
    assert stats.overlap == N_YEARS - 1


def test_radial_width_series_from_circles():
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    rings = [np.stack([r * np.cos(t), r * np.sin(t)], 1) for r in (10, 25, 30)]
    series = radial_width_series(rings, (0, 0), [0, 90, 180, 270], outer_year=2000, unit_per_px=0.1)
    assert [s.series_id for s in series] == ["R000", "R090", "R180", "R270"]
    for s in series:
        assert s.first_year == 1998
        np.testing.assert_allclose(s.widths_mm, [1.0, 1.5, 0.5], atol=0.01)


def test_compass_series_ids_and_directions():
    from tras.analysis.rings import compass_series

    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    # Eccentric outer ring: wider towards +x (east).
    rings = [
        np.stack([10 * np.cos(t), 10 * np.sin(t)], 1),
        np.stack([5 + 20 * np.cos(t), 20 * np.sin(t)], 1),
    ]
    series = compass_series(rings, (0, 0), 2001, 0.5, "Disc 07/A")
    assert [s.series_id for s in series] == ["Disc_07N", "Disc_07E", "Disc_07S", "Disc_07W"]
    by_id = {s.series_id[-1]: s for s in series}
    assert by_id["E"].widths_mm[1] == pytest.approx((25 - 10) * 0.5, abs=0.05)
    assert by_id["W"].widths_mm[1] == pytest.approx((15 - 10) * 0.5, abs=0.05)
    assert all(s.first_year == 2000 for s in series)
