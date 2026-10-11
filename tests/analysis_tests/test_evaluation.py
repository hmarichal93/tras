"""Tests for ring and boundary evaluation metrics."""

import numpy as np
import pytest

from tras.evaluation.cross_section import score_rings
from tras.evaluation.path import match_boundaries
from tras.evaluation.path import score_boundaries


def circle(radius, center=(0.0, 0.0), n=360):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], 1)


GT = [circle(r) for r in (10, 20, 30, 40)]


def test_perfect_detection():
    s = score_rings(GT, GT, (0, 0))
    assert (s.tp, s.fp, s.fn) == (4, 0, 0)
    assert s.f1 == 1.0
    assert s.rmse_px == pytest.approx(0.0, abs=1e-9)


def test_missed_and_extra_rings():
    detected = [circle(10.5), circle(29), circle(44.5), circle(48)]  # 20 missed; 48 extra
    s = score_rings(detected, GT, (0, 0))
    assert (s.tp, s.fp, s.fn) == (3, 1, 1)
    assert s.precision == pytest.approx(0.75)
    assert s.recall == pytest.approx(0.75)


def test_duplicate_detection_counts_once():
    s = score_rings([circle(20), circle(21)], GT, (0, 0))
    assert (s.tp, s.fp) == (1, 1)


def test_partial_ring_below_fraction_is_rejected():
    # Half of the ring sits on GT ring 2, half on ring 3: no zone gets >= 60 % of rays.
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = np.where(t < np.pi, 20, 30)
    half = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    s = score_rings([half], GT, (0, 0))
    assert s.tp == 0


def test_empty_inputs():
    assert score_rings([], GT, (0, 0)).fn == 4
    assert score_rings(GT, [], (0, 0)).fp == 4


def test_boundary_matching_is_one_to_one():
    pairs = match_boundaries([10.0, 10.5, 30.0], [10.2, 20.0, 29.0], tolerance=1.5)
    assert sorted(pairs) == [(0, 0), (2, 2)] or sorted(pairs) == [(1, 0), (2, 2)]
    s = score_boundaries([10.0, 10.5, 30.0], [10.2, 20.0, 29.0], tolerance=1.5)
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    assert s.mae < 1.0
