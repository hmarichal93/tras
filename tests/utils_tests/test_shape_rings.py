"""Tests for the canvas-shape → ring-measurement bridge."""

import numpy as np
import pytest

from tras.utils.shape_rings import ring_property_rows
from tras.utils.shape_rings import rings_inner_to_outer
from tras.utils.shape_rings import split_ring_shapes


class Point:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def x(self):
        return self._x

    def y(self):
        return self._y


class FakeShape:
    def __init__(self, label, xy, shape_type="polygon"):
        self.label = label
        self.shape_type = shape_type
        self.points = [Point(x, y) for x, y in xy]


def circle(r, n=360):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def square(h):
    return [(-h, -h), (h, -h), (h, h), (-h, h)]


def test_split_and_order():
    shapes = [
        FakeShape("b", circle(20)),
        FakeShape("exclusion", square(3)),
        FakeShape("a", circle(10)),
        FakeShape("path", [(0, 0), (5, 5)], "linestrip"),
    ]
    rings, exclusions = split_ring_shapes(shapes)
    assert [s.label for s in rings] == ["b", "a"]
    assert [s.label for s in exclusions] == ["exclusion"]
    assert [s.label for s in rings_inner_to_outer(rings)] == ["a", "b"]


def test_rows_subtract_exclusions_and_keep_plain_areas():
    rings = [FakeShape("2001", circle(20)), FakeShape("2000", circle(10))]
    rows = ring_property_rows(rings, [FakeShape("exclusion", square(3))], pith_xy=(0, 0))
    assert [r["label"] for r in rows] == ["2000", "2001"]
    assert rows[0]["area_no_exclusion"] - rows[0]["area"] == pytest.approx(36, rel=1e-6)
    assert rows[1]["area"] == pytest.approx(np.pi * 300, rel=1e-3)
    assert rows[1]["mean_width"] == pytest.approx(10, abs=0.05)
    assert rows[0]["radial_width_px"] is None


def test_rows_without_pith_use_innermost_centroid():
    offset = np.array([50.0, 40.0])
    rings = [FakeShape("1", circle(10) + offset), FakeShape("2", circle(20) + offset)]
    rows = ring_property_rows(rings)
    assert rows[1]["mean_width"] == pytest.approx(10, abs=0.05)
    assert "area_no_exclusion" not in rows[0]
