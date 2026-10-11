"""Tests for ring geometry, measurements and QC flags."""

import numpy as np
import pytest

from tras.analysis.geometry import polygon_area
from tras.analysis.geometry import polygon_centroid
from tras.analysis.geometry import polygon_perimeter
from tras.analysis.geometry import ray_angles
from tras.analysis.geometry import ray_polygon_distances
from tras.analysis.qc import flag_rings
from tras.analysis.rings import measure_rings
from tras.analysis.rings import radial_profiles
from tras.analysis.rings import radial_widths


def circle(radius, center=(0.0, 0.0), n=720):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], 1)


def square(half):
    return np.array([[-half, -half], [half, -half], [half, half], [-half, half]], float)


def test_square_area_perimeter_centroid():
    sq = square(2) + 5
    assert polygon_area(sq) == pytest.approx(16)
    assert polygon_perimeter(sq) == pytest.approx(16)
    assert polygon_centroid(sq) == pytest.approx((5, 5))


def test_ray_distances_on_square():
    d = ray_polygon_distances(square(1), (0, 0), np.array([0, np.pi / 4, np.pi / 2]))
    np.testing.assert_allclose(d, [1, np.sqrt(2), 1])


def test_ray_distance_takes_outer_hit_on_concave_ring():
    # U-shape opening upwards; the ray towards -y crosses two edges.
    shape = np.array(
        [[-3, -3], [3, -3], [3, 3], [1, 3], [1, -1], [-1, -1], [-1, 3], [-3, 3]], float
    )
    d = ray_polygon_distances(shape, (0, 1), np.array([-np.pi / 2]))
    assert d[0] == pytest.approx(4.0)


def test_ray_missing_ring_gives_nan():
    d = ray_polygon_distances(square(1) + 10, (0, 0), ray_angles(4))
    assert np.isnan(d).sum() >= 3


def test_concentric_circles_widths():
    rings = [circle(30), circle(10), circle(20)]  # unordered on purpose
    m = measure_rings(rings, (0, 0), outer_year=2020)
    assert [r.index for r in m] == [1, 2, 3]
    assert [r.year for r in m] == [2018, 2019, 2020]
    for r in m:
        assert r.mean_width == pytest.approx(10, abs=0.05)
        assert r.min_width == pytest.approx(10, abs=0.05)
    assert m[0].area == pytest.approx(np.pi * 100, rel=1e-3)
    assert m[1].area == pytest.approx(np.pi * (400 - 100), rel=1e-3)
    assert m[2].mean_radius == pytest.approx(30, abs=0.05)
    assert m[2].eccentricity == pytest.approx(0, abs=1e-6)


def test_eccentric_ring_has_uneven_width():
    m = measure_rings([circle(10), circle(20, center=(5, 0))], (0, 0))
    assert m[1].max_width > m[1].min_width + 9
    assert m[1].eccentricity == pytest.approx(5, abs=1e-3)


def test_exclusion_zone_reduces_area():
    exclusion = square(5)  # 100 px² fully inside the first ring
    m = measure_rings([circle(10), circle(20)], (0, 0), exclusions=[exclusion])
    assert m[0].excluded_area == pytest.approx(100, rel=1e-6)
    assert m[0].area == pytest.approx(np.pi * 100 - 100, rel=1e-3)
    # The second ring's own band is unaffected.
    assert m[1].area == pytest.approx(np.pi * 300, rel=1e-3)


def test_custom_labels_follow_ordering():
    m = measure_rings([circle(20), circle(10)], (0, 0), labels=["outer", "inner"])
    assert [r.label for r in m] == ["inner", "outer"]


def test_scaled_converts_units():
    r = measure_rings([circle(10)], (0, 0))[0]
    scaled = r.scaled(0.5)
    assert scaled["mean_width"] == pytest.approx(r.mean_width * 0.5)
    assert scaled["area"] == pytest.approx(r.area * 0.25)


def test_qc_flags_narrow_ring_and_crossing():
    radii = [10, 20, 30, 40, 41, 50, 60, 70, 80]  # ring 5 (41) is a 1 px false ring
    rings = [circle(r) for r in radii]
    m = measure_rings(rings, (0, 0))
    flags = flag_rings(m)
    assert [(f.ring_index, f.code) for f in flags] == [(5, "narrow")]

    crossing = [circle(10), circle(12, center=(4, 0))]
    profiles = radial_profiles(crossing, (0, 0), 360)
    m = measure_rings(crossing, (0, 0))
    codes = {f.code for f in flag_rings(m, radial_widths(profiles))}
    assert "crossing" in codes


def test_qc_no_flags_on_regular_growth():
    rings = [circle(r) for r in np.cumsum([12, 11, 10, 10, 9, 8, 8, 7, 7, 6])]
    assert flag_rings(measure_rings(rings, (0, 0))) == []
