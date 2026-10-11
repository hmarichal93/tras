"""Tests for automatic core ring detection on synthetic cores (classical evidence only)."""

from pathlib import Path

import numpy as np
import pytest

from tras.cores import CoreParams
from tras.cores import analyze_core_image
from tras.cores import detect_core_rings
from tras.cores import find_cores
from tras.cores import to_label_shapes
from tras.cores import to_pos
from tras.cores import to_series
from tras.cores import write_csv
from tras.cores.ribbon import extract_ribbon
from tras.cores.ribbon import sample_path
from tras.io.pos import read_pos
from tras.io.pos import write_pos

BOUNDARIES = np.array([40.0, 95, 140, 200, 245, 280, 330, 370, 400, 445])
CLASSICAL = CoreParams(evidence="classical", half_width=8)


def synthetic_core(length=500, height=40, boundaries=BOUNDARIES, tilt_deg=0.0, flip=False):
    """Conifer-like core along +x: earlywood light, darkening to latewood, sharp boundary.

    Pith on the left (x=0) unless ``flip``. Boundaries tilt by ``tilt_deg`` from vertical.
    """
    x = np.arange(length, dtype=np.float64)[None, :]
    y = np.arange(height, dtype=np.float64)[:, None] - height / 2
    shifted = x - y * np.tan(np.deg2rad(tilt_deg))
    edges = np.r_[0.0, boundaries, length * 2]
    idx = np.clip(np.searchsorted(edges, shifted, side="right") - 1, 0, len(edges) - 2)
    start, end = edges[idx], edges[idx + 1]
    phase = np.clip((shifted - start) / (end - start), 0, 1)
    intensity = 210 - 120 * phase**3  # latewood darkens at the end of each ring
    rng = np.random.default_rng(0)
    intensity = intensity + rng.normal(0, 3, intensity.shape)
    image = np.repeat(np.clip(intensity, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    return image[:, ::-1] if flip else image


def horizontal_path(image, margin=0):
    h, w = image.shape[:2]
    return np.array([[margin, h / 2], [w - 1 - margin, h / 2]], dtype=np.float64)


def test_sample_path_is_unit_speed():
    path = sample_path([[0, 0], [10, 0], [10, 10]])
    assert path.length == pytest.approx(20, abs=1)
    np.testing.assert_allclose(np.linalg.norm(path.tangent, axis=1), 1)


def test_ribbon_of_straight_path_is_a_crop():
    image = np.arange(100 * 100, dtype=np.float64).reshape(100, 100)
    ribbon = extract_ribbon(image, [[10, 50], [60, 50]], half_width=3)
    assert ribbon.values.shape == (7, 51)
    np.testing.assert_allclose(ribbon.values[3], image[50, 10:61])


def test_detects_boundaries_with_known_pith_side():
    image = synthetic_core()
    result = detect_core_rings(image, horizontal_path(image), CoreParams(**{**vars(CLASSICAL), "pith_side": "start"}))
    found = np.array([b.s for b in result.boundaries])
    assert len(found) == len(BOUNDARIES)
    np.testing.assert_allclose(found, BOUNDARIES, atol=2)


@pytest.mark.parametrize("flip", [False, True])
def test_auto_orientation_puts_pith_first(flip):
    image = synthetic_core(flip=flip)
    result = detect_core_rings(image, horizontal_path(image), CLASSICAL)
    pith_x = image.shape[1] - 1 if flip else 0
    assert abs(result.path.xy[0][0] - pith_x) < 1
    assert len(result.boundaries) == len(BOUNDARIES)


def test_tilted_boundaries_give_perpendicular_widths():
    tilt = 30.0
    image = synthetic_core(height=60, tilt_deg=tilt)
    result = detect_core_rings(image, horizontal_path(image), CoreParams(evidence="classical", half_width=20))
    assert len(result.boundaries) == len(BOUNDARIES)
    tilts = np.rad2deg([abs(b.tilt) for b in result.boundaries])
    assert np.median(tilts) == pytest.approx(tilt, abs=6)
    expected = np.diff(BOUNDARIES) * np.cos(np.deg2rad(tilt))
    np.testing.assert_allclose(result.widths_px, expected, atol=3)


def test_years_and_mm_widths():
    image = synthetic_core()
    result = detect_core_rings(image, horizontal_path(image), CLASSICAL, mm_per_px=0.01, outer_year=2020)
    assert result.years[-1] == 2020
    assert len(result.years) == len(BOUNDARIES) - 1
    np.testing.assert_allclose(result.widths_mm, result.widths_px * 0.01)


def place_on_background(core, canvas_shape=(400, 700), top_left=(50, 100), background=30):
    canvas = np.full((*canvas_shape, 3), background, dtype=np.uint8)
    y, x = top_left
    canvas[y : y + core.shape[0], x : x + core.shape[1]] = core
    return canvas


def test_find_cores_on_dark_background():
    core = synthetic_core()
    canvas = place_on_background(core, canvas_shape=(400, 700))
    canvas[250:290, 100:600] = synthetic_core()[:, :500]
    cores = find_cores(canvas)
    assert len(cores) == 2
    assert cores[0].path_xy[:, 1].mean() == pytest.approx(70, abs=4)
    assert cores[0].diameter_px == pytest.approx(40, abs=6)
    assert cores[0].length_px > 400


def test_analyze_core_image_end_to_end():
    canvas = place_on_background(synthetic_core())
    (result,) = analyze_core_image(canvas, params=CLASSICAL, mm_per_px=0.02, outer_year=2010)
    found = np.sort([b.xy[0] - 100 for b in result.boundaries])
    matched = [np.min(np.abs(found - b)) < 3 for b in BOUNDARIES[1:-1]]
    assert np.mean(matched) >= 0.9  # core tips are trimmed, so the ends may be lost


def test_exports_round_trip(tmp_path: Path):
    image = synthetic_core(height=60, tilt_deg=20)
    result = detect_core_rings(
        image, horizontal_path(image), CoreParams(evidence="classical", half_width=20),
        mm_per_px=0.01, outer_year=2020,
    )

    pos_path = tmp_path / "core.pos"
    write_pos(pos_path, to_pos(result, image_file="core.png"))
    pos = read_pos(pos_path)
    assert pos.dated_year == 2020
    # CooRecorder recomputes the same perpendicular widths from the double points.
    np.testing.assert_allclose(pos.chronological_widths_mm(pith_first=False), result.widths_mm, atol=1e-3)

    series = to_series(result, "CORE1")
    assert series.last_year == 2020
    np.testing.assert_allclose(series.widths_mm, result.widths_mm)

    csv_path = tmp_path / "core.csv"
    write_csv(result, csv_path)
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("ring,year,width_mm")
    assert len(lines) == len(result.widths_px) + 1

    shapes = to_label_shapes(result)
    assert shapes[0]["shape_type"] == "linestrip"
    assert shapes[-1]["label"] == "core1_2020"
    assert all(len(s["points"]) == 2 for s in shapes[1:])
