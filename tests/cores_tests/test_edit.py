"""Core measurements rebuilt from edited canvas shapes (tras.cores.edit)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tras.cores import CoreParams
from tras.cores import detect_core_rings
from tras.cores import to_label_shapes
from tras.cores.edit import CORE_BOUNDARY_LABEL
from tras.cores.edit import MAX_HAND_TILT_DEG
from tras.cores.edit import boundary_labels
from tras.cores.edit import cores_from_shapes
from tras.cores.edit import duplicate_boundaries
from tras.cores.edit import export_summary
from tras.cores.edit import export_warnings
from tras.cores.edit import unassigned_boundaries
from tras.cores.synthetic import centre_path
from tras.cores.synthetic import synthetic_core

PARAMS = CoreParams(evidence="classical", pith_side="start", min_ring_width=10)


@pytest.fixture(scope="module")
def detected():
    image = synthetic_core(height=60, tilt_deg=20)
    return detect_core_rings(image, centre_path(image), PARAMS, mm_per_px=0.01, outer_year=2020)


def _line(label, centre, half=20.0, angle=np.pi / 2):
    d = half * np.array([np.cos(angle), np.sin(angle)])
    c = np.asarray(centre, dtype=float)
    return {"label": label, "points": [list(c - d), list(c + d)], "shape_type": "line"}


def test_unedited_shapes_reproduce_detection(detected):
    shapes = to_label_shapes(detected, core_index=1)
    rebuilt = cores_from_shapes(shapes, mm_per_px=0.01, outer_year=2020)[1]

    assert len(rebuilt.boundaries) == len(detected.boundaries)
    np.testing.assert_allclose(
        [b.s for b in rebuilt.boundaries], [b.s for b in detected.boundaries], atol=1.0
    )
    np.testing.assert_allclose(rebuilt.widths_mm, detected.widths_mm, rtol=0.02, atol=0.01)
    assert rebuilt.years == detected.years


def test_deleted_boundary_merges_rings(detected):
    shapes = to_label_shapes(detected, core_index=1)
    del shapes[3]  # path is shapes[0]; drop the third boundary
    rebuilt = cores_from_shapes(shapes, mm_per_px=0.01, outer_year=2020)[1]

    assert len(rebuilt.widths_px) == len(detected.widths_px) - 1
    assert rebuilt.widths_px.sum() == pytest.approx(detected.widths_px.sum(), rel=0.02)
    assert rebuilt.years[-1] == 2020


def test_added_boundary_splits_ring_and_relabels_years(detected):
    shapes = to_label_shapes(detected, core_index=1)
    first, second = detected.boundaries[1], detected.boundaries[2]
    middle = (np.array(first.xy) + np.array(second.xy)) / 2
    shapes.append(_line(CORE_BOUNDARY_LABEL, middle))

    rebuilt = cores_from_shapes(shapes, mm_per_px=0.01, outer_year=2020)[1]
    assert len(rebuilt.widths_px) == len(detected.widths_px) + 1
    added = next(b for b in rebuilt.boundaries if np.isnan(b.strength))
    assert added.tilt == pytest.approx(0.0, abs=1e-6)  # drawn perpendicular to the path

    labels = boundary_labels(shapes, outer_year=2020)
    assert labels[len(shapes) - 1] == f"core1_{rebuilt.years[1]}"
    assert sorted(labels.values())[-1] == "core1_b0"  # innermost closes no ring
    assert labels[len(shapes) - 2] == "core1_2020"  # outermost boundary


def test_dragged_boundary_moves_width(detected):
    shapes = to_label_shapes(detected, core_index=1)
    for point in shapes[2]["points"]:  # second boundary, moved 5 px towards the bark
        point[0] += 5.0
    rebuilt = cores_from_shapes(shapes, mm_per_px=0.01)[1]
    assert rebuilt.boundaries[1].s == pytest.approx(detected.boundaries[1].s + 5.0, abs=1.0)


def test_lines_off_the_path_and_other_shapes_are_ignored(detected):
    shapes = to_label_shapes(detected, core_index=1)
    shapes.append(_line(CORE_BOUNDARY_LABEL, (100, 500)))  # far from the path
    shapes.append(_line("radial_measurement_line", (100, 30)))
    shapes.append({"label": "core1_x", "points": [[1, 1], [1, 1]], "shape_type": "line"})
    rebuilt = cores_from_shapes(shapes)[1]
    assert len(rebuilt.boundaries) == len(detected.boundaries)


def test_new_boundary_joins_the_path_it_crosses(detected):
    other = detected.path.xy + np.array([0.0, 200.0])
    shapes = [
        {"label": "core_path_1", "points": detected.path.xy.tolist(), "shape_type": "linestrip"},
        {"label": "core_path_2", "points": other.tolist(), "shape_type": "linestrip"},
        _line(CORE_BOUNDARY_LABEL, other[50]),
        _line(CORE_BOUNDARY_LABEL, other[150]),
    ]
    cores = cores_from_shapes(shapes)
    assert len(cores[1].boundaries) == 0
    assert len(cores[2].boundaries) == 2


def _path_shape(label, xy):
    return {"label": label, "points": np.asarray(xy).tolist(), "shape_type": "linestrip"}


def test_new_boundary_joins_the_core_crossed_nearest_its_midpoint(detected):
    """Regression: a line crossing two paths joined the first core, not the nearest."""
    other = detected.path.xy + np.array([0.0, 30.0])
    centre = other[100] + np.array([0.0, -5.0])  # spans both paths, closest to core 2
    shapes = [
        _path_shape("core_path_1", detected.path.xy),
        _path_shape("core_path_2", other),
        _line(CORE_BOUNDARY_LABEL, centre, half=40.0),
    ]
    cores = cores_from_shapes(shapes)
    assert len(cores[1].boundaries) == 0
    assert len(cores[2].boundaries) == 1


def test_near_duplicate_keeps_the_hand_drawn_line(detected):
    shapes = to_label_shapes(detected, core_index=1)
    target = detected.boundaries[2]
    shapes.append(_line(CORE_BOUNDARY_LABEL, np.array(target.xy) + [1.0, 0.0]))

    rebuilt = cores_from_shapes(shapes)[1]
    assert len(rebuilt.boundaries) == len(detected.boundaries)
    kept = rebuilt.boundaries[2]
    assert np.isnan(kept.strength)
    assert kept.s == pytest.approx(target.s + 1.0, abs=0.5)
    assert duplicate_boundaries(shapes) == [3]  # the detected line (path is shapes[0])
    assert 3 not in boundary_labels(shapes)


def test_distinct_boundaries_are_not_duplicates(detected):
    shapes = to_label_shapes(detected, core_index=1)
    assert duplicate_boundaries(shapes) == []


def test_steep_hand_line_takes_the_median_tilt_of_the_core(detected):
    shapes = to_label_shapes(detected, core_index=1)
    first, second = detected.boundaries[1], detected.boundaries[2]
    middle = (np.array(first.xy) + np.array(second.xy)) / 2
    steep = np.deg2rad(10.0)  # 80 degrees from the path normal
    shapes.append(_line(CORE_BOUNDARY_LABEL, middle, angle=steep))

    added = next(b for b in cores_from_shapes(shapes)[1].boundaries if np.isnan(b.strength))
    others = [b.tilt for b in detected.boundaries]
    assert added.tilt == pytest.approx(np.median(others), abs=1e-6)
    assert abs(np.rad2deg(added.tilt)) <= MAX_HAND_TILT_DEG


def test_steep_hand_line_alone_gets_zero_tilt(detected):
    shapes = [
        _path_shape("core_path_1", detected.path.xy),
        _line(CORE_BOUNDARY_LABEL, detected.path.xy[100], angle=np.deg2rad(5.0)),
    ]
    (boundary,) = cores_from_shapes(shapes)[1].boundaries
    assert boundary.tilt == 0.0


def test_unassigned_boundaries_lists_lines_off_every_path(detected):
    shapes = to_label_shapes(detected, core_index=1)
    end = detected.path.xy[-1]
    shapes.append(_line(CORE_BOUNDARY_LABEL, end + [30.0, 0.0]))  # beyond the path end
    shapes.append(_line(CORE_BOUNDARY_LABEL, (100, 500)))  # far from the path
    between = (np.array(detected.boundaries[1].xy) + np.array(detected.boundaries[2].xy)) / 2
    shapes.append(_line(CORE_BOUNDARY_LABEL, between))  # on the path

    n = len(shapes)
    assert unassigned_boundaries(shapes) == [n - 3, n - 2]
    rebuilt = cores_from_shapes(shapes)[1]
    assert len(rebuilt.boundaries) == len(detected.boundaries) + 1


def test_deleting_outermost_boundary_keeps_outer_year(detected):
    shapes = to_label_shapes(detected, core_index=1)
    del shapes[-1]
    rebuilt = cores_from_shapes(shapes, outer_year=2020)[1]
    assert len(rebuilt.widths_px) == len(detected.widths_px) - 1
    assert rebuilt.years[-1] == 2020
    assert boundary_labels(shapes, outer_year=2020)[len(shapes) - 1] == "core1_2020"


def test_export_summary_lists_rings_warnings_and_files(detected):
    undated = replace(detected, mm_per_px=None, outer_year=None)
    warnings = export_warnings([detected, undated], n_unassigned=2, n_duplicates=1)
    assert len(warnings) == 4
    assert warnings[0].startswith("No scale")
    assert warnings[1].startswith("Undated")
    assert warnings[2].startswith("2 hand-drawn boundary line(s)")
    assert warnings[3].startswith("1 boundary line(s) duplicate")

    n = len(detected.widths_px)
    lines = export_summary([detected, undated], [Path("a.csv")], warnings)
    assert lines[0] == f"2 cores, {2 * n} rings, {2020 - n + 1}–2020"
    assert lines[1] == f"Core 1: {n} rings, {2020 - n + 1}–2020"
    assert lines[2] == f"Core 2: {n} rings, undated"
    assert "Year labels on the image updated." in lines
    assert lines[-2:] == ["Files written:", "a.csv"]


def test_export_without_problems_has_no_warnings(detected):
    assert export_warnings([detected]) == []
