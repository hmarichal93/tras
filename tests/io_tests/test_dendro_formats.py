"""Round-trip and parsing tests for .pos and Tucson .rwl files."""

from pathlib import Path

import pytest

from tras.io.pos import GAP
from tras.io.pos import RING
from tras.io.pos import SEASONWOOD
from tras.io.pos import PosFile
from tras.io.pos import PosPoint
from tras.io.pos import read_pos
from tras.io.pos import write_pos
from tras.io.tucson import RingWidthSeries
from tras.io.tucson import read_rwl
from tras.io.tucson import sanitize_series_id
from tras.io.tucson import write_rwl

COORECORDER_SAMPLE = """#DENDRO (Cybis Dendro program compatible format) Coordinate file written as x.pos 2024-11-08 20:59:11
#Imagefile C11C_east.jpg
#DPI 163,0
#All coordinates in millimeters (mm)
SCALE 1
#C DATED 2012
#C PithCoordinates=0.000,10.000; DistanceToPith=0.5; YearsToPith=3;
#C CooRecorder=9.8.1 March 23 2022;
#C licensedTo=Someone;
10.000,0.000
7.000,0.000  7.000,1.000
5.000,1.000
D4.000,1.000 #W#
3.000,1.000
D2.500,1.000 #%gap
D2.000,1.000 #%gap
1.000,1.000 #false?
"""


def test_read_coorecorder_sample(tmp_path: Path):
    path = tmp_path / "sample.pos"
    path.write_text(COORECORDER_SAMPLE)
    pos = read_pos(path)

    assert pos.dpi == 163.0
    assert pos.image_file == "C11C_east.jpg"
    assert pos.dated_year == 2012
    assert pos.pith_xy_mm == (0.0, 10.0)
    assert pos.years_to_pith == 3
    kinds = [p.kind for p in pos.points]
    assert kinds == [RING, RING, RING, SEASONWOOD, RING, GAP, GAP, RING]
    assert pos.points[1].end == (7.0, 1.0)
    assert pos.points[-1].label == "false?"
    # 10→7 (3); double point jump not counted, 7,1→5,1 (2); 5→4→3 (2);
    # 3→2.5 counted (0.5), 2.5→2 skipped (gap pair), 2→1 counted (1).
    assert pos.ring_widths_mm() == pytest.approx([3.0, 2.0, 2.0, 1.5])


def test_single_gap_skips_following_segment():
    points = (
        PosPoint(10, 0),
        PosPoint(8, 0, kind=GAP, label="%gap"),
        PosPoint(5, 0),
    )
    assert PosFile(points).ring_widths_mm() == pytest.approx([2.0])


def test_direction_from_pith_coordinates():
    bark_first = PosFile((PosPoint(10, 0), PosPoint(5, 0), PosPoint(1, 0)), pith_xy_mm=(0, 0))
    assert bark_first.pith_first() is False
    assert bark_first.chronological_widths_mm() == pytest.approx([4.0, 5.0])
    pith_first = PosFile((PosPoint(1, 0), PosPoint(5, 0), PosPoint(10, 0)), pith_xy_mm=(0, 0))
    assert pith_first.pith_first() is True
    assert pith_first.chronological_widths_mm() == pytest.approx([4.0, 5.0])


def test_pos_round_trip(tmp_path: Path):
    original = PosFile(
        points=(
            PosPoint(12.5, 3.25),
            PosPoint(10.0, 3.0, end=(10.1, 3.4)),
            PosPoint(9.0, 3.0, kind=SEASONWOOD, label="W#"),
            PosPoint(7.75, 3.5, label="check"),
        ),
        dpi=2400.0,
        image_file="core.tif",
        dated_year=2023,
        pith_xy_mm=(1.0, 3.0),
        distance_to_pith_mm=6.75,
        years_to_pith=4,
    )
    path = tmp_path / "core.pos"
    write_pos(path, original)
    text = path.read_text()
    assert text.startswith("#DENDRO (Cybis Dendro program compatible format)")
    assert "#C DATED 2023" in text
    assert "10.000,3.000  10.100,3.400" in text

    loaded = read_pos(path)
    assert loaded.points == original.points
    assert loaded.dated_year == 2023
    assert loaded.pith_xy_mm == (1.0, 3.0)
    assert loaded.ring_widths_mm() == pytest.approx(original.ring_widths_mm())


def test_rwl_round_trip_both_precisions(tmp_path: Path):
    series = [
        RingWidthSeries("TRS01A", 1995, (1.23, 0.98, 2.5, 0.0, 1.11, 3.07, 0.5)),
        RingWidthSeries("TRS01B", 2001, tuple(0.1 * i for i in range(1, 24))),
    ]
    for precision in (0.01, 0.001):
        path = tmp_path / f"out_{precision}.rwl"
        write_rwl(path, series, precision=precision)
        loaded = {s.series_id: s for s in read_rwl(path)}
        for s in series:
            got = loaded[s.series_id]
            assert got.first_year == s.first_year
            assert got.widths_mm == pytest.approx(s.widths_mm, abs=precision / 2)


def test_rwl_line_layout(tmp_path: Path):
    path = tmp_path / "x.rwl"
    write_rwl(path, [RingWidthSeries("A1", 1998, (1.0, 2.0, 3.0))])
    lines = path.read_text().splitlines()
    assert lines[0] == "A1      1998   100   200"
    assert lines[1] == "A1      2000   300   999"


def test_rwl_avoids_stop_marker_collision(tmp_path: Path):
    path = tmp_path / "x.rwl"
    write_rwl(path, [RingWidthSeries("A", 2000, (9.99, 1.0))])
    (s,) = read_rwl(path)
    assert len(s.widths_mm) == 2


def test_sanitize_series_id():
    assert sanitize_series_id("Core 12/North#1") == "Core_12N"
    assert sanitize_series_id("") == "TRAS"


def test_radial_export_is_coorecorder_compatible(tmp_path: Path):
    from tras.utils.pos_exporter import export_to_pos

    measurements = {
        "ring_1": {"intersection_point": (110.0, 100.0), "distance_from_pith": 10.0},
        "ring_3": {"intersection_point": (150.0, 100.0), "distance_from_pith": 50.0},
        "ring_2": {"intersection_point": (125.0, 100.0), "distance_from_pith": 25.0},
    }
    path = tmp_path / "radial.pos"
    ok = export_to_pos(
        path,
        measurements,
        pith_xy=(100.0, 100.0),
        direction_xy=(200.0, 100.0),
        image_scale={"value": 0.02, "unit": "mm"},
        metadata={"sample_code": "C11", "harvested_year": 2022},
    )
    assert ok
    pos = read_pos(path)
    assert pos.dpi == pytest.approx(1270.0)
    assert pos.dated_year == 2022
    assert pos.pith_first() is False  # bark first, CooRecorder convention
    # Widths oldest first; the pith closes the innermost ring.
    assert pos.chronological_widths_mm() == pytest.approx([0.2, 0.3, 0.5])


def test_rwl_value_999_is_data_at_0001_mm(tmp_path: Path):
    """Regression: 999 (0.999 mm) was taken as the 0.01 mm stop marker."""
    path = tmp_path / "x.rwl"
    write_rwl(path, [RingWidthSeries("A", 2000, (1.2, 0.999, 1.5, 2.0))], precision=0.001)
    (s,) = read_rwl(path)
    assert s.widths_mm == pytest.approx((1.2, 0.999, 1.5, 2.0))
