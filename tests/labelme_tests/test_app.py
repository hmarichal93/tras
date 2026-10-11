import os.path as osp
import shutil
import tempfile

import numpy as np
import pytest
from PIL import Image
from pytestqt.qtbot import QtBot

import tras.app
import tras.config
import tras.testing
from tras.api import _save_detection_results
from tras.cores.synthetic import synthetic_core


@pytest.fixture
def data_dir(tmp_path):
    """Two raw images and one annotated label file, generated on the fly."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for i, name in enumerate(("2011_000003.jpg", "2011_000006.jpg")):
        image = synthetic_core(length=320, height=240, seed=i)
        Image.fromarray(image).save(raw / name)
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    ring = np.array([[60, 60], [200, 60], [200, 180], [60, 180]], dtype=np.float32)
    _save_detection_results(
        annotated / "2011_000003.json",
        synthetic_core(length=320, height=240),
        (130.0, 120.0),
        [ring],
    )
    return str(tmp_path)


def _show_window_and_wait_for_imagedata(qtbot: QtBot, win: tras.app.MainWindow) -> None:
    win.show()

    def check_imageData():
        assert hasattr(win, "imageData")
        assert win.imageData is not None

    qtbot.waitUntil(check_imageData)  # wait for loadFile


@pytest.mark.gui
def test_MainWindow_open(qtbot: QtBot) -> None:
    win: tras.app.MainWindow = tras.app.MainWindow()
    qtbot.addWidget(win)
    win.show()
    win.close()


@pytest.mark.gui
def test_MainWindow_open_img(qtbot: QtBot, data_dir) -> None:
    img_file: str = osp.join(data_dir, "raw/2011_000003.jpg")
    win: tras.app.MainWindow = tras.app.MainWindow(filename=img_file)
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    win.close()


@pytest.mark.gui
def test_MainWindow_open_json(qtbot: QtBot, data_dir):
    json_files: list[str] = [osp.join(data_dir, "annotated/2011_000003.json")]
    json_file: str
    for json_file in json_files:
        tras.testing.assert_labelfile_sanity(json_file)

        win: tras.app.MainWindow = tras.app.MainWindow(filename=json_file)
        qtbot.addWidget(win)
        _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
        win.close()


@pytest.mark.gui
def test_MainWindow_openNextAndPrevImg(qtbot: QtBot, data_dir) -> None:
    directory: str = osp.join(data_dir, "raw")
    win: tras.app.MainWindow = tras.app.MainWindow(filename=directory)
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    first_image_name: str = "2011_000003.jpg"
    second_image_name: str = "2011_000006.jpg"

    assert osp.basename(win.imagePath) == first_image_name
    win.openPrevImg()
    qtbot.wait(100)
    assert osp.basename(win.imagePath) == first_image_name

    win.openNextImg()
    qtbot.wait_until(lambda: osp.basename(win.imagePath) != first_image_name)
    assert osp.basename(win.imagePath) == second_image_name
    win.openPrevImg()
    qtbot.wait_until(lambda: osp.basename(win.imagePath) != second_image_name)
    assert osp.basename(win.imagePath) == first_image_name


@pytest.mark.gui
def test_MainWindow_annotate_jpg(qtbot: QtBot, data_dir) -> None:
    """Shapes on the canvas are saved to a valid label file."""
    from PyQt5.QtCore import QPointF

    from tras.shape import Shape

    tmp_dir: str = tempfile.mkdtemp()
    input_file: str = osp.join(data_dir, "raw/2011_000003.jpg")
    out_file: str = osp.join(tmp_dir, "2011_000003.json")

    config: dict = tras.config._get_default_config_and_create_labelmerc()
    win: tras.app.MainWindow = tras.app.MainWindow(
        config=config,
        filename=input_file,
        output_file=out_file,
    )
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    shape = Shape(label="whole", shape_type="polygon")
    for x, y in [(80, 60), (240, 60), (240, 180), (80, 180)]:
        shape.addPoint(QPointF(x, y))
    shape.close()
    win.loadShapes([shape], replace=False)

    assert len(win.canvas.shapes) == 1
    assert win.canvas.shapes[0].label == "whole"
    assert win.canvas.shapes[0].shape_type == "polygon"

    win.saveFile()

    tras.testing.assert_labelfile_sanity(out_file)
    shutil.rmtree(tmp_dir)


@pytest.mark.gui
def test_exclusion_zones_are_not_rings(qtbot: QtBot, data_dir) -> None:
    """Exclusion polygons are subtracted from areas, never measured or relabelled as rings."""
    from PyQt5.QtCore import QPointF

    from tras.shape import Shape

    win = tras.app.MainWindow(filename=osp.join(data_dir, "raw/2011_000003.jpg"))
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    def polygon(label, xy):
        shape = Shape(label=label, shape_type="polygon")
        for x, y in xy:
            shape.addPoint(QPointF(x, y))
        shape.close()
        return shape

    inner = polygon("1", [(140, 100), (180, 100), (180, 140), (140, 140)])
    outer = polygon("2", [(120, 80), (200, 80), (200, 160), (120, 160)])
    exclusion = polygon("exclusion", [(150, 110), (160, 110), (160, 120), (150, 120)])
    win.loadShapes([inner, outer, exclusion], replace=False)
    win.pith_xy = (160.0, 120.0)
    win.sample_metadata = {"harvested_year": 2020}

    ring_properties, _, _ = win._prepare_ring_properties_data()

    assert [p["label"] for p in ring_properties] == ["2019", "2020"]
    assert ring_properties[0]["area"] == pytest.approx(1600 - 100)
    assert exclusion.label == "exclusion"
    win.setClean()  # otherwise close() asks to save (modal) and blocks
    win.close()


@pytest.mark.gui
def test_export_dialog_writes_rwl(qtbot: QtBot, data_dir, tmp_path) -> None:
    import numpy as np
    from PyQt5.QtCore import QPointF

    from tras.io.tucson import read_rwl
    from tras.shape import Shape
    from tras.widgets import ExportDialog

    win = tras.app.MainWindow(filename=osp.join(data_dir, "raw/2011_000003.jpg"))
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    shapes = []
    for label, r in (("1", 20), ("2", 40), ("3", 60)):
        shape = Shape(label=label, shape_type="polygon")
        for a in t:
            shape.addPoint(QPointF(160 + r * np.cos(a), 120 + r * np.sin(a)))
        shape.close()
        shapes.append(shape)
    win.loadShapes(shapes, replace=False)
    win.pith_xy = (160.0, 120.0)
    win.image_scale = {"value": 0.1, "unit": "mm"}
    win.sample_metadata = {"harvested_year": 2022, "sample_code": "T1"}

    dialog = ExportDialog(parent=win)
    qtbot.addWidget(dialog)
    assert dialog.export_rwl_checkbox.isEnabled()
    path = dialog._export_rwl(tmp_path)

    series = read_rwl(path)
    assert [s.series_id for s in series] == ["T1N", "T1E", "T1S", "T1W"]
    assert series[0].widths_mm == pytest.approx((2.0, 2.0, 2.0), abs=0.05)
    win.setClean()
    win.close()


@pytest.mark.gui
def test_tools_menu_separates_cross_sections_and_cores(qtbot: QtBot, data_dir, monkeypatch) -> None:
    from PyQt5.QtCore import QPointF

    from tras.shape import Shape

    win = tras.app.MainWindow(filename=osp.join(data_dir, "raw/2011_000003.jpg"))
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    def texts(menu):
        return [a.text() for a in menu.actions() if not a.isSeparator()]

    tools = [a.text() for a in win.menus.tools.actions() if not a.isSeparator()]
    assert tools == ["Cross-section", "Increment Core", "Batch Processing (Cross-sections)..."]
    assert "Core Analysis..." not in texts(win.menus.tools_cross_section)
    assert texts(win.menus.tools_core) == [
        "Sample Metadata",
        "Set Image Scale",
        "Draw Core Path",
        "Core Analysis...",
        "Draw Core Boundary",
        "Export Core Measurements...",
    ]

    # Only core paths (not open rings drawn as linestrips) are offered to Core Analysis.
    def linestrip(label):
        shape = Shape(label=label, shape_type="linestrip")
        for x, y in [(10, 50), (300, 60)]:
            shape.addPoint(QPointF(x, y))
        return shape

    win.loadShapes([linestrip("ring_1"), linestrip("core_path")], replace=False)
    captured = {}

    class FakeDialog:
        def __init__(self, image, drawn_paths, *args, **kwargs):
            captured["paths"] = drawn_paths
            self.results = []

        def exec_(self):
            return 0

    monkeypatch.setattr(tras.app, "CoreDialog", FakeDialog)
    win._action_core_analysis()
    assert len(captured["paths"]) == 1

    # Core-path draw mode labels new linestrips automatically.
    win.toggleDrawMode(False, createMode="linestrip", core_path_mode=True)
    assert win._core_path_mode
    win.toggleDrawMode(True)
    assert not win._core_path_mode
    win.setClean()
    win.close()


def _core_window(qtbot: QtBot, data_dir) -> tras.app.MainWindow:
    """Main window with one measured core (path + 5 boundary lines) on the canvas."""
    from tras.cores import CoreBoundary
    from tras.cores import CoreResult
    from tras.cores import to_label_shapes
    from tras.cores.ribbon import sample_path

    win = tras.app.MainWindow(filename=osp.join(data_dir, "raw/2011_000003.jpg"))
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    path = sample_path([[20.0, 120.0], [300.0, 120.0]])
    boundaries = tuple(
        CoreBoundary(s, (20.0 + s, 120.0), 0.0, 5.0) for s in (20.0, 60.0, 100.0, 140.0, 180.0)
    )
    result = CoreResult(path=path, boundaries=boundaries, half_width=20)
    win._replace_core_shapes(to_label_shapes(result, core_index=1), tras.app.CORE_PATH_LABEL)
    return win


def _draw_line(win: tras.app.MainWindow, p1, p2) -> None:
    """Finish a line on the canvas as a mouse drag would (no event sequence)."""
    from PyQt5.QtCore import QPointF

    from tras.shape import Shape

    shape = Shape(shape_type="line")
    shape.addPoint(QPointF(*p1))
    shape.addPoint(QPointF(*p2))
    win.canvas.current = shape
    win.canvas.finalise()  # emits newShape → MainWindow.newShape


@pytest.mark.gui
def test_draw_core_boundary_labels_new_lines(qtbot: QtBot, data_dir) -> None:
    win = _core_window(qtbot, data_dir)
    assert win.actions.drawCoreBoundary.isEnabled()
    assert win.actions.drawCoreBoundary.shortcut().toString() == "Ctrl+B"

    win.actions.drawCoreBoundary.trigger()
    assert win._core_boundary_mode
    assert win.actions.drawCoreBoundary.isChecked()
    assert win.canvas.createMode == "line"
    _draw_line(win, (100.0, 100.0), (100.0, 140.0))

    added = win.canvas.shapes[-1]
    assert added.label == "core_boundary"
    assert added.description == "Ring boundary added by hand"
    win.setEditMode()
    assert not win._core_boundary_mode
    assert not win.actions.drawCoreBoundary.isChecked()
    win.setClean()
    win.close()


@pytest.mark.gui
def test_export_core_measurements_uses_canvas_edits(
    qtbot: QtBot, data_dir, tmp_path, monkeypatch
) -> None:
    from PyQt5 import QtWidgets

    win = _core_window(qtbot, data_dir)
    win.image_scale = {"value": 0.01, "unit": "mm"}
    win.sample_metadata = {"harvested_year": 2021, "outer_ring_year": 2020}
    boundary = win.canvas.shapes[2]  # second boundary (s = 60), moved 10 px to the bark
    for point in boundary.points:
        point.setX(point.x() + 10.0)
    win.toggleDrawMode(False, createMode="line", core_boundary_mode=True)
    _draw_line(win, (180.0, 100.0), (180.0, 140.0))  # splits the s = 140..180 ring
    _draw_line(win, (100.0, 10.0), (100.0, 40.0))  # crosses no core path

    out = tmp_path / "export"
    out.mkdir()
    messages = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(out)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "exec_", lambda box: messages.append(box.text()) or 0
    )
    win._action_export_core_measurements()

    csv_lines = (out / "2011_000003_core1.csv").read_text().splitlines()[1:]
    widths = [float(line.split(",")[3]) for line in csv_lines]
    years = [int(line.split(",")[1]) for line in csv_lines]
    assert widths == pytest.approx([50.0, 30.0, 40.0, 20.0, 20.0], abs=0.5)
    assert years == [2016, 2017, 2018, 2019, 2020]
    assert (out / "2011_000003_core1.pos").exists()
    assert (out / "2011_000003.rwl").exists()

    labels = [shape.label for shape in win.canvas.shapes]
    assert labels[-2] == "core1_2019"  # the hand-drawn boundary got its year
    assert labels[-1] == "core_boundary"  # off the path: kept, ignored and selected
    assert win.canvas.selectedShapes == [win.canvas.shapes[-1]]
    assert "1 cores, 5 rings, 2016–2020" in messages[0]
    assert "do not cross a core path" in messages[0]
    win.setClean()
    win.close()


@pytest.mark.gui
def test_export_core_measurements_without_core_shows_error(
    qtbot: QtBot, data_dir, monkeypatch
) -> None:
    win = tras.app.MainWindow(filename=osp.join(data_dir, "raw/2011_000003.jpg"))
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    errors = []
    monkeypatch.setattr(win, "errorMessage", lambda title, text: errors.append(text))
    win._action_export_core_measurements()
    assert errors and errors[0].startswith("No measured core on the image.")
    win.close()


@pytest.mark.gui
def test_core_analysis_stores_outer_ring_year(qtbot: QtBot, data_dir, monkeypatch) -> None:
    from PyQt5 import QtWidgets

    win = tras.app.MainWindow(filename=osp.join(data_dir, "raw/2011_000003.jpg"))
    qtbot.addWidget(win)
    _show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    win.sample_metadata = {"harvested_year": 2021, "sample_code": "T1"}
    captured = {}

    class FakeDialog:
        def __init__(self, image, drawn_paths, mm_per_px, outer_year, **kwargs):
            captured["outer_year"] = outer_year
            self.results = ["one core"]

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def outer_ring_year(self):
            return 2020

        def label_shapes(self):
            return []

    monkeypatch.setattr(tras.app, "CoreDialog", FakeDialog)
    win._action_core_analysis()
    assert captured["outer_year"] == 2021  # harvested year until set
    assert win.sample_metadata == {
        "harvested_year": 2021,
        "sample_code": "T1",
        "outer_ring_year": 2020,
    }
    assert win.otherData["sample_metadata"]["outer_ring_year"] == 2020
    win._action_core_analysis()
    assert captured["outer_year"] == 2020
    win.setClean()
    win.close()


def _export_cores(win: tras.app.MainWindow, folder, monkeypatch) -> list[str]:
    """Run Export Core Measurements into ``folder``; return the summary texts shown."""
    from PyQt5 import QtWidgets

    messages: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(folder)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "exec_", lambda box: messages.append(box.text()) or 0
    )
    win._action_export_core_measurements()
    return messages


@pytest.mark.gui
def test_export_after_deleting_a_core_keeps_core_numbers(
    qtbot: QtBot, data_dir, tmp_path, monkeypatch
) -> None:
    from tras.cores import CoreBoundary
    from tras.cores import CoreResult
    from tras.cores import to_label_shapes
    from tras.cores.ribbon import sample_path
    from tras.io.pos import read_pos
    from tras.io.tucson import read_rwl

    win = _core_window(qtbot, data_dir)
    win.image_scale = {"value": 0.01, "unit": "mm"}
    win.sample_metadata = {"outer_ring_year": 2020}
    shapes = []
    for k, y in ((1, 40.0), (2, 120.0), (3, 200.0)):
        path = sample_path([[20.0, y], [300.0, y]])
        boundaries = tuple(CoreBoundary(s, (20.0 + s, y), 0.0, 5.0) for s in (20.0, 60.0, 100.0))
        result = CoreResult(path=path, boundaries=boundaries, half_width=15)
        shapes += to_label_shapes(result, core_index=k)
    win._replace_core_shapes(shapes, tras.app.CORE_PATH_LABEL)
    win.canvas.selectShapes([s for s in win.canvas.shapes if s.group_id == 2])
    win.remLabels(win.canvas.deleteSelected())  # the user deletes core 2

    out = tmp_path / "export"
    messages = _export_cores(win, out, monkeypatch)
    win.setClean()  # a failing assert must not leave a "save changes?" prompt open

    assert sorted(p.name for p in out.iterdir()) == [
        "2011_000003.rwl",
        "2011_000003_core1.csv",
        "2011_000003_core1.pos",
        "2011_000003_core3.csv",
        "2011_000003_core3.pos",
    ]
    assert [s.series_id for s in read_rwl(out / "2011_000003.rwl")] == ["2011_01", "2011_03"]
    assert read_pos(out / "2011_000003_core3.pos").image_file == "2011_000003.jpg"
    assert "Core 3: 2 rings, 2019–2020" in messages[0]
    win.close()


@pytest.mark.gui
def test_undo_after_core_export_relabel(qtbot: QtBot, data_dir, tmp_path, monkeypatch) -> None:
    win = _core_window(qtbot, data_dir)
    win.sample_metadata = {"outer_ring_year": 2020}
    win.toggleDrawMode(False, createMode="line", core_boundary_mode=True)
    _draw_line(win, (180.0, 100.0), (180.0, 140.0))
    before = [shape.label for shape in win.canvas.shapes]
    _export_cores(win, tmp_path / "export", monkeypatch)
    relabelled = [shape.label for shape in win.canvas.shapes]
    assert relabelled[-1] == "core1_2019"

    _draw_line(win, (250.0, 100.0), (250.0, 140.0))  # a later edit...
    win.undoShapeEdit()  # ...undone alone: the year labels stay
    assert [shape.label for shape in win.canvas.shapes] == relabelled
    win.undoShapeEdit()  # the relabel itself is undoable
    assert [shape.label for shape in win.canvas.shapes] == before
    win.setClean()
    win.close()


@pytest.mark.gui
def test_rerun_core_analysis_keeps_hand_drawn_boundaries(qtbot: QtBot, data_dir) -> None:
    """A hand-drawn boundary survives re-running Core Analysis, even after export relabelled it."""
    from PyQt5.QtCore import QPointF

    from tras.shape import Shape

    win = _core_window(qtbot, data_dir)
    win.setClean()  # a failing assertion must not leave a "save changes?" prompt open
    win.actions.drawCoreBoundary.trigger()
    _draw_line(win, (100.0, 100.0), (100.0, 140.0))
    hand = win.canvas.shapes[-1]
    hand.label = "core1_2019"  # as relabelled by an export
    detected = Shape(label="core1_2018", shape_type="line", description="strength=3.00")
    detected.addPoint(QPointF(50.0, 100.0))
    detected.addPoint(QPointF(50.0, 140.0))
    win.loadShapes([detected], replace=False)

    new_path = {
        "label": "core_path_1",
        "points": [[0.0, 120.0], [200.0, 120.0]],
        "shape_type": "linestrip",
        "group_id": 1,
        "description": "",
    }
    win._replace_core_shapes([new_path], "core_path")

    labels = [s.label for s in win.canvas.shapes]
    win.setClean()
    win.close()
    assert "core1_2019" in labels  # hand-drawn kept
    assert "core1_2018" not in labels  # old detection replaced
