import os.path as osp
import shutil
import tempfile

import pytest
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from pytestqt.qtbot import QtBot

import tras.app
import tras.config
import tras.testing

import numpy as np
from PIL import Image

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
        annotated / "2011_000003.json", synthetic_core(length=320, height=240), (130.0, 120.0), [ring]
    )
    return str(tmp_path)


def _show_window_and_wait_for_imagedata(
    qtbot: QtBot, win: tras.app.MainWindow
) -> None:
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
