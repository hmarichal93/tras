"""Ring overlay readability (T10): paint Shapes into a QImage and inspect the pixels."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest
from PyQt5 import QtCore
from PyQt5 import QtGui

from tras.app import DEFAULT_RING_PALETTE
from tras.app import load_ring_palette
from tras.app import style_shape
from tras.shape import Shape

CANVAS_PX = 200  # side of the square test image
CENTER_PX = 100.0
RADIUS_PX = 60.0  # ring radius on screen
N_RING_POINTS = 361  # as produced by ring resampling
RING_RGB = (86, 180, 233)
BACKGROUND = QtGui.QColor(128, 128, 128)


@pytest.fixture(autouse=True)
def reset_shape_scale() -> Iterator[None]:
    """Shape.scale is a class attribute set by the canvas; restore it after each test."""
    yield
    Shape.scale = 1.0


def make_ring(scale: float = 1.0, label: str = "ring_1") -> Shape:
    """Closed 361-point circle that appears with RADIUS_PX on screen at ``scale``."""
    shape = Shape(label=label, shape_type="polygon")
    for angle in np.linspace(0, 2 * math.pi, N_RING_POINTS, endpoint=False):
        x = CENTER_PX + RADIUS_PX * math.cos(angle)
        y = CENTER_PX + RADIUS_PX * math.sin(angle)
        shape.addPoint(QtCore.QPointF(x / scale, y / scale))
    shape.close()
    style_shape(shape, RING_RGB)
    return shape


def paint(shape: Shape, scale: float = 1.0) -> np.ndarray:
    """Paint ``shape`` on a grey image the way the canvas does; return (H, W, 3) RGB."""
    Shape.scale = scale
    image = QtGui.QImage(CANVAS_PX, CANVAS_PX, QtGui.QImage.Format_RGB32)
    image.fill(BACKGROUND)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    shape.paint(painter)
    painter.end()
    ptr = image.constBits()
    ptr.setsize(image.byteCount())
    bgra = np.frombuffer(ptr, np.uint8).reshape(CANVAS_PX, CANVAS_PX, 4)
    return bgra[..., 2::-1].copy()


def changed_mask(rgb: np.ndarray) -> np.ndarray:
    return np.any(rgb != BACKGROUND.getRgb()[:3], axis=2)


def distance_to_ring_px() -> np.ndarray:
    yy, xx = np.mgrid[0:CANVAS_PX, 0:CANVAS_PX] + 0.5
    return np.abs(np.hypot(xx - CENTER_PX, yy - CENTER_PX) - RADIUS_PX)


def radial_stroke_width_px(rgb: np.ndarray) -> int:
    """Number of changed pixels on the horizontal ray right of the centre."""
    row = changed_mask(rgb)[int(CENTER_PX), int(CENTER_PX) :]
    return int(row.sum())


def test_stroke_width_is_constant_in_screen_px_across_zoom():
    widths = [radial_stroke_width_px(paint(make_ring(s), s)) for s in (0.25, 3.0)]

    assert widths[0] == widths[1]
    assert widths[0] <= Shape.line_width + Shape.HALO_EXTRA_WIDTH + 2


def test_idle_ring_leaves_pixels_away_from_outline_untouched():
    rgb = paint(make_ring())

    far = distance_to_ring_px() > Shape.line_width + 2
    assert not changed_mask(rgb)[far].any()


def test_selected_ring_is_not_filled():
    ring = make_ring()
    ring.selected = True
    ring.fill = True

    rgb = paint(ring)

    interior = np.hypot(*(np.mgrid[0:CANVAS_PX, 0:CANVAS_PX] + 0.5 - CENTER_PX)) < (RADIUS_PX - 10)
    assert not changed_mask(rgb)[interior].any()


def test_selected_ring_outline_is_white():
    ring = make_ring()
    ring.selected = True

    rgb = paint(ring)

    assert (rgb == 255).all(axis=2).any()


def test_exclusion_area_interior_is_filled():
    area = make_ring(label="exclusion")
    area.fill = True

    rgb = paint(area)

    center = rgb[int(CENTER_PX), int(CENTER_PX)]
    assert center[0] > center[1]  # red tint over grey


def test_points_always_draw_vertices():
    pith = Shape(label="pith", shape_type="point")
    pith.addPoint(QtCore.QPointF(CENTER_PX, CENTER_PX))
    pith.close()

    assert pith._should_draw_vertices()
    assert changed_mask(paint(pith))[int(CENTER_PX), int(CENTER_PX)]


def test_idle_ring_draws_no_vertices():
    assert not make_ring()._should_draw_vertices()


@pytest.mark.parametrize("state", ["selected", "hovered"])
def test_active_ring_draws_vertices_only_when_zoomed_in(state: str):
    # 361 points on a 60 px radius are ~1 px apart; at 10x they are ~10 px apart
    ring = make_ring(scale=1.0)
    setattr(ring, state, True)

    Shape.scale = 1.0
    assert not ring._should_draw_vertices()
    Shape.scale = 10.0
    assert ring._should_draw_vertices()


def test_boundary_line_shows_endpoints_when_selected():
    line = Shape(label="core_boundary", shape_type="line")
    line.addPoint(QtCore.QPointF(10, 10))
    line.addPoint(QtCore.QPointF(12, 10))
    line.close()

    assert not line._should_draw_vertices()
    line.selected = True
    assert line._should_draw_vertices()


def test_ring_being_drawn_shows_vertices():
    shape = Shape(label="ring_1", shape_type="polygon")
    shape.addPoint(QtCore.QPointF(10, 10))
    shape.addPoint(QtCore.QPointF(11, 10))

    assert shape._should_draw_vertices()


def test_ring_palette_has_no_black_and_neighbours_differ():
    palette = load_ring_palette({})

    np.testing.assert_array_equal(palette, DEFAULT_RING_PALETTE)
    assert palette.sum(axis=1).min() > 0
    following = np.roll(palette, -1, axis=0)  # includes the wrap-around pair
    assert np.any(palette != following, axis=1).all()


def test_ring_palette_reads_config():
    palette = load_ring_palette({"ring_palette": [[1, 2, 3], [4, 5, 6]]})

    np.testing.assert_array_equal(palette, [[1, 2, 3], [4, 5, 6]])
