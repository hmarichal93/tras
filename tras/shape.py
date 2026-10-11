import copy

import numpy as np
import skimage.measure
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui

import tras.utils

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.

# Shapes whose few vertices are control points (corners, centre, endpoints): show them
# whenever the shape is edited, however close together they are on screen.
CONTROL_POINT_SHAPES = ("line", "rectangle", "circle")

# Fill of negative prompt points ("points" shapes with label 0).
NEGATIVE_VERTEX_COLOR = QtGui.QColor(255, 0, 0, 255)


def _cosmetic_pen(color: QtGui.QColor, width: float) -> QtGui.QPen:
    """Pen whose width is in screen px whatever the painter transform, with round joins."""
    pen = QtGui.QPen(color, width)
    pen.setCosmetic(True)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    return pen


class Shape:
    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    PEN_WIDTH = 2  # mask outlines only

    # Ring overlay readability (T10). Widths are screen px (cosmetic pens).
    EMPHASIS_LINE_WIDTH = 4  # selected or hovered shape
    HALO_EXTRA_WIDTH = 2  # halo is this much wider than the stroke it sits under

    # The following class variables influence the drawing of all shape objects.
    line_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 128)
    fill_color: QtGui.QColor = QtGui.QColor(0, 0, 0, 64)
    vertex_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 255)
    select_line_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)
    select_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 64)
    hvertex_fill_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)
    halo_color: QtGui.QColor = QtGui.QColor(0, 0, 0, 150)
    line_width: int = 2  # idle stroke, screen px (config shape.line_width)

    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    _current_vertex_fill_color: QtGui.QColor = vertex_fill_color

    def __init__(
        self,
        label=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        description=None,
        mask=None,
    ):
        self.label = label
        self.group_id = group_id
        self.points = []
        self.point_labels = []
        self.shape_type = shape_type
        self._shape_raw = None
        self._points_raw = []
        self._shape_type_raw = None
        self.fill = False
        self.selected = False
        self.hovered = False  # set by the canvas each paint: mouse is over this shape
        self.flags = flags
        self.description = description
        self.other_data = {}
        self.mask = mask

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def _scale_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)

    def setShapeRefined(self, shape_type, points, point_labels, mask=None):
        self._shape_raw = (self.shape_type, self.points, self.point_labels)
        self.shape_type = shape_type
        self.points = points
        self.point_labels = point_labels
        self.mask = mask

    def restoreShapeRaw(self):
        if self._shape_raw is None:
            return
        self.shape_type, self.points, self.point_labels = self._shape_raw
        self._shape_raw = None

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "points",
            "mask",
        ]:
            raise ValueError(f"Unexpected shape_type: {value}")
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point, label=1):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)
            self.point_labels.append(label)

    def canAddPoint(self):
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        if self.points:
            if self.point_labels:
                self.point_labels.pop()
            return self.points.pop()
        return None

    def insertPoint(self, i, point, label=1):
        self.points.insert(i, point)
        self.point_labels.insert(i, label)

    def canRemovePoint(self) -> bool:
        if not self.canAddPoint():
            return False

        if self.shape_type == "polygon" and len(self.points) <= 3:
            return False

        if self.shape_type == "linestrip" and len(self.points) <= 2:
            return False

        return True

    def removePoint(self, i: int):
        if not self.canRemovePoint():
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        self.points.pop(i)
        self.point_labels.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def paint(self, painter):
        """Draw the shape in screen pixels: haloed stroke, optional fill, vertex dots.

        The canvas undoes its zoom before painting, so pen widths are screen px;
        ``Shape.scale`` maps image px to screen px for the coordinates.
        """
        if self.mask is None and not self.points:
            return
        if self.mask is not None:
            self._paint_mask(painter)
        if not self.points:
            return

        line_path = self._build_line_path()
        emphasised = self.selected or self.hovered
        width = self.EMPHASIS_LINE_WIDTH if emphasised else self.line_width
        color = self.select_line_color if self.selected else self.line_color
        self._stroke_with_halo(painter, line_path, color, width)
        if self._should_fill():
            fill = self.select_fill_color if self.selected else self.fill_color
            painter.fillPath(line_path, fill)
        self._paint_vertices(painter)

    def _paint_mask(self, painter):
        color = self.select_line_color if self.selected else self.line_color
        pen = QtGui.QPen(color)
        pen.setWidth(self.PEN_WIDTH)
        painter.setPen(pen)

        image_to_draw = np.zeros(self.mask.shape + (4,), dtype=np.uint8)
        fill_color = (
            self.select_fill_color.getRgb()
            if self.selected
            else self.fill_color.getRgb()
        )
        image_to_draw[self.mask] = fill_color
        qimage = QtGui.QImage.fromData(tras.utils.img_arr_to_data(image_to_draw))
        qimage = qimage.scaled(
            qimage.size() * self.scale,
            QtCore.Qt.IgnoreAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        painter.drawImage(self._scale_point(point=self.points[0]), qimage)

        line_path = QtGui.QPainterPath()
        contours = skimage.measure.find_contours(np.pad(self.mask, pad_width=1))
        for contour in contours:
            contour += [self.points[0].y(), self.points[0].x()]
            line_path.moveTo(
                self._scale_point(QtCore.QPointF(contour[0, 1], contour[0, 0]))
            )
            for point in contour[1:]:
                line_path.lineTo(self._scale_point(QtCore.QPointF(point[1], point[0])))
        painter.drawPath(line_path)

    def _build_line_path(self) -> QtGui.QPainterPath:
        """Outline of the shape in screen px (empty for point shapes)."""
        path = QtGui.QPainterPath()
        if self.shape_type in ["rectangle", "mask"]:
            if len(self.points) == 2:
                path.addRect(
                    QtCore.QRectF(
                        self._scale_point(self.points[0]),
                        self._scale_point(self.points[1]),
                    )
                )
        elif self.shape_type == "circle":
            if len(self.points) == 2:
                radius = tras.utils.distance(
                    self._scale_point(self.points[0] - self.points[1])
                )
                path.addEllipse(self._scale_point(self.points[0]), radius, radius)
        elif self.shape_type not in ["point", "points"]:
            path.moveTo(self._scale_point(self.points[0]))
            for p in self.points[1:]:
                path.lineTo(self._scale_point(p))
            if self.shape_type == "polygon" and self.isClosed():
                path.closeSubpath()
        return path

    def _stroke_with_halo(self, painter, path, color, width) -> None:
        """Stroke ``path`` with a dark halo underneath so it reads on light and dark wood."""
        if path.isEmpty():
            return
        opaque = QtGui.QColor(color)
        opaque.setAlpha(255)
        painter.setPen(_cosmetic_pen(self.halo_color, width + self.HALO_EXTRA_WIDTH))
        painter.drawPath(path)
        painter.setPen(_cosmetic_pen(opaque, width))
        painter.drawPath(path)

    def _should_fill(self) -> bool:
        """Fill only areas whose style asks for it (exclusions); rings have alpha-0 fills."""
        if not self.fill or self.shape_type in ["line", "linestrip", "points", "mask"]:
            return False
        color = self.select_fill_color if self.selected else self.fill_color
        return color.alpha() > 0

    def _should_draw_vertices(self) -> bool:
        """Vertex dots: always for points; for other shapes only while edited.

        A shape is being edited when it is still open (being drawn), selected or
        hovered. Dense outlines (a 361-point ring) only get dots once zoomed in far
        enough that the median spacing between dots is at least ``point_size``.
        """
        if self.shape_type in ["point", "points"]:
            return True
        editing = self.selected or self.hovered or not self.isClosed()
        if not editing:
            return False
        if self.shape_type in CONTROL_POINT_SHAPES or not self.isClosed():
            return True
        return self._median_vertex_spacing_px() >= self.point_size

    def _median_vertex_spacing_px(self) -> float:
        """Median distance between consecutive vertices, in screen px."""
        if len(self.points) < 2:
            return float("inf")
        xy = np.array([(p.x(), p.y()) for p in self.points], dtype=np.float64)
        steps = np.hypot(*np.diff(xy, axis=0).T)
        return float(np.median(steps)) * self.scale

    def _paint_vertices(self, painter) -> None:
        """Draw vertex dots (and the highlighted vertex, so dragging stays visible)."""
        vrtx_path = QtGui.QPainterPath()
        negative_vrtx_path = QtGui.QPainterPath()
        if self._should_draw_vertices():
            labels = self.point_labels if self.shape_type == "points" else []
            for i in range(len(self.points)):
                is_negative = i < len(labels) and labels[i] != 1
                self.drawVertex(negative_vrtx_path if is_negative else vrtx_path, i)
        elif self._highlightIndex is not None and self._highlightIndex < len(self.points):
            self.drawVertex(vrtx_path, self._highlightIndex)

        vertex_outline = QtGui.QPen(self.halo_color, 1)
        vertex_outline.setCosmetic(True)
        painter.setPen(vertex_outline)
        if not vrtx_path.isEmpty():
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self._current_vertex_fill_color)
        if not negative_vrtx_path.isEmpty():
            painter.drawPath(negative_vrtx_path)
            painter.fillPath(negative_vrtx_path, NEGATIVE_VERTEX_COLOR)

    def drawVertex(self, path, i):
        d = self.point_size
        shape = self.point_type
        point = self._scale_point(self.points[i])
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size  # type: ignore[assignment]
        if self._highlightIndex is not None:
            self._current_vertex_fill_color = self.hvertex_fill_color
        else:
            self._current_vertex_fill_color = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        min_distance = float("inf")
        min_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i, p in enumerate(self.points):
            p = QtCore.QPointF(p.x() * self.scale, p.y() * self.scale)
            dist = tras.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float("inf")
        post_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i in range(len(self.points)):
            start = self.points[i - 1]
            end = self.points[i]
            start = QtCore.QPointF(start.x() * self.scale, start.y() * self.scale)
            end = QtCore.QPointF(end.x() * self.scale, end.y() * self.scale)
            line = [start, end]
            dist = tras.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point) -> bool:
        if self.shape_type in ["line", "linestrip", "points"]:
            return False
        if self.mask is not None:
            y = np.clip(
                int(round(point.y() - self.points[0].y())),
                0,
                self.mask.shape[0] - 1,
            )
            x = np.clip(
                int(round(point.x() - self.points[0].x())),
                0,
                self.mask.shape[1] - 1,
            )
            return self.mask[y, x]
        return self.makePath().contains(point)

    def makePath(self):
        if self.shape_type in ["rectangle", "mask"]:
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                path.addRect(QtCore.QRectF(self.points[0], self.points[1]))
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                raidus = tras.utils.distance(self.points[0] - self.points[1])
                path.addEllipse(self.points[0], raidus, raidus)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        """Clear the highlighted point"""
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
