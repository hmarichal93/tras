from __future__ import annotations

import functools
import html
import io
import math
import os
import os.path as osp
import re
import types
import webbrowser
from datetime import datetime
from dateutil.relativedelta import relativedelta

import imgviz
import natsort
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from tras import __appname__
from tras import __version__
from tras._label_file import LabelFile
from tras._label_file import LabelFileError
from tras._label_file import ShapeDict
from tras.config import get_config
from tras.shape import Shape
from tras.widgets import BrightnessContrastDialog
from tras.widgets import Canvas
from tras.widgets import FileDialogPreview
from tras.widgets import LabelDialog
from tras.widgets import LabelListWidget
from tras.widgets import LabelListWidgetItem
from tras.widgets import ToolBar
from tras.widgets import UniqueLabelQListWidget
from tras.widgets import ZoomWidget
from tras.widgets import TreeRingDialog
from tras.widgets import PreprocessDialog
from tras.widgets import RingPropertiesDialog
from tras.widgets import MetadataDialog

from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP: NDArray[np.uint8] = imgviz.label_colormap()


class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    # NB: this tells Mypy etc. that `actions` here
    #     is a different type cf. the parent class
    #     (where it is Callable[[QWidget], list[QAction]]).
    actions: types.SimpleNamespace  # type: ignore[assignment]

    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super().__init__()
        self.setWindowTitle(f"{__appname__} - Tree Ring Analyzer Suite")

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False

        self._copied_shapes = None

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Ring Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr("Select label to start annotating for it. Press 'Esc' to deselect.")
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                self.uniqLabelList.add_label_item(
                    label=label, color=self._get_rgb_by_label(label=label)
                )
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.mouseMoved.connect(
            lambda pos: self.status_right.setText(f"x={pos.x():.3f}, y={pos.y():.3f}")
        )
        self.canvas.statusUpdated.connect(lambda text: self.status_left.setText(text))

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("📁 &Open"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Step 1: Load wood cross-section image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("💾 &Save"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Step 8: Export labels and data to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("💾 Save &As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Step 8: Export labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(
            self.tr("Draw Ring Manually"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Manually draw a tree ring by clicking points"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        editMode = action(
            self.tr("Edit Rings"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected rings"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Rings"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected rings"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Rings"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected rings"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Rings"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected rings to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Rings"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied rings"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from ring"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nRings"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all rings"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nRings"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all rings"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nRings"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all rings"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(f"{shortcuts['zoom_in']},{shortcuts['zoom_out']}"),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected ring"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Ring"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill ring while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = types.SimpleNamespace(
            about=action(
                text=f"&About {__appname__}",
                slot=functools.partial(
                    QMessageBox.about,
                    self,
                    f"About {__appname__}",
                    f"""
<h3>{__appname__}</h3>
<p>Image Polygonal Annotation with Python</p>
<p>Version: {__version__}</p>
<p>Author: Kentaro Wada</p>
<p>
    <a href="https://labelme.io">Homepage</a> |
    <a href="https://labelme.io/docs">Documentation</a> |
    <a href="https://labelme.io/docs/troubleshoot">Troubleshooting</a>
</p>
<p>
    <a href="https://github.com/wkentaro/labelme">GitHub</a> |
    <a href="https://x.com/labelmeai">Twitter/X</a>
</p>
""",
                ),
            ),
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            toggle_keep_prev_brightness_contrast=action(
                text=self.tr("Keep Previous Brightness/Contrast"),
                slot=lambda: self._config.__setitem__(
                    "keep_prev_brightness_contrast",
                    not self._config["keep_prev_brightness_contrast"],
                ),
                checkable=True,
                checked=self._config["keep_prev_brightness_contrast"],
            ),
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
        )
        self.on_shapes_present_actions = (saveAs, hideAll, showAll, toggleAll)

        self.draw_actions: list[tuple[str, QtWidgets.QAction]] = [
            ("polygon", createMode),
            ("rectangle", createRectangleMode),
            ("circle", createCircleMode),
            ("point", createPointMode),
            ("line", createLineMode),
            ("linestrip", createLineStripMode),
        ]

        # Tree Ring Detection action
        detectTreeRings = action(
            self.tr("🎯 Detect Tree Rings"),
            self._action_detect_rings,
            None,
            "objects",
            self.tr("Step 4: Automatically detect rings using APD, CS-TRD, or DeepCS-TRD"),
            enabled=False,
        )
        
        # Preprocess Image action
        preprocessImage = action(
            self.tr("🖼️ Preprocess Image"),
            self._action_preprocess_image,
            None,
            "edit",
            self.tr("Step 3: Crop, resize, or remove background from image"),
            enabled=False,
        )
        
        # Clear All Rings action
        clearAllRings = action(
            self.tr("Clear All Rings"),
            self._action_clear_all_rings,
            None,
            "cancel",
            self.tr("Remove all detected tree ring polygons"),
            enabled=False,
        )
        
        # Ring Properties action
        ringProperties = action(
            self.tr("📊 View Ring Properties"),
            self._action_ring_properties,
            None,
            "objects",
            self.tr("Step 7: Compute and view ring area, perimeter, and cumulative measurements"),
            enabled=False,
        )
        
        # Metadata action
        metadata = action(
            self.tr("📝 Sample Metadata"),
            self._action_metadata,
            None,
            "edit",
            self.tr("Input harvested year, sample code, and observations (can be done at any time)"),
            enabled=False,
        )
        
        # Scale calibration action
        setScale = action(
            self.tr("📏 Set Image Scale"),
            self._action_set_scale,
            None,
            "ruler",
            self.tr("Step 2: Set physical scale for measurements (mm, cm, μm per pixel)"),
            enabled=False,
        )
        
        # Measure ring width along radial line
        measureRadialWidth = action(
            self.tr("📐 Measure Ring Width"),
            self._action_measure_radial_width,
            None,
            "line",
            self.tr("Step 6: Measure ring widths along a radial line from pith"),
            enabled=False,
        )
        

        # Group zoom controls into a list for easier toggling.
        self.zoom_actions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.on_load_active_actions = (
            close,
            createMode,
            createRectangleMode,
            createCircleMode,
            createLineMode,
            createPointMode,
            createLineStripMode,
            editMode,
            brightnessContrast,
            detectTreeRings,  # Keep enabled but not in toolbar
            preprocessImage,
            clearAllRings,
            ringProperties,
            metadata,
            setScale,
            measureRadialWidth,
        )
        # menu shown at right click
        self.context_menu_actions = (
            createMode,
            createRectangleMode,
            createCircleMode,
            createLineMode,
            createPointMode,
            createLineStripMode,
            editMode,
            edit,
            duplicate,
            copy,
            paste,
            delete,
            undo,
            undoLastPoint,
            removePoint,
        )
        # XXX: need to add some actions here to activate the shortcut
        self.edit_menu_actions = (
            edit,
            duplicate,
            copy,
            paste,
            delete,
            None,
            undo,
            undoLastPoint,
            None,
            removePoint,
            None,
            toggle_keep_prev_mode,
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = types.SimpleNamespace(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            tools=self.menu(self.tr("&Tools")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help, self.actions.about))
        # Tools menu organized by workflow order
        utils.addActions(self.menus.tools, (
            metadata,           # Step 1: Sample Metadata (optional, before starting)
            None,
            setScale,           # Step 2: Set Scale
            preprocessImage,    # Step 3: Preprocess
            detectTreeRings,    # Step 4: Detect Rings
            None,
            measureRadialWidth, # Step 6: Measure Width
            ringProperties,     # Step 7: View Properties
        ))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
                self.actions.toggle_keep_prev_brightness_contrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.context_menu_actions)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        # Toolbar setup
        self.tools = self.toolbar("Tools")
        self.toolbar_actions = (
            open_,
            opendir,
            openPrevImg,
            openNextImg,
            save,
            deleteFile,
            None,
            createMode,
            editMode,
            clearAllRings,
            duplicate,
            delete,
            undo,
            brightnessContrast,
            None,
            fitWindow,
            zoom,
        )

        self.status_left = QtWidgets.QLabel(self.tr("%s started.") % __appname__)
        self.status_right = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status_left, 1)
        self.statusBar().addWidget(self.status_right, 0)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.labelFile: LabelFile | None = None
        self.imagePath: str | None = None
        self.recentFiles: list[str] = []
        self.maxRecent = 7
        self.otherData = None
        self.sample_metadata = None  # Store harvested year, sample code, observation
        self.image_scale = None  # Store scale: {'value': float, 'unit': str} e.g., {'value': 0.02, 'unit': 'mm'}
        self.imageArray = None  # Store preprocessed image as numpy array (bypasses QImage corruption)
        self.radial_line_measurements = None  # Store radial width measurements: {ring_label: {'radial_width': float, ...}}
        self.pith_xy = None  # Store pith coordinates from detection: (x, y)
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {  # type: ignore[var-annotated]
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("tras", "tras")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(f"{title}ToolBar")
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        self.tools.clear()
        utils.addActions(self.tools, self.toolbar_actions)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], self.context_menu_actions)
        self.menus.edit.clear()
        actions = (
            *[draw_action for _, draw_action in self.draw_actions],
            self.actions.editMode,
            *self.edit_menu_actions,
        )
        utils.addActions(self.menus.edit, actions)

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            assert self.imagePath
            label_file = f"{osp.splitext(self.imagePath)[0]}.json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = f"{title} - {self.filename}*"
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        for _, action in self.draw_actions:
            action.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = f"{title} - {self.filename}"
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.zoom_actions:
            z.setEnabled(value)
        for action in self.on_load_active_actions:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def show_status_message(self, message, delay=2000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.imageArray = None  # Clear preprocessed array
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def _action_detect_rings(self) -> None:
        if self.image.isNull():
            self.errorMessage(self.tr("No image"), self.tr("Please open an image first."))
            return
        
        # Convert image to RGB888 format first
        # Use stored numpy array if available (from preprocessing), otherwise extract from QImage
        if self.imageArray is not None:
            # Use the stored preprocessed array (bypasses QImage corruption)
            image_np = self.imageArray
            logger.info(f"✓ Using stored preprocessed numpy array: {image_np.shape} ({image_np.dtype})")
        else:
            # Convert QImage to numpy array
            if self.image.format() != QtGui.QImage.Format_RGB888:
                qimage_rgb = self.image.convertToFormat(QtGui.QImage.Format_RGB888)
            else:
                qimage_rgb = self.image
            
            # Extract image as numpy array
            image_np = utils.img_qt_to_arr(qimage_rgb)[:, :, :3]
            logger.info(f"Extracted image from QImage: {image_np.shape} ({image_np.dtype})")
        
        # Log image info
        logger.info(f"Tree ring detection image: {image_np.shape} ({image_np.dtype})")
        logger.info(f"  Image stats: min={image_np.min()}, max={image_np.max()}, mean={image_np.mean():.2f}")
        logger.info(f"  First pixel: {image_np[0, 0]}")
        if self.otherData and "preprocessing" in self.otherData:
            preprocessing_info = self.otherData["preprocessing"]
            logger.info(f"Image was preprocessed: scale={preprocessing_info.get('scale_factor', 1.0)}, "
                       f"crop={preprocessing_info.get('crop_rect') is not None}, "
                       f"bg_removed={preprocessing_info.get('background_removed', False)}")
        else:
            logger.info("Image is original (no preprocessing applied)")
        
        # Show detection dialog (loop to handle click pith mode)
        clicked_cx, clicked_cy = None, None
        while True:
            dlg = TreeRingDialog(
                image_width=self.image.width(), 
                image_height=self.image.height(), 
                parent=self, 
                image_np=image_np,
                initial_cx=clicked_cx,
                initial_cy=clicked_cy
            )
            result = dlg.exec_()
            
            if result == QtWidgets.QDialog.Accepted:
                # User clicked detection button - proceed
                break
            elif result == 2:  # Click pith mode requested
                # Set up single-click handler to get pith coordinates
                logger.info("Entering click-to-set-pith mode")
                QtWidgets.QMessageBox.information(
                    self,
                    self.tr("Click to Set Pith"),
                    self.tr("Click on the image where the pith (center) is located.\n\n"
                           "The dialog will reappear with the coordinates.")
                )
                
                # Set up one-time click handler
                clicked_coords = self._wait_for_pith_click()
                if clicked_coords:
                    clicked_cx, clicked_cy = clicked_coords
                    logger.info(f"Pith clicked at ({clicked_cx:.1f}, {clicked_cy:.1f})")
                    # Loop will reopen dialog with these coordinates
                else:
                    # User cancelled click
                    return
            else:
                # User cancelled
                return
        
        # Get rings from CS-TRD or DeepCS-TRD
        cstrd_rings = dlg.get_cstrd_rings()
        deepcstrd_rings = dlg.get_deepcstrd_rings()
        pith_xy = dlg.get_pith_xy()
        
        if cstrd_rings is not None:
            rings = cstrd_rings
        elif deepcstrd_rings is not None:
            rings = deepcstrd_rings
        else:
            # No detection method was used
            self.errorMessage(self.tr("No detection"), self.tr("Please use CS-TRD or DeepCS-TRD detection buttons."))
            return
        
        # Store pith coordinates for later use (e.g., radial width measurement)
        if pith_xy:
            self.pith_xy = pith_xy
            logger.info(f"Stored pith coordinates: ({pith_xy[0]:.1f}, {pith_xy[1]:.1f})")
        
        if not rings:
            self.show_status_message(self.tr("No rings detected."))
            return
        
        # Convert to shapes
        shapes: list[Shape] = []
        
        # Use year labels if metadata exists, otherwise use numeric labels
        if self.sample_metadata and 'harvested_year' in self.sample_metadata:
            harvested_year = self.sample_metadata['harvested_year']
            n_rings = len(rings)
            
            # Calculate innermost year using datetime
            harvested_date = datetime(harvested_year, 1, 1)
            innermost_date = harvested_date - relativedelta(years=n_rings - 1)
            innermost_year = innermost_date.year
            
            logger.info(f"Labeling {n_rings} rings with years: outermost={harvested_year}, innermost={innermost_year}")
            
            # Detection returns rings from innermost (pith) to outermost (bark)
            # So rings[0] = innermost (oldest), rings[-1] = outermost (newest)
            # Outermost ring (last in list) gets harvested_year
            for i, ring in enumerate(rings, start=0):
                # Calculate year using datetime: innermost gets oldest year, outermost gets harvested year
                ring_date = harvested_date - relativedelta(years=(n_rings - 1 - i))
                year = ring_date.year
                shape = Shape(label=f"ring_{year}", shape_type="polygon")
                for x, y in ring:
                    shape.addPoint(QtCore.QPointF(float(x), float(y)))
                shape.close()
                shapes.append(shape)
        else:
            logger.info(f"Labeling rings with numeric labels (no metadata set)")
            for i, ring in enumerate(rings, start=1):
                shape = Shape(label=f"ring_{i}", shape_type="polygon")
                for x, y in ring:
                    shape.addPoint(QtCore.QPointF(float(x), float(y)))
                shape.close()
                shapes.append(shape)
        
        self.canvas.storeShapes()
        self.loadShapes(shapes, replace=False)
        self.setDirty()

    def _wait_for_pith_click(self):
        """Wait for user to click on canvas to select pith coordinates"""
        clicked_coords = [None]  # Use list to allow modification in nested function
        
        def single_click_handler(event):
            """Handle single click to capture pith coordinates"""
            try:
                # Get click position in image coordinates
                pos = self.canvas.transformPos(event.pos())
                x, y = pos.x(), pos.y()
                
                # Store coordinates
                clicked_coords[0] = (x, y)
                logger.info(f"Pith clicked at ({x:.1f}, {y:.1f})")
                
                # Restore original handler
                self.canvas.mousePressEvent = original_handler
            except Exception as e:
                logger.error(f"Error capturing pith click: {e}")
                # Restore original handler
                self.canvas.mousePressEvent = original_handler
        
        # Store original handler
        original_handler = self.canvas.mousePressEvent
        
        # Set up one-time click handler
        self.canvas.mousePressEvent = single_click_handler
        
        # Wait for click by entering nested event loop
        loop = QtCore.QEventLoop()
        
        # Set up timer to check if click was captured
        def check_click():
            if clicked_coords[0] is not None:
                loop.quit()
        
        timer = QtCore.QTimer()
        timer.timeout.connect(check_click)
        timer.start(100)  # Check every 100ms
        
        # Also allow ESC key to cancel
        original_key_handler = self.canvas.keyPressEvent
        def cancel_on_esc(event):
            if event.key() == QtCore.Qt.Key_Escape:
                logger.info("Pith click cancelled by user")
                self.canvas.mousePressEvent = original_handler
                self.canvas.keyPressEvent = original_key_handler
                loop.quit()
            else:
                original_key_handler(event)
        self.canvas.keyPressEvent = cancel_on_esc
        
        # Wait for click or cancel (with 30 second timeout)
        QtCore.QTimer.singleShot(30000, loop.quit)
        loop.exec_()
        
        timer.stop()
        
        # Restore handlers
        self.canvas.mousePressEvent = original_handler
        self.canvas.keyPressEvent = original_key_handler
        
        return clicked_coords[0]
    
    def _action_clear_all_rings(self) -> None:
        """Remove all ring polygons from the canvas"""
        if not self.canvas.shapes:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Shapes"),
                self.tr("There are no shapes to clear.")
            )
            return
        
        # Count ring polygons
        ring_shapes = [s for s in self.canvas.shapes if s.label and s.label.startswith("ring_")]
        
        if not ring_shapes:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Rings"),
                self.tr("There are no ring polygons to clear.")
            )
            return
        
        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("Clear All Rings"),
            self.tr(f"Remove all {len(ring_shapes)} ring polygons?\n\nThis action cannot be undone."),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Remove all ring shapes
            self.canvas.storeShapes()  # Save to undo stack
            remaining_shapes = [s for s in self.canvas.shapes if not (s.label and s.label.startswith("ring_"))]
            self.loadShapes(remaining_shapes, replace=True)
            self.setDirty()
            logger.info(f"Cleared {len(ring_shapes)} ring polygons")
            self.show_status_message(self.tr(f"Cleared {len(ring_shapes)} ring polygons"))
    
    def _action_metadata(self) -> None:
        """Input and store sample metadata (harvested year, sample code, observations)"""
        if self.image.isNull():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Image"),
                self.tr("Please open an image first.")
            )
            return
        
        # Show metadata dialog
        dlg = MetadataDialog(existing_metadata=self.sample_metadata, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        # Store metadata
        self.sample_metadata = dlg.get_metadata()
        
        logger.info(f"Sample metadata updated:")
        logger.info(f"  - Harvested Year: {self.sample_metadata['harvested_year']}")
        logger.info(f"  - Sample Code: {self.sample_metadata['sample_code']}")
        logger.info(f"  - Observation: {self.sample_metadata['observation'][:50]}..." if len(self.sample_metadata['observation']) > 50 else f"  - Observation: {self.sample_metadata['observation']}")
        
        # Store in otherData for saving to JSON
        if self.otherData is None:
            self.otherData = {}
        self.otherData["sample_metadata"] = self.sample_metadata
        
        # Mark as dirty to prompt save
        self.setDirty()
        
        self.show_status_message(
            self.tr(f"Metadata saved: {self.sample_metadata['sample_code']} ({self.sample_metadata['harvested_year']})")
        )
        
        # If there are existing rings, ask if user wants to relabel them with years
        ring_shapes = [s for s in self.canvas.shapes if s.label and s.label.startswith("ring_")]
        if ring_shapes:
            reply = QtWidgets.QMessageBox.question(
                self,
                self.tr("Relabel Existing Rings?"),
                self.tr(f"You have {len(ring_shapes)} existing rings.\n\n"
                       f"Do you want to relabel them with years based on harvested year {self.sample_metadata['harvested_year']}?\n\n"
                       f"The outermost ring (which circumscribes all others) will be labeled {self.sample_metadata['harvested_year']}, "
                       f"and inner rings will have decreasing years toward the pith."),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self._relabel_rings_with_years()
    
    def _relabel_rings_with_years(self):
        """Relabel existing rings with years based on harvested year"""
        if not self.sample_metadata or 'harvested_year' not in self.sample_metadata:
            return
        
        # Get all ring shapes and sort by label (ring_1, ring_2, ...)
        ring_shapes = [s for s in self.canvas.shapes if s.label and s.label.startswith("ring_")]
        
        def extract_ring_number(shape):
            try:
                # Handle both "ring_123" and "ring_2020" formats
                parts = shape.label.split('_')
                if len(parts) >= 2:
                    return int(parts[1])
                return 0
            except (IndexError, ValueError):
                return 0
        
        ring_shapes.sort(key=extract_ring_number)
        
        # Relabel: Sorted rings go from innermost to outermost (ring_1 to ring_N)
        # Innermost (ring_1) gets oldest year, outermost (ring_N) gets harvested_year
        harvested_year = self.sample_metadata['harvested_year']
        n_rings = len(ring_shapes)
        
        # Calculate innermost year using datetime
        harvested_date = datetime(harvested_year, 1, 1)
        innermost_date = harvested_date - relativedelta(years=n_rings - 1)
        innermost_year = innermost_date.year
        
        logger.info(f"Relabeling {n_rings} rings: innermost={innermost_year}, outermost={harvested_year}")
        
        for i, shape in enumerate(ring_shapes):
            # innermost (i=0) gets oldest year, outermost (i=n-1) gets harvested year
            ring_date = harvested_date - relativedelta(years=(n_rings - 1 - i))
            new_year = ring_date.year
            new_label = f"ring_{new_year}"
            logger.info(f"Relabeling {shape.label} → {new_label}")
            shape.label = new_label
        
        # Update label list
        self.loadShapes(self.canvas.shapes, replace=True)
        self.setDirty()
        
        logger.info(f"✓ Relabeled {len(ring_shapes)} rings with years")
        self.show_status_message(self.tr(f"Relabeled {len(ring_shapes)} rings with years"))
    
    def _action_set_scale(self) -> None:
        """Set physical scale/calibration for measurements"""
        if self.image.isNull():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Image"),
                self.tr("Please open an image first.")
            )
            return
        
        from tras.widgets import ScaleDialog
        
        # Get current scale if exists
        current_scale = self.image_scale['value'] if self.image_scale else None
        current_unit = self.image_scale['unit'] if self.image_scale else 'mm'
        
        # Show scale dialog
        dlg = ScaleDialog(parent=self, current_scale=current_scale, current_unit=current_unit)
        result = dlg.exec_()
        
        if result == 2:  # Draw line mode
            # User wants to draw a line
            self._enter_scale_line_mode()
            return
        elif result != QtWidgets.QDialog.Accepted:
            return
        
        # Direct input mode
        if dlg.get_method() == 'direct':
            scale_value = dlg.get_scale_value()
            unit = dlg.get_unit()
            
            self.image_scale = {
                'value': scale_value,
                'unit': unit
            }
            
            logger.info(f"Scale set (direct input): {scale_value:.6f} {unit}/pixel")
            
            # Store in otherData
            if self.otherData is None:
                self.otherData = {}
            self.otherData["image_scale"] = self.image_scale
            
            self.setDirty()
            self.show_status_message(
                self.tr(f"Scale set: {scale_value:.6f} {unit}/pixel")
            )
    
    def _enter_scale_line_mode(self):
        """Enter mode for drawing a calibration line"""
        # Set flag so newShape() knows this is a calibration line
        self._skip_next_label = True
        self._waiting_for_scale_line = True
        self.toggleDrawMode(False, createMode="line")
        
        self.show_status_message(
            self.tr("Draw a line of known length, then specify its physical size...")
        )
    
    def _handle_scale_line(self, shape):
        """Handle a drawn scale calibration line"""
        from tras.widgets import LineCalibrationDialog
        
        # Calculate line length in pixels
        if len(shape.points) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Invalid Line"),
                self.tr("Please draw a line with at least 2 points.")
            )
            # Remove the invalid shape
            self.canvas.shapes.remove(shape)
            self.loadShapes(self.canvas.shapes, replace=True)
            return
        
        # Calculate line length (Euclidean distance)
        p1 = shape.points[0]
        p2 = shape.points[-1]
        line_length_pixels = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
        
        logger.info(f"Scale calibration line drawn: {line_length_pixels:.2f} pixels")
        
        # Show calibration dialog
        dlg = LineCalibrationDialog(line_length_pixels, parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            scale = dlg.get_scale(line_length_pixels)
            unit = dlg.get_unit()
            
            if scale:
                self.image_scale = {
                    'value': scale,
                    'unit': unit
                }
                
                logger.info(f"Scale calibrated: {scale:.6f} {unit}/pixel")
                
                # Store in otherData
                if self.otherData is None:
                    self.otherData = {}
                self.otherData["image_scale"] = self.image_scale
                
                self.setDirty()
                self.show_status_message(
                    self.tr(f"Scale calibrated: {scale:.6f} {unit}/pixel")
                )
        
        # Remove the calibration line (it's just for measurement)
        self.canvas.shapes.remove(shape)
        self.loadShapes(self.canvas.shapes, replace=True)
    
    def _action_measure_radial_width(self) -> None:
        """Measure ring widths along a radial line from pith - opens dialog"""
        from tras.widgets import RadialWidthDialog
        
        # Check if we have rings
        ring_shapes = [s for s in self.canvas.shapes if s.label and s.label.startswith("ring_")]
        
        if not ring_shapes:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Rings"),
                self.tr("No tree rings found.\n\n"
                       "Please detect rings first using Tools > Tree Ring Detection.")
            )
            return
        
        # Open dialog in a loop to handle multiple actions
        while True:
            dlg = RadialWidthDialog(
                parent=self,
                pith_xy=self.pith_xy,
                has_measurement=(self.radial_line_measurements is not None),
                measurement_data=self.radial_line_measurements
            )
            
            result = dlg.exec_()
            
            if result == QtWidgets.QDialog.Rejected:
                # User closed dialog
                return
            
            action = dlg.get_action()
            
            if action == 'set_direction':
                # Set/change measurement direction
                success = self._perform_radial_measurement(ring_shapes)
                if not success:
                    return  # Error or cancelled
                # Continue loop to show dialog again with updated measurement
            
            elif action == 'clear':
                # Clear measurement line
                self._clear_radial_measurement_lines()
                self.radial_line_measurements = None
                if self.otherData and "radial_line_measurements" in self.otherData:
                    del self.otherData["radial_line_measurements"]
                self.setDirty()
                self.show_status_message(self.tr("Cleared radial measurement line"))
                # Continue loop to show dialog again
            
            elif action == 'export':
                # Export to .POS format
                self._export_radial_to_pos()
                # Continue loop to show dialog again
            
            else:
                # Unknown action
                return
    
    def _perform_radial_measurement(self, ring_shapes):
        """Perform the actual radial measurement (pith + direction clicks)"""
        
        # Check if we have stored pith coordinates from detection
        has_detected_pith = self.pith_xy is not None
        
        # Ask user whether to use detected pith or click custom pith
        if has_detected_pith:
            # Offer choice between detected pith and custom pith
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle(self.tr("Pith Location"))
            msg.setText(self.tr(f"Detected pith at ({self.pith_xy[0]:.1f}, {self.pith_xy[1]:.1f})\n\n"
                               "How do you want to set the pith (origin of radial line)?"))
            msg.setIcon(QtWidgets.QMessageBox.Question)
            
            use_detected_btn = msg.addButton(self.tr("Use Detected Pith"), QtWidgets.QMessageBox.AcceptRole)
            custom_pith_btn = msg.addButton(self.tr("Click to Set Custom Pith"), QtWidgets.QMessageBox.ActionRole)
            cancel_btn = msg.addButton(QtWidgets.QMessageBox.Cancel)
            
            msg.exec_()
            
            if msg.clickedButton() == cancel_btn:
                return False
            elif msg.clickedButton() == use_detected_btn:
                pith_xy = self.pith_xy
                logger.info(f"Using detected pith: ({pith_xy[0]:.1f}, {pith_xy[1]:.1f})")
            else:  # custom_pith_btn
                # Let user click to set pith
                QtWidgets.QMessageBox.information(
                    self,
                    self.tr("Set Custom Pith"),
                    self.tr("Click on the image to set the pith (tree center).\n\n"
                           "This will be used as the origin of the radial measurement line.")
                )
                self.show_status_message(self.tr("Click on the pith (tree center)..."))
                pith_xy = self._wait_for_single_click("custom pith")
                if pith_xy is None:
                    self.show_status_message(self.tr("Radial measurement cancelled"))
                    return False
                logger.info(f"Using custom pith: ({pith_xy[0]:.1f}, {pith_xy[1]:.1f})")
        else:
            # No detected pith, must click
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Set Pith Location"),
                self.tr("No pith detected. Click on the image to set the pith (tree center).\n\n"
                       "This will be used as the origin of the radial measurement line.")
            )
            self.show_status_message(self.tr("Click on the pith (tree center)..."))
            pith_xy = self._wait_for_single_click("pith")
            if pith_xy is None:
                self.show_status_message(self.tr("Radial measurement cancelled"))
                return False
            logger.info(f"Using clicked pith: ({pith_xy[0]:.1f}, {pith_xy[1]:.1f})")
        
        # Inform user about direction click
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Set Direction"),
            self.tr(f"Pith set at ({pith_xy[0]:.1f}, {pith_xy[1]:.1f})\n\n"
                   "Now click on the image to define the radial line direction.\n\n"
                   "The system will measure ring widths along this transect.")
        )
        
        # Get direction click
        self.show_status_message(self.tr("Click to define the radial line direction..."))
        direction_xy = self._wait_for_single_click("direction")
        
        if direction_xy is None:
            self.show_status_message(self.tr("Radial measurement cancelled"))
            return False
        
        logger.info(f"Radial measurement: Direction point at ({direction_xy[0]:.1f}, {direction_xy[1]:.1f})")
        
        # Compute ring widths along the line
        from tras.utils.ring_width_measurer import compute_ring_widths_along_line
        
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            measurements = compute_ring_widths_along_line(ring_shapes, pith_xy, direction_xy)
            
            QtWidgets.QApplication.restoreOverrideCursor()
            
            if not measurements:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("No Intersections"),
                    self.tr("No ring intersections found along the specified line.\n\n"
                           "Try a different direction or check that rings are properly detected.")
                )
                return False
            
            # Store measurements
            self.radial_line_measurements = {
                'pith_xy': pith_xy,
                'direction_xy': direction_xy,
                'measurements': measurements
            }
            
            # Store in otherData for persistence
            if self.otherData is None:
                self.otherData = {}
            self.otherData["radial_line_measurements"] = self.radial_line_measurements
            
            # Draw the radial line on the canvas for visual feedback
            self._draw_radial_measurement_line(pith_xy, direction_xy)
            
            self.setDirty()
            
            logger.info(f"✓ Measured {len(measurements)} rings along radial line")
            
            # Show summary
            message = f"Measured {len(measurements)} rings along radial line:\n\n"
            
            # Show first few measurements
            sorted_rings = sorted(measurements.items(), key=lambda x: measurements[x[0]]['distance_from_pith'])
            for i, (label, data) in enumerate(sorted_rings[:10]):
                width = data['radial_width']
                if self.image_scale:
                    # Convert to physical units
                    scale = self.image_scale['value']
                    unit = self.image_scale['unit']
                    width_physical = width * scale
                    message += f"{label}: {width_physical:.4f} {unit}\n"
                else:
                    message += f"{label}: {width:.2f} px\n"
            
            if len(measurements) > 10:
                message += f"\n... and {len(measurements) - 10} more rings"
            
            message += f"\n\nMeasurement saved! Use the dialog to export to .POS format."
            
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Radial Width Measurement Complete"),
                message
            )
            
            self.show_status_message(
                self.tr(f"✓ Measured {len(measurements)} rings along radial line")
            )
            
            return True
        
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Measurement Failed"),
                self.tr(f"Failed to measure ring widths:\n\n{str(e)}")
            )
            logger.error(f"Radial width measurement failed: {e}", exc_info=True)
            return False
    
    def _export_radial_to_pos(self):
        """Export radial measurements to .POS format for CooRecorder"""
        from tras.utils.pos_exporter import export_to_pos
        
        if not self.radial_line_measurements:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("No Measurements"),
                self.tr("No radial measurements to export.\n\n"
                       "Please set a measurement direction first.")
            )
            return
        
        # Get default filename from sample code if available
        default_filename = "ring_measurements.pos"
        if self.sample_metadata and 'sample_code' in self.sample_metadata:
            sample_code = self.sample_metadata['sample_code']
            safe_code = "".join(c for c in sample_code if c.isalnum() or c in ('-', '_'))
            default_filename = f"{safe_code}.pos"
        
        # Ask user for filename
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export Radial Measurements to .POS"),
            default_filename,
            self.tr("CooRecorder POS Files (*.pos);;All Files (*)")
        )
        
        if not filename:
            return
        
        # Export
        success = export_to_pos(
            filepath=filename,
            measurements=self.radial_line_measurements['measurements'],
            pith_xy=self.radial_line_measurements['pith_xy'],
            direction_xy=self.radial_line_measurements['direction_xy'],
            image_scale=self.image_scale,
            metadata=self.sample_metadata
        )
        
        if success:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Export Successful"),
                self.tr(f"Radial measurements exported to:\n{filename}\n\n"
                       f"This .POS file can be opened in CooRecorder for further analysis.")
            )
            logger.info(f"Exported radial measurements to {filename}")
        else:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Export Failed"),
                self.tr("Failed to export measurements to .POS format.\n\n"
                       "Check console for error details.")
            )
    
    def _wait_for_single_click(self, click_type="point"):
        """Wait for a single click on the canvas and return coordinates"""
        from PyQt5 import QtCore
        
        clicked_point = [None]
        loop = QtCore.QEventLoop()
        
        def single_click_handler(ev):
            # Transform click to image coordinates
            pos = self.canvas.transformPos(ev.pos())
            clicked_point[0] = (pos.x(), pos.y())
            logger.info(f"{click_type} clicked at ({pos.x():.1f}, {pos.y():.1f})")
            loop.quit()
        
        def key_handler(ev):
            if ev.key() == QtCore.Qt.Key_Escape:
                logger.info(f"{click_type} click cancelled")
                loop.quit()
        
        # Temporarily override mouse press event
        original_mouse_press = self.canvas.mousePressEvent
        original_key_press = self.canvas.keyPressEvent
        
        self.canvas.mousePressEvent = single_click_handler
        self.canvas.keyPressEvent = key_handler
        
        # Set a timeout (30 seconds)
        timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(30000)
        
        # Wait for click or timeout
        loop.exec_()
        
        # Restore original handlers
        self.canvas.mousePressEvent = original_mouse_press
        self.canvas.keyPressEvent = original_key_press
        
        timer.stop()
        
        return clicked_point[0]
    
    def _clear_radial_measurement_lines(self):
        """Remove all radial measurement lines from canvas"""
        radial_lines = [s for s in self.canvas.shapes if s.label == "radial_measurement_line"]
        if radial_lines:
            for line in radial_lines:
                self.canvas.shapes.remove(line)
                # Also remove from label list
                item = self.labelList.findItemByShape(line)
                if item:
                    self.labelList.removeItem(item)
            self.canvas.update()
            logger.info(f"Cleared {len(radial_lines)} radial measurement line(s)")
            return len(radial_lines)
        return 0
    
    def _draw_radial_measurement_line(self, pith_xy, direction_xy):
        """Draw a visual line showing the radial measurement transect"""
        from tras.shape import Shape
        import math
        
        # First, clear any existing radial measurement lines
        self._clear_radial_measurement_lines()
        
        # Extend the line to the edge of the image
        # Calculate direction vector
        dx = direction_xy[0] - pith_xy[0]
        dy = direction_xy[1] - pith_xy[1]
        
        # Normalize and extend to a very long distance
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1:
            return
        
        dx /= length
        dy /= length
        
        # Extend to 10x image diagonal (ensures it reaches edge)
        max_dim = max(self.image.width(), self.image.height())
        extension = max_dim * math.sqrt(2) * 10
        
        end_x = pith_xy[0] + dx * extension
        end_y = pith_xy[1] + dy * extension
        
        # Create a line shape
        line_shape = Shape(label="radial_measurement_line", shape_type="line")
        line_shape.points = [
            QtCore.QPointF(pith_xy[0], pith_xy[1]),
            QtCore.QPointF(end_x, end_y)
        ]
        line_shape.close()
        
        # Set distinctive color (cyan/bright blue)
        line_shape.line_color = QtGui.QColor(0, 255, 255)  # Cyan
        line_shape.fill_color = QtGui.QColor(0, 255, 255, 0)  # Transparent fill
        
        # Add to canvas
        self.canvas.shapes.append(line_shape)
        self.addLabel(line_shape)
        self.canvas.update()
        
        logger.info(f"Drew radial measurement line from ({pith_xy[0]:.1f}, {pith_xy[1]:.1f}) to ({end_x:.1f}, {end_y:.1f})")
    
    def _action_ring_properties(self) -> None:
        """Compute and display ring properties (area, perimeter, etc.)"""
        if not self.canvas.shapes:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Shapes"),
                self.tr("There are no shapes to analyze.")
            )
            return
        
        # Filter ring shapes and sort by label (ring_1, ring_2, ...)
        ring_shapes = [s for s in self.canvas.shapes if s.label and s.label.startswith("ring_")]
        
        if not ring_shapes:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Rings"),
                self.tr("There are no ring polygons to analyze.\n\n"
                       "Please detect rings first using Tools > Tree Ring Detection.")
            )
            return
        
        # Automatically re-label rings before computing properties
        # (in case user added/removed rings manually)
        logger.info("Auto-relabeling rings before computing properties...")
        self._relabel_rings_with_years()
        
        # Sort rings by label number
        # If using years: sort descending (2020, 2019, 2018... = outermost to innermost)
        # If using numbers: sort ascending (ring_1, ring_2, ring_3... = outermost to innermost)
        def extract_ring_number(shape):
            try:
                return int(shape.label.split('_')[1])
            except (IndexError, ValueError):
                return 0
        
        # Determine if we're using year labels (numbers > 1000) or sequential labels
        first_num = extract_ring_number(ring_shapes[0]) if ring_shapes else 0
        using_years = first_num > 1000
        
        # Sort appropriately
        ring_shapes.sort(key=extract_ring_number, reverse=using_years)
        
        logger.info(f"Computing properties for {len(ring_shapes)} rings...")
        
        # Compute properties for each ring
        ring_properties = []
        cumulative_area = 0.0
        
        for i, shape in enumerate(ring_shapes):
            # Get polygon points
            points = [(p.x(), p.y()) for p in shape.points]
            
            if len(points) < 3:
                logger.warning(f"Skipping {shape.label}: too few points ({len(points)})")
                continue
            
            # Compute area using Shoelace formula
            area = 0.0
            n = len(points)
            for j in range(n):
                x1, y1 = points[j]
                x2, y2 = points[(j + 1) % n]
                area += x1 * y2 - x2 * y1
            area = abs(area) / 2.0
            
            # Compute perimeter
            perimeter = 0.0
            for j in range(n):
                x1, y1 = points[j]
                x2, y2 = points[(j + 1) % n]
                perimeter += ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            
            cumulative_area += area
            
            # Build properties dict (no centroid width)
            props = {
                'label': shape.label,
                'area': area,
                'cumulative_area': cumulative_area,
                'perimeter': perimeter
            }
            
            # Add radial width if available
            if self.radial_line_measurements:
                measurements = self.radial_line_measurements.get('measurements', {})
                if shape.label in measurements:
                    radial_data = measurements[shape.label]
                    props['radial_width_px'] = radial_data['radial_width']
                else:
                    props['radial_width_px'] = None
            else:
                props['radial_width_px'] = None
            
            ring_properties.append(props)
        
        if not ring_properties:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("No Valid Rings"),
                self.tr("No rings have enough points for analysis.")
            )
            return
        
        logger.info(f"✓ Computed properties for {len(ring_properties)} rings")
        
        # Prepare metadata including scale and sample info
        metadata = {}
        if self.sample_metadata:
            metadata.update(self.sample_metadata)
        if self.image_scale:
            metadata['scale'] = self.image_scale
        
        # Show dialog with results
        dlg = RingPropertiesDialog(ring_properties, parent=self, metadata=metadata if metadata else None)
        dlg.exec_()
    
    def _action_preprocess_image(self) -> None:
        """Preprocess the current image (resize, crop, remove background)"""
        if self.image.isNull():
            self.errorMessage(self.tr("No image"), self.tr("Please open an image first."))
            return
        
        # Convert current image to numpy array using PIL for robustness
        from PIL import Image as PILImage
        import tempfile
        import os
        
        logger.info(f"Original QImage format: {self.image.format()} (RGB888={QtGui.QImage.Format_RGB888})")
        
        # Save QImage to temp file and reload with PIL (most robust method)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        self.image.save(tmp_path, 'PNG')
        pil_img = PILImage.open(tmp_path)
        image_np = np.array(pil_img, dtype=np.uint8)
        
        # Ensure RGB (not RGBA)
        if image_np.ndim == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        logger.info(f"Image array shape: {image_np.shape}, dtype: {image_np.dtype}")
        
        # Check for crop region (last rectangle drawn)
        crop_rect = None
        if self.canvas.shapes:
            # Find the last rectangle shape
            for shape in reversed(self.canvas.shapes):
                if shape.shape_type == "rectangle" and len(shape.points) >= 2:
                    # Get bounding box
                    pts = np.array([[p.x(), p.y()] for p in shape.points])
                    x_min, y_min = np.min(pts, axis=0).astype(int)
                    x_max, y_max = np.max(pts, axis=0).astype(int)
                    w, h = x_max - x_min, y_max - y_min
                    crop_rect = (x_min, y_min, w, h)
                    logger.info(f"Found crop rectangle: {crop_rect}")
                    break
        
        # Show preprocessing dialog (with potential loop for crop drawing)
        while True:
            dlg = PreprocessDialog(image=image_np, crop_rect=crop_rect, parent=self)
            result = dlg.exec_()
            
            if result == 2:  # User clicked "Draw Crop Rectangle"
                # Set canvas to rectangle mode
                self._skip_next_label = True
                self._waiting_for_crop_rect = True  # Flag to auto-reopen dialog after drawing
                self.toggleDrawMode(False, createMode="rectangle")
                
                # Instruct user
                self.show_status_message(self.tr("Draw a rectangle to define the crop region..."))
                return
            elif result != QtWidgets.QDialog.Accepted:
                return
            else:
                break  # User accepted
        
        # Get processed image
        processed_img = dlg.get_processed_image()
        preprocessing_info = dlg.get_preprocessing_info()
        
        # LOG: Check processed image before converting to QImage
        logger.info(f"Processed image from dialog: shape={processed_img.shape}, dtype={processed_img.dtype}")
        logger.info(f"  Stats: min={processed_img.min()}, max={processed_img.max()}, mean={processed_img.mean():.2f}")
        logger.info(f"  First pixel: {processed_img[0, 0]}")
        logger.info(f"  Contiguous: {processed_img.flags['C_CONTIGUOUS']}")
        
        # Confirm replacement
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("Replace Image"),
            self.tr(f"Replace current image with processed version?\n\n"
                   f"Original: {preprocessing_info['original_size'][0]} x {preprocessing_info['original_size'][1]}\n"
                   f"Processed: {preprocessing_info['processed_size'][0]} x {preprocessing_info['processed_size'][1]}\n"
                   f"Scale: {preprocessing_info['scale_factor']:.2f}\n\n"
                   f"Note: All existing annotations will be cleared."),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        # Clear existing shapes
        self.canvas.shapes = []
        self.labelList.clear()
        
        # Convert processed image to QImage using PIL as intermediate
        # This ensures proper data handling without corruption
        from PIL import Image as PILImage
        
        # Ensure image is uint8 and contiguous
        processed_img = np.ascontiguousarray(processed_img, dtype=np.uint8)
        
        # Save to temporary file and load - most reliable method
        # This ensures Qt's image loader handles everything correctly
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            PILImage.fromarray(processed_img).save(tmp_path, format='PNG')
        
        # Load into QImage and explicitly convert to RGB888 format
        self.image = QtGui.QImage(tmp_path)
        
        # CRITICAL: Ensure the loaded image is in RGB888 format (not BGR or other formats)
        if self.image.format() != QtGui.QImage.Format_RGB888:
            logger.info(f"Converting loaded image from format {self.image.format()} to RGB888")
            self.image = self.image.convertToFormat(QtGui.QImage.Format_RGB888)
        
        # Clean up temp file
        import os
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        # CRITICAL FIX: Store the preprocessed numpy array directly
        # This bypasses ALL QImage corruption issues
        self.imageArray = processed_img.copy()  # Store pristine numpy array
        logger.info(f"✓ Stored preprocessed image as numpy array: shape={self.imageArray.shape}, dtype={self.imageArray.dtype}")
        
        # LOG: Verify QImage format
        logger.info(f"QImage format: {self.image.format()} (RGB888=13), size={self.image.width()}x{self.image.height()}")
        
        # CRITICAL: Also update imageData to match the preprocessed image
        # This ensures that everything (canvas, detection, save) uses the preprocessed version
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(processed_img).save(buffer, format='PNG')
        self.imageData = buffer.getvalue()
        logger.info(f"Updated self.imageData with preprocessed image ({len(self.imageData)} bytes)")
        
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(self.image))
        
        # IMPORTANT: Adjust scale/zoom for the new image size and enable canvas
        self.canvas.setEnabled(True)
        self.toggleActions(True)  # Enable actions like detection, etc.
        self.adjustScale(initial=True)
        self.paintCanvas()
        self.canvas.setFocus()  # Give focus to canvas for mouse/keyboard events
        
        # Log preprocessing applied
        logger.info(f"✓ Image replaced with preprocessed version:")
        logger.info(f"  - Original size: {preprocessing_info['original_size'][0]}x{preprocessing_info['original_size'][1]}")
        logger.info(f"  - New size: {preprocessing_info['processed_size'][0]}x{preprocessing_info['processed_size'][1]}")
        if preprocessing_info.get('crop_rect'):
            logger.info(f"  - Crop: {preprocessing_info['crop_rect']}")
        if preprocessing_info.get('scale_factor', 1.0) != 1.0:
            logger.info(f"  - Scale: {preprocessing_info['scale_factor']:.2f}")
        if preprocessing_info.get('background_removed'):
            logger.info(f"  - Background removed: {preprocessing_info.get('background_method', 'yes')}")
        logger.info(f"  → Detection methods will now use this preprocessed image")
        
        # Store preprocessing info in otherData
        if self.otherData is None:
            self.otherData = {}
        self.otherData["preprocessing"] = preprocessing_info
        
        # Adjust image scale if it was set and image was resized
        if self.image_scale and preprocessing_info.get('scale_factor', 1.0) != 1.0:
            scale_factor = preprocessing_info['scale_factor']
            old_scale = self.image_scale['value']
            # When image is smaller, each pixel represents more physical distance
            new_scale = old_scale / scale_factor
            
            logger.info(f"Adjusting scale for resize: {old_scale:.6f} → {new_scale:.6f} {self.image_scale['unit']}/pixel")
            
            self.image_scale['value'] = new_scale
            self.otherData["image_scale"] = self.image_scale
        
        # Mark as modified
        self.setDirty()
        self.show_status_message(
            self.tr(f"Image preprocessed: {preprocessing_info['processed_size'][0]}x{preprocessing_info['processed_size'][1]}")
        )
        
        logger.info(f"Image preprocessed: {preprocessing_info}")

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            for _, draw_action in self.draw_actions:
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in self.draw_actions:
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, f"&{i + 1} {QtCore.QFileInfo(f).fileName()}", self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)  # type: ignore[attr-defined,union-attr]
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
            flags_disabled=not edit_flags,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        for item in items:
            shape: Shape = item.shape()  # type: ignore[no-redef]

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            if shape.group_id is None:
                r, g, b = shape.fill_color.getRgb()[:3]
                item.setText(
                    f"{html.escape(shape.label)} "
                    f'<font color="#{r:02x}{g:02x}{b:02x}">●</font>'
                )
            else:
                item.setText(f"{shape.label} ({shape.group_id})")
            self.setDirty()
            if self.uniqLabelList.find_label_item(shape.label) is None:
                self.uniqLabelList.add_label_item(
                    label=shape.label, color=self._get_rgb_by_label(label=shape.label)
                )

    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = f"{shape.label} ({shape.group_id})"
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.find_label_item(shape.label) is None:
            self.uniqLabelList.add_label_item(
                label=shape.label, color=self._get_rgb_by_label(label=shape.label)
            )
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.on_shapes_present_actions:
            action.setEnabled(True)

        self._update_shape_color(shape)
        r, g, b = shape.fill_color.getRgb()[:3]
        label_list_item.setText(
            f'{html.escape(text)} <font color="#{r:02x}{g:02x}{b:02x}">●</font>'
        )

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label: str) -> tuple[int, int, int]:
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.find_label_item(label)
            item_index: int = (
                self.uniqLabelList.indexFromItem(item).row()
                if item
                else self.uniqLabelList.count()
            )
            label_id: int = (
                1  # skip black color by default
                + item_index
                + self._config["shift_auto_shape_color"]
            )
            rgb: tuple[int, int, int] = tuple(
                LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)].tolist()
            )
            return rgb
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            if not (
                len(self._config["label_colors"][label]) == 3
                and all(0 <= c <= 255 for c in self._config["label_colors"][label])
            ):
                raise ValueError(
                    "Color for label must be 0-255 RGB tuple, but got: "
                    f"{self._config['label_colors'][label]}"
                )
            return tuple(self._config["label_colors"][label])
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        if replace:
            self.labelList.clear()
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def _load_shape_dicts(self, shape_dicts: list[ShapeDict]) -> None:
        shapes: list[Shape] = []
        shape_dict: ShapeDict
        for shape_dict in shape_dicts:
            shape: Shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                group_id=shape_dict["group_id"],
                description=shape_dict["description"],
                mask=shape_dict["mask"],
            )
            for x, y in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, shape.label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(shape_dict["flags"])
            shape.other_data = shape_dict["other_data"]

            shapes.append(shape)
        self.loadShapes(shapes=shapes)

    def loadFlags(self, flags):
        self.flag_widget.clear()  # type: ignore[union-attr]
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)  # type: ignore[union-attr]

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):  # type: ignore[union-attr]
            item = self.flag_widget.item(i)  # type: ignore[union-attr]
            assert item
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            assert self.imagePath
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        # Check if we should skip label prompt (for crop rectangles or scale lines)
        skip_label = getattr(self, '_skip_next_label', False)
        if skip_label:
            self._skip_next_label = False  # Reset flag
            # Check if this is a scale calibration line or crop rectangle
            if getattr(self, '_waiting_for_scale_line', False):
                text = "scale_calibration"
            else:
                text = "crop_region"
            flags = {}
            group_id = None
            description = ""
        else:
            items = self.uniqLabelList.selectedItems()
            text = None
            if items:
                text = items[0].data(Qt.UserRole)
            flags = {}
            group_id = None
            description = ""
            if self._config["display_label_popup"] or not text:
                previous_text = self.labelDialog.edit.text()
                text, flags, group_id, description = self.labelDialog.popUp(text)
                if not text:
                    self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
            
            # Check if we just drew a crop rectangle - auto-reopen preprocess dialog
            if getattr(self, '_waiting_for_crop_rect', False) and text == "crop_region":
                self._waiting_for_crop_rect = False
                # Use QTimer to reopen dialog after current event loop completes
                QtCore.QTimer.singleShot(100, self._action_preprocess_image)
            
            # Check if we just drew a scale calibration line
            elif getattr(self, '_waiting_for_scale_line', False) and text == "scale_calibration":
                self._waiting_for_scale_line = False
                # Calculate line length and show calibration dialog
                QtCore.QTimer.singleShot(100, lambda: self._handle_scale_line(shape))
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()
            # Reset flags if drawing was cancelled
            if getattr(self, '_waiting_for_crop_rect', False):
                self._waiting_for_crop_rect = False
                self.show_status_message(self.tr("Crop cancelled"))
            if getattr(self, '_waiting_for_scale_line', False):
                self._waiting_for_scale_line = False
                self.show_status_message(self.tr("Scale calibration cancelled"))

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value: bool, is_initial_load: bool = False):
        del value

        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData).convert("RGB"),
            self.onNewBrightnessContrast,
            parent=self,
        )

        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if is_initial_load:
            prev_filename: str = self.recentFiles[0] if self.recentFiles else ""
            if self._config["keep_prev_brightness_contrast"] and prev_filename:
                brightness, contrast = self.brightnessContrast_values.get(
                    prev_filename, (None, None)
                )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)

        if is_initial_load:
            dialog.onNewValue(None)
        else:
            dialog.exec_()
            brightness = dialog.slider_brightness.value()
            contrast = dialog.slider_contrast.value()

        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def loadFile(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.show_status_message(self.tr("Loading %s...") % osp.basename(str(filename)))
        label_file = f"{osp.splitext(filename)[0]}.json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p><p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.show_status_message(self.tr("Error reading %s") % label_file)
                return False
            assert self.labelFile is not None
            self.imageData = self.labelFile.imageData
            assert self.labelFile.imagePath
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self.otherData = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        assert self.imageData is not None
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                f"*.{fmt.data().decode()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.show_status_message(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self._load_shape_dicts(shape_dicts=self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        self.brightnessContrast(value=False, is_initial_load=True)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.show_status_message(self.tr("Loaded %s") % osp.basename(filename))
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super().resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            f"*.{fmt.data().decode()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + [f"*{LabelFile.suffix}"]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            return filename[0]
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = f"{osp.splitext(self.filename)[0]}.json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info(f"Label file is removed: {label_file}")

            item = self.fileListWidget.currentItem()
            if item:
                item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, f"<p><b>{title}</b></p>{message}"
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if self.canvas.hShape and not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.on_shapes_present_actions:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.on_shapes_present_actions:
                    action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.importDirImages(targetDirPath)

    @property
    def imageList(self) -> list[str]:
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            assert item
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = f"{osp.splitext(file)[0]}.json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self.openNextImg()
