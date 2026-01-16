"""Shared module for exporting annotations to JSON format.

This module provides a single canonical implementation for exporting
annotations that is used by both File > Save and Tools > Export Data.
"""

from __future__ import annotations

import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from tras._label_file import LabelFile, LabelFileError

if TYPE_CHECKING:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QListWidget, QListWidgetItem


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def export_annotations_json(window, output_file: Path | str) -> None:
    """Export annotations to JSON file using canonical format.
    
    This function implements the standard annotation export format used
    by TRAS. It extracts shapes, flags, metadata, and preprocessing info
    from the window and saves them to a JSON file.
    
    Args:
        window: MainWindow instance with annotations, metadata, etc.
        output_file: Path to output JSON file (will be created if needed)
        
    Raises:
        LabelFileError: If export fails (e.g., invalid image path)
        OSError: If output directory cannot be created
        
    Note:
        - Exports even if there are zero shapes (empty shapes list)
        - Includes preprocessing info if present in otherData
        - Includes sample_metadata, image_scale, and pith_xy if available
        - Respects window._config["store_data"] for imageData inclusion
    """
    output_file = Path(output_file)
    
    # Ensure output directory exists
    if output_file.parent and not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    lf = LabelFile()
    
    def format_shape(s):
        """Format a shape for JSON export."""
        data = s.other_data.copy() if hasattr(s, 'other_data') and s.other_data else {}
        data.update(
            dict(
                label=s.label,
                points=[(float(p.x()), float(p.y())) for p in s.points],
                group_id=s.group_id if hasattr(s, 'group_id') else None,
                description=s.description if hasattr(s, 'description') else "",
                shape_type=s.shape_type,
                flags=s.flags if hasattr(s, 'flags') else {},
                mask=None
                if not hasattr(s, 'mask') or s.mask is None
                else _img_arr_to_b64(s.mask.astype(np.uint8)),
            )
        )
        # Convert numpy types in shape data
        return _convert_numpy_types(data)
    
    # Extract shapes from labelList
    shapes = []
    if hasattr(window, 'labelList') and window.labelList:
        shapes = [format_shape(item.shape()) for item in window.labelList]
    
    # Extract flags from flag_widget
    flags = {}
    if hasattr(window, 'flag_widget') and window.flag_widget:
        from PyQt5.QtCore import Qt
        for i in range(window.flag_widget.count()):
            item = window.flag_widget.item(i)
            if item:
                key = item.text()
                flag = item.checkState() == Qt.Checked
                flags[key] = flag
    
    # Prepare image path (relative to JSON file)
    if not hasattr(window, 'imagePath') or not window.imagePath:
        raise LabelFileError("Cannot export: imagePath is not set")
    
    try:
        imagePath = osp.relpath(window.imagePath, osp.dirname(str(output_file)))
    except ValueError:
        # relpath can fail if paths are on different drives (Windows)
        # Fall back to basename
        logger.warning(f"Could not make imagePath relative, using basename")
        imagePath = osp.basename(window.imagePath)
    
    # Include imageData based on config
    imageData = None
    if hasattr(window, '_config') and window._config.get("store_data", True):
        imageData = getattr(window, 'imageData', None)
    elif not hasattr(window, '_config'):
        # If config doesn't exist, default to storing data
        imageData = getattr(window, 'imageData', None)
    
    # Prepare otherData - start with existing otherData or empty dict
    otherData = {}
    if hasattr(window, 'otherData') and window.otherData:
        otherData = window.otherData.copy()
    
    # Augment otherData with current state (only if present)
    # Note: pith_xy and radial measurements should be cleared on preprocess,
    # but we include them here if they exist
    
    if hasattr(window, 'pith_xy') and window.pith_xy is not None:
        otherData["pith_xy"] = window.pith_xy
    
    if hasattr(window, 'sample_metadata') and window.sample_metadata is not None:
        otherData["sample_metadata"] = window.sample_metadata
    
    if hasattr(window, 'image_scale') and window.image_scale is not None:
        otherData["image_scale"] = window.image_scale
    
    # Preprocessing info is already in otherData if it was set, so we don't need to add it again
    
    # Convert numpy types to native Python types for JSON serialization
    # This is critical for preprocessing info which may contain numpy int64 values
    otherData = _convert_numpy_types(otherData)
    
    # Get image dimensions
    if hasattr(window, 'image') and window.image:
        imageHeight = window.image.height()
        imageWidth = window.image.width()
    else:
        raise LabelFileError("Cannot export: image is not available")
    
    # Save to file
    lf.save(
        filename=str(output_file),
        shapes=shapes,
        imagePath=imagePath,
        imageData=imageData,
        imageHeight=imageHeight,
        imageWidth=imageWidth,
        otherData=otherData,
        flags=flags,
    )
    
    logger.info(f"Exported annotations to {output_file} ({len(shapes)} shapes)")


def _img_arr_to_b64(img_arr: np.ndarray) -> str:
    """Convert numpy image array to base64-encoded PNG string.
    
    This is a helper function that wraps the utils function.
    """
    from tras import utils
    return utils.img_arr_to_b64(img_arr)

