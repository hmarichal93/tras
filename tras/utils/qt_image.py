"""Robust QImage <-> numpy conversion.

Direct buffer access (``QImage.bits()`` + ``reshape``) ignores QImage's 32-bit
row alignment and silently warps or mis-channels images for some widths/formats
(see ``img_qt_to_arr`` in :mod:`tras.utils.image`). The GUI therefore standardized
on a PIL round-trip via a temporary PNG, which lets Qt and PIL handle stride and
format correctly. This module centralizes that conversion.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image as PILImage
from PyQt5 import QtGui


def qimage_to_numpy(qimage: QtGui.QImage) -> np.ndarray:
    """Convert a QImage to an RGB ``uint8`` numpy array (H x W x 3).

    Saves the QImage to a temporary PNG and reloads it with PIL. Slower than
    direct buffer access but avoids QImage stride/format pitfalls.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        qimage.save(tmp_path, "PNG")
        with PILImage.open(tmp_path) as pil_img:
            return np.array(pil_img.convert("RGB"), dtype=np.uint8)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
