"""Shared image-preprocessing helpers for the ring-detection methods.

Every detection method (CS-TRD, DeepCS-TRD, INBD) and the pith detector (APD)
expects the input image as an RGB ``uint8`` array. The edge-based methods also
pad the image when the pith sits too close to a border, so the detector has room
to work. Centralizing that logic here keeps the per-method helpers in
``tras/utils/*_helper.py`` in sync.

All callers pass RGB images (``api._load_image`` and the GUI both decode via
PIL ``.convert("RGB")``), so 3-channel input is treated as RGB and never
channel-swapped.
"""

from __future__ import annotations

import cv2
import numpy as np

# Minimum free margin (in pixels) required between the pith and each image
# border before edge-based ring detection. If the pith is closer than this, the
# image is padded with a white border to create the margin.
EDGE_DETECTION_MIN_MARGIN = 100

# (top, bottom, left, right) padding amounts, in pixels.
Padding = tuple[int, int, int, int]


def normalize_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as a 3-channel RGB ``uint8`` array.

    Grayscale is expanded to RGB, a 4th (alpha) channel is dropped, and
    3-channel input is assumed to already be RGB (callers decode to RGB).
    Non-``uint8`` data is rescaled into the 0-255 range.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (255 * image).astype(np.uint8)
        else:
            image = (255 * (image.astype(np.float32) / image.max())).astype(np.uint8)

    return image


def pad_for_edge_detection(
    image: np.ndarray,
    center_xy: tuple[float, float],
    min_margin: int = EDGE_DETECTION_MIN_MARGIN,
) -> tuple[np.ndarray, tuple[float, float], Padding]:
    """Pad ``image`` with a white border if the pith is too close to an edge.

    Returns the (possibly padded) image, the pith coordinates adjusted for the
    padding, and the ``(top, bottom, left, right)`` padding amounts so callers
    can shift detected ring coordinates back into the original frame.
    """
    cx, cy = center_xy
    h, w = image.shape[:2]

    pad_top = max(0, min_margin - int(cy))
    pad_bottom = max(0, min_margin - (h - int(cy)))
    pad_left = max(0, min_margin - int(cx))
    pad_right = max(0, min_margin - (w - int(cx)))

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),  # White padding (background)
        )
        cx += pad_left
        cy += pad_top

    return image, (cx, cy), (pad_top, pad_bottom, pad_left, pad_right)
