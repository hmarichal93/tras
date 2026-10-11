"""Synthetic fixtures for the image/shape utility tests (no files needed in git)."""

import io

import numpy as np
import PIL.Image

from tras.utils import image as image_module
from tras.utils import shape as shape_module

IMAGE_SHAPE = (120, 160, 3)


def make_image() -> np.ndarray:
    """Deterministic RGB test image with smooth gradients (JPEG-friendly)."""
    yy, xx = np.mgrid[0 : IMAGE_SHAPE[0], 0 : IMAGE_SHAPE[1]]
    img = np.stack([xx * 255 // IMAGE_SHAPE[1], yy * 255 // IMAGE_SHAPE[0], (xx + yy) % 256], -1)
    return img.astype(np.uint8)


def make_jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    PIL.Image.fromarray(make_image()).save(buffer, format="JPEG")
    return buffer.getvalue()


def get_img_and_data():
    """Return (image, labelme-style data dict) with two polygons and embedded image."""
    data = {
        "imageData": image_module.img_arr_to_b64(make_image()),
        "shapes": [
            {"label": "ring_1", "points": [[20, 20], [60, 20], [60, 60], [20, 60]]},
            {"label": "ring_2", "points": [[80, 30], [140, 30], [110, 100]]},
        ],
    }
    img = image_module.img_b64_to_arr(data["imageData"])
    return img, data


def get_img_and_lbl():
    img, data = get_img_and_data()

    label_name_to_value = {"__background__": 0}
    for shape in data["shapes"]:
        label_name_to_value[shape["label"]] = len(label_name_to_value)

    label_names = [None] * len(label_name_to_value)
    for label_name, label_value in label_name_to_value.items():
        label_names[label_value] = label_name

    lbl, _ = shape_module.shapes_to_label(img.shape, data["shapes"], label_name_to_value)
    return img, lbl, label_names
