"""Preprocessing utilities for image processing (resize, background removal).

This module provides pure functions for preprocessing images without PyQt5 dependencies,
making them usable in both GUI and CLI contexts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage


def preprocess_image(
    image: npt.NDArray[np.uint8],
    scale_factor: float = 1.0,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    remove_background: bool = False,
) -> npt.NDArray[np.uint8]:
    """
    Apply preprocessing steps to an image.

    Args:
        image: Input image as numpy array (H x W x 3, uint8, RGB)
        scale_factor: Resize scale (0.1 to 1.0). 1.0 means no resize.
        crop_rect: Optional crop rectangle as (x, y, width, height). None means no crop.
        remove_background: Whether to remove background using U2Net.

    Returns:
        Processed image as numpy array (H x W x 3, uint8, RGB)

    Raises:
        ValueError: If scale_factor is not in valid range
        RuntimeError: If background removal fails
    """
    if not (0.1 <= scale_factor <= 1.0):
        raise ValueError(f"scale_factor must be between 0.1 and 1.0, got {scale_factor}")

    img = image.copy()

    # Ensure image is in RGB format and contiguous
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # Remove alpha channel

    img = np.ascontiguousarray(img, dtype=np.uint8)

    # 1. Crop if region is selected
    if crop_rect is not None:
        x, y, w, h = crop_rect
        img = img[y : y + h, x : x + w]
        img = np.ascontiguousarray(img, dtype=np.uint8)

    # 2. Resize if needed
    if scale_factor != 1.0:
        h, w = img.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Use PIL for resize to avoid orientation issues
        pil_img = PILImage.fromarray(img, mode="RGB")
        pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.BICUBIC)
        img = np.array(pil_img, dtype=np.uint8)

        # Ensure output is contiguous
        img = np.ascontiguousarray(img, dtype=np.uint8)

    # 3. Remove background using U2Net if enabled
    if remove_background:
        try:
            from urudendro import remove_salient_object

            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / "input.png"
                output_path = Path(temp_dir) / "output.png"

                # Save current image (convert RGB to BGR for cv2.imwrite)
                cv2.imwrite(str(input_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # Run U2Net background removal
                remove_salient_object(str(input_path), str(output_path))

                # Load result (convert BGR back to RGB)
                result = cv2.imread(str(output_path))
                if result is not None:
                    img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    img = np.ascontiguousarray(img, dtype=np.uint8)
                else:
                    raise RuntimeError("U2Net did not produce output")

        except ImportError as e:
            raise RuntimeError(
                f"U2Net background removal requires urudendro package: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"U2Net background removal failed: {str(e)}") from e

    return img


def get_preprocessing_info(
    original_shape: Tuple[int, int],
    crop_rect: Optional[Tuple[int, int, int, int]],
    scale_factor: float,
    background_removed: bool,
) -> dict:
    """
    Get preprocessing metadata.

    Args:
        original_shape: Original image shape as (height, width)
        crop_rect: Crop rectangle as (x, y, width, height) or None
        scale_factor: Applied scale factor
        background_removed: Whether background was removed

    Returns:
        Dictionary with preprocessing metadata
    """
    return {
        "crop_rect": crop_rect,
        "scale_factor": scale_factor,
        "background_removed": background_removed,
        "background_method": "u2net" if background_removed else None,
        "original_size": [int(original_shape[1]), int(original_shape[0])],
    }




