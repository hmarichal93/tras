"""Tests for preprocess_helper utilities."""

import numpy as np
import pytest

from tras.utils.preprocess_helper import preprocess_image


def test_preprocess_image_no_changes():
    """Test preprocessing with no changes."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_image(image, scale_factor=1.0, remove_background=False)
    assert result.shape == image.shape
    assert np.array_equal(result, image)


def test_preprocess_image_resize():
    """Test preprocessing with resize."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_image(image, scale_factor=0.5, remove_background=False)
    assert result.shape == (50, 50, 3)


def test_preprocess_image_invalid_scale():
    """Test preprocessing with invalid scale factor."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess_image(image, scale_factor=2.0, remove_background=False)
    with pytest.raises(ValueError):
        preprocess_image(image, scale_factor=0.05, remove_background=False)


def test_preprocess_image_grayscale():
    """Test preprocessing with grayscale image."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = preprocess_image(image, scale_factor=1.0, remove_background=False)
    assert result.shape == (100, 100, 3)  # Should be converted to RGB


def test_preprocess_image_rgba():
    """Test preprocessing with RGBA image."""
    image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    result = preprocess_image(image, scale_factor=1.0, remove_background=False)
    assert result.shape == (100, 100, 3)  # Should be converted to RGB




