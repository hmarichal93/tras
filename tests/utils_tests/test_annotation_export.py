"""Tests for annotation export utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tras._label_file import LabelFile
from tras.utils.annotation_export import export_annotations_json


class MockShape:
    """Mock shape object for testing."""
    
    def __init__(self, label="test", points=None, shape_type="polygon"):
        self.label = label
        self.points = points or []
        self.group_id = None
        self.description = ""
        self.shape_type = shape_type
        self.flags = {}
        self.mask = None
        self.other_data = {}


class MockPoint:
    """Mock point object for testing."""
    
    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    def x(self):
        return self._x
    
    def y(self):
        return self._y


class MockWindow:
    """Mock window object for testing annotation export."""
    
    def __init__(self):
        self.imagePath = "/test/image.png"
        self.imageData = b"fake_image_data"
        self.image = MagicMock()
        self.image.height.return_value = 100
        self.image.width.return_value = 200
        self._config = {"store_data": True}
        self.otherData = {}
        self.sample_metadata = None
        self.image_scale = None
        self.pith_xy = None
        self.labelList = []
        self.flag_widget = MagicMock()
        self.flag_widget.count.return_value = 0


def test_export_empty_shapes():
    """Test exporting with zero shapes."""
    window = MockWindow()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Verify file was created
        assert output_file.exists()
        
        # Load and verify structure
        label_file = LabelFile(str(output_file))
        assert label_file.shapes == []
        assert label_file.imagePath == "image.png"  # Relative path
        assert label_file.imageData is not None


def test_export_with_shapes():
    """Test exporting with shapes."""
    window = MockWindow()
    
    # Create mock shapes
    shape1 = MockShape("ring1", [MockPoint(10, 20), MockPoint(30, 40)])
    shape2 = MockShape("ring2", [MockPoint(50, 60), MockPoint(70, 80)])
    
    # Create mock label list items
    item1 = MagicMock()
    item1.shape.return_value = shape1
    item2 = MagicMock()
    item2.shape.return_value = shape2
    
    window.labelList = [item1, item2]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Verify file was created
        assert output_file.exists()
        
        # Load and verify
        label_file = LabelFile(str(output_file))
        assert len(label_file.shapes) == 2
        assert label_file.shapes[0]["label"] == "ring1"
        assert label_file.shapes[1]["label"] == "ring2"


def test_export_preserves_preprocessing_info():
    """Test that preprocessing info is preserved in export."""
    window = MockWindow()
    window.otherData = {
        "preprocessing": {
            "crop_rect": (10, 20, 100, 150),
            "scale_factor": 0.8,
            "background_removed": False,
            "original_size": [200, 100],
            "processed_size": [160, 80],
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Load and verify preprocessing info is present
        label_file = LabelFile(str(output_file))
        assert "preprocessing" in label_file.otherData
        assert label_file.otherData["preprocessing"]["crop_rect"] == (10, 20, 100, 150)
        assert label_file.otherData["preprocessing"]["scale_factor"] == 0.8


def test_export_preserves_sample_metadata():
    """Test that sample metadata is preserved in export."""
    window = MockWindow()
    window.sample_metadata = {
        "harvested_year": 2023,
        "sample_code": "TEST001",
        "observation": "Test observation"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Load and verify sample metadata is present
        label_file = LabelFile(str(output_file))
        assert "sample_metadata" in label_file.otherData
        assert label_file.otherData["sample_metadata"]["harvested_year"] == 2023
        assert label_file.otherData["sample_metadata"]["sample_code"] == "TEST001"


def test_export_preserves_image_scale():
    """Test that image scale is preserved in export."""
    window = MockWindow()
    window.image_scale = {
        "value": 0.02,
        "unit": "mm"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Load and verify image scale is present
        label_file = LabelFile(str(output_file))
        assert "image_scale" in label_file.otherData
        assert label_file.otherData["image_scale"]["value"] == 0.02
        assert label_file.otherData["image_scale"]["unit"] == "mm"


def test_export_preserves_pith_xy():
    """Test that pith_xy is preserved in export when present."""
    window = MockWindow()
    window.pith_xy = (50, 75)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Load and verify pith_xy is present
        label_file = LabelFile(str(output_file))
        assert "pith_xy" in label_file.otherData
        assert label_file.otherData["pith_xy"] == (50, 75)


def test_export_respects_store_data_config():
    """Test that imageData inclusion respects store_data config."""
    window = MockWindow()
    window._config = {"store_data": False}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Load and verify imageData is None
        label_file = LabelFile(str(output_file), enforce_image_data=False)
        assert label_file.imageData is None


def test_export_empty_shapes_with_metadata():
    """Test exporting empty shapes but with preprocessing and metadata."""
    window = MockWindow()
    window.otherData = {
        "preprocessing": {
            "crop_rect": (10, 20, 100, 150),
            "scale_factor": 0.8,
            "background_removed": False,
            "original_size": [200, 100],
            "processed_size": [160, 80],
        }
    }
    window.sample_metadata = {
        "harvested_year": 2023,
        "sample_code": "TEST001",
        "observation": "Test"
    }
    window.image_scale = {"value": 0.02, "unit": "mm"}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Load and verify all metadata is present even with empty shapes
        label_file = LabelFile(str(output_file))
        assert label_file.shapes == []
        assert "preprocessing" in label_file.otherData
        assert "sample_metadata" in label_file.otherData
        assert "image_scale" in label_file.otherData


def test_export_with_flags():
    """Test exporting with image-level flags."""
    window = MockWindow()
    
    # Mock flag widget with flags
    flag_item1 = MagicMock()
    flag_item1.text.return_value = "verified"
    flag_item1.checkState.return_value = 2  # Qt.Checked
    
    flag_item2 = MagicMock()
    flag_item2.text.return_value = "reviewed"
    flag_item2.checkState.return_value = 0  # Qt.Unchecked
    
    window.flag_widget.count.return_value = 2
    window.flag_widget.item.side_effect = lambda i: [flag_item1, flag_item2][i]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.json"
        export_annotations_json(window, output_file)
        
        # Load and verify flags
        label_file = LabelFile(str(output_file))
        assert label_file.flags["verified"] is True
        assert label_file.flags["reviewed"] is False


def test_export_creates_output_directory():
    """Test that export creates output directory if it doesn't exist."""
    window = MockWindow()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "subdir" / "test.json"
        export_annotations_json(window, output_file)
        
        # Verify directory and file were created
        assert output_file.parent.exists()
        assert output_file.exists()

