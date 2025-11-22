"""Tests for tras_export_json CLI command."""

import json
from pathlib import Path

from tras.cli.export_json import main


def test_export_json_with_valid_labels(tmp_path: Path):
    """Test export_json with valid labels."""
    # Create a test JSON file with valid labels
    test_json = tmp_path / "test.json"
    
    # Create JSON directly
    json_data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "ring_1",
                "points": [[10, 20], [30, 40], [50, 60]],
                "shape_type": "polygon",
            },
            {
                "label": "ring_2",
                "points": [[70, 80], [90, 100]],
                "shape_type": "polygon",
            },
        ],
        "imagePath": "test.jpg",
        "imageData": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 PNG
        "imageHeight": 1,
        "imageWidth": 1,
    }
    
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    # Run export
    import sys
    from io import StringIO

    old_argv = sys.argv
    old_stdout = sys.stdout
    try:
        sys.argv = ["tras_export_json", str(test_json)]
        sys.stdout = StringIO()
        main()
        output = sys.stdout.getvalue()
        assert "Saved to:" in output
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout

    # Check output directory was created
    output_dir = tmp_path / "test"
    assert output_dir.exists()
    assert (output_dir / "label_names.txt").exists()


def test_export_json_with_none_label(tmp_path: Path):
    """Test export_json handles None labels gracefully."""
    # Create a test JSON file with None label
    test_json = tmp_path / "test.json"
    
    # Create JSON directly with None label (simulating corrupted data)
    json_data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": None,  # None label
                "points": [[10, 20], [30, 40]],
                "shape_type": "polygon",
            }
        ],
        "imagePath": "test.jpg",
        "imageData": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 PNG
        "imageHeight": 1,
        "imageWidth": 1,
    }
    
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    # Run export - should not crash
    import sys
    from io import StringIO

    old_argv = sys.argv
    old_stdout = sys.stdout
    try:
        sys.argv = ["tras_export_json", str(test_json)]
        sys.stdout = StringIO()
        main()
        output = sys.stdout.getvalue()
        assert "Saved to:" in output
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout

    # Check output directory was created
    output_dir = tmp_path / "test"
    assert output_dir.exists()


def test_export_json_with_empty_label(tmp_path: Path):
    """Test export_json handles empty labels gracefully."""
    test_json = tmp_path / "test.json"
    
    json_data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "",  # Empty label
                "points": [[10, 20], [30, 40]],
                "shape_type": "polygon",
            }
        ],
        "imagePath": "test.jpg",
        "imageData": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "imageHeight": 1,
        "imageWidth": 1,
    }
    
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    # Run export - should not crash
    import sys
    from io import StringIO

    old_argv = sys.argv
    old_stdout = sys.stdout
    try:
        sys.argv = ["tras_export_json", str(test_json)]
        sys.stdout = StringIO()
        main()
        output = sys.stdout.getvalue()
        assert "Saved to:" in output
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout

