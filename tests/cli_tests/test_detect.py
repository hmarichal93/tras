"""Tests for unified tras_detect CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_detect_routes_file_correctly(tmp_path):
    """Test that file path routes to single-image processing."""
    from tras.cli.detect import _process_single_image

    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake image data")

    with patch("tras.cli.detect.detect_api") as mock_detect:
        mock_result = MagicMock()
        mock_result.output_path = tmp_path / "test_detected.json"
        mock_result.pith_xy = (100.0, 200.0)
        mock_result.rings = []
        mock_detect.return_value = mock_result

        _process_single_image(
            image_path=image_file,
            output=None,
            pith_x=None,
            pith_y=None,
            auto_pith=True,
            pith_method="apd_dl",
            ring_method="deepcstrd",
            scale=None,
            remove_background=False,
            sampling_nr=360,
            cstrd_sigma=3.0,
            cstrd_th_low=5.0,
            cstrd_th_high=20.0,
            cstrd_alpha=30,
            cstrd_nr=360,
            deepcstrd_model="generic",
            deepcstrd_tile_size=0,
            deepcstrd_alpha=45,
            deepcstrd_nr=360,
            deepcstrd_rotations=5,
            deepcstrd_threshold=0.5,
            inbd_model="INBD_EH",
            inbd_auto_pith=True,
        )

        mock_detect.assert_called_once()
        call_kwargs = mock_detect.call_args[1]
        assert call_kwargs["image_path"] == image_file
        assert call_kwargs["ring_method"] == "deepcstrd"


def test_detect_routes_directory_correctly(tmp_path):
    """Test that directory path routes to batch processing."""
    from tras.cli.detect import _process_folder

    # Create test images
    (tmp_path / "image1.jpg").write_bytes(b"fake")
    (tmp_path / "image2.png").write_bytes(b"fake")
    (tmp_path / "image3.jpeg").write_bytes(b"fake")
    (tmp_path / "not_image.txt").write_text("not an image")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("tras.cli.detect.detect_api") as mock_detect, patch(
        "tras.cli.detect.compute_ring_properties"
    ) as mock_props, patch("tras.cli.detect.write_ring_properties_csv") as mock_csv, patch(
        "tras.cli.detect.generate_pdf_report"
    ) as mock_pdf:
        mock_result = MagicMock()
        mock_result.rings = [MagicMock()]  # Non-empty rings
        mock_result.pith_xy = (100.0, 200.0)
        mock_result.image = b"fake image"
        mock_detect.return_value = mock_result
        mock_props.return_value = [{"label": "ring_1", "area": 100.0}]

        _process_folder(
            input_dir=tmp_path,
            output_dir=output_dir,
            config=None,
            scale_value=0.0213,
            scale_unit="mm",
            preprocess_resize_scale=None,
            preprocess_remove_background=None,
            postprocess_sampling_nr=None,
            pith_auto=None,
            pith_method=None,
            ring_method=None,
            cstrd_sigma=None,
            cstrd_th_low=None,
            cstrd_th_high=None,
            cstrd_alpha=None,
            cstrd_nr=None,
            deepcstrd_model=None,
            deepcstrd_tile_size=None,
            deepcstrd_alpha=None,
            deepcstrd_nr=None,
            deepcstrd_rotations=None,
            deepcstrd_threshold=None,
            inbd_model=None,
            inbd_auto_pith=None,
        )

        # Should process 3 images (jpg, png, jpeg) but not txt
        assert mock_detect.call_count == 3


def test_image_extension_filtering(tmp_path):
    """Test that only JPEG/PNG images are processed."""
    from tras.cli.detect import _process_folder

    # Create various files
    (tmp_path / "image1.jpg").write_bytes(b"fake")
    (tmp_path / "image2.JPG").write_bytes(b"fake")  # Uppercase
    (tmp_path / "image3.png").write_bytes(b"fake")
    (tmp_path / "image4.jpeg").write_bytes(b"fake")
    (tmp_path / "image5.JPEG").write_bytes(b"fake")  # Uppercase
    (tmp_path / "image6.gif").write_bytes(b"fake")  # Not supported
    (tmp_path / "image7.tiff").write_bytes(b"fake")  # Not supported
    (tmp_path / "readme.txt").write_text("text file")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    processed_files = []

    def capture_detect_call(*args, **kwargs):
        processed_files.append(kwargs.get("image_path"))

    with patch("tras.cli.detect.detect_api", side_effect=capture_detect_call), patch(
        "tras.cli.detect.compute_ring_properties"
    ) as mock_props, patch("tras.cli.detect.write_ring_properties_csv"), patch(
        "tras.cli.detect.generate_pdf_report"
    ):
        mock_props.return_value = [{"label": "ring_1", "area": 100.0}]

        # This will fail validation, but we can check which files were attempted
        try:
            _process_folder(
                input_dir=tmp_path,
                output_dir=output_dir,
                config=None,
                scale_value=0.0213,
                scale_unit="mm",
                preprocess_resize_scale=None,
                preprocess_remove_background=None,
                postprocess_sampling_nr=None,
                pith_auto=None,
                pith_method=None,
                ring_method=None,
                cstrd_sigma=None,
                cstrd_th_low=None,
                cstrd_th_high=None,
                cstrd_alpha=None,
                cstrd_nr=None,
                deepcstrd_model=None,
                deepcstrd_tile_size=None,
                deepcstrd_alpha=None,
                deepcstrd_nr=None,
                deepcstrd_rotations=None,
                deepcstrd_threshold=None,
                inbd_model=None,
                inbd_auto_pith=None,
            )
        except Exception:
            pass  # Expected to fail due to missing config, but we check files first

    # Check that only JPEG/PNG files were processed (case-insensitive)
    processed_names = {f.name for f in processed_files if f}
    assert "image1.jpg" in processed_names
    assert "image2.JPG" in processed_names
    assert "image3.png" in processed_names
    assert "image4.jpeg" in processed_names
    assert "image5.JPEG" in processed_names
    assert "image6.gif" not in processed_names
    assert "image7.tiff" not in processed_names
    assert "readme.txt" not in processed_names

