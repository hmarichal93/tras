"""Tests for the unified tras_detect CLI command."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tras.cli.detect import _config_overrides
from tras.cli.detect import _process_folder
from tras.cli.detect import _process_single_image
from tras.cli.detect import app
from tras.config.detection_defaults import get_detection_defaults
from tras.params import DetectionParams


def _mock_detection(tmp_path: Path) -> MagicMock:
    result = MagicMock()
    result.output_path = tmp_path / "test_detected.json"
    result.pith_xy = (100.0, 200.0)
    result.rings = []
    return result


def test_single_image_passes_params_to_api(tmp_path):
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake image data")
    params = DetectionParams(ring_method="cstrd")

    with patch("tras.cli.detect.detect_api") as mock_detect:
        mock_detect.return_value = _mock_detection(tmp_path)
        _process_single_image(image_file, None, params)

    call_kwargs = mock_detect.call_args.kwargs
    assert call_kwargs["image_path"] == image_file
    assert call_kwargs["params"].ring_method == "cstrd"
    assert call_kwargs["output"] == tmp_path / "test_detected.json"


def test_single_image_cli_uses_yaml_defaults(tmp_path):
    """Regression: the CLI used to hardcode DeepCS-TRD defaults that disagreed with the YAML."""
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake")
    defaults = get_detection_defaults()

    with patch("tras.cli.detect.detect_api") as mock_detect:
        mock_detect.return_value = _mock_detection(tmp_path)
        result = CliRunner().invoke(app, [str(image_file)])

    assert result.exit_code == 0, result.output
    params = mock_detect.call_args.kwargs["params"]
    assert params.deepcstrd.alpha == defaults["deepcstrd"]["alpha"]
    assert params.deepcstrd.rotations == defaults["deepcstrd"]["rotations"]
    assert params.deepcstrd.threshold == defaults["deepcstrd"]["threshold"]
    assert params.ring_method == defaults["method"]


def test_single_image_cli_flags_override_defaults(tmp_path):
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake")

    with patch("tras.cli.detect.detect_api") as mock_detect:
        mock_detect.return_value = _mock_detection(tmp_path)
        result = CliRunner().invoke(
            app,
            [
                str(image_file),
                "--ring-method",
                "cstrd",
                "--cstrd-sigma",
                "2.5",
                "--pith-x",
                "10",
                "--pith-y",
                "20",
            ],
        )

    assert result.exit_code == 0, result.output
    params = mock_detect.call_args.kwargs["params"]
    assert params.ring_method == "cstrd"
    assert params.cstrd.sigma == 2.5
    assert params.pith.manual_xy == (10.0, 20.0)


def test_invalid_ring_method_exits_with_error(tmp_path):
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake")
    result = CliRunner().invoke(app, [str(image_file), "--ring-method", "bogus"])
    assert result.exit_code == 1


def test_config_overrides_translate_flag_names():
    overrides = _config_overrides(
        {"scale": 0.5, "auto_pith": False, "cstrd_sigma": 2.0, "ring_method": None}
    )
    assert overrides == {
        "preprocess_resize_scale": 0.5,
        "pith_auto": False,
        "cstrd_sigma": 2.0,
    }


def test_folder_mode_routes_to_run_batch(tmp_path):
    with patch("tras.cli.detect.run_batch") as mock_batch:
        mock_batch.return_value = MagicMock(total=3, processed=3, skipped=0, errors=0)
        _process_folder(
            tmp_path,
            tmp_path / "out",
            None,
            {"scale_value": 0.0213, "scale_unit": "mm", "ring_method": "cstrd"},
        )

    args, kwargs = mock_batch.call_args
    assert args[0] == tmp_path
    assert args[1] == tmp_path / "out"
    config = args[2]
    assert config["physical_scale"] == {"value": 0.0213, "unit": "mm"}
    assert config["rings"]["method"] == "cstrd"
    assert kwargs["per_image_reports"] is True


def test_folder_mode_processes_supported_images(tmp_path, monkeypatch):
    """Folder mode runs detection on every supported image (case-insensitive extensions)."""
    from tras.utils import batch_helper

    for name in ("a.jpg", "b.JPG", "c.png", "d.tif", "e.gif", "readme.txt"):
        (tmp_path / name).write_bytes(b"fake")

    seen: list[str] = []

    def fake_detect(image_path, output=None, params=None):
        seen.append(Path(image_path).name)
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(batch_helper, "detect", fake_detect)
    _process_folder(tmp_path, tmp_path / "out", None, {})

    assert sorted(seen) == ["a.jpg", "b.JPG", "c.png", "d.tif"]


def test_outer_year_flag_reaches_batch_config(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"fake")
    with patch("tras.cli.detect.run_batch") as mock_batch:
        mock_batch.return_value = MagicMock(total=1, processed=1, skipped=0, errors=0)
        result = CliRunner().invoke(
            app,
            [str(tmp_path), "--outer-year", "2021", "--scale-value", "0.02", "--scale-unit", "mm"],
        )
    assert result.exit_code == 0, result.output
    config = mock_batch.call_args.args[2]
    assert config["dating"] == {"outer_year": 2021}


# Regression: --pith-x/--pith-y were silently ignored in folder mode.
def test_pith_flags_are_rejected_in_folder_mode(tmp_path, monkeypatch):
    from tras.utils import batch_helper

    (tmp_path / "a.jpg").write_bytes(b"fake")
    monkeypatch.setattr(batch_helper, "detect", lambda *a, **k: pytest.fail("detect ran"))
    result = CliRunner().invoke(app, [str(tmp_path), "--pith-x", "10", "--pith-y", "20"])
    assert result.exit_code == 1
    assert not (tmp_path / "tras_out").exists()


# Regression: a misspelled key in --config (rings.deepcstrd.treshold) was silently dropped.
def test_unknown_config_key_exits_with_error_in_folder_mode(tmp_path, monkeypatch):
    from tras.utils import batch_helper

    (tmp_path / "a.jpg").write_bytes(b"fake")
    config = tmp_path / "config.yml"
    config.write_text("rings:\n  deepcstrd:\n    treshold: 0.3\n", encoding="utf-8")
    monkeypatch.setattr(batch_helper, "detect", lambda *a, **k: pytest.fail("detect ran"))
    result = CliRunner().invoke(app, [str(tmp_path), "--config", str(config)])
    assert result.exit_code == 1
