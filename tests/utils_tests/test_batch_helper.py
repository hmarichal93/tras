"""Tests for the Qt-free batch processing runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tras.api import DetectionResult
from tras.api import _save_detection_results
from tras.utils import batch_helper


def _write_png(path: Path) -> None:
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)


def _fake_result(output: Path | None) -> DetectionResult:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    rings = [np.array([[2, 2], [12, 2], [12, 12], [2, 12]], dtype=np.float32)]
    pith = (8.0, 8.0)
    if output is not None:
        _save_detection_results(Path(output), image, pith, rings)
    return DetectionResult(
        image=image, pith_xy=pith, rings=rings, preprocessing=None, output_path=output
    )


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "imgs"
    folder.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        _write_png(folder / name)
    return folder


CONFIG = {"rings": {"method": "deepcstrd"}}


def test_run_batch_writes_outputs(input_dir: Path, tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_detect(image_path, output=None, **kwargs):
        calls.append(Path(image_path).name)
        return _fake_result(output)

    monkeypatch.setattr(batch_helper, "detect", fake_detect)
    out = tmp_path / "out"

    summary = batch_helper.run_batch(input_dir, out, CONFIG)

    assert sorted(calls) == ["a.png", "b.png", "c.png"]
    for name in ("a", "b", "c"):
        assert (out / f"{name}.json").exists()
    assert (out / "summary.pdf").exists()
    assert (out / "batch_config.yml").exists()
    assert summary.processed == 3
    assert summary.skipped == 0
    assert summary.errors == 0


def test_run_batch_skips_existing_json(input_dir: Path, tmp_path: Path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    # Pre-create b.json so it should be skipped (not re-detected) but still summarized.
    _fake_result(out / "b.json")

    calls: list[str] = []

    def fake_detect(image_path, output=None, **kwargs):
        calls.append(Path(image_path).name)
        return _fake_result(output)

    monkeypatch.setattr(batch_helper, "detect", fake_detect)

    summary = batch_helper.run_batch(input_dir, out, CONFIG)

    assert calls == ["a.png", "c.png"]  # b.png skipped
    assert summary.processed == 2
    assert summary.skipped == 1
    assert summary.errors == 0
    assert (out / "summary.pdf").exists()


def test_run_batch_continues_after_error(input_dir: Path, tmp_path: Path, monkeypatch):
    def fake_detect(image_path, output=None, **kwargs):
        if Path(image_path).name == "b.png":
            raise RuntimeError("boom")
        return _fake_result(output)

    monkeypatch.setattr(batch_helper, "detect", fake_detect)
    out = tmp_path / "out"

    summary = batch_helper.run_batch(input_dir, out, CONFIG)

    assert summary.processed == 2
    assert summary.errors == 1
    assert summary.error_names == ["b.png"]
    assert (out / "a.json").exists()
    assert (out / "c.json").exists()
    assert not (out / "b.json").exists()
    assert (out / "summary.pdf").exists()


def test_find_images_filters_and_sorts(tmp_path: Path):
    for name in ("z.png", "a.JPG", "note.txt", "b.tiff"):
        (tmp_path / name).write_bytes(b"x")
    names = [p.name for p in batch_helper.find_images(tmp_path)]
    assert names == ["a.JPG", "b.tiff", "z.png"]


def _fake_concentric(output: Path | None) -> DetectionResult:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    rings = [
        np.stack([60 + r * np.cos(t), 60 + r * np.sin(t)], 1).astype(np.float32)
        for r in (10, 25, 40)
    ]
    if output is not None:
        _save_detection_results(Path(output), image, (60.0, 60.0), rings)
    return DetectionResult(
        image=image, pith_xy=(60.0, 60.0), rings=rings, preprocessing=None, output_path=output
    )


def test_run_batch_writes_radii_rwl_when_dated(input_dir: Path, tmp_path: Path, monkeypatch):
    from tras.io.tucson import read_rwl

    monkeypatch.setattr(
        batch_helper, "detect", lambda image_path, output=None, **kw: _fake_concentric(output)
    )
    config = {
        "rings": {"method": "deepcstrd"},
        "physical_scale": {"value": 0.5, "unit": "mm"},
        "dating": {"outer_year": 2020},
    }
    summary = batch_helper.run_batch(input_dir, tmp_path / "out", config)

    assert summary.rwl_path == tmp_path / "out" / "radii.rwl"
    series = read_rwl(summary.rwl_path)
    assert len(series) == 3 * 4  # three images x N/E/S/W
    assert all(s.first_year == 2018 and s.last_year == 2020 for s in series)
    assert series[0].widths_mm == pytest.approx((5.0, 7.5, 7.5), abs=0.01)


def test_run_batch_skips_rwl_without_scale(input_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        batch_helper, "detect", lambda image_path, output=None, **kw: _fake_concentric(output)
    )
    config = {"rings": {"method": "deepcstrd"}, "dating": {"outer_year": 2020}}
    assert batch_helper.run_batch(input_dir, tmp_path / "out", config).rwl_path is None


def test_radii_rwl_with_colliding_names_and_resize(tmp_path: Path):
    """Regression: names sharing 7 characters crashed run_batch; resize ignored the scale."""
    from tras.io.tucson import read_rwl

    results = [
        {"name": n, "image": None, "pith_xy": (60.0, 60.0), "rings": _fake_concentric(None).rings}
        for n in ("disc_001.jpg", "disc_002.jpg")
    ]
    config = {
        "physical_scale": {"value": 0.5, "unit": "mm"},
        "preprocess": {"resize_scale": 0.5},
        "dating": {"outer_year": 2020},
    }
    path = batch_helper.write_radii_rwl(tmp_path / "radii.rwl", results, config)

    series = read_rwl(path)
    assert len({s.series_id for s in series}) == 8
    assert (tmp_path / "radii_ids.csv").read_text().splitlines()[1] == "R0001,disc_001"
    # 0.5 mm per original px, image halved: 1 mm per processed px; first ring 10 px.
    assert series[0].widths_mm[0] == pytest.approx(10.0, abs=0.05)


# Regression: --pith-x/--pith-y (pith.x/y) were silently ignored in folder mode. One pixel
# position cannot be the pith of every image, so a batch now refuses it up front.
def test_run_batch_rejects_manual_pith(input_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(batch_helper, "detect", lambda *a, **k: pytest.fail("detect ran"))
    out = tmp_path / "out"
    config = {**CONFIG, "pith": {"x": 8.0, "y": 8.0}}

    with pytest.raises(ValueError, match="manual pith"):
        batch_helper.run_batch(input_dir, out, config)
    assert not out.exists()


# Regression: unknown YAML keys were silently ignored by the batch runner.
def test_run_batch_rejects_unknown_config_key(input_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(batch_helper, "detect", lambda *a, **k: pytest.fail("detect ran"))
    config = {"rings": {"method": "deepcstrd", "deepcstrd": {"treshold": 0.3}}}

    with pytest.raises(ValueError, match="treshold"):
        batch_helper.run_batch(input_dir, tmp_path / "out", config)
