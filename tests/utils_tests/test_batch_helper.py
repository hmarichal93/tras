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
