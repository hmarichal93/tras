"""Tests for the batch summary PDF generator."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tras.utils.report_pdf import generate_summary_pdf


def _fake_result(name: str) -> dict:
    return {
        "name": name,
        "image": np.zeros((16, 16, 3), dtype=np.uint8),
        "pith_xy": (8.0, 8.0),
        "rings": [np.array([[2, 2], [12, 2], [12, 12], [2, 12]], dtype=np.float32)],
    }


def test_generate_summary_pdf_one_page_per_image(tmp_path: Path):
    results = [_fake_result("a.png"), _fake_result("b.png")]
    out = tmp_path / "summary.pdf"

    generate_summary_pdf(out, results)

    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    # One "/Type /Page" object per page ("/Type /Pages" excluded via word boundary).
    page_count = len(re.findall(rb"/Type\s*/Page\b", data))
    assert page_count == 2
