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


def _circle(radius: float, n: int = 180) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([100 + radius * np.cos(t), 100 + radius * np.sin(t)], 1).astype(np.float32)


def test_single_report_has_width_qc_page(tmp_path: Path):
    from tras.utils.report_pdf import _qc_text
    from tras.utils.report_pdf import generate_pdf_report
    from tras.utils.ring_properties import compute_ring_properties

    radii = [10, 20, 30, 40, 41, 50, 60, 70, 80]  # ring 5 is a 1 px false ring
    rings = [_circle(r) for r in radii]
    out = tmp_path / "report.pdf"
    generate_pdf_report(
        out,
        np.full((200, 200, 3), 200, dtype=np.uint8),
        (100.0, 100.0),
        rings,
        compute_ring_properties(rings, (100.0, 100.0)),
        0.1,
        "mm",
        sample_code="S1",
        harvested_year=2020,
    )
    pages = len(re.findall(rb"/Type\s*/Page\b", out.read_bytes()))
    assert pages == 5  # cover, 2 overlays, analysis plots, widths & QC

    from tras.analysis.qc import QCFlag

    text = _qc_text([QCFlag(5, "narrow", "ring_5 is unusually narrow")])
    assert "ring_5 is unusually narrow" in text
    assert _qc_text([]).endswith("no ring flagged.")
