"""Tests for the one-call core analysis pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from loguru import logger

from tras.cores import pipeline
from tras.cores.detect import CoreParams

IMAGE = np.zeros((32, 200, 3), dtype=np.uint8)
PATH = np.array([[0.0, 16.0], [199.0, 16.0]])


@pytest.fixture
def warnings_logged():
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING", format="{message}")
    yield messages
    logger.remove(sink)


def _run_with_boundaries(monkeypatch, n_boundaries: int, evidence: str) -> None:
    result = MagicMock(boundaries=tuple(range(n_boundaries)))
    monkeypatch.setattr(pipeline, "detect_core_rings", lambda *a, **k: result)
    params = CoreParams(evidence=evidence)
    pipeline.analyze_core_image(IMAGE, paths=[PATH], params=params)


# Regression: deep evidence finding 0 boundaries wrote empty CSVs without any hint.
@pytest.mark.parametrize("n_boundaries", [0, 1])
def test_empty_deep_core_warns_to_try_classical(monkeypatch, warnings_logged, n_boundaries):
    _run_with_boundaries(monkeypatch, n_boundaries, evidence="deep")
    assert len(warnings_logged) == 1
    assert "--evidence classical" in warnings_logged[0]


def test_no_warning_when_deep_core_has_rings(monkeypatch, warnings_logged):
    _run_with_boundaries(monkeypatch, 2, evidence="deep")
    assert warnings_logged == []


def test_no_classical_hint_for_classical_evidence(monkeypatch, warnings_logged):
    _run_with_boundaries(monkeypatch, 0, evidence="classical")
    assert warnings_logged == []
