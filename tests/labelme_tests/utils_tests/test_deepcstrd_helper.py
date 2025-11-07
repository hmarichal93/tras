import numpy as np
import pytest

from tras.utils import deepcstrd_helper


def test_detect_rings_deepcstrd_batch_passes_parameters(monkeypatch):
    images = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]
    centers = [(1.0, 2.0), (3.0, 4.0)]

    calls = []
    results_iter = iter([
        [np.array([[0.0, 0.0], [1.0, 1.0]])],
        [np.array([[2.0, 2.0], [3.0, 3.0]])],
    ])

    def patched_detect(image, center_xy, **kwargs):
        calls.append((image, center_xy, kwargs))
        return next(results_iter)

    monkeypatch.setattr(deepcstrd_helper, "detect_rings_deepcstrd", patched_detect)

    batch_results = deepcstrd_helper.detect_rings_deepcstrd_batch(
        images,
        centers,
        model_id="generic",
        tile_size=256,
        alpha=30,
        nr=180,
        total_rotations=3,
        prediction_map_threshold=0.4,
        width=1280,
        height=720,
        batch_size=4,
    )

    assert len(batch_results) == 2
    assert np.array_equal(batch_results[0][0], np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert np.array_equal(batch_results[1][0], np.array([[2.0, 2.0], [3.0, 3.0]]))

    assert len(calls) == 2
    assert calls[0][1] == centers[0]
    assert calls[0][2]["batch_size"] == 4
    assert calls[0][2]["model_id"] == "generic"
    assert calls[1][1] == centers[1]
    assert calls[1][2]["tile_size"] == 256
    assert calls[1][2]["batch_size"] == 4


def test_detect_rings_deepcstrd_batch_length_mismatch():
    images = [np.zeros((5, 5, 3), dtype=np.uint8)]
    centers = [(1.0, 1.0), (2.0, 2.0)]

    with pytest.raises(ValueError):
        deepcstrd_helper.detect_rings_deepcstrd_batch(images, centers)
