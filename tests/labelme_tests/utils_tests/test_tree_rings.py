import numpy as np

from tras._automation.tree_rings import RingDetectParams, detect_tree_rings


def _synthetic_ring_image(h: int = 400, w: int = 400, cx: float = 200.0, cy: float = 200.0) -> np.ndarray:
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # Create alternating bright/dark rings using a sine; normalize to [0,1]
    img = (np.sin(r / 6.0) * 0.5 + 0.5).astype(np.float32)
    img = (img * 255).astype(np.uint8)
    return np.dstack([img, img, img])


def test_detect_tree_rings_on_synthetic_image():
    img = _synthetic_ring_image()
    params = RingDetectParams(
        angular_steps=360,
        min_radius=5.0,
        relative_threshold=0.25,
        min_peak_distance=3,
        min_coverage=0.5,
        max_rings=10,
    )
    rings = detect_tree_rings(img, center_xy=(200.0, 200.0), params=params)
    # Expect some rings to be detected on the synthetic pattern
    assert len(rings) >= 3
    # Each ring should provide a polyline with one point per angle step
    for ring in rings:
        assert ring.shape[1] == 2
        assert ring.shape[0] == params.angular_steps
