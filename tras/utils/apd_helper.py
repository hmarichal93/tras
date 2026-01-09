from typing import Tuple
from pathlib import Path
import tempfile
import numpy as np
import cv2

from tras.tree_ring_methods.apd.automatic_wood_pith_detector import apd, apd_dl


def _get_apd_yolo_weights_path() -> Path:
    """Get path to APD YOLO weights file."""
    repo_root = Path(__file__).parent.parent.parent
    
    # Primary location: downloaded_assets/apd/yolo/all_best_yolov8.pt
    primary_path = repo_root / "downloaded_assets" / "apd" / "yolo" / "all_best_yolov8.pt"
    if primary_path.exists():
        return primary_path
    
    # Fallback location: checkpoints/yolo/all_best_yolov8.pt (for compatibility)
    fallback_path = repo_root / "checkpoints" / "yolo" / "all_best_yolov8.pt"
    if fallback_path.exists():
        return fallback_path
    
    # Model not found - provide detailed setup instructions
    raise FileNotFoundError(
        f"APD YOLO weights not found.\n\n"
        f"Searched in:\n"
        f"  - {primary_path}\n"
        f"  - {fallback_path}\n\n"
        f"SETUP REQUIRED:\n"
        f"Run `python tools/download_release_assets.py` to download the YOLO weights."
    )


def detect_pith_apd(image: np.ndarray, method: str = "apd_dl") -> Tuple[float, float]:
    """
    Run APD (Automatic Pith Detection) on the given image.
    Returns (x, y) coordinates of the detected pith center.
    
    Args:
        image: Input image as numpy array
        method: Method to use ("apd", "apd_pcl", or "apd_dl")
    
    Returns:
        Tuple of (x, y) coordinates of pith center
    """
    # Validate method
    if method not in ["apd", "apd_pcl", "apd_dl"]:
        raise ValueError(f"Unknown APD method: {method}. Must be one of: 'apd', 'apd_pcl', 'apd_dl'")
    
    # Ensure RGB format
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (255 * (image.astype(np.float32) / image.max())).astype(np.uint8)
    
    # Handle deep learning method
    if method == "apd_dl":
        weights_path = _get_apd_yolo_weights_path()
        # Create temporary output directory for YOLO predictions
        with tempfile.TemporaryDirectory(prefix="apd_dl_") as temp_dir:
            output_dir = Path(temp_dir)
            peak = apd_dl(image, output_dir, str(weights_path))
            x, y = peak
            return float(x), float(y)
    
    # Handle classical methods (apd, apd_pcl)
    # APD parameters
    st_sigma = 1.2
    st_window = 3
    lo_w = 3
    percent_lo = 0.7
    max_iter = 11
    rf = 7
    epsilon = 1e-3
    
    # Run APD
    peak = apd(image, st_sigma, st_window, lo_w, percent_lo, max_iter, rf, epsilon, 
               pclines=(method == "apd_pcl"), debug=False, output_dir=None)
    
    # peak is (x, y) where x is horizontal (column) and y is vertical (row)
    x, y = peak
    return float(x), float(y)
