from typing import Tuple
import numpy as np
import cv2

from labelme.tree_ring_methods.apd.automatic_wood_pith_detector import apd

def detect_pith_apd(image: np.ndarray, method: str = "apd") -> Tuple[float, float]:
    """
    Run APD (Automatic Pith Detection) on the given image.
    Returns (x, y) coordinates of the detected pith center.
    
    Args:
        image: Input image as numpy array
        method: Method to use ("apd" or "apd_pcl")
    
    Returns:
        Tuple of (x, y) coordinates of pith center
    """
    # Ensure RGB format
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (255 * (image.astype(np.float32) / image.max())).astype(np.uint8)
    
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
    
    # peak is (y, x) in APD
    y, x = peak
    return float(x), float(y)
