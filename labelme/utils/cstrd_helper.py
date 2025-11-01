import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path

# Add tree ring methods to path
_tree_ring_path = Path(__file__).parent.parent / "tree_ring_methods" / "cstrd"
sys.path.insert(0, str(_tree_ring_path))

from cross_section_tree_ring_detection.cross_section_tree_ring_detection import TreeRingDetection

def detect_rings_cstrd(
    image: np.ndarray,
    center_xy: Tuple[float, float],
    sigma: float = 3.0,
    th_low: float = 5.0,
    th_high: float = 20.0,
    alpha: int = 30,
    nr: int = 360,
    min_chain_length: int = 2
) -> List[np.ndarray]:
    """
    Run CS-TRD (Classical tree ring detection) on the given image.
    
    Args:
        image: Input image as numpy array (H x W x 3)
        center_xy: Pith center coordinates (x, y)
        sigma: Gaussian smoothing sigma
        th_low: Low threshold for Canny edge detection
        th_high: High threshold for Canny edge detection
        alpha: Alpha parameter for angular sampling
        nr: Number of radial samples (360 = 1 degree resolution)
        min_chain_length: Minimum chain length to keep
    
    Returns:
        List of ring polylines, each as Nx2 array of (x, y) points
    """
    # Ensure RGB format
    if image.ndim == 2:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (255 * (image.astype(np.float32) / image.max())).astype(np.uint8)
    
    cx, cy = center_xy
    
    # Run CS-TRD
    result = TreeRingDetection(
        im_in=image,
        cy=int(cy),
        cx=int(cx),
        sigma=sigma,
        th_low=th_low,
        th_high=th_high,
        hsize=0,
        wsize=0,
        alpha=alpha,
        nr=nr,
        min_chain_length=min_chain_length,
        debug=False,
        im_path=None,
        output_dir=None
    )
    
    # Extract rings from result
    rings = []
    if result and 'rings' in result:
        for ring_data in result['rings']:
            if isinstance(ring_data, np.ndarray) and ring_data.shape[1] == 2:
                rings.append(ring_data.astype(np.float32))
            elif 'points' in ring_data:
                pts = np.array(ring_data['points'], dtype=np.float32)
                if pts.shape[1] == 2:
                    rings.append(pts)
    
    return rings

