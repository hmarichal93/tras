import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path

# Add tree ring methods to path
_deepcstrd_path = Path(__file__).parent.parent / "tree_ring_methods" / "deepcstrd"
_cstrd_path = Path(__file__).parent.parent / "tree_ring_methods" / "cstrd"
sys.path.insert(0, str(_deepcstrd_path))
sys.path.insert(0, str(_cstrd_path))

from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection

def detect_rings_deepcstrd(
    image: np.ndarray, 
    center_xy: Tuple[float, float],
    model_id: str = "generic",
    tile_size: int = 0,
    alpha: int = 45,
    nr: int = 360,
    total_rotations: int = 5,
    prediction_map_threshold: float = 0.5
) -> List[np.ndarray]:
    """
    Run DeepCS-TRD on the given image and return a list of ring polylines (Nx2 arrays).
    
    Args:
        image: Input image as numpy array (H x W x 3)
        center_xy: Pith center coordinates (x, y)
        model_id: Model identifier ("generic", "pinus_v1", "pinus_v2", "gleditsia", "salix")
        tile_size: Tile size for processing (0 for no tiling, 256 for tiling)
        alpha: Alpha parameter for angular sampling
        nr: Number of radial samples (360 = 1 degree resolution)
        total_rotations: Number of rotations for test-time augmentation
        prediction_map_threshold: Threshold for binary prediction map
    
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
    
    # Get model path
    model_path = _get_model_path(model_id, tile_size)
    
    cx, cy = center_xy
    
    # Run DeepCS-TRD
    result = DeepTreeRingDetection(
        im_in=image,
        cy=int(cy),
        cx=int(cx),
        height=0,
        width=0,
        alpha=alpha,
        nr=nr,
        mc=2,
        weights_path=model_path,
        total_rotations=total_rotations,
        debug_image_input_path=None,
        debug_output_dir=None,
        tile_size=tile_size,
        prediction_map_threshold=prediction_map_threshold
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

def _get_model_path(model_id: str, tile_size: int = 0) -> str:
    """Get path to DeepCS-TRD model weights."""
    base_path = Path(__file__).parent.parent / "tree_ring_methods" / "deepcstrd" / "models" / "deep_cstrd"
    
    # Normalize tile size
    tile_size = 0 if tile_size not in [0, 256] else tile_size
    
    model_map = {
        "pinus_v1": f"{tile_size}_pinus_v1_1504.pth",
        "pinus_v2": f"{tile_size}_pinus_v2_1504.pth",
        "gleditsia": f"{tile_size}_gleditsia_1504.pth",
        "salix": f"{tile_size}_salix_1504.pth",
        "generic": "0_all_1504.pth",
    }
    
    if model_id not in model_map:
        model_id = "generic"
    
    return str(base_path / model_map[model_id])
