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
    
    # Check if pith is too close to edges and add padding if needed
    import cv2
    h, w = image.shape[:2]
    min_margin = 100  # DeepCS-TRD needs margin for edge detection
    
    pad_top = max(0, min_margin - int(cy))
    pad_bottom = max(0, min_margin - (h - int(cy)))
    pad_left = max(0, min_margin - int(cx))
    pad_right = max(0, min_margin - (w - int(cx)))
    
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        # Add padding with white color (background)
        image = cv2.copyMakeBorder(
            image, 
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)  # White padding
        )
        # Adjust pith coordinates
        cx += pad_left
        cy += pad_top
        print(f"DeepCS-TRD: Added padding - top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        print(f"DeepCS-TRD: Adjusted pith: ({cx:.1f}, {cy:.1f})")
    
    # Create temporary output directory for DeepCS-TRD
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="deepcstrd_")
    
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
        debug_output_dir=temp_dir,
        tile_size=tile_size,
        prediction_map_threshold=prediction_map_threshold
    )
    
    # Extract rings from result
    # DeepTreeRingDetection returns a tuple: (im_seg, im_pre, ch_e, ch_f, ch_s, ch_c, ch_p, rings_dict)
    rings = []
    
    if result is not None and isinstance(result, tuple) and len(result) >= 8:
        # Extract the rings dictionary (last element of tuple)
        rings_dict = result[7]
        
        if isinstance(rings_dict, dict) and 'shapes' in rings_dict:
            for shape in rings_dict['shapes']:
                if 'points' in shape and isinstance(shape['points'], list):
                    pts = np.array(shape['points'], dtype=np.float32)
                    if len(pts.shape) == 2 and pts.shape[1] == 2:
                        # Adjust coordinates back if padding was added
                        if any([pad_top, pad_bottom, pad_left, pad_right]):
                            pts[:, 0] -= pad_left  # x coordinates
                            pts[:, 1] -= pad_top   # y coordinates
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
