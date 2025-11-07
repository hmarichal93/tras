import numpy as np
from typing import List, Sequence, Tuple
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
    prediction_map_threshold: float = 0.5,
    width: int = 0,
    height: int = 0,
    batch_size: int = 1,
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
        width: Resize width (0 = no resize) (default: 0)
        height: Resize height (0 = no resize) (default: 0)
        batch_size: Number of tiles/images to process simultaneously inside the
            neural network forward pass. Increasing this value can improve GPU
            utilization when tiling is enabled. (default: 1)
    
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
        height=height,
        width=width,
        alpha=alpha,
        nr=nr,
        mc=2,
        weights_path=model_path,
        total_rotations=total_rotations,
        debug_image_input_path=None,
        debug_output_dir=temp_dir,
        tile_size=tile_size,
        prediction_map_threshold=prediction_map_threshold,
        batch_size=batch_size
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


def detect_rings_deepcstrd_batch(
    images: Sequence[np.ndarray],
    centers_xy: Sequence[Tuple[float, float]],
    *,
    model_id: str = "generic",
    tile_size: int = 0,
    alpha: int = 45,
    nr: int = 360,
    total_rotations: int = 5,
    prediction_map_threshold: float = 0.5,
    width: int = 0,
    height: int = 0,
    batch_size: int = 1,
) -> List[List[np.ndarray]]:
    """Run DeepCS-TRD on multiple images.

    Args:
        images: Sequence of RGB images.
        centers_xy: Sequence of (x, y) tuples with the pith location for each
            image. Must match ``images`` in length and order.
        model_id: Model identifier shared across the batch. (default: "generic")
        tile_size: Tile size for processing (0 for no tiling). (default: 0)
        alpha: Alpha parameter for angular sampling. (default: 45)
        nr: Number of radial samples. (default: 360)
        total_rotations: Number of rotations for test-time augmentation.
            (default: 5)
        prediction_map_threshold: Threshold for the binary prediction map.
            (default: 0.5)
        width: Resize width for preprocessing. (default: 0)
        height: Resize height for preprocessing. (default: 0)
        batch_size: Number of tiles/images processed together inside the neural
            network forward pass. Passed directly to
            :func:`detect_rings_deepcstrd`. (default: 1)

    Returns:
        A list with one entry per image. Each entry contains the list of rings
        detected for that image. When an image produces no detections, the
        corresponding entry is an empty list.

    Raises:
        ValueError: If ``images`` and ``centers_xy`` have different lengths.
    """

    if len(images) != len(centers_xy):
        raise ValueError("images and centers_xy must have the same length")

    results: List[List[np.ndarray]] = []
    for image, center_xy in zip(images, centers_xy):
        rings = detect_rings_deepcstrd(
            image,
            center_xy=center_xy,
            model_id=model_id,
            tile_size=tile_size,
            alpha=alpha,
            nr=nr,
            total_rotations=total_rotations,
            prediction_map_threshold=prediction_map_threshold,
            width=width,
            height=height,
            batch_size=batch_size,
        )
        results.append(rings)

    return results

def _get_model_path(model_id: str, tile_size: int = 0) -> str:
    """Get path to DeepCS-TRD model weights."""
    base_path = Path(__file__).parent.parent / "tree_ring_methods" / "deepcstrd" / "models" / "deep_cstrd"
    
    # Normalize tile size
    tile_size = 0 if tile_size not in [0, 256] else tile_size
    
    # Special case: generic model is always full image
    if model_id == "generic":
        model_path = base_path / "0_all_1504.pth"
    else:
        # Try to find model file with specified tile size
        model_path = base_path / f"{tile_size}_{model_id}_1504.pth"
        
        # If not found and tile_size=256, fallback to tile_size=0
        if not model_path.exists() and tile_size == 256:
            print(f"Warning: Tiled model for {model_id} not found, using full image model")
            model_path = base_path / f"0_{model_id}_1504.pth"
    
    # If still not found, fallback to generic
    if not model_path.exists():
        print(f"Warning: Model {model_id} not found, using generic model")
        model_path = base_path / "0_all_1504.pth"
    
    return str(model_path)
