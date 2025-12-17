import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# Add INBD to path
_inbd_path = Path(__file__).parent.parent / "tree_ring_methods" / "inbd" / "src"
sys.path.insert(0, str(_inbd_path))

def detect_rings_inbd(
    image: np.ndarray, 
    center_xy: Optional[Tuple[float, float]] = None,
    model_id: str = "INBD_EH",
    output_format: str = "polylines"
) -> List[np.ndarray]:
    """
    Run INBD on the given image and return a list of ring polylines (Nx2 arrays).
    
    INBD (Iterative Next Boundary Detection) is a CVPR 2023 method for instance
    segmentation of tree rings in microscopy images.
    
    Args:
        image: Input image as numpy array (H x W x 3)
        center_xy: Pith center coordinates (x, y). Optional - if None, INBD will 
                   attempt to detect the pith automatically or use its own method.
        model_id: Model identifier ("INBD_EH", "INBD_DO", "INBD_VM", "INBD_UruDendro")
            - INBD_EH: Empetrum hermaphroditum (shrub)
            - INBD_DO: Dryas octopetala (shrub)
            - INBD_VM: Vaccinium myrtillus (shrub)
            - INBD_UruDendro: Pinus taeda (tree)
        output_format: Output format ("polylines" or "masks")
    
    Returns:
        List of ring polylines, each as Nx2 array of (x, y) points
    
    Raises:
        ImportError: If INBD source code is not available
        FileNotFoundError: If model file is not found
    """
    # Validate INBD installation
    if not _inbd_path.exists():
        raise ImportError(
            f"INBD source code not found at {_inbd_path}\n\n"
            "SETUP REQUIRED:\n"
            "1. Clone INBD repository:\n"
            "   cd tras/tree_ring_methods/inbd\n"
            "   git clone https://github.com/hmarichal93/INBD.git src\n\n"
            "2. Install dependencies:\n"
            "   cd src\n"
            "   pip install -r requirements.txt\n\n"
            "3. Download models:\n"
            "   python fetch_pretrained_models.py\n\n"
            "See tras/tree_ring_methods/inbd/README.md for complete instructions."
        )
    
    # INBD uses a CLI interface through main.py, so we'll call it as a subprocess
    # rather than trying to import modules directly
    
    # Import cv2 at the beginning
    import cv2
    
    # Ensure RGB format (convert common OpenCV BGR to RGB)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        # Assume it's already RGB (TRAS typically provides RGB)
        pass
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (255 * image).astype(np.uint8)
        else:
            image = (255 * (image.astype(np.float32) / image.max())).astype(np.uint8)
    
    # Get model path
    model_path = _get_model_path(model_id)
    
    # Handle optional pith coordinates
    cx, cy = None, None
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    
    if center_xy is not None:
        cx, cy = center_xy
        
        # Check if pith is too close to edges and add padding if needed
        h, w = image.shape[:2]
        min_margin = 100  # INBD needs margin for edge detection
        
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
            print(f"INBD: Added padding - top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
            print(f"INBD: Adjusted pith: ({cx:.1f}, {cy:.1f})")
    else:
        print("INBD: Running without explicit pith coordinates (auto-detection mode)")
    
    # Create temporary output directory for INBD
    import tempfile
    import os
    import subprocess
    import json
    
    temp_dir = tempfile.mkdtemp(prefix="inbd_")
    temp_image_path = os.path.join(temp_dir, "input_image.jpg")
    temp_output_path = os.path.join(temp_dir, "output.json")
    
    # Save image temporarily
    cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    try:
        # Run INBD as subprocess
        # INBD CLI: python main.py inference model_path image_path --cx X --cy Y
        main_py = _inbd_path / "main.py"
        
        # Build command - INBD expects --cy before --cx based on their examples
        cmd = [
            "python",
            str(main_py),
            "inference",
            model_path,
            temp_image_path
        ]
        
        # Add pith coordinates only if provided
        if cx is not None and cy is not None:
            cmd.extend(["--cy", str(int(cy))])
            cmd.extend(["--cx", str(int(cx))])
            print(f"INBD: Running with pith coordinates: cx={int(cx)}, cy={int(cy)}")
        else:
            print("INBD: Running without pith coordinates (auto-detection)")
        
        print(f"INBD: Command: {' '.join(cmd)}")
        print(f"INBD: Working directory: {_inbd_path}")
        
        # Run INBD
        result = subprocess.run(
            cmd,
            cwd=str(_inbd_path),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Check if INBD completed (note: it may fail at contour conversion but still create labelmap)
        print(f"INBD: Process returned with code {result.returncode}")
        print(f"INBD stdout: {result.stdout}")
        
        if result.returncode != 0:
            # INBD may fail at the contour conversion step but still create the labelmap
            # Check if it's the cy/cx NoneType error
            if "TypeError: must be real number, not NoneType" in result.stderr or "cy" in result.stderr.lower():
                print("INBD: Process failed at contour conversion (expected with auto-detection)")
                print("INBD: Will attempt to process labelmap directly")
            else:
                # Other errors should be raised
                raise RuntimeError(
                    f"INBD process failed with return code {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )
        else:
            print(f"INBD: Process completed successfully")
        
        # INBD outputs labelmap PNG files, not JSON
        # Look for labelmap output in the inference directory
        inference_dir = _inbd_path / "inference"
        labelmap_files = []
        
        if inference_dir.exists():
            # Look for .labelmap.png files
            labelmap_files = list(inference_dir.rglob("*.labelmap.png"))
        
        if labelmap_files:
            # Use the most recent labelmap file
            labelmap_file = max(labelmap_files, key=lambda p: p.stat().st_mtime)
            print(f"INBD: Found labelmap at {labelmap_file}")
            
            # If pith coordinates weren't provided, compute them from the innermost region
            if cx is None or cy is None:
                print("INBD: Computing pith from innermost region...")
                computed_cx, computed_cy = _compute_pith_from_labelmap(labelmap_file)
                if computed_cx is not None and computed_cy is not None:
                    cx, cy = computed_cx, computed_cy
                    print(f"INBD: Computed pith at ({cx:.1f}, {cy:.1f})")
            
            # Extract rings from labelmap
            rings = _extract_rings_from_labelmap(labelmap_file, cx, cy, pad_left, pad_top)
        else:
            # Try to find JSON output files as fallback
            output_files = list(Path(temp_dir).glob("*.json"))
            if not output_files:
                output_files = list((_inbd_path / "output").glob("*.json"))
            
            if output_files:
                output_file = max(output_files, key=lambda p: p.stat().st_mtime)
                with open(output_file, 'r') as f:
                    inbd_result = json.load(f)
                rings = _extract_rings_from_inbd_result(inbd_result, pad_left, pad_top)
            else:
                print("INBD: No output files found")
                rings = []
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return rings
        
    except subprocess.TimeoutExpired:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("INBD inference timed out after 5 minutes")
    except Exception as e:
        # Clean up on error
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"INBD inference failed: {str(e)}")


def _get_model_path(model_id: str) -> str:
    """Get path to INBD model checkpoint."""
    # Look in downloaded_assets first (for consistency with DeepCS-TRD)
    base_path = Path(__file__).parent.parent.parent / "downloaded_assets" / "inbd"
    
    # Also check in INBD source directory
    inbd_checkpoints = Path(__file__).parent.parent / "tree_ring_methods" / "inbd" / "src" / "checkpoints"
    
    # Model filename format: INBD_EH/model.pt.zip
    model_file = f"{model_id}/model.pt.zip"
    
    # Try downloaded_assets first
    model_path = base_path / model_file
    if model_path.exists():
        return str(model_path)
    
    # Try INBD checkpoints directory
    model_path = inbd_checkpoints / model_file
    if model_path.exists():
        return str(model_path)
    
    # Model not found - provide detailed setup instructions
    raise FileNotFoundError(
        f"INBD model '{model_id}' not found.\n\n"
        f"Searched in:\n"
        f"  - {base_path}\n"
        f"  - {inbd_checkpoints}\n\n"
        f"SETUP REQUIRED:\n"
        f"1. Clone INBD repository:\n"
        f"   cd tras/tree_ring_methods/inbd\n"
        f"   git clone https://github.com/hmarichal93/INBD.git src\n\n"
        f"2. Download pre-trained models:\n"
        f"   cd src\n"
        f"   python fetch_pretrained_models.py\n\n"
        f"Available models after setup:\n"
        f"  - INBD_EH: Empetrum hermaphroditum (shrub)\n"
        f"  - INBD_DO: Dryas octopetala (shrub)\n"
        f"  - INBD_VM: Vaccinium myrtillus (shrub)\n"
        f"  - INBD_UruDendro: Pinus taeda (tree)"
    )


def _extract_rings_from_inbd_result(
    result: any,
    pad_left: int = 0,
    pad_top: int = 0
) -> List[np.ndarray]:
    """
    Extract ring polylines from INBD result.
    
    INBD returns results in a format that we need to convert to our standard
    polyline format (list of Nx2 arrays).
    """
    rings = []
    
    # The exact format depends on INBD's implementation
    # We'll need to adapt this based on what INBD actually returns
    
    if result is None:
        return rings
    
    # Handle different possible return formats from INBD
    if isinstance(result, dict):
        # If result is a dict with 'rings' or 'shapes' key
        if 'rings' in result:
            ring_data = result['rings']
        elif 'shapes' in result:
            ring_data = result['shapes']
        else:
            ring_data = result
    elif isinstance(result, list):
        ring_data = result
    else:
        # Unknown format
        print(f"Warning: Unexpected INBD result format: {type(result)}")
        return rings
    
    # Convert to polylines
    for ring in ring_data:
        if isinstance(ring, dict):
            if 'points' in ring:
                pts = np.array(ring['points'], dtype=np.float32)
            elif 'coordinates' in ring:
                pts = np.array(ring['coordinates'], dtype=np.float32)
            else:
                continue
        elif isinstance(ring, (list, np.ndarray)):
            pts = np.array(ring, dtype=np.float32)
        else:
            continue
        
        # Ensure proper shape
        if len(pts.shape) == 1:
            pts = pts.reshape(-1, 2)
        
        if len(pts.shape) == 2 and pts.shape[1] == 2:
            # Adjust coordinates back if padding was added
            if pad_left > 0 or pad_top > 0:
                pts[:, 0] -= pad_left  # x coordinates
                pts[:, 1] -= pad_top   # y coordinates
            rings.append(pts)
    
    return rings


def _parse_rings_from_stdout(
    stdout: str,
    pad_left: int = 0,
    pad_top: int = 0
) -> List[np.ndarray]:
    """
    Parse ring polylines from INBD stdout output.
    
    This is a fallback method if INBD doesn't create output files.
    """
    rings = []
    
    # Try to find JSON-like output in stdout
    import json
    import re
    
    # Look for JSON blocks in the output
    json_matches = re.findall(r'\{[^{}]*\}', stdout, re.MULTILINE | re.DOTALL)
    
    for json_str in json_matches:
        try:
            data = json.loads(json_str)
            extracted = _extract_rings_from_inbd_result(data, pad_left, pad_top)
            rings.extend(extracted)
        except json.JSONDecodeError:
            continue
    
    return rings


def _compute_pith_from_labelmap(labelmap_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute pith coordinates from the innermost region in the labelmap.
    
    Args:
        labelmap_path: Path to the labelmap PNG file
        
    Returns:
        Tuple of (cx, cy) or (None, None) if computation fails
    """
    try:
        import cv2
        from scipy import ndimage
        
        # Load labelmap
        labelmap = cv2.imread(str(labelmap_path), cv2.IMREAD_GRAYSCALE)
        if labelmap is None:
            print(f"Warning: Could not load labelmap from {labelmap_path}")
            return None, None
        
        # Find the innermost region (smallest non-zero label, typically 1)
        unique_labels = np.unique(labelmap)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
        
        if len(unique_labels) == 0:
            print("Warning: No labeled regions found in labelmap")
            return None, None
        
        # The innermost ring should be the first label (smallest label number)
        innermost_label = min(unique_labels)
        innermost_mask = (labelmap == innermost_label)
        
        # Compute center of mass (centroid) of the innermost region
        center_of_mass = ndimage.center_of_mass(innermost_mask)
        
        if center_of_mass is not None and len(center_of_mass) == 2:
            cy, cx = center_of_mass  # ndimage returns (row, col) = (y, x)
            return float(cx), float(cy)
        
        return None, None
        
    except Exception as e:
        print(f"Error computing pith from labelmap: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _extract_rings_from_labelmap(
    labelmap_path: Path,
    cx: Optional[float],
    cy: Optional[float],
    pad_left: int = 0,
    pad_top: int = 0
) -> List[np.ndarray]:
    """
    Extract ring contours from INBD labelmap.
    
    Args:
        labelmap_path: Path to the labelmap PNG file
        cx: Pith x coordinate (optional)
        cy: Pith y coordinate (optional)
        pad_left: Padding to adjust coordinates
        pad_top: Padding to adjust coordinates
        
    Returns:
        List of ring polylines as Nx2 arrays
    """
    try:
        import cv2
        
        # Load labelmap
        labelmap = cv2.imread(str(labelmap_path), cv2.IMREAD_GRAYSCALE)
        if labelmap is None:
            print(f"Warning: Could not load labelmap from {labelmap_path}")
            return []
        
        rings = []
        
        # Get unique labels (each label represents a ring)
        unique_labels = np.unique(labelmap)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        print(f"INBD: Found {len(unique_labels)} ring labels in labelmap")
        
        # Extract contour for each ring
        for label in sorted(unique_labels):
            # Create binary mask for this ring
            mask = (labelmap == label).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Convert to Nx2 array
                pts = largest_contour.squeeze()
                
                if len(pts.shape) == 2 and pts.shape[1] == 2:
                    # Adjust for padding if needed
                    if pad_left > 0 or pad_top > 0:
                        pts = pts.astype(np.float32)
                        pts[:, 0] -= pad_left  # x
                        pts[:, 1] -= pad_top   # y
                    
                    rings.append(pts.astype(np.float32))
        
        print(f"INBD: Extracted {len(rings)} ring contours")
        return rings
        
    except Exception as e:
        print(f"Error extracting rings from labelmap: {e}")
        import traceback
        traceback.print_exc()
        return []

