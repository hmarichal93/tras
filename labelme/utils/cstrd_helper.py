import numpy as np
from typing import List, Tuple
import subprocess
import tempfile
import json
import sys
import os
from pathlib import Path
from PIL import Image

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
    
    This method calls CS-TRD as a subprocess following the TRAS approach.
    
    Args:
        image: Input image as numpy array (H x W x 3)
        center_xy: Pith center coordinates (x, y)
        sigma: Gaussian smoothing sigma (default: 3.0)
        th_low: Low threshold for Canny edge detection (default: 5.0)
        th_high: High threshold for Canny edge detection (default: 20.0)
        alpha: Alpha parameter for angular sampling (default: 30)
        nr: Number of radial samples (default: 360 = 1 degree resolution)
        min_chain_length: Minimum chain length to keep (default: 2)
    
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
    
    # Create temporary directory for CS-TRD execution
    temp_dir = Path(tempfile.mkdtemp(prefix="cstrd_"))
    temp_image = temp_dir / "input_image.png"
    
    try:
        # Save image to temp file
        Image.fromarray(image).save(temp_image)
        
        # Get CS-TRD root path
        cstrd_root = Path(__file__).parent.parent / "tree_ring_methods" / "cstrd"
        main_py = cstrd_root / "main.py"
        
        # Build command following TRAS approach
        cmd = [
            sys.executable,  # Use current Python interpreter
            str(main_py),
            "--input", str(temp_image),
            "--cy", str(int(cy)),
            "--cx", str(int(cx)),
            "--output_dir", str(temp_dir),
            "--root", str(cstrd_root),
            "--sigma", str(sigma),
            "--th_low", str(th_low),
            "--th_high", str(th_high),
            "--alpha", str(alpha),
            "--nr", str(nr),
            "--min_chain_length", str(min_chain_length),
            "--save_imgs", "0",  # Don't save debug images
            "--debug", "0"
        ]
        
        # Set PYTHONPATH to include CS-TRD directory and preserve existing env
        env = os.environ.copy()
        env["PYTHONPATH"] = str(cstrd_root)
        env["QT_QPA_PLATFORM"] = "offscreen"  # Run Qt in headless mode
        env["MPLBACKEND"] = "Agg"  # Use non-interactive matplotlib backend
        
        # Run CS-TRD
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=str(cstrd_root)
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"CS-TRD failed: {result.stderr}")
        
        # Read output JSON
        output_json = temp_dir / "labelme.json"
        if not output_json.exists():
            raise FileNotFoundError(f"CS-TRD output not found: {output_json}")
        
        with open(output_json, 'r') as f:
            data = json.load(f)
        
        # Extract rings from JSON
        rings = []
        if 'shapes' in data:
            for shape in data['shapes']:
                if 'points' in shape and isinstance(shape['points'], list):
                    pts = np.array(shape['points'], dtype=np.float32)
                    if len(pts.shape) == 2 and pts.shape[1] == 2:
                        rings.append(pts)
        
        return rings
        
    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

