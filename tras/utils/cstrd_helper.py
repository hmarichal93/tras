import numpy as np
from typing import List, Optional, Tuple
import subprocess
import tempfile
import json
import sys
import os
from pathlib import Path
from PIL import Image
from loguru import logger

from tras.params import CstrdParams
from tras.utils.image_preprocessing import normalize_to_rgb_uint8
from tras.utils.image_preprocessing import pad_for_edge_detection


def detect_rings_cstrd(
    image: np.ndarray,
    center_xy: Tuple[float, float],
    sigma: Optional[float] = None,
    th_low: Optional[float] = None,
    th_high: Optional[float] = None,
    alpha: Optional[int] = None,
    nr: Optional[int] = None,
    min_chain_length: int = 2,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Run CS-TRD (Classical tree ring detection) on the given image.

    Every parameter left as ``None`` takes its default from
    ``tras/config/detection_defaults.yaml`` (via ``CstrdParams``), the same values the
    API, CLI and GUI use.

    This method calls CS-TRD as a subprocess following the TRAS approach.

    Args:
        image: Input image as numpy array (H x W x 3)
        center_xy: Pith center coordinates (x, y)
        sigma: Gaussian smoothing sigma
        th_low: Low threshold for Canny edge detection
        th_high: High threshold for Canny edge detection
        alpha: Alpha parameter for angular sampling
        nr: Number of radial samples (360 = 1 degree resolution)
        min_chain_length: Minimum chain length to keep (default: 2)
        width: Resize width (0 = no resize)
        height: Resize height (0 = no resize)

    Returns:
        List of ring polylines, each as Nx2 array of (x, y) points
    """
    defaults = CstrdParams()
    sigma = defaults.sigma if sigma is None else sigma
    th_low = defaults.th_low if th_low is None else th_low
    th_high = defaults.th_high if th_high is None else th_high
    alpha = defaults.alpha if alpha is None else alpha
    nr = defaults.nr if nr is None else nr
    width = defaults.width if width is None else width
    height = defaults.height if height is None else height

    # Ensure RGB uint8 format
    image = normalize_to_rgb_uint8(image)

    # Check if pith is too close to edges and add padding if needed
    image, (cx, cy), (pad_top, pad_bottom, pad_left, pad_right) = pad_for_edge_detection(
        image, center_xy
    )
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        logger.debug(
            "CS-TRD: Added padding - top={}, bottom={}, left={}, right={}; adjusted pith=({:.1f}, {:.1f})",
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cx,
            cy,
        )

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
            "--input",
            str(temp_image),
            "--cy",
            str(int(cy)),
            "--cx",
            str(int(cx)),
            "--output_dir",
            str(temp_dir),
            "--root",
            str(cstrd_root),
            "--sigma",
            str(sigma),
            "--th_low",
            str(th_low),
            "--th_high",
            str(th_high),
            "--alpha",
            str(alpha),
            "--nr",
            str(nr),
            "--min_chain_length",
            str(min_chain_length),
            "--wsize",
            str(width),
            "--hsize",
            str(height),
            "--save_imgs",
            "0",  # Don't save debug images
            "--debug",
            "0",
        ]

        # Set PYTHONPATH to include CS-TRD directory and preserve existing env
        env = os.environ.copy()
        env["PYTHONPATH"] = str(cstrd_root)
        env["QT_QPA_PLATFORM"] = "offscreen"  # Run Qt in headless mode
        env["MPLBACKEND"] = "Agg"  # Use non-interactive matplotlib backend
        # Keep the run's config in its temp dir: never rewrite the packaged general.json.
        env["CSTRD_RUNTIME_CONFIG"] = str(temp_dir / "general.json")

        # Run CS-TRD
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(cstrd_root))

        if result.returncode != 0:
            error_msg = f"CS-TRD detection failed.\n\n"

            # Check for common issues
            if "AssertionError" in result.stderr:
                error_msg += (
                    "This usually happens when:\n"
                    "• Image is cropped too tightly around the wood section\n"
                    "• Pith is too close to image edges\n"
                    "• Wood section extends beyond image borders\n\n"
                    "Recommendations:\n"
                    "• Leave >100px margin around the wood section when cropping\n"
                    "• Use resize instead of crop for smaller images\n"
                    "• Try DeepCS-TRD which may handle edge cases better\n\n"
                )

            error_msg += f"Technical details:\n{result.stderr[-500:]}"  # Last 500 chars
            raise RuntimeError(error_msg)

        # Read output JSON
        output_json = temp_dir / "labelme.json"
        if not output_json.exists():
            raise FileNotFoundError(f"CS-TRD output not found: {output_json}")

        with open(output_json, "r") as f:
            data = json.load(f)

        # Extract rings from JSON
        rings = []
        if "shapes" in data:
            for shape in data["shapes"]:
                if "points" in shape and isinstance(shape["points"], list):
                    pts = np.array(shape["points"], dtype=np.float32)
                    if len(pts.shape) == 2 and pts.shape[1] == 2:
                        # Adjust coordinates back if padding was added
                        if any([pad_top, pad_bottom, pad_left, pad_right]):
                            pts[:, 0] -= pad_left  # x coordinates
                            pts[:, 1] -= pad_top  # y coordinates
                        rings.append(pts)

        return rings

    finally:
        # Cleanup temp files
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
