"""CLI command for automatic pith and ring detection.

This command performs automatic detection of tree ring pith and rings from an image.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from tras.api import detect as detect_api

app = typer.Typer(
    name="detect",
    help="Detect tree ring pith and rings from an image",
)


@app.command()
def main(
    image_path: Path = typer.Argument(..., help="Path to input image"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file path"),
    pith_x: Optional[float] = typer.Option(None, "--pith-x", help="Manual pith X coordinate"),
    pith_y: Optional[float] = typer.Option(None, "--pith-y", help="Manual pith Y coordinate"),
    auto_pith: bool = typer.Option(True, "--auto-pith/--no-auto-pith", help="Auto-detect pith with APD"),
    pith_method: str = typer.Option("apd_dl", "--pith-method", help="APD method: 'apd', 'apd_pcl', or 'apd_dl'"),
    ring_method: str = typer.Option(
        "deepcstrd", "--ring-method", help="Ring detection method: 'cstrd', 'deepcstrd', or 'inbd'"
    ),
    # Preprocessing options
    scale: Optional[float] = typer.Option(None, "--scale", help="Resize scale factor (0.1-1.0)"),
    remove_background: bool = typer.Option(False, "--remove-background", help="Remove background with U2Net"),
    # Postprocessing options
    sampling_nr: int = typer.Option(360, "--sampling-nr", help="Number of radial samples for postprocess resampling"),
    # CS-TRD parameters
    cstrd_sigma: float = typer.Option(3.0, "--cstrd-sigma", help="CS-TRD Gaussian sigma"),
    cstrd_th_low: float = typer.Option(5.0, "--cstrd-th-low", help="CS-TRD low threshold"),
    cstrd_th_high: float = typer.Option(20.0, "--cstrd-th-high", help="CS-TRD high threshold"),
    cstrd_alpha: int = typer.Option(30, "--cstrd-alpha", help="CS-TRD alpha parameter"),
    cstrd_nr: int = typer.Option(360, "--cstrd-nr", help="CS-TRD radial samples"),
    # DeepCS-TRD parameters
    deepcstrd_model: str = typer.Option("generic", "--deepcstrd-model", help="DeepCS-TRD model ID"),
    deepcstrd_tile_size: int = typer.Option(0, "--deepcstrd-tile-size", help="DeepCS-TRD tile size (0 or 256)"),
    deepcstrd_alpha: int = typer.Option(45, "--deepcstrd-alpha", help="DeepCS-TRD alpha parameter"),
    deepcstrd_nr: int = typer.Option(360, "--deepcstrd-nr", help="DeepCS-TRD radial samples (method-internal)"),
    deepcstrd_rotations: int = typer.Option(5, "--deepcstrd-rotations", help="DeepCS-TRD test-time augmentations"),
    deepcstrd_threshold: float = typer.Option(0.5, "--deepcstrd-threshold", help="DeepCS-TRD prediction threshold"),
    # INBD parameters
    inbd_model: str = typer.Option("INBD_EH", "--inbd-model", help="INBD model ID"),
    inbd_auto_pith: bool = typer.Option(True, "--inbd-auto-pith/--no-inbd-auto-pith", help="INBD auto-detect pith"),
) -> None:
    """
    Detect tree ring pith and rings from an image.

    This command performs:
    1. Optional preprocessing (resize, background removal)
    2. Pith detection (automatic with APD or manual coordinates)
    3. Ring detection (CS-TRD or DeepCS-TRD)
    4. Save results to LabelFile JSON format

    Examples:
        # Auto-detect pith and rings with DeepCS-TRD
        tras_detect image.jpg -o output.json

        # Use CS-TRD with custom pith coordinates
        tras_detect image.jpg --pith-x 100 --pith-y 200 --ring-method cstrd --no-auto-pith

        # Use INBD with auto pith detection
        tras_detect image.jpg --ring-method inbd --inbd-model INBD_UruDendro1

        # Preprocess image before detection
        tras_detect image.jpg --scale 0.8 --remove-background
        
        # Custom sampling NR for postprocess resampling
        tras_detect image.jpg --sampling-nr 720
    """
    try:
        # Validate input
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            sys.exit(1)

        # Determine output path
        if output is None:
            output = image_path.parent / f"{image_path.stem}_detected.json"
        else:
            output = Path(output)

        try:
            detection = detect_api(
                image_path=image_path,
                output=output,
                pith_x=pith_x,
                pith_y=pith_y,
                auto_pith=auto_pith,
                pith_method=pith_method,
                ring_method=ring_method,
                scale=scale,
                remove_background=remove_background,
                sampling_nr=sampling_nr,
                cstrd_sigma=cstrd_sigma,
                cstrd_th_low=cstrd_th_low,
                cstrd_th_high=cstrd_th_high,
                cstrd_alpha=cstrd_alpha,
                cstrd_nr=cstrd_nr,
                deepcstrd_model=deepcstrd_model,
                deepcstrd_tile_size=deepcstrd_tile_size,
                deepcstrd_alpha=deepcstrd_alpha,
                deepcstrd_nr=deepcstrd_nr,
                deepcstrd_rotations=deepcstrd_rotations,
                deepcstrd_threshold=deepcstrd_threshold,
                inbd_model=inbd_model,
                inbd_auto_pith=inbd_auto_pith,
            )
        except ValueError as exc:
            logger.error(f"Invalid configuration: {exc}")
            sys.exit(1)

        result_path = detection.output_path or output

        logger.info("✓ Detection complete!")
        print(f"\nResults saved to: {result_path}")
        print(f"  Pith: ({detection.pith_xy[0]:.1f}, {detection.pith_xy[1]:.1f})")
        print(f"  Rings: {len(detection.rings)}")

    except KeyboardInterrupt:
        logger.warning("Detection interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    app()



