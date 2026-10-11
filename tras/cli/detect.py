"""``tras_detect``: automatic pith and ring detection from the command line.

The input decides the mode:

* **image file** → detect rings on that image and write one TRAS label file (JSON);
* **directory** → batch-process every image in it (JSON + CSV + PDF per image, a combined
  ``summary.pdf`` and the ``batch_config.yml`` actually used).

Parameter precedence, lowest to highest: ``tras/config/detection_defaults.yaml`` →
``--config`` YAML → command-line flags.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from typing import Optional

import typer
from loguru import logger

from tras.api import detect as detect_api
from tras.params import DetectionParams
from tras.utils.batch_helper import run_batch
from tras.utils.cli_config import ConfigError
from tras.utils.cli_config import load_config
from tras.utils.cli_config import merge_config

app = typer.Typer(
    name="detect",
    help="Detect tree ring pith and rings from an image or batch process a folder",
)

# CLI flag name -> key in the YAML config schema understood by ``merge_config``.
_FLAG_TO_CONFIG_KEY = {
    "scale": "preprocess_resize_scale",
    "remove_background": "preprocess_remove_background",
    "sampling_nr": "postprocess_sampling_nr",
    "auto_pith": "pith_auto",
}


def _process_single_image(
    image_path: Path,
    output: Optional[Path],
    params: DetectionParams,
) -> None:
    """Detect rings on one image and write its label file."""
    if not image_path.is_file():
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)

    output = output or image_path.parent / f"{image_path.stem}_detected.json"
    try:
        detection = detect_api(image_path=image_path, output=output, params=params)
    except ValueError as exc:
        logger.error(f"Invalid configuration: {exc}")
        sys.exit(1)

    print(f"\nResults saved to: {detection.output_path or output}")
    print(f"  Pith: ({detection.pith_xy[0]:.1f}, {detection.pith_xy[1]:.1f})")
    print(f"  Rings: {len(detection.rings)}")


def _process_folder(
    input_dir: Path,
    output_dir: Optional[Path],
    config: Optional[Path],
    overrides: dict[str, Any],
) -> None:
    """Batch-process every image in ``input_dir``.

    ``overrides`` holds the flags given on the command line, already translated to
    config keys (see ``_config_overrides``).
    """
    if not input_dir.is_dir():
        logger.error(f"Input directory not found or not a directory: {input_dir}")
        sys.exit(1)

    output_dir = Path(output_dir) if output_dir else input_dir / "tras_out"
    try:
        merged = merge_config(load_config(config), overrides)
    except ConfigError as exc:
        logger.error(f"Configuration error: {exc}")
        sys.exit(1)

    if "physical_scale" not in merged:
        logger.warning(
            "No physical scale given (--scale-value/--scale-unit or physical_scale in "
            "--config); CSV and PDF reports will be in pixels."
        )

    summary = run_batch(input_dir, output_dir, merged, per_image_reports=True)
    logger.info(
        f"Processed {summary.processed}, skipped {summary.skipped}, "
        f"errors {summary.errors} of {summary.total} image(s). Output: {output_dir}"
    )
    if summary.total == 0:
        logger.warning(f"No images found in {input_dir}")


def _config_overrides(flags: dict[str, Any]) -> dict[str, Any]:
    """Translate given (non-None) CLI flags to ``merge_config`` override keys."""
    return {
        _FLAG_TO_CONFIG_KEY.get(name, name): value
        for name, value in flags.items()
        if value is not None
    }


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="Path to input image file or directory containing images",
    ),
    # Single image options
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output JSON file path (single image mode only)"
    ),
    pith_x: Optional[float] = typer.Option(None, "--pith-x", help="Manual pith X coordinate"),
    pith_y: Optional[float] = typer.Option(None, "--pith-y", help="Manual pith Y coordinate"),
    auto_pith: Optional[bool] = typer.Option(
        None, "--auto-pith/--no-auto-pith", help="Auto-detect pith with APD"
    ),
    pith_method: Optional[str] = typer.Option(
        None, "--pith-method", help="APD method: 'apd', 'apd_pcl', or 'apd_dl'"
    ),
    ring_method: Optional[str] = typer.Option(
        None, "--ring-method", help="Ring detection method: 'cstrd', 'deepcstrd', or 'inbd'"
    ),
    # Preprocessing options
    scale: Optional[float] = typer.Option(None, "--scale", help="Resize scale factor (0.1-1.0)"),
    remove_background: Optional[bool] = typer.Option(
        None, "--remove-background/--no-remove-background", help="Remove background with U2Net"
    ),
    # Postprocessing options
    sampling_nr: Optional[int] = typer.Option(
        None, "--sampling-nr", help="Number of radial samples for postprocess resampling"
    ),
    # CS-TRD parameters
    cstrd_sigma: Optional[float] = typer.Option(None, "--cstrd-sigma", help="CS-TRD Gaussian sigma"),
    cstrd_th_low: Optional[float] = typer.Option(None, "--cstrd-th-low", help="CS-TRD low threshold"),
    cstrd_th_high: Optional[float] = typer.Option(
        None, "--cstrd-th-high", help="CS-TRD high threshold"
    ),
    cstrd_alpha: Optional[int] = typer.Option(None, "--cstrd-alpha", help="CS-TRD alpha parameter"),
    cstrd_nr: Optional[int] = typer.Option(None, "--cstrd-nr", help="CS-TRD radial samples"),
    # DeepCS-TRD parameters
    deepcstrd_model: Optional[str] = typer.Option(
        None, "--deepcstrd-model", help="DeepCS-TRD model ID or path"
    ),
    deepcstrd_tile_size: Optional[int] = typer.Option(
        None, "--deepcstrd-tile-size", help="DeepCS-TRD tile size (0 or 256)"
    ),
    deepcstrd_alpha: Optional[int] = typer.Option(
        None, "--deepcstrd-alpha", help="DeepCS-TRD alpha parameter"
    ),
    deepcstrd_nr: Optional[int] = typer.Option(
        None, "--deepcstrd-nr", help="DeepCS-TRD radial samples (method-internal)"
    ),
    deepcstrd_rotations: Optional[int] = typer.Option(
        None, "--deepcstrd-rotations", help="DeepCS-TRD test-time augmentations"
    ),
    deepcstrd_threshold: Optional[float] = typer.Option(
        None, "--deepcstrd-threshold", help="DeepCS-TRD prediction threshold"
    ),
    # INBD parameters
    inbd_model: Optional[str] = typer.Option(None, "--inbd-model", help="INBD model ID or path"),
    inbd_auto_pith: Optional[bool] = typer.Option(
        None, "--inbd-auto-pith/--no-inbd-auto-pith", help="INBD auto-detect pith"
    ),
    # Batch processing options (only used when input is a directory)
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for batch processing (default: <input_dir>/tras_out)",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML configuration file (batch processing)"
    ),
    scale_value: Optional[float] = typer.Option(
        None, "--scale-value", help="Physical scale value (unit per pixel) for batch processing"
    ),
    scale_unit: Optional[str] = typer.Option(
        None, "--scale-unit", help="Physical scale unit (e.g., mm, cm) for batch processing"
    ),
) -> None:
    """
    Detect tree ring pith and rings from an image or batch process a folder.

    Single Image Mode Examples:
        # Auto-detect pith and rings with DeepCS-TRD
        tras_detect image.jpg -o output.json

        # Use CS-TRD with custom pith coordinates
        tras_detect image.jpg --pith-x 100 --pith-y 200 --ring-method cstrd --no-auto-pith

        # Use INBD with auto pith detection
        tras_detect image.jpg --ring-method inbd --inbd-model INBD_UruDendro1

    Batch Processing Mode Examples:
        # Process folder with YAML config
        tras_detect /path/to/images --config config.yml

        # Process folder with CLI-only config
        tras_detect /path/to/images --scale-value 0.0213 --scale-unit mm

        # Override config with CLI flags
        tras_detect /path/to/images --config config.yml --ring-method cstrd
    """
    detection_flags = {
        "pith_x": pith_x,
        "pith_y": pith_y,
        "auto_pith": auto_pith,
        "pith_method": pith_method,
        "ring_method": ring_method,
        "scale": scale,
        "remove_background": remove_background,
        "sampling_nr": sampling_nr,
        "cstrd_sigma": cstrd_sigma,
        "cstrd_th_low": cstrd_th_low,
        "cstrd_th_high": cstrd_th_high,
        "cstrd_alpha": cstrd_alpha,
        "cstrd_nr": cstrd_nr,
        "deepcstrd_model": deepcstrd_model,
        "deepcstrd_tile_size": deepcstrd_tile_size,
        "deepcstrd_alpha": deepcstrd_alpha,
        "deepcstrd_nr": deepcstrd_nr,
        "deepcstrd_rotations": deepcstrd_rotations,
        "deepcstrd_threshold": deepcstrd_threshold,
        "inbd_model": inbd_model,
        "inbd_auto_pith": inbd_auto_pith,
    }
    try:
        if input_path.is_file():
            params = DetectionParams().with_flat_overrides(**detection_flags)
            _process_single_image(input_path, output, params)
        elif input_path.is_dir():
            folder_flags = {
                **detection_flags,
                "scale_value": scale_value,
                "scale_unit": scale_unit,
            }
            _process_folder(input_path, output_dir, config, _config_overrides(folder_flags))
        else:
            logger.error(f"Input path not found: {input_path}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(130)
    except (ValueError, TypeError) as exc:
        logger.error(f"Invalid parameters: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    app()
