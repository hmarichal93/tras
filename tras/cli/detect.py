"""Unified CLI command for automatic pith and ring detection.

Automatically detects if input is a single image file or a directory of images,
and processes accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from tras.api import detect as detect_api
from tras.utils.cli_config import (
    ConfigError,
    get_detection_params,
    load_config,
    merge_config,
    validate_config,
)
from tras.utils.ring_properties import compute_ring_properties, write_ring_properties_csv
from tras.utils.report_pdf import generate_pdf_report

app = typer.Typer(
    name="detect",
    help="Detect tree ring pith and rings from an image or batch process a folder",
)


def _process_single_image(
    image_path: Path,
    output: Optional[Path],
    pith_x: Optional[float],
    pith_y: Optional[float],
    auto_pith: bool,
    pith_method: str,
    ring_method: str,
    scale: Optional[float],
    remove_background: bool,
    sampling_nr: int,
    cstrd_sigma: float,
    cstrd_th_low: float,
    cstrd_th_high: float,
    cstrd_alpha: int,
    cstrd_nr: int,
    deepcstrd_model: str,
    deepcstrd_tile_size: int,
    deepcstrd_alpha: int,
    deepcstrd_nr: int,
    deepcstrd_rotations: int,
    deepcstrd_threshold: float,
    inbd_model: str,
    inbd_auto_pith: bool,
) -> None:
    """Process a single image file."""
    # Validate input
    if not image_path.exists() or not image_path.is_file():
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


def _process_folder(
    input_dir: Path,
    output_dir: Optional[Path],
    config: Optional[Path],
    scale_value: Optional[float],
    scale_unit: Optional[str],
    preprocess_resize_scale: Optional[float],
    preprocess_remove_background: Optional[bool],
    postprocess_sampling_nr: Optional[int],
    pith_auto: Optional[bool],
    pith_method: Optional[str],
    ring_method: Optional[str],
    cstrd_sigma: Optional[float],
    cstrd_th_low: Optional[float],
    cstrd_th_high: Optional[float],
    cstrd_alpha: Optional[int],
    cstrd_nr: Optional[int],
    deepcstrd_model: Optional[str],
    deepcstrd_tile_size: Optional[int],
    deepcstrd_alpha: Optional[int],
    deepcstrd_nr: Optional[int],
    deepcstrd_rotations: Optional[int],
    deepcstrd_threshold: Optional[float],
    inbd_model: Optional[str],
    inbd_auto_pith: Optional[bool],
) -> None:
    """Process all images in a folder."""
    # Validate input directory
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory not found or not a directory: {input_dir}")
        sys.exit(1)

    # Determine output directory
    if output_dir is None:
        output_dir = input_dir / "tras_out"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load and merge configuration
    yaml_config = load_config(config)

    # Collect CLI overrides (only non-None values)
    normalized_overrides = {}
    if scale_value is not None:
        normalized_overrides["scale_value"] = scale_value
    if scale_unit is not None:
        normalized_overrides["scale_unit"] = scale_unit
    if preprocess_resize_scale is not None:
        normalized_overrides["preprocess_resize_scale"] = preprocess_resize_scale
    if preprocess_remove_background is not None:
        normalized_overrides["preprocess_remove_background"] = preprocess_remove_background
    if postprocess_sampling_nr is not None:
        normalized_overrides["postprocess_sampling_nr"] = postprocess_sampling_nr
    if pith_auto is not None:
        normalized_overrides["pith_auto"] = pith_auto
    if pith_method is not None:
        normalized_overrides["pith_method"] = pith_method
    if ring_method is not None:
        normalized_overrides["ring_method"] = ring_method
    if cstrd_sigma is not None:
        normalized_overrides["cstrd_sigma"] = cstrd_sigma
    if cstrd_th_low is not None:
        normalized_overrides["cstrd_th_low"] = cstrd_th_low
    if cstrd_th_high is not None:
        normalized_overrides["cstrd_th_high"] = cstrd_th_high
    if cstrd_alpha is not None:
        normalized_overrides["cstrd_alpha"] = cstrd_alpha
    if cstrd_nr is not None:
        normalized_overrides["cstrd_nr"] = cstrd_nr
    if deepcstrd_model is not None:
        normalized_overrides["deepcstrd_model"] = deepcstrd_model
    if deepcstrd_tile_size is not None:
        normalized_overrides["deepcstrd_tile_size"] = deepcstrd_tile_size
    if deepcstrd_alpha is not None:
        normalized_overrides["deepcstrd_alpha"] = deepcstrd_alpha
    if deepcstrd_nr is not None:
        normalized_overrides["deepcstrd_nr"] = deepcstrd_nr
    if deepcstrd_rotations is not None:
        normalized_overrides["deepcstrd_rotations"] = deepcstrd_rotations
    if deepcstrd_threshold is not None:
        normalized_overrides["deepcstrd_threshold"] = deepcstrd_threshold
    if inbd_model is not None:
        normalized_overrides["inbd_model"] = inbd_model
    if inbd_auto_pith is not None:
        normalized_overrides["inbd_auto_pith"] = inbd_auto_pith

    merged_config = merge_config(yaml_config, normalized_overrides)

    # Validate that physical scale is present
    try:
        validate_config(merged_config)
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        logger.error(
            "Physical scale is required for batch processing. "
            "Provide via --config YAML file or --scale-value/--scale-unit CLI flags."
        )
        sys.exit(1)

    scale_value_final = float(merged_config["physical_scale"]["value"])
    scale_unit_final = merged_config["physical_scale"]["unit"]

    # Get detection parameters
    detect_params = get_detection_params(merged_config)

    # Find all JPEG/PNG images
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        logger.warning(f"No JPEG/PNG images found in {input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(image_files)} image(s) to process")

    # Process each image
    success_count = 0
    error_count = 0

    for img_idx, image_path in enumerate(sorted(image_files), 1):
        logger.info(f"\n[{img_idx}/{len(image_files)}] Processing {image_path.name}")

        try:
            # Determine output filenames (stem-based, no suffixes)
            stem = image_path.stem
            json_path = output_dir / f"{stem}.json"
            csv_path = output_dir / f"{stem}.csv"
            pdf_path = output_dir / f"{stem}.pdf"

            # Run detection
            logger.info(f"  Detecting pith and rings...")
            detection = detect_api(image_path=image_path, output=json_path, **detect_params)

            if not detection.rings:
                logger.warning(f"  No rings detected in {image_path.name}, skipping CSV/PDF")
                continue

            # Compute ring properties
            logger.info(f"  Computing ring properties...")
            ring_props = compute_ring_properties(detection.rings, detection.pith_xy)

            # Write CSV
            logger.info(f"  Writing CSV...")
            write_ring_properties_csv(csv_path, ring_props, scale_value_final, scale_unit_final)

            # Generate PDF
            logger.info(f"  Generating PDF report...")
            generate_pdf_report(
                output_path=pdf_path,
                image=detection.image,
                pith_xy=detection.pith_xy,
                rings=detection.rings,
                ring_properties=ring_props,
                scale_value=scale_value_final,
                scale_unit=scale_unit_final,
                sample_code=stem,
                harvested_year=None,
            )

            logger.info(f"  ✓ Complete: {stem}.json, {stem}.csv, {stem}.pdf")
            success_count += 1

        except KeyboardInterrupt:
            logger.warning("\nProcessing interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.error(f"  ✗ Failed to process {image_path.name}: {e}")
            import traceback

            traceback.print_exc()
            error_count += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"  Successful: {success_count}/{len(image_files)}")
    if error_count > 0:
        logger.warning(f"  Errors: {error_count}/{len(image_files)}")
    logger.info(f"  Output directory: {output_dir}")


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
    cstrd_th_high: Optional[float] = typer.Option(None, "--cstrd-th-high", help="CS-TRD high threshold"),
    cstrd_alpha: Optional[int] = typer.Option(None, "--cstrd-alpha", help="CS-TRD alpha parameter"),
    cstrd_nr: Optional[int] = typer.Option(None, "--cstrd-nr", help="CS-TRD radial samples"),
    # DeepCS-TRD parameters
    deepcstrd_model: Optional[str] = typer.Option(
        None, "--deepcstrd-model", help="DeepCS-TRD model ID or path"
    ),
    deepcstrd_tile_size: Optional[int] = typer.Option(
        None, "--deepcstrd-tile-size", help="DeepCS-TRD tile size (0 or 256)"
    ),
    deepcstrd_alpha: Optional[int] = typer.Option(None, "--deepcstrd-alpha", help="DeepCS-TRD alpha parameter"),
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
        None, "--output-dir", help="Output directory for batch processing (default: <input_dir>/tras_out)"
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

    Automatically detects if input is a single image file or a directory:
    - **Single image**: Detects rings and saves JSON output
    - **Directory**: Batch processes all JPEG/PNG images, generates JSON/CSV/PDF per image

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
        tras_detect /path/to/images --scale-value 0.0213 --scale-unit mm --ring-method deepcstrd

        # Override config with CLI flags
        tras_detect /path/to/images --config config.yml --scale-value 0.025 --ring-method cstrd
    """
    try:
        input_path = Path(input_path)

        # Check if input exists
        if not input_path.exists():
            logger.error(f"Input path not found: {input_path}")
            sys.exit(1)

        # Determine if it's a file or directory
        if input_path.is_file():
            # Single image mode - use defaults for None values
            logger.info(f"Processing single image: {input_path}")
            _process_single_image(
                image_path=input_path,
                output=output,
                pith_x=pith_x,
                pith_y=pith_y,
                auto_pith=auto_pith if auto_pith is not None else True,
                pith_method=pith_method if pith_method is not None else "apd_dl",
                ring_method=ring_method if ring_method is not None else "deepcstrd",
                scale=scale,
                remove_background=remove_background if remove_background is not None else False,
                sampling_nr=sampling_nr if sampling_nr is not None else 360,
                cstrd_sigma=cstrd_sigma if cstrd_sigma is not None else 3.0,
                cstrd_th_low=cstrd_th_low if cstrd_th_low is not None else 5.0,
                cstrd_th_high=cstrd_th_high if cstrd_th_high is not None else 20.0,
                cstrd_alpha=cstrd_alpha if cstrd_alpha is not None else 30,
                cstrd_nr=cstrd_nr if cstrd_nr is not None else 360,
                deepcstrd_model=deepcstrd_model if deepcstrd_model is not None else "generic",
                deepcstrd_tile_size=deepcstrd_tile_size if deepcstrd_tile_size is not None else 0,
                deepcstrd_alpha=deepcstrd_alpha if deepcstrd_alpha is not None else 45,
                deepcstrd_nr=deepcstrd_nr if deepcstrd_nr is not None else 360,
                deepcstrd_rotations=deepcstrd_rotations if deepcstrd_rotations is not None else 5,
                deepcstrd_threshold=deepcstrd_threshold if deepcstrd_threshold is not None else 0.5,
                inbd_model=inbd_model if inbd_model is not None else "INBD_EH",
                inbd_auto_pith=inbd_auto_pith if inbd_auto_pith is not None else True,
            )
        elif input_path.is_dir():
            # Batch processing mode
            logger.info(f"Processing folder: {input_path}")
            _process_folder(
                input_dir=input_path,
                output_dir=output_dir,
                config=config,
                scale_value=scale_value,
                scale_unit=scale_unit,
                preprocess_resize_scale=scale,  # Map --scale to preprocess_resize_scale
                preprocess_remove_background=remove_background,
                postprocess_sampling_nr=sampling_nr,
                pith_auto=auto_pith,
                pith_method=pith_method,
                ring_method=ring_method,
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
        else:
            logger.error(f"Input path is neither a file nor a directory: {input_path}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    app()
