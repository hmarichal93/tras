"""Qt-free batch processing runner for the TRAS GUI.

Shared by the GUI batch dialog and the ``tras_detect`` folder mode. It writes one
``.json`` label file per image, then a single combined ``summary.pdf`` (one image per
page) and a CLI-compatible ``batch_config.yml`` recording the settings used.
Already-processed images (those with an existing ``.json``) are skipped so a batch can
be resumed.

This module is intentionally free of any Qt dependency so it can be tested headless.
The Qt threading wrapper lives in ``tras/widgets/batch_dialog.py``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
import numpy.typing as npt
from loguru import logger
from PIL import Image as PILImage

from tras._label_file import LabelFile
from tras.analysis.rings import compass_series
from tras.api import detect
from tras.io.tucson import sanitize_series_id
from tras.io.tucson import write_rwl
from tras.params import DetectionParams
from tras.utils.cli_config import dump_config
from tras.utils.cli_config import get_detection_params
from tras.utils.pos_exporter import mm_per_pixel
from tras.utils.report_pdf import generate_pdf_report
from tras.utils.report_pdf import generate_summary_pdf
from tras.utils.ring_properties import compute_ring_properties
from tras.utils.ring_properties import write_ring_properties_csv

# Broader than the CLI's {.jpg, .jpeg, .png}: the GUI also opens TIFF/BMP images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# progress(current_index, total, image_name)
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class BatchSummary:
    """Outcome of a batch run."""

    total: int
    processed: int
    skipped: int
    errors: int
    output_dir: Path
    summary_pdf: Path
    config_path: Path
    error_names: list[str] = field(default_factory=list)
    rwl_path: Path | None = None  # radii.rwl, written when dated and scaled


def find_images(input_dir: Path) -> list[Path]:
    """Return image files in ``input_dir`` (non-recursive), sorted by name."""
    return sorted(
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _scale_from_config(config: dict[str, Any]) -> tuple[float | None, str | None]:
    """Optional (value, unit) physical scale of the *processed* image's pixels.

    ``physical_scale`` refers to the original image; after resizing by ``resize_scale``
    every processed pixel covers ``1 / resize_scale`` original pixels.
    """
    scale = config.get("physical_scale")
    if not isinstance(scale, dict):
        return None, None
    try:
        value, unit = float(scale["value"]), str(scale["unit"])
    except (KeyError, TypeError, ValueError):
        return None, None
    resize = (config.get("preprocess") or {}).get("resize_scale")
    return (value / float(resize) if resize else value), unit


def _decode_embedded_image(image_data: bytes) -> npt.NDArray[np.uint8]:
    """Decode the base64-embedded image bytes stored in a label file into RGB array."""
    pil_img = PILImage.open(io.BytesIO(image_data)).convert("RGB")
    return np.array(pil_img, dtype=np.uint8)


def _load_result_from_json(
    json_path: Path,
) -> tuple[npt.NDArray[np.uint8], tuple[float, float], list[npt.NDArray[np.float32]]]:
    """Recover (image, pith_xy, rings) from an existing TRAS label file.

    Used to include already-processed (skipped) images in the summary PDF.
    """
    label = LabelFile(str(json_path))
    if label.imageData is None:
        raise ValueError(f"{json_path} has no embedded image")
    image = _decode_embedded_image(label.imageData)

    pith_xy: tuple[float, float] = (0.0, 0.0)
    rings: list[npt.NDArray[np.float32]] = []
    for shape in label.shapes:
        if shape["label"] == "pith" and shape["points"]:
            point = shape["points"][0]
            pith_xy = (float(point[0]), float(point[1]))
        elif shape["shape_type"] == "polygon":
            rings.append(np.array(shape["points"], dtype=np.float32))
    return image, pith_xy, rings


def _effective_config(config: dict[str, Any], params: DetectionParams) -> dict[str, Any]:
    """Config actually used: every detection parameter spelled out, plus the scale."""
    effective: dict[str, Any] = {}
    for key in ("physical_scale", "dating"):
        if key in config:
            effective[key] = config[key]
    effective.update(params.to_config())
    return effective


def write_radii_rwl(
    path: Path, results: list[dict[str, Any]], config: dict[str, Any]
) -> Path | None:
    """Tucson file with N/E/S/W radius series of every image, when dated and scaled.

    Needs ``dating.outer_year`` and a physical scale in ``config``; returns None otherwise.
    Series IDs are ``<name[:7]><N|E|S|W>``; if two names share their first 7 characters,
    IDs become ``R<index>`` and ``radii_ids.csv`` maps them back to image names.
    """
    outer_year = (config.get("dating") or {}).get("outer_year")
    mm_px = output_mm_per_pixel(config)
    usable = [r for r in results if len(r["rings"]) >= 2]
    if not outer_year or mm_px is None or not usable:
        return None
    names = [Path(r["name"]).stem for r in usable]
    stems = [sanitize_series_id(n)[:7] for n in names]
    if len(set(stems)) < len(stems):
        stems = [f"R{i:04d}" for i in range(1, len(usable) + 1)]
        id_map = path.with_name("radii_ids.csv")
        id_map.write_text(
            "series_stem,image\n" + "".join(f"{s},{n}\n" for s, n in zip(stems, names)),
            encoding="utf-8",
        )
    series = [
        s
        for r, name, stem in zip(usable, names, stems)
        for s in compass_series(
            r["rings"], r["pith_xy"], int(outer_year), mm_px, name, id_stem=stem
        )
    ]
    write_rwl(path, series)
    logger.info(f"Wrote {len(series)} radius series to {path}")
    return path


def output_mm_per_pixel(config: dict[str, Any]) -> float | None:
    """Millimetres per pixel of the *processed* image the rings are measured on.

    ``physical_scale`` refers to the original image; resizing by ``resize_scale`` makes
    each processed pixel ``1 / resize_scale`` times larger.
    """
    mm_px = mm_per_pixel(config.get("physical_scale"))
    if mm_px is None:
        return None
    resize = (config.get("preprocess") or {}).get("resize_scale")
    return mm_px / float(resize) if resize else mm_px


def write_image_reports(
    output_stem: Path,
    image: npt.NDArray[np.uint8],
    pith_xy: tuple[float, float],
    rings: list[npt.NDArray[np.float32]],
    scale_value: float | None,
    scale_unit: str | None,
) -> None:
    """Write ``<stem>.csv`` (ring properties) and ``<stem>.pdf`` (report) for one image."""
    if not rings:
        logger.warning(f"No rings detected for {output_stem.name}; skipping CSV/PDF")
        return
    value, unit = (scale_value, scale_unit) if scale_value and scale_unit else (1.0, "px")
    ring_props = compute_ring_properties(rings, pith_xy)
    write_ring_properties_csv(output_stem.with_suffix(".csv"), ring_props, value, unit)
    generate_pdf_report(
        output_path=output_stem.with_suffix(".pdf"),
        image=image,
        pith_xy=pith_xy,
        rings=rings,
        ring_properties=ring_props,
        scale_value=value,
        scale_unit=unit,
        sample_code=output_stem.name,
        harvested_year=None,
    )


def run_batch(
    input_dir: Path | str,
    output_dir: Path | str,
    config: dict[str, Any],
    progress: ProgressCallback | None = None,
    *,
    per_image_reports: bool = False,
) -> BatchSummary:
    """Detect rings for every image in a folder and write batch outputs.

    For each image: skip if its ``.json`` already exists (still included in the
    summary), otherwise run detection and write its ``.json``. Per-image failures are
    logged and counted but never abort the batch. After the loop, writes
    ``summary.pdf`` and ``batch_config.yml`` into ``output_dir``.

    With ``per_image_reports=True`` (the CLI folder mode) each newly processed image also
    gets a ``<stem>.csv`` of ring properties and a ``<stem>.pdf`` report. Without a
    physical scale in ``config`` those are written in pixel units.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = get_detection_params(config)
    scale_value, scale_unit = _scale_from_config(config)

    images = find_images(input_dir)
    total = len(images)
    logger.info(f"Batch: found {total} image(s) in {input_dir}")

    results: list[dict[str, Any]] = []
    processed = skipped = errors = 0
    error_names: list[str] = []

    for idx, image_path in enumerate(images, 1):
        name = image_path.name
        if progress is not None:
            progress(idx, total, name)

        json_path = output_dir / f"{image_path.stem}.json"
        try:
            if json_path.exists():
                logger.info(f"[{idx}/{total}] Skipping {name} (JSON already exists)")
                image, pith_xy, rings = _load_result_from_json(json_path)
                skipped += 1
            else:
                logger.info(f"[{idx}/{total}] Processing {name}")
                detection = detect(image_path=image_path, output=json_path, params=params)
                image, pith_xy, rings = (
                    detection.image,
                    detection.pith_xy,
                    detection.rings,
                )
                processed += 1
                if per_image_reports:
                    write_image_reports(
                        output_dir / image_path.stem, image, pith_xy, rings, scale_value, scale_unit
                    )
            results.append({"name": name, "image": image, "pith_xy": pith_xy, "rings": rings})
        except Exception as exc:
            logger.error(f"[{idx}/{total}] Failed to process {name}: {exc}")
            errors += 1
            error_names.append(name)

    summary_pdf = output_dir / "summary.pdf"
    if results:
        generate_summary_pdf(summary_pdf, results, scale_value, scale_unit)
    else:
        logger.warning("Batch produced no results; skipping summary PDF")

    config_path = output_dir / "batch_config.yml"
    dump_config(_effective_config(config, params), config_path)
    rwl_path = write_radii_rwl(output_dir / "radii.rwl", results, config)

    logger.info(f"Batch complete: {processed} processed, {skipped} skipped, {errors} error(s)")
    return BatchSummary(
        total=total,
        processed=processed,
        skipped=skipped,
        errors=errors,
        output_dir=output_dir,
        summary_pdf=summary_pdf,
        config_path=config_path,
        error_names=error_names,
        rwl_path=rwl_path,
    )
