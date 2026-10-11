"""``tras_core``: automatic ring detection on increment-core images.

For each image: find the cores (or use ``--path`` / ``--path-from-pos``), detect ring
boundaries, and write per core a CooRecorder ``.pos`` and a ``.csv``, plus per image a
TRAS label file (``.json``, editable in the GUI) and, when dated, a Tucson ``.rwl``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from loguru import logger
from PIL import Image

from tras.cores import CoreResult
from tras.cores import analyze_core_image
from tras.cores import default_core_params
from tras.cores import export_results
from tras.cores import to_label_shapes
from tras.io.pos import MM_PER_INCH
from tras.io.pos import RING
from tras.io.pos import read_pos
from tras.utils.batch_helper import IMAGE_EXTENSIONS

app = typer.Typer(name="core", help="Detect tree rings on increment-core images")

# Gap length (× neighbouring ring widths) searched by --fill-gaps (docs/benchmarks.md).
GAP_FILL_RATIO = 2.2


def mm_per_px_from_dpi(image_path: Path) -> Optional[float]:
    """Pixel size from the DPI scanners store in TIFF/JPEG/PNG metadata."""
    with Image.open(image_path) as img:
        dpi = img.info.get("dpi")
    if not dpi:
        return None
    value = float(dpi[0])
    # 72/96 dpi are display defaults, not scan resolutions.
    return MM_PER_INCH / value if value > 150 else None


def parse_path(text: str) -> np.ndarray:
    """``"x1,y1;x2,y2;..."`` → ``(N, 2)`` array."""
    points = [tuple(float(v) for v in pair.split(",")) for pair in text.split(";") if pair.strip()]
    if len(points) < 2:
        raise typer.BadParameter("A path needs at least two 'x,y' points separated by ';'.")
    return np.array(points, dtype=np.float64)


def path_from_pos(pos_file: Path) -> np.ndarray:
    """Re-use the path of an existing CooRecorder measurement (in pixels)."""
    pos = read_pos(pos_file)
    if not pos.dpi:
        raise typer.BadParameter(f"{pos_file} has no DPI; cannot convert mm to pixels.")
    return np.array([pos.to_pixels((p.x, p.y)) for p in pos.points if p.kind == RING])


def write_outputs(
    image_path: Path,
    image: np.ndarray,
    results: list[CoreResult],
    output_dir: Path,
) -> None:
    """Write CSV/.pos per core, one .rwl if dated, and an editable TRAS label file."""
    from tras._label_file import LabelFile
    from tras.utils import img_arr_to_data

    export_results(results, output_dir, image_path.stem, image_file=image_path.name)
    shapes = []
    for k, result in enumerate(results, start=1):
        shapes += to_label_shapes(result, core_index=k)
        n_boundaries, n_rings = len(result.boundaries), len(result.widths_px)
        logger.info(f"{image_path.stem}_core{k}: {n_boundaries} boundaries, {n_rings} rings")
    LabelFile(filename=None, enforce_image_data=False).save(
        filename=str(output_dir / f"{image_path.stem}.json"),
        shapes=shapes,
        imagePath=image_path.name,
        imageHeight=image.shape[0],
        imageWidth=image.shape[1],
        imageData=img_arr_to_data(image),
        flags={},
    )


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Core image or folder of core images"),
    output_dir: Optional[Path] = typer.Option(
        None, "-o", "--output-dir", help="Default: <input folder>/tras_cores"
    ),
    path: Optional[str] = typer.Option(
        None, "--path", help="Measurement path 'x1,y1;x2,y2;...' (px)"
    ),
    path_from_pos_file: Optional[Path] = typer.Option(
        None, "--path-from-pos", help="Use the path of a CooRecorder .pos"
    ),
    dpi: Optional[float] = typer.Option(
        None, "--dpi", help="Scan resolution (default: image metadata)"
    ),
    mm_per_px: Optional[float] = typer.Option(
        None, "--mm-per-px", help="Pixel size in mm (overrides --dpi)"
    ),
    outer_year: Optional[int] = typer.Option(
        None, "--outer-year", help="Year of the outermost ring"
    ),
    pith_side: str = typer.Option("auto", "--pith-side", help="auto | start | end of the path"),
    evidence: Optional[str] = typer.Option(
        None, "--evidence", help="classical | deep | hybrid (default: best available)"
    ),
    half_width: Optional[int] = typer.Option(None, "--half-width", help="Ribbon half-width (px)"),
    min_ring_width: Optional[float] = typer.Option(
        None, "--min-ring-width", help="Minimum ring width (px)"
    ),
    sensitivity: Optional[float] = typer.Option(
        None, "--sensitivity", help="Classical threshold (noise units)"
    ),
    deep_threshold: Optional[float] = typer.Option(
        None, "--deep-threshold", help="Deep probability threshold"
    ),
    fill_gaps: bool = typer.Option(
        False,
        "--fill-gaps",
        help="Recover rings the deep model missed from ring-width continuity "
        "(closer ring counts, slightly lower precision)",
    ),
) -> None:
    """Detect ring boundaries on increment cores and export CooRecorder/Tucson files.

    Examples:
        tras_core scan.tif --outer-year 2023            # find cores automatically
        tras_core scan.tif --path "120,40;3900,85"       # measure along a given path
        tras_core cores/ --dpi 2400 --outer-year 2023    # whole folder
    """
    overrides = {
        "pith_side": pith_side,
        "evidence": evidence,
        "half_width": half_width,
        "min_ring_width": min_ring_width,
        "sensitivity": sensitivity,
        "deep_threshold": deep_threshold,
        "gap_fill_ratio": GAP_FILL_RATIO if fill_gaps else None,
    }
    params = default_core_params(**{k: v for k, v in overrides.items() if v is not None})
    paths = None
    if path:
        paths = [parse_path(path)]
    elif path_from_pos_file:
        paths = [path_from_pos(path_from_pos_file)]

    images = _collect_images(input_path)
    out = output_dir or (input_path if input_path.is_dir() else input_path.parent) / "tras_cores"
    for image_path in images:
        scale = mm_per_px or (MM_PER_INCH / dpi if dpi else mm_per_px_from_dpi(image_path))
        if scale is None:
            logger.warning(
                f"{image_path.name}: no scale (--dpi/--mm-per-px); widths in px only, no .pos"
            )
        image = np.array(Image.open(image_path).convert("RGB"))
        results = analyze_core_image(
            image, paths=paths, params=params, mm_per_px=scale, outer_year=outer_year
        )
        write_outputs(image_path, image, results, out)
    print(f"Results written to {out}")


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if input_path.is_file():
        return [input_path]
    raise typer.BadParameter(f"Input not found: {input_path}")


if __name__ == "__main__":
    app()
