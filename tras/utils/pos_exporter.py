"""Export a radial ring-width measurement to CooRecorder ``.pos`` (see ``tras.io.pos``)."""

from __future__ import annotations

from typing import Any
from typing import Optional

from loguru import logger

from tras.io.pos import MM_PER_INCH
from tras.io.pos import PosFile
from tras.io.pos import PosPoint
from tras.io.pos import write_pos

# Physical units the GUI scale dialog offers, in millimetres.
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "um": 1e-3, "µm": 1e-3, "m": 1000.0, "in": MM_PER_INCH}


def mm_per_pixel(image_scale: Optional[dict[str, Any]]) -> Optional[float]:
    """Millimetres per pixel from a GUI scale dict ``{'value': unit/px, 'unit': str}``."""
    if not image_scale or not image_scale.get("value"):
        return None
    factor = UNIT_TO_MM.get(str(image_scale.get("unit", "mm")))
    return None if factor is None else float(image_scale["value"]) * factor


def export_to_pos(
    filepath,
    measurements: dict[str, dict[str, Any]],
    pith_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    image_scale: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    image_file: Optional[str] = None,
) -> bool:
    """Write the boundary crossings of one radial line as a CooRecorder ``.pos`` file.

    Args:
        measurements: ``{label: {'intersection_point': (x, y), 'distance_from_pith': px}}``
            as produced by ``tras.utils.ring_width_measurer``.
        pith_xy: Pith in pixels.
        direction_xy: A point on the measurement line (kept as a comment).
        image_scale: GUI scale dict. Without it 1 px is written as 1 mm (DPI 25.4).
        metadata: Optional ``sample_code`` and ``harvested_year`` (year of outer ring).

    Returns:
        True on success.
    """
    try:
        pos = build_radial_pos(
            measurements, pith_xy, direction_xy, image_scale, metadata or {}, image_file
        )
        write_pos(filepath, pos)
        return True
    except Exception as exc:  # surfaced to the user by the GUI
        logger.exception(f"Error exporting to .POS format: {exc}")
        return False


def build_radial_pos(
    measurements: dict[str, dict[str, Any]],
    pith_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    image_scale: Optional[dict[str, Any]],
    metadata: dict[str, Any],
    image_file: Optional[str] = None,
) -> PosFile:
    """Build a bark-first ``PosFile`` from radial-line boundary crossings."""
    mm_px = mm_per_pixel(image_scale)
    if mm_px is None:
        logger.warning(".POS export without a physical scale: writing 1 px = 1 mm")
        mm_px = 1.0

    crossings = sorted(
        measurements.values(), key=lambda m: m["distance_from_pith"], reverse=True
    )
    boundaries = [
        PosPoint(m["intersection_point"][0] * mm_px, m["intersection_point"][1] * mm_px)
        for m in crossings
    ]
    # The line starts at the pith, so the pith closes the innermost ring.
    points = tuple(boundaries + [PosPoint(pith_xy[0] * mm_px, pith_xy[1] * mm_px, label="pith")])
    comments = [f"TRAS radial line from pith towards ({direction_xy[0]:.1f}, {direction_xy[1]:.1f}) px"]
    if metadata.get("sample_code"):
        comments.insert(0, str(metadata["sample_code"]))

    harvested = metadata.get("harvested_year")
    return PosFile(
        points=points,
        dpi=MM_PER_INCH / mm_px,
        image_file=image_file or metadata.get("image_file"),
        dated_year=int(harvested) if harvested else None,
        pith_xy_mm=(pith_xy[0] * mm_px, pith_xy[1] * mm_px),
        distance_to_pith_mm=0.0,
        years_to_pith=0,
        comments=tuple(comments),
    )
