"""Programmatic entry point for cross-section pith and ring detection.

``detect()`` runs the whole pipeline on one image::

    load → preprocess → locate pith → detect rings → resample rings → (save)

Parameters come from a ``DetectionParams`` (see ``tras/params.py``). For backward
compatibility, the legacy flat keywords (``ring_method="cstrd"``, ``cstrd_sigma=2.5``, ...)
are still accepted and override the given or default parameters.

Detector back-ends import torch and the method packages, so they are imported lazily
inside the step functions: ``import tras.api`` stays fast and works without a GPU stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from PIL import Image as PILImage

from tras._label_file import LabelFile
from tras.params import DetectionParams
from tras.utils import img_arr_to_data
from tras.utils.ring_sampling import resample_rings_by_rays

RingList = list[npt.NDArray[np.float32]]


@dataclass
class PreprocessingInfo:
    scale_factor: float
    remove_background: bool
    original_size: tuple[int, int]
    processed_size: tuple[int, int]

    def to_flag_payload(self) -> dict[str, Any]:
        return {
            "scale_factor": self.scale_factor,
            "remove_background": self.remove_background,
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size),
        }


@dataclass
class DetectionResult:
    image: npt.NDArray[np.uint8]
    pith_xy: tuple[float, float]
    rings: RingList
    preprocessing: PreprocessingInfo | None
    output_path: Path | None = None
    params: DetectionParams | None = None


def detect(
    image_path: Path | str,
    *,
    output: Path | str | None = None,
    params: DetectionParams | None = None,
    **flat_overrides: Any,
) -> DetectionResult:
    """Detect the pith and the annual rings of one cross-section image.

    Args:
        image_path: Image file (any format PIL reads).
        output: If given, write a TRAS label file (JSON) with rings and pith here.
        params: Full parameter set. Defaults to ``DetectionParams()``.
        **flat_overrides: Legacy flat keywords (``ring_method``, ``cstrd_sigma``,
            ``pith_x``, ...). See ``tras.params.flat_parameter_names()``.

    Returns:
        ``DetectionResult`` with the (preprocessed) image, pith and rings. Each ring is an
        ``(sampling_nr, 2)`` array of ``(x, y)`` points ordered by angle around the pith.
    """
    params = (params or DetectionParams()).with_flat_overrides(**flat_overrides)
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image, preprocessing = preprocess(load_image(path), params)
    pith_xy = locate_pith(image, params)
    rings, pith_xy = detect_rings(image, pith_xy, params)
    logger.info("Detected {} rings; resampling to {} points", len(rings), params.sampling_nr)
    rings = resample_rings_by_rays(rings, pith_xy, params.sampling_nr)

    output_path = Path(output) if output is not None else None
    if output_path is not None:
        payload = preprocessing.to_flag_payload() if preprocessing else None
        _save_detection_results(output_path, image, pith_xy, rings, payload)

    return DetectionResult(
        image=image,
        pith_xy=pith_xy,
        rings=rings,
        preprocessing=preprocessing,
        output_path=output_path,
        params=params,
    )


def load_image(image_path: Path) -> npt.NDArray[np.uint8]:
    """Load an image from disk as an RGB uint8 array."""
    logger.info("Loading image from {}", image_path)
    with PILImage.open(image_path) as pil_img:
        return np.array(pil_img.convert("RGB"), dtype=np.uint8)


# Kept for callers that used the private name.
_load_image = load_image


def preprocess(
    image: npt.NDArray[np.uint8], params: DetectionParams
) -> tuple[npt.NDArray[np.uint8], PreprocessingInfo | None]:
    """Resize and/or remove the background when requested; otherwise return as-is."""
    if params.resize_scale is None and not params.remove_background:
        return image, None

    from tras.utils.preprocess_helper import preprocess_image

    scale = params.resize_scale or 1.0
    logger.info("Preprocessing (scale={}, remove_background={})", scale, params.remove_background)
    processed = preprocess_image(
        image,
        scale_factor=scale,
        crop_rect=None,
        remove_background=params.remove_background,
    )
    info = PreprocessingInfo(
        scale_factor=scale,
        remove_background=params.remove_background,
        original_size=(int(image.shape[1]), int(image.shape[0])),
        processed_size=(int(processed.shape[1]), int(processed.shape[0])),
    )
    return processed, info


def locate_pith(
    image: npt.NDArray[np.uint8], params: DetectionParams
) -> tuple[float, float] | None:
    """Return the pith from manual coordinates or APD.

    Returns ``None`` when INBD will locate the pith itself, so APD is not run for nothing.
    """
    if params.pith.manual_xy is not None:
        logger.info("Using manual pith {}", params.pith.manual_xy)
        return params.pith.manual_xy
    if params.ring_method == "inbd" and params.inbd.auto_pith:
        return None
    if not params.pith.auto:
        raise ValueError(
            "Pith coordinates required. Provide pith_x/pith_y or enable auto pith detection."
        )

    from tras.utils.apd_helper import detect_pith_apd

    logger.info("Detecting pith with {}", params.pith.method)
    return detect_pith_apd(image, method=params.pith.method)


def detect_rings(
    image: npt.NDArray[np.uint8],
    pith_xy: tuple[float, float] | None,
    params: DetectionParams,
) -> tuple[RingList, tuple[float, float]]:
    """Run the selected ring detector. Returns ``(rings, pith_xy)``.

    The pith is returned because INBD may refine or detect it.
    """
    logger.info("Detecting rings with {}", params.ring_method)
    if params.ring_method == "inbd":
        return _detect_rings_inbd(image, pith_xy, params)
    if pith_xy is None:
        raise ValueError(f"{params.ring_method} needs a pith location.")
    if params.ring_method == "cstrd":
        return _detect_rings_cstrd(image, pith_xy, params), pith_xy
    return _detect_rings_deepcstrd(image, pith_xy, params), pith_xy


def _detect_rings_cstrd(
    image: npt.NDArray[np.uint8], pith_xy: tuple[float, float], params: DetectionParams
) -> RingList:
    from tras.utils.cstrd_helper import detect_rings_cstrd

    p = params.cstrd
    return detect_rings_cstrd(
        image,
        pith_xy,
        sigma=p.sigma,
        th_low=p.th_low,
        th_high=p.th_high,
        alpha=p.alpha,
        nr=p.nr,
    )


def _detect_rings_deepcstrd(
    image: npt.NDArray[np.uint8], pith_xy: tuple[float, float], params: DetectionParams
) -> RingList:
    from tras.utils.deepcstrd_helper import detect_rings_deepcstrd

    p = params.deepcstrd
    return detect_rings_deepcstrd(
        image,
        pith_xy,
        model_id=p.model,
        tile_size=p.tile_size,
        alpha=p.alpha,
        nr=p.nr,
        total_rotations=p.rotations,
        prediction_map_threshold=p.threshold,
        width=p.width,
        height=p.height,
    )


def _detect_rings_inbd(
    image: npt.NDArray[np.uint8],
    pith_xy: tuple[float, float] | None,
    params: DetectionParams,
) -> tuple[RingList, tuple[float, float]]:
    from tras.utils.inbd_helper import detect_rings_inbd

    center = None if params.inbd.auto_pith else pith_xy
    rings, pith = detect_rings_inbd(  # type: ignore[misc]  # return_pith=True → tuple
        image, center_xy=center, model_id=params.inbd.model, return_pith=True
    )
    return list(rings), (float(pith[0]), float(pith[1]))


def build_label_shapes(pith_xy: tuple[float, float], rings: RingList) -> list[dict[str, Any]]:
    """Convert rings and pith to labelme-compatible shape dicts."""
    shapes: list[dict[str, Any]] = [
        _shape_dict(f"ring_{idx}", [[float(x), float(y)] for x, y in ring], "polygon")
        for idx, ring in enumerate(rings, start=1)
    ]
    shapes.append(_shape_dict("pith", [[float(pith_xy[0]), float(pith_xy[1])]], "point"))
    return shapes


def _shape_dict(label: str, points: list[list[float]], shape_type: str) -> dict[str, Any]:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": shape_type,
        "flags": {},
        "description": "",
    }


def _save_detection_results(
    output_path: Path,
    image: npt.NDArray[np.uint8],
    pith_xy: tuple[float, float],
    rings: RingList,
    preprocessing_info: dict | None = None,
) -> None:
    """Serialize detection results to the TRAS label file (JSON) format."""
    flags = {}
    if preprocessing_info:
        flags["preprocessing"] = json.dumps(preprocessing_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_file = LabelFile(filename=None, enforce_image_data=False)
    label_file.save(
        filename=str(output_path),
        shapes=build_label_shapes(pith_xy, rings),
        imagePath=str(output_path.parent / output_path.stem),
        imageHeight=image.shape[0],
        imageWidth=image.shape[1],
        imageData=img_arr_to_data(image),
        flags=flags,
    )
    logger.info("Saved detection results to {}", output_path)
