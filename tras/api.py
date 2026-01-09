from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from loguru import logger
from PIL import Image as PILImage

from tras._label_file import LabelFile
from tras.utils import img_arr_to_b64
from tras.utils.apd_helper import detect_pith_apd
from tras.utils.cstrd_helper import detect_rings_cstrd
from tras.utils.deepcstrd_helper import detect_rings_deepcstrd
from tras.utils.inbd_helper import detect_rings_inbd
from tras.utils.preprocess_helper import preprocess_image
from tras.utils.ring_sampling import resample_rings_by_rays


@dataclass
class PreprocessingInfo:
    scale_factor: float
    remove_background: bool
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


@dataclass
class DetectionResult:
    image: npt.NDArray[np.uint8]
    pith_xy: tuple[float, float]
    rings: list[npt.NDArray[np.float32]]
    preprocessing: Optional[PreprocessingInfo]
    output_path: Optional[Path] = None


def _load_image(image_path: Path) -> npt.NDArray[np.uint8]:
    """Load image from disk using PIL with RGB conversion."""
    try:
        pil_img = PILImage.open(image_path)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - PIL error surfaces to caller
        logger.error(f"Failed to load image from %s: %s", image_path, exc)
        raise


def _save_detection_results(
    output_path: Path,
    image: npt.NDArray[np.uint8],
    pith_xy: tuple[float, float],
    rings: list[npt.NDArray[np.float32]],
    preprocessing_info: Optional[dict] = None,
) -> None:
    """Serialize detection results to TRAS LabelFile JSON format."""
    shapes = []
    for ring_idx, ring in enumerate(rings):
        points = [[float(p[0]), float(p[1])] for p in ring]
        shapes.append(
            {
                "label": f"ring_{ring_idx + 1}",
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "description": "",
            }
        )

    shapes.append(
        {
            "label": "pith",
            "points": [[float(pith_xy[0]), float(pith_xy[1])]],
            "group_id": None,
            "shape_type": "point",
            "flags": {},
            "description": "",
        }
    )

    label_file = LabelFile(
        filename=str(output_path),
        imagePath=str(output_path.parent / output_path.stem),
        imageData=img_arr_to_b64(image),
        shapes=shapes,
        flags={},
        imageHeight=image.shape[0],
        imageWidth=image.shape[1],
    )

    if preprocessing_info:
        label_file.flags["preprocessing"] = json.dumps(preprocessing_info)

    label_file.save(str(output_path))
    logger.info("Saved detection results to %s", output_path)


def detect(
    image_path: Path | str,
    *,
    output: Optional[Path] = None,
    pith_x: Optional[float] = None,
    pith_y: Optional[float] = None,
    auto_pith: bool = True,
    pith_method: str = "apd_dl",
    ring_method: str = "deepcstrd",
    scale: Optional[float] = None,
    remove_background: bool = False,
    sampling_nr: int = 360,
    cstrd_sigma: float = 3.0,
    cstrd_th_low: float = 5.0,
    cstrd_th_high: float = 20.0,
    cstrd_alpha: int = 30,
    cstrd_nr: int = 360,
    deepcstrd_model: str = "generic",
    deepcstrd_tile_size: int = 0,
    deepcstrd_alpha: int = 45,
    deepcstrd_nr: int = 360,
    deepcstrd_rotations: int = 5,
    deepcstrd_threshold: float = 0.5,
    inbd_model: str = "INBD_EH",
    inbd_auto_pith: bool = True,
) -> DetectionResult:
    """
    Perform pith and ring detection for an input image.

    This mirrors the CLI workflow but exposes a pure Python API.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    logger.info("Loading image from %s", path)
    image = _load_image(path)
    original_shape = image.shape
    preprocessing_meta: Optional[PreprocessingInfo] = None

    if scale is not None or remove_background:
        logger.info("Applying preprocessing (scale=%s, remove_background=%s)", scale, remove_background)
        processed = preprocess_image(
            image,
            scale_factor=scale or 1.0,
            crop_rect=None,
            remove_background=remove_background,
        )
        preprocessing_meta = PreprocessingInfo(
            scale_factor=scale or 1.0,
            remove_background=remove_background,
            original_size=(int(original_shape[1]), int(original_shape[0])),
            processed_size=(int(processed.shape[1]), int(processed.shape[0])),
        )
        image = processed

    if pith_x is not None and pith_y is not None:
        pith_xy = (float(pith_x), float(pith_y))
        logger.info("Using manual pith coordinates: (%.1f, %.1f)", pith_xy[0], pith_xy[1])
    elif auto_pith:
        logger.info("Detecting pith using %s", pith_method)
        pith_xy = detect_pith_apd(image, method=pith_method)
    else:
        raise ValueError("Pith coordinates required. Provide --pith-x/--pith-y or enable auto detection.")

    logger.info("Detecting rings via %s", ring_method)
    pith_for_sampling = pith_xy
    
    if ring_method == "cstrd":
        rings = detect_rings_cstrd(
            image,
            pith_xy,
            sigma=cstrd_sigma,
            th_low=cstrd_th_low,
            th_high=cstrd_th_high,
            alpha=cstrd_alpha,
            nr=cstrd_nr,
        )
    elif ring_method == "deepcstrd":
        rings = detect_rings_deepcstrd(
            image,
            pith_xy,
            model_id=deepcstrd_model,
            tile_size=deepcstrd_tile_size,
            alpha=deepcstrd_alpha,
            nr=deepcstrd_nr,
            total_rotations=deepcstrd_rotations,
            prediction_map_threshold=deepcstrd_threshold,
        )
    elif ring_method == "inbd":
        if inbd_auto_pith:
            rings, pith_for_sampling = detect_rings_inbd(
                image,
                center_xy=None,
                model_id=inbd_model,
                return_pith=True,
            )
        else:
            rings, pith_for_sampling = detect_rings_inbd(
                image,
                center_xy=pith_xy,
                model_id=inbd_model,
                return_pith=True,
            )
    else:
        raise ValueError(f"Unknown ring detection method: {ring_method}")

    logger.info("Detected %d rings", len(rings))
    
    # Apply postprocess resampling
    logger.info("Resampling rings to %d points using pith (%.1f, %.1f)", sampling_nr, pith_for_sampling[0], pith_for_sampling[1])
    rings = resample_rings_by_rays(rings, pith_for_sampling, sampling_nr)
    logger.info("Resampled to %d rings", len(rings))

    preprocessing_payload = None
    if preprocessing_meta:
        preprocessing_payload = {
            "scale_factor": preprocessing_meta.scale_factor,
            "remove_background": preprocessing_meta.remove_background,
            "original_size": list(preprocessing_meta.original_size),
            "processed_size": list(preprocessing_meta.processed_size),
        }

    if output is not None:
        _save_detection_results(Path(output), image, pith_for_sampling, rings, preprocessing_payload)

    return DetectionResult(
        image=image,
        pith_xy=pith_for_sampling,
        rings=rings,
        preprocessing=preprocessing_meta,
        output_path=Path(output) if output is not None else None,
    )
