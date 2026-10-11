"""One-call core analysis: find cores (or use given paths) and detect their rings."""

from __future__ import annotations

import dataclasses
from typing import Optional
from typing import Sequence

import numpy as np
import numpy.typing as npt
from loguru import logger

from tras.cores.axis import find_cores
from tras.cores.detect import CoreParams
from tras.cores.detect import CoreResult
from tras.cores.detect import detect_core_rings


def deep_model_available(model_id: str = "generic") -> bool:
    """True if torch, segmentation_models_pytorch and the model weights are present."""
    try:
        import segmentation_models_pytorch  # noqa: F401
        import torch  # noqa: F401

        from tras.utils.deepcstrd_helper import _get_model_path

        _get_model_path(model_id, tile_size=0)
        return True
    except Exception:
        return False


def default_core_params(**overrides) -> CoreParams:
    """Best available parameters: learned evidence when its model is installed.

    Defaults chosen on the UruDendro virtual-core tuning split and confirmed on held-out
    CooRecorder measurements (``docs/benchmarks.md``).
    """
    params = CoreParams(**overrides)
    if "evidence" not in overrides and deep_model_available(params.model):
        params = dataclasses.replace(params, evidence="deep")
    return params


def analyze_core_image(
    image: npt.NDArray,
    *,
    paths: Optional[Sequence[npt.ArrayLike]] = None,
    params: Optional[CoreParams] = None,
    mm_per_px: Optional[float] = None,
    outer_year: Optional[int] = None,
) -> list[CoreResult]:
    """Detect ring boundaries on every core in ``image``.

    Args:
        image: RGB or grayscale image of one or more cores (or a cross-section).
        paths: Measurement paths (polylines, px). ``None`` finds the cores
            automatically and uses their centrelines.
        params: Detector parameters (default ``default_core_params()``).
        mm_per_px: Pixel size for millimetre widths and exports.
        outer_year: Calendar year of the outermost ring (same for every core).
    """
    params = params or default_core_params()
    if paths is None:
        found = find_cores(image)
        logger.info("Found {} core(s)", len(found))
        # Stay inside the core: never wider than ~30 % of its diameter on each side.
        jobs = [
            (core.path_xy, dataclasses.replace(params, half_width=min(params.half_width, core.half_width)))
            for core in found
        ]
    else:
        jobs = [(np.asarray(p, dtype=np.float64), params) for p in paths]
    return [
        detect_core_rings(image, path, job_params, mm_per_px=mm_per_px, outer_year=outer_year)
        for path, job_params in jobs
    ]
