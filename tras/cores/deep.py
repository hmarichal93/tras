"""Learned boundary evidence: the DeepCS-TRD U-Net run on a core ribbon.

The U-Net (``segmentation_models_pytorch`` U-Net, ResNet-18 encoder) was trained on
cross-sections resized to 1504 px to output a per-pixel ring-boundary probability.
Locally a boundary looks the same on a core, so the same weights are reused. The ribbon
is resized by ``scale`` so rings have the pixel size the model saw in training, then the
probability map is resized back to the ribbon grid.

torch and the model package are imported lazily so ``tras.cores`` works without them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from tras.cores.ribbon import Ribbon

# The U-Net downsamples 5 times: input sides must be multiples of 32.
_UNET_MULTIPLE = 32
# Background value used to pad the ribbon (white, as in DeepCS-TRD preprocessing).
_PAD_VALUE = 255
# Images were resized to this side before training.
TRAINING_SIDE_PX = 1504


def training_scale(image_shape: tuple[int, ...]) -> float:
    """Resize factor DeepCS-TRD would apply to a cross-section of this size."""
    return TRAINING_SIDE_PX / max(image_shape[:2])


def boundary_probability(ribbon: Ribbon, model_id: str = "generic", scale: float = 1.0) -> npt.NDArray[np.float64]:
    """Per-pixel boundary probability on the ribbon grid (same shape as the ribbon)."""
    rgb = np.clip(ribbon.values[..., :3], 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)  # DeepCS-TRD was trained on BGR input
    padded, (ph, pw) = _pad_to_multiple(bgr, _UNET_MULTIPLE)

    probability = _predict(padded, model_id)[:ph, :pw]
    return cv2.resize(probability.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)


def boundary_probability_along_path(
    image: npt.NDArray, path_xy: npt.ArrayLike, half_width: int, model_id: str = "generic",
    scale: float = 1.0, context_px: int = 96,
) -> Ribbon:
    """Run the U-Net on the image crop around the path, then straighten the result.

    Unlike ``boundary_probability`` on a ribbon, the network sees the wood in its
    natural geometry (curved rings, surrounding context), as in training.

    Returns a single-channel ``Ribbon`` of probabilities.
    """
    from tras.cores.ribbon import extract_ribbon

    path = np.asarray(path_xy, dtype=np.float64)
    margin = half_width + int(round(context_px / scale))
    h, w = image.shape[:2]
    x0, y0 = np.maximum(np.floor(path.min(axis=0)) - margin, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(path.max(axis=0)) + margin + 1, [w, h]).astype(int)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1, :3]).astype(np.uint8)

    ch, cw = crop.shape[:2]
    resized = cv2.resize(crop, (max(1, round(cw * scale)), max(1, round(ch * scale))), interpolation=cv2.INTER_AREA)
    padded, (ph, pw) = _pad_to_multiple(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), _UNET_MULTIPLE)
    probability = _predict(padded, model_id)[:ph, :pw]
    probability = cv2.resize(probability.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR)
    return extract_ribbon(probability, path - [x0, y0], half_width)


def _pad_to_multiple(image: npt.NDArray[np.uint8], multiple: int) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
    h, w = image.shape[:2]
    H = int(np.ceil(h / multiple) * multiple)
    W = int(np.ceil(w / multiple) * multiple)
    padded = np.full((H, W, 3), _PAD_VALUE, dtype=np.uint8)
    padded[:h, :w] = image
    return padded, (h, w)


def _predict(image_bgr: npt.NDArray[np.uint8], model_id: str) -> npt.NDArray[np.float32]:
    import torch

    model, device = _load_model(model_id)
    tensor = torch.from_numpy(image_bgr).permute(2, 0, 1)[None].float().to(device) / 255.0
    with torch.no_grad():
        logits = model(tensor)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


@lru_cache(maxsize=2)
def _load_model(model_id: str) -> tuple[Any, Any]:
    import segmentation_models_pytorch as smp
    import torch

    from tras.utils.deepcstrd_helper import _get_model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    state = torch.load(_get_model_path(model_id, tile_size=0), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval(), device
