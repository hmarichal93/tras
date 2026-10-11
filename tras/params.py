"""Typed detection parameters shared by the API, CLI, batch runner, and GUI.

Every default comes from ``tras/config/detection_defaults.yaml`` (via
``get_detection_defaults``), so no entry point can drift from the others.

Two equivalent spellings are supported:

* nested (``DetectionParams.cstrd.sigma``): the canonical form used in code;
* flat (``cstrd_sigma``): the legacy keyword form of ``tras.api.detect`` and the CLI flags.

``DetectionParams.with_flat_overrides`` converts the flat form into the nested one, and
``DetectionParams.from_config`` reads the YAML schema of ``examples/cli/process_config.yml``.
This module is Qt- and torch-free so it can be imported anywhere.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from tras.config.detection_defaults import get_detection_defaults

RING_METHODS = ("cstrd", "deepcstrd", "inbd")
PITH_METHODS = ("apd", "apd_pcl", "apd_dl")


def _section(name: str) -> dict[str, Any]:
    return get_detection_defaults()[name]


@dataclass(frozen=True)
class CstrdParams:
    sigma: float = field(default_factory=lambda: float(_section("cstrd")["sigma"]))
    th_low: float = field(default_factory=lambda: float(_section("cstrd")["th_low"]))
    th_high: float = field(default_factory=lambda: float(_section("cstrd")["th_high"]))
    alpha: int = field(default_factory=lambda: int(_section("cstrd")["alpha"]))
    nr: int = field(default_factory=lambda: int(_section("cstrd")["nr"]))
    width: int = field(default_factory=lambda: int(_section("cstrd")["width"]))
    height: int = field(default_factory=lambda: int(_section("cstrd")["height"]))


@dataclass(frozen=True)
class DeepCstrdParams:
    model: str = field(default_factory=lambda: str(_section("deepcstrd")["model"]))
    tile_size: int = field(default_factory=lambda: int(_section("deepcstrd")["tile_size"]))
    alpha: int = field(default_factory=lambda: int(_section("deepcstrd")["alpha"]))
    nr: int = field(default_factory=lambda: int(_section("deepcstrd")["nr"]))
    rotations: int = field(default_factory=lambda: int(_section("deepcstrd")["rotations"]))
    threshold: float = field(default_factory=lambda: float(_section("deepcstrd")["threshold"]))
    width: int = field(default_factory=lambda: int(_section("deepcstrd")["width"]))
    height: int = field(default_factory=lambda: int(_section("deepcstrd")["height"]))


@dataclass(frozen=True)
class InbdParams:
    model: str = field(default_factory=lambda: str(_section("inbd")["model"]))
    auto_pith: bool = field(default_factory=lambda: bool(_section("inbd")["auto_pith"]))


@dataclass(frozen=True)
class PithParams:
    auto: bool = field(default_factory=lambda: bool(_section("pith")["auto"]))
    method: str = field(default_factory=lambda: str(_section("pith")["method"]))
    x: float | None = None
    y: float | None = None

    @property
    def manual_xy(self) -> tuple[float, float] | None:
        """Manual pith coordinates, or ``None`` if not both given."""
        if self.x is None or self.y is None:
            return None
        return (float(self.x), float(self.y))


@dataclass(frozen=True)
class DetectionParams:
    """Complete parameter set for one cross-section detection run."""

    ring_method: str = field(default_factory=lambda: str(get_detection_defaults()["method"]))
    pith: PithParams = field(default_factory=PithParams)
    cstrd: CstrdParams = field(default_factory=CstrdParams)
    deepcstrd: DeepCstrdParams = field(default_factory=DeepCstrdParams)
    inbd: InbdParams = field(default_factory=InbdParams)
    sampling_nr: int = field(
        default_factory=lambda: int(get_detection_defaults()["sampling"]["nr"])
    )
    resize_scale: float | None = None
    remove_background: bool = False

    def __post_init__(self) -> None:
        validate_params(self)

    def with_flat_overrides(self, **flat: Any) -> DetectionParams:
        """Return a copy with legacy flat keyword overrides applied.

        ``None`` values are ignored, so CLI options that were not given keep the default.
        Unknown keys raise ``TypeError`` (a typo must not be silently dropped).
        """
        return _apply_flat_overrides(self, {k: v for k, v in flat.items() if v is not None})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DetectionParams:
        """Build parameters from the YAML config schema (``process_config.yml``)."""
        return _params_from_config(cls(), config)

    def to_config(self) -> dict[str, Any]:
        """Serialize to the YAML config schema (round-trips with ``from_config``)."""
        return _params_to_config(self)


def validate_params(params: DetectionParams) -> None:
    """Raise ``ValueError`` on values no detector can run with."""
    if params.ring_method not in RING_METHODS:
        raise ValueError(
            f"Unknown ring method '{params.ring_method}'. Choose one of {RING_METHODS}."
        )
    if params.pith.method not in PITH_METHODS:
        raise ValueError(
            f"Unknown pith method '{params.pith.method}'. Choose one of {PITH_METHODS}."
        )
    if params.sampling_nr < 3:
        raise ValueError(f"sampling_nr must be >= 3, got {params.sampling_nr}.")
    if params.resize_scale is not None and not 0 < params.resize_scale <= 1:
        raise ValueError(f"resize_scale must be in (0, 1], got {params.resize_scale}.")


# Flat keyword -> (nested section or None for top level, field name).
_FLAT_KEYS: dict[str, tuple[str | None, str]] = {
    "ring_method": (None, "ring_method"),
    "sampling_nr": (None, "sampling_nr"),
    "scale": (None, "resize_scale"),
    "resize_scale": (None, "resize_scale"),
    "remove_background": (None, "remove_background"),
    "auto_pith": ("pith", "auto"),
    "pith_method": ("pith", "method"),
    "pith_x": ("pith", "x"),
    "pith_y": ("pith", "y"),
    "inbd_model": ("inbd", "model"),
    "inbd_auto_pith": ("inbd", "auto_pith"),
}
for _f in dataclasses.fields(CstrdParams):
    _FLAT_KEYS[f"cstrd_{_f.name}"] = ("cstrd", _f.name)
for _f in dataclasses.fields(DeepCstrdParams):
    _FLAT_KEYS[f"deepcstrd_{_f.name}"] = ("deepcstrd", _f.name)


def flat_parameter_names() -> list[str]:
    """All flat keyword names accepted by ``with_flat_overrides``."""
    return sorted(_FLAT_KEYS)


def _apply_flat_overrides(params: DetectionParams, flat: dict[str, Any]) -> DetectionParams:
    unknown = sorted(set(flat) - set(_FLAT_KEYS))
    if unknown:
        raise TypeError(f"Unknown detection parameter(s): {', '.join(unknown)}")

    top_level: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {}
    for key, value in flat.items():
        section, name = _FLAT_KEYS[key]
        if section is None:
            top_level[name] = value
        else:
            nested.setdefault(section, {})[name] = value

    for section, changes in nested.items():
        top_level[section] = dataclasses.replace(getattr(params, section), **changes)
    return dataclasses.replace(params, **top_level)


def _params_from_config(base: DetectionParams, config: dict[str, Any]) -> DetectionParams:
    preprocess = config.get("preprocess") or {}
    postprocess = config.get("postprocess") or {}
    pith = config.get("pith") or {}
    rings = config.get("rings") or {}

    flat: dict[str, Any] = {
        "resize_scale": preprocess.get("resize_scale"),
        "remove_background": preprocess.get("remove_background"),
        "sampling_nr": postprocess.get("sampling_nr"),
        "auto_pith": pith.get("auto"),
        "pith_method": pith.get("method"),
        "ring_method": rings.get("method"),
    }
    for method in ("cstrd", "deepcstrd", "inbd"):
        for name, value in (rings.get(method) or {}).items():
            flat[f"{method}_{name}"] = value
    known = {k: v for k, v in flat.items() if k in _FLAT_KEYS}
    return base.with_flat_overrides(**known)


def _params_to_config(params: DetectionParams) -> dict[str, Any]:
    config: dict[str, Any] = {
        "preprocess": {"remove_background": params.remove_background},
        "postprocess": {"sampling_nr": params.sampling_nr},
        "pith": {"auto": params.pith.auto, "method": params.pith.method},
        "rings": {
            "method": params.ring_method,
            "cstrd": dataclasses.asdict(params.cstrd),
            "deepcstrd": dataclasses.asdict(params.deepcstrd),
            "inbd": dataclasses.asdict(params.inbd),
        },
    }
    if params.resize_scale is not None:
        config["preprocess"]["resize_scale"] = params.resize_scale
    return config
