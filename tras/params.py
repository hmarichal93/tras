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
import difflib
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
        """Build parameters from the YAML config schema (``process_config.yml``).

        Raises ``ValueError`` naming any unknown key in the detection sections
        (``DETECTION_CONFIG_KEYS``); missing keys keep their defaults.
        """
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
    if (params.pith.x is None) != (params.pith.y is None):
        raise ValueError(
            f"Manual pith needs both x and y, got x={params.pith.x}, y={params.pith.y}."
        )


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


def _method_keys(params_class: type) -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(params_class))


# The detection part of the YAML config schema (``process_config.yml``): section -> keys.
# The single list of valid keys; ``from_config`` rejects anything else.
DETECTION_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "preprocess": ("resize_scale", "remove_background"),
    "postprocess": ("sampling_nr",),
    "pith": ("auto", "method", "x", "y"),
    "rings": ("method", *RING_METHODS),
    "rings.cstrd": _method_keys(CstrdParams),
    "rings.deepcstrd": _method_keys(DeepCstrdParams),
    "rings.inbd": _method_keys(InbdParams),
}


def check_section_keys(section: str, values: Any, valid: tuple[str, ...]) -> None:
    """Raise ``ValueError`` if mapping ``values`` has a key not in ``valid``.

    The message names the key, its section, the closest valid key and every valid key,
    so a typo such as ``treshold`` cannot silently fall back to the default.
    """
    if values is None:
        return
    if not isinstance(values, dict):
        raise ValueError(f"Config section '{section}' must be a mapping, got {values!r}.")
    for key in values:
        if key not in valid:
            raise ValueError(_unknown_key_message(str(key), section, valid))


def _unknown_key_message(key: str, section: str, valid: tuple[str, ...]) -> str:
    close = difflib.get_close_matches(key, valid, n=1)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    return (
        f"Unknown key '{key}' in config section '{section}'.{hint} Valid keys: {', '.join(valid)}."
    )


def check_detection_config_keys(config: dict[str, Any]) -> None:
    """Reject unknown keys in the detection sections of a config dict.

    Other top-level sections (``physical_scale``, ``dating``) are not detection
    parameters and are left to their own readers.
    """
    for section, valid in DETECTION_CONFIG_KEYS.items():
        check_section_keys(section, _config_section(config, section), valid)


def _config_section(config: dict[str, Any], dotted: str) -> Any:
    """Value at a dotted section path (``rings.cstrd``), ``None`` if any level is missing."""
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _params_from_config(base: DetectionParams, config: dict[str, Any]) -> DetectionParams:
    check_detection_config_keys(config)
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
        "pith_x": pith.get("x"),
        "pith_y": pith.get("y"),
        "ring_method": rings.get("method"),
    }
    for method in RING_METHODS:
        for name, value in (rings.get(method) or {}).items():
            flat[f"{method}_{name}"] = value
    return base.with_flat_overrides(**flat)


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
    if params.pith.manual_xy is not None:
        config["pith"]["x"], config["pith"]["y"] = params.pith.manual_xy
    return config
