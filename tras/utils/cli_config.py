"""YAML configuration loader and validator for CLI batch processing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from tras.params import DetectionParams


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file, or None

    Returns:
        Configuration dictionary, empty if config_path is None
    """
    if config_path is None:
        return {}

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}
        logger.info(f"Loaded config from {config_path}")
        return config
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}") from e
    except Exception as e:
        raise ConfigError(f"Failed to load config file: {e}") from e


def dump_config(config: dict[str, Any], path: Path) -> None:
    """Write a config dict to a YAML file in the schema ``load_config`` reads.

    The output is CLI-compatible: it can be passed back to ``tras_detect`` via
    ``--config``. ``sort_keys=False`` preserves the logical section order.

    Args:
        config: Configuration dictionary (same shape as ``process_config.yml``).
        path: Destination YAML file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    logger.info(f"Wrote config to {path}")


def validate_config(config: dict[str, Any]) -> None:
    """Validate that required configuration fields are present.

    Args:
        config: Configuration dictionary

    Raises:
        ConfigError: If required fields are missing
    """
    if "physical_scale" not in config:
        raise ConfigError("Required field 'physical_scale' missing from config")

    scale = config["physical_scale"]
    if not isinstance(scale, dict):
        raise ConfigError("'physical_scale' must be a dictionary")

    if "value" not in scale:
        raise ConfigError("Required field 'physical_scale.value' missing")
    if "unit" not in scale:
        raise ConfigError("Required field 'physical_scale.unit' missing")

    try:
        float(scale["value"])
    except (ValueError, TypeError):
        raise ConfigError("'physical_scale.value' must be a number")

    if not isinstance(scale["unit"], str) or not scale["unit"]:
        raise ConfigError("'physical_scale.unit' must be a non-empty string")


def normalize_model_path(model_value: str, method: str) -> str:
    """Normalize model value to either model ID or path.

    If model_value is an existing filesystem path, return it as-is.
    Otherwise, treat it as a model ID.

    Args:
        model_value: Model ID or path string
        method: Detection method ('deepcstrd' or 'inbd')

    Returns:
        Model path if file exists, otherwise model ID
    """
    if not isinstance(model_value, str):
        return model_value

    # Check if it's a filesystem path
    model_path = Path(model_value)
    if model_path.exists() and model_path.is_file():
        logger.info(f"Using model path for {method}: {model_path}")
        return str(model_path.resolve())

    # Treat as model ID
    logger.debug(f"Treating '{model_value}' as model ID for {method}")
    return model_value


def merge_config(config: dict[str, Any], cli_overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge flat CLI overrides into a YAML config dict (CLI wins).

    Override keys use the CLI spelling: ``scale_value``, ``scale_unit``,
    ``preprocess_<name>``, ``postprocess_<name>``, ``pith_<name>``, ``ring_method`` and
    ``<method>_<name>`` for ``cstrd``/``deepcstrd``/``inbd``. Neither input is modified.
    Model values that point to existing files are resolved to absolute paths.
    """
    merged = copy.deepcopy(config)
    for key, value in cli_overrides.items():
        section, name = _config_location(key)
        target = merged
        for part in section:
            target = target.setdefault(part, {})
        target[name] = value

    _normalize_model_paths(merged.get("rings") or {})
    return merged


_SECTION_PREFIXES: list[tuple[str, tuple[str, ...]]] = [
    ("preprocess_", ("preprocess",)),
    ("postprocess_", ("postprocess",)),
    ("pith_", ("pith",)),
    ("deepcstrd_", ("rings", "deepcstrd")),
    ("cstrd_", ("rings", "cstrd")),
    ("inbd_", ("rings", "inbd")),
]


def _config_location(key: str) -> tuple[tuple[str, ...], str]:
    """Map a flat override key to its (section path, field name) in the YAML schema."""
    if key == "scale_value":
        return ("physical_scale",), "value"
    if key == "scale_unit":
        return ("physical_scale",), "unit"
    if key == "ring_method":
        return ("rings",), "method"
    for prefix, section in _SECTION_PREFIXES:
        if key.startswith(prefix):
            return section, key[len(prefix) :]
    return (), key


def _normalize_model_paths(rings_config: dict[str, Any]) -> None:
    for method in ("deepcstrd", "inbd"):
        method_config = rings_config.get(method)
        if isinstance(method_config, dict) and "model" in method_config:
            method_config["model"] = normalize_model_path(method_config["model"], method)


def get_detection_params(config: dict[str, Any]) -> DetectionParams:
    """Build typed detection parameters from a merged config dict.

    Missing values fall back to ``tras/config/detection_defaults.yaml``.
    """
    params = DetectionParams.from_config(config)
    logger.info("Detection parameters: {}", params)
    return params
