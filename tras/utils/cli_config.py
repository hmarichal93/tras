"""YAML configuration loader and validator for CLI batch processing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


def load_config(config_path: Optional[Path]) -> dict[str, Any]:
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
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}
        logger.info(f"Loaded config from {config_path}")
        # Debug: log deepcstrd model value if present
        if "rings" in config and "deepcstrd" in config.get("rings", {}):
            deepcstrd_model = config["rings"]["deepcstrd"].get("model", "NOT_SET")
            logger.info(f"After loading YAML, deepcstrd.model = '{deepcstrd_model}'")
        return config
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}") from e
    except Exception as e:
        raise ConfigError(f"Failed to load config file: {e}") from e


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
    """Merge CLI overrides into config (CLI wins).

    Args:
        config: Base configuration from YAML
        cli_overrides: CLI-provided overrides

    Returns:
        Merged configuration dictionary
    """
    # Use deep copy to avoid modifying nested dicts in the original config
    merged = copy.deepcopy(config)
    
    # Debug: log deepcstrd model value right after copy
    if "rings" in merged and "deepcstrd" in merged.get("rings", {}):
        logger.debug(f"After deepcopy, deepcstrd.model = '{merged['rings']['deepcstrd'].get('model', 'NOT_SET')}'")
    
    # Debug: log CLI overrides
    if cli_overrides:
        logger.debug(f"CLI overrides keys: {list(cli_overrides.keys())}")
        deepcstrd_overrides = {k: v for k, v in cli_overrides.items() if k.startswith("deepcstrd_")}
        if deepcstrd_overrides:
            logger.debug(f"deepcstrd CLI overrides: {deepcstrd_overrides}")

    # Handle physical_scale separately
    if "scale_value" in cli_overrides or "scale_unit" in cli_overrides:
        if "physical_scale" not in merged:
            merged["physical_scale"] = {}
        if "scale_value" in cli_overrides:
            merged["physical_scale"]["value"] = cli_overrides.pop("scale_value")
        if "scale_unit" in cli_overrides:
            merged["physical_scale"]["unit"] = cli_overrides.pop("scale_unit")

    # Handle preprocess section
    if any(k.startswith("preprocess_") for k in cli_overrides):
        if "preprocess" not in merged:
            merged["preprocess"] = {}
        for key, value in list(cli_overrides.items()):
            if key.startswith("preprocess_"):
                merged["preprocess"][key.replace("preprocess_", "")] = value
                cli_overrides.pop(key)

    # Handle postprocess section
    if any(k.startswith("postprocess_") for k in cli_overrides):
        if "postprocess" not in merged:
            merged["postprocess"] = {}
        for key, value in list(cli_overrides.items()):
            if key.startswith("postprocess_"):
                merged["postprocess"][key.replace("postprocess_", "")] = value
                cli_overrides.pop(key)

    # Handle pith section
    if any(k.startswith("pith_") for k in cli_overrides):
        if "pith" not in merged:
            merged["pith"] = {}
        for key, value in list(cli_overrides.items()):
            if key.startswith("pith_"):
                merged["pith"][key.replace("pith_", "")] = value
                cli_overrides.pop(key)

    # Handle rings section
    if "ring_method" in cli_overrides:
        if "rings" not in merged:
            merged["rings"] = {}
        merged["rings"]["method"] = cli_overrides.pop("ring_method")

    # Handle method-specific parameters
    for method in ["cstrd", "deepcstrd", "inbd"]:
        prefix = f"{method}_"
        if any(k.startswith(prefix) for k in cli_overrides):
            if "rings" not in merged:
                merged["rings"] = {}
            # Don't overwrite existing method config - merge into it
            if method not in merged["rings"]:
                merged["rings"][method] = {}
            else:
                # Debug: log existing config before merging CLI overrides
                logger.debug(f"Before merging CLI overrides, {method} config: {merged['rings'][method]}")
            for key, value in list(cli_overrides.items()):
                if key.startswith(prefix):
                    param_name = key.replace(prefix, "")
                    logger.debug(f"Merging {method}.{param_name} = {value}")
                    merged["rings"][method][param_name] = value
                    cli_overrides.pop(key)
            # Debug: log config after merging CLI overrides
            if method == "deepcstrd":
                logger.debug(f"After merging CLI overrides, deepcstrd config: {merged['rings'][method]}")

    # Normalize model paths for methods that use models
    if "rings" in merged:
        rings_config = merged["rings"]
        logger.debug(f"After merge, rings_config keys: {list(rings_config.keys())}")
        
        # Log final values for each method to verify YAML config is preserved
        for method in ["cstrd", "deepcstrd", "inbd"]:
            if method in rings_config:
                method_config = rings_config[method]
                logger.info(f"Final {method.upper()} config values: {method_config}")
        
        if "deepcstrd" in rings_config:
            logger.debug(f"deepcstrd keys: {list(rings_config['deepcstrd'].keys())}")
            logger.debug(f"deepcstrd.model value before normalization: {rings_config['deepcstrd'].get('model', 'NOT_SET')}")

        # DeepCS-TRD model
        if "deepcstrd" in rings_config and "model" in rings_config["deepcstrd"]:
            model_value = rings_config["deepcstrd"]["model"]
            logger.info(f"Normalizing DeepCS-TRD model: '{model_value}'")
            normalized_model = normalize_model_path(model_value, "deepcstrd")
            logger.info(f"Normalized DeepCS-TRD model: '{normalized_model}'")
            rings_config["deepcstrd"]["model"] = normalized_model
        elif "deepcstrd" in rings_config:
            logger.warning(f"deepcstrd section exists but 'model' key is missing. Keys: {list(rings_config['deepcstrd'].keys())}")

        # INBD model
        if "inbd" in rings_config and "model" in rings_config["inbd"]:
            model_value = rings_config["inbd"]["model"]
            rings_config["inbd"]["model"] = normalize_model_path(model_value, "inbd")

    # Any remaining CLI overrides go to top level
    merged.update(cli_overrides)

    return merged


def get_detection_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract detection parameters from merged config for tras.api.detect().

    Args:
        config: Merged configuration dictionary

    Returns:
        Dictionary of parameters for tras.api.detect()
    """
    params = {}

    # Physical scale (required, but not passed to detect API)
    # It's used separately for CSV/PDF generation

    # Preprocessing
    preprocess = config.get("preprocess", {})
    if "resize_scale" in preprocess:
        params["scale"] = float(preprocess["resize_scale"])
    params["remove_background"] = preprocess.get("remove_background", False)

    # Postprocessing
    postprocess = config.get("postprocess", {})
    params["sampling_nr"] = postprocess.get("sampling_nr", 360)

    # Pith detection
    pith = config.get("pith", {})
    params["auto_pith"] = pith.get("auto", True)
    params["pith_method"] = pith.get("method", "apd_dl")

    # Ring detection method
    rings_config = config.get("rings", {})
    params["ring_method"] = rings_config.get("method", "deepcstrd")

    # CS-TRD parameters
    if "cstrd" in rings_config:
        cstrd = rings_config["cstrd"]
        logger.info(f"CS-TRD config values: sigma={cstrd.get('sigma', 3.0)}, th_low={cstrd.get('th_low', 5.0)}, th_high={cstrd.get('th_high', 20.0)}, alpha={cstrd.get('alpha', 30)}, nr={cstrd.get('nr', 360)}")
        params["cstrd_sigma"] = cstrd.get("sigma", 3.0)
        params["cstrd_th_low"] = cstrd.get("th_low", 5.0)
        params["cstrd_th_high"] = cstrd.get("th_high", 20.0)
        params["cstrd_alpha"] = cstrd.get("alpha", 30)
        params["cstrd_nr"] = cstrd.get("nr", 360)

    # DeepCS-TRD parameters
    if "deepcstrd" in rings_config:
        deepcstrd = rings_config["deepcstrd"]
        # Get model value - should be set after normalization in merge_config
        model_value = deepcstrd.get("model", "generic")
        tile_size_value = deepcstrd.get("tile_size", 0)
        logger.info(f"DeepCS-TRD config values: model='{model_value}', tile_size={tile_size_value}, width={deepcstrd.get('width', 0)}, height={deepcstrd.get('height', 0)}, alpha={deepcstrd.get('alpha', 45)}, nr={deepcstrd.get('nr', 360)}, rotations={deepcstrd.get('rotations', 5)}, threshold={deepcstrd.get('threshold', 0.5)}")
        if model_value == "generic" and "model" in deepcstrd:
            logger.warning(f"DeepCS-TRD model was 'generic' but 'model' key exists in config with value: {deepcstrd.get('model')}")
        params["deepcstrd_model"] = model_value
        params["deepcstrd_tile_size"] = tile_size_value
        params["deepcstrd_alpha"] = deepcstrd.get("alpha", 45)
        params["deepcstrd_nr"] = deepcstrd.get("nr", 360)
        params["deepcstrd_rotations"] = deepcstrd.get("rotations", 5)
        params["deepcstrd_threshold"] = deepcstrd.get("threshold", 0.5)
        params["deepcstrd_width"] = deepcstrd.get("width", 0)
        params["deepcstrd_height"] = deepcstrd.get("height", 0)

    # INBD parameters
    if "inbd" in rings_config:
        inbd = rings_config["inbd"]
        logger.info(f"INBD config values: model='{inbd.get('model', 'INBD_EH')}', auto_pith={inbd.get('auto_pith', True)}")
        params["inbd_model"] = inbd.get("model", "INBD_EH")
        params["inbd_auto_pith"] = inbd.get("auto_pith", True)

    return params

