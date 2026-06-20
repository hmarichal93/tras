"""Loader for the single source of truth of default detection parameters.

Kept lightweight (pathlib + yaml only, no torch/Qt) so it can be imported from the API,
the CLI, and the GUI alike. Values live in ``detection_defaults.yaml`` beside this file.
"""

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS_FILE = Path(__file__).parent / "detection_defaults.yaml"


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    with open(_DEFAULTS_FILE) as f:
        return yaml.safe_load(f)


def get_detection_defaults() -> dict[str, Any]:
    """Return default detection parameters shared by the GUI, API, and CLI.

    Returns a deep copy so callers may read/modify the result without affecting the
    cached parse or other callers.
    """
    return copy.deepcopy(_load())
