"""Tests for the YAML config layer of the CLI and batch runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from tras.params import DetectionParams
from tras.utils.cli_config import check_config_keys
from tras.utils.cli_config import dump_config
from tras.utils.cli_config import get_detection_params
from tras.utils.cli_config import load_config

EXAMPLE_CONFIG = Path(__file__).parents[2] / "examples" / "cli" / "process_config.yml"


def test_example_process_config_loads():
    params = get_detection_params(load_config(EXAMPLE_CONFIG))
    assert isinstance(params, DetectionParams)


def test_effective_batch_config_round_trips(tmp_path: Path):
    params = DetectionParams().with_flat_overrides(ring_method="cstrd", scale=0.5)
    config = {
        "physical_scale": {"value": 0.02, "unit": "mm"},
        "dating": {"outer_year": 2020},
        **params.to_config(),
    }
    path = tmp_path / "batch_config.yml"
    dump_config(config, path)
    assert get_detection_params(load_config(path)) == params


# Regression: unknown YAML keys were silently ignored (here a misspelled section).
@pytest.mark.parametrize(
    ("config", "bad_key"),
    [
        ({"ring": {"method": "cstrd"}}, "ring"),
        ({"physical_scale": {"value": 0.02, "units": "mm"}}, "units"),
        ({"dating": {"outer_yr": 2020}}, "outer_yr"),
    ],
)
def test_unknown_config_key_is_rejected(config, bad_key):
    with pytest.raises(ValueError, match=f"Unknown key '{bad_key}'"):
        get_detection_params(config)


def test_top_level_typo_suggests_section():
    with pytest.raises(ValueError, match="Did you mean 'rings'"):
        check_config_keys({"ring": {}})
