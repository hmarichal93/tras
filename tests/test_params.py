"""Tests for typed detection parameters."""

import pytest

from tras.config.detection_defaults import get_detection_defaults
from tras.params import DetectionParams
from tras.params import flat_parameter_names


def test_defaults_match_yaml():
    defaults = get_detection_defaults()
    params = DetectionParams()
    assert params.ring_method == defaults["method"]
    assert params.sampling_nr == defaults["sampling"]["nr"]
    assert params.pith.method == defaults["pith"]["method"]
    assert params.cstrd.sigma == defaults["cstrd"]["sigma"]
    assert params.deepcstrd.threshold == defaults["deepcstrd"]["threshold"]
    assert params.inbd.model == defaults["inbd"]["model"]


def test_flat_overrides_map_to_nested_fields():
    params = DetectionParams().with_flat_overrides(
        ring_method="cstrd", cstrd_sigma=2.0, deepcstrd_rotations=3, pith_x=1, pith_y=2, scale=0.5
    )
    assert params.ring_method == "cstrd"
    assert params.cstrd.sigma == 2.0
    assert params.deepcstrd.rotations == 3
    assert params.pith.manual_xy == (1.0, 2.0)
    assert params.resize_scale == 0.5


def test_flat_overrides_ignore_none():
    base = DetectionParams()
    assert base.with_flat_overrides(cstrd_sigma=None) == base


def test_unknown_flat_key_is_rejected():
    with pytest.raises(TypeError, match="cstrd_sigmaa"):
        DetectionParams().with_flat_overrides(cstrd_sigmaa=1.0)


@pytest.mark.parametrize(
    "overrides",
    [{"ring_method": "nope"}, {"pith_method": "nope"}, {"sampling_nr": 1}, {"scale": 2.0}],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        DetectionParams().with_flat_overrides(**overrides)


def test_config_round_trip():
    params = DetectionParams().with_flat_overrides(
        ring_method="inbd", inbd_model="INBD_UruDendro1", scale=0.75, sampling_nr=180
    )
    assert DetectionParams.from_config(params.to_config()) == params


def test_from_config_reads_method_sections():
    config = {"rings": {"method": "cstrd", "cstrd": {"sigma": 4.0, "width": 800, "height": 800}}}
    params = DetectionParams.from_config(config)
    assert params.cstrd.sigma == 4.0
    assert params.cstrd.width == 800


def test_flat_names_cover_every_method_field():
    names = flat_parameter_names()
    assert "deepcstrd_width" in names
    assert "cstrd_nr" in names
    assert "inbd_auto_pith" in names


# Regression: unknown YAML keys were silently ignored, e.g. ``rings.deepcstrd.treshold``
# (typo) left the threshold at its default and the user published different parameters.
def test_unknown_yaml_key_is_not_silently_ignored():
    config = {"rings": {"deepcstrd": {"treshold": 0.3}}}
    with pytest.raises(ValueError, match="treshold") as excinfo:
        DetectionParams.from_config(config)
    message = str(excinfo.value)
    assert "rings.deepcstrd" in message
    assert "Did you mean 'threshold'?" in message
    assert "rotations" in message  # lists the valid keys


@pytest.mark.parametrize(
    "config",
    [
        {"preprocess": {"resize": 0.5}},
        {"postprocess": {"sampling": 180}},
        {"pith": {"metod": "apd"}},
        {"rings": {"methd": "cstrd"}},
        {"rings": {"deepcstrdd": {"alpha": 30}}},
        {"rings": {"cstrd": {"sigmaa": 2.0}}},
        {"rings": {"inbd": {"modell": "INBD_EH"}}},
    ],
)
def test_unknown_key_in_any_detection_section_is_rejected(config):
    with pytest.raises(ValueError, match="Unknown key"):
        DetectionParams.from_config(config)


def test_detection_section_must_be_a_mapping():
    with pytest.raises(ValueError, match="'rings' must be a mapping"):
        DetectionParams.from_config({"rings": "cstrd"})


def test_from_config_ignores_non_detection_sections():
    config = {"physical_scale": {"value": 0.02, "unit": "mm"}, "dating": {"outer_year": 2020}}
    assert DetectionParams.from_config(config) == DetectionParams()


# Regression: ``pith.x``/``pith.y`` in a config were never read.
def test_from_config_reads_manual_pith():
    params = DetectionParams.from_config({"pith": {"x": 10, "y": 20}})
    assert params.pith.manual_xy == (10.0, 20.0)
    assert DetectionParams.from_config(params.to_config()) == params


def test_manual_pith_needs_both_coordinates():
    with pytest.raises(ValueError, match="both x and y"):
        DetectionParams().with_flat_overrides(pith_x=10.0)
