"""Regression: detector helpers owned defaults that disagreed with detection_defaults.yaml.

``detect_rings_deepcstrd`` used alpha=45, rotations=5, threshold=0.5 and no resize, so a
direct caller got different rings than the API/CLI/GUI (and ran out of memory on large
images). Every helper default must now come from the YAML.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from tras.config.detection_defaults import get_detection_defaults

IMAGE = np.zeros((64, 64, 3), dtype=np.uint8)
PITH = (32.0, 32.0)


def test_deepcstrd_helper_defaults_come_from_yaml():
    from tras.utils import deepcstrd_helper

    defaults = get_detection_defaults()["deepcstrd"]
    with patch.object(deepcstrd_helper, "DeepTreeRingDetection", return_value=None) as detector:
        with patch.object(deepcstrd_helper, "_get_model_path", return_value="w.pth") as model_path:
            deepcstrd_helper.detect_rings_deepcstrd(IMAGE, PITH)

    kwargs = detector.call_args.kwargs
    assert kwargs["alpha"] == defaults["alpha"]
    assert kwargs["nr"] == defaults["nr"]
    assert kwargs["total_rotations"] == defaults["rotations"]
    assert kwargs["prediction_map_threshold"] == defaults["threshold"]
    assert kwargs["width"] == defaults["width"]
    assert kwargs["height"] == defaults["height"]
    assert kwargs["tile_size"] == defaults["tile_size"]
    assert model_path.call_args.args == (defaults["model"], defaults["tile_size"])


def test_deepcstrd_helper_explicit_values_win():
    from tras.utils import deepcstrd_helper

    with patch.object(deepcstrd_helper, "DeepTreeRingDetection", return_value=None) as detector:
        with patch.object(deepcstrd_helper, "_get_model_path", return_value="w.pth"):
            deepcstrd_helper.detect_rings_deepcstrd(IMAGE, PITH, alpha=10, width=0)

    assert detector.call_args.kwargs["alpha"] == 10
    assert detector.call_args.kwargs["width"] == 0


def _cli_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_cstrd_helper_defaults_come_from_yaml():
    from tras.utils import cstrd_helper

    defaults = get_detection_defaults()["cstrd"]
    failed = MagicMock(returncode=1, stderr="stop")
    with patch.object(cstrd_helper.subprocess, "run", return_value=failed) as run:
        with pytest.raises(RuntimeError):
            cstrd_helper.detect_rings_cstrd(IMAGE, PITH)

    cmd = run.call_args.args[0]
    assert float(_cli_value(cmd, "--sigma")) == defaults["sigma"]
    assert float(_cli_value(cmd, "--th_low")) == defaults["th_low"]
    assert float(_cli_value(cmd, "--th_high")) == defaults["th_high"]
    assert int(_cli_value(cmd, "--alpha")) == defaults["alpha"]
    assert int(_cli_value(cmd, "--nr")) == defaults["nr"]
    assert int(_cli_value(cmd, "--wsize")) == defaults["width"]
    assert int(_cli_value(cmd, "--hsize")) == defaults["height"]


def test_inbd_helper_default_model_comes_from_yaml():
    from tras.utils import inbd_helper

    if not inbd_helper._inbd_path.exists():
        pytest.skip("INBD sources not installed")
    with patch.object(inbd_helper, "_get_model_path", side_effect=LookupError) as model_path:
        with pytest.raises(LookupError):
            inbd_helper.detect_rings_inbd(IMAGE, PITH)

    assert model_path.call_args.args == (get_detection_defaults()["inbd"]["model"],)
