"""Canonical filesystem locations for downloaded detection model weights.

Kept dependency-free (pathlib only, no torch) so it can be imported from lightweight
modules such as the dependency checker without pulling in heavy ML libraries.
"""

from pathlib import Path


def get_deepcstrd_models_dir() -> Path:
    """Return the directory where DeepCS-TRD ``.pth`` weights live.

    This is the repo-root ``downloaded_assets/`` folder, populated by
    ``tools/download_release_assets.py`` and where uploaded custom models are stored.
    """
    # tras/utils/model_paths.py -> tras/utils -> tras -> repo root -> downloaded_assets/
    return Path(__file__).parent.parent.parent / "downloaded_assets"


def get_inbd_checkpoints_dir() -> Path:
    """Return the directory where INBD ``model.pt.zip`` checkpoints live.

    Each model is stored in its own ``INBD_<name>/model.pt.zip`` subdirectory.
    """
    # tras/utils/model_paths.py -> tras/utils -> tras -> tree_ring_methods/inbd/...
    return (
        Path(__file__).parent.parent
        / "tree_ring_methods"
        / "inbd"
        / "src"
        / "checkpoints"
    )
