"""Utilities for fetching and caching the bundled U2-Net weights."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

MODEL_FILENAME = "u2net.pth"
LOCAL_ASSET_DIR = "downloaded_assets"
DEFAULT_RELEASE_TAG = "u2net-v1"
DOWNLOAD_URL_ENV = "URUDENDRO_U2NET_URL"
RELEASE_TAG_ENV = "URUDENDRO_U2NET_TAG"


def get_model_path() -> Path:
    """Return the on-disk path where the weight file should live."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / LOCAL_ASSET_DIR / MODEL_FILENAME


def get_download_url() -> str:
    """Return the URL that hosts the release asset."""
    env_url = os.environ.get(DOWNLOAD_URL_ENV)
    if env_url:
        return env_url
    tag = os.environ.get(RELEASE_TAG_ENV, DEFAULT_RELEASE_TAG)
    return f"https://github.com/hmarichal93/uruDendro/releases/download/{tag}/{MODEL_FILENAME}"


def download_u2net_weights(url: str | None = None, quiet: bool = False) -> Path:
    """Download the release asset into the package directory."""
    target_path = get_model_path()
    download_url = url or get_download_url()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_name = None
    try:
        with urlopen(download_url) as response, tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copyfileobj(response, tmp_file)
            tmp_name = tmp_file.name
        os.replace(tmp_name, target_path)
    except URLError as exc:
        with suppress(FileNotFoundError):
            if tmp_name:
                os.remove(tmp_name)
        raise RuntimeError(
            f"Failed to download {MODEL_FILENAME} from {download_url}. "
            "Set URUDENDRO_U2NET_URL to override the download location."
        ) from exc

    if not quiet:
        print(f"Downloaded {MODEL_FILENAME} from {download_url}")
    return target_path


def ensure_u2net_weights() -> Path:
    """Ensure the U2-Net weights are present locally and return their path."""
    model_path = get_model_path()
    if model_path.exists():
        return model_path
        
    raise FileNotFoundError(
        f"{MODEL_FILENAME} is missing. Run `python tools/download_release_assets.py` to download the model."
    )


__all__ = [
    "DEFAULT_RELEASE_TAG",
    "DOWNLOAD_URL_ENV",
    "MODEL_FILENAME",
    "RELEASE_TAG_ENV",
    "download_u2net_weights",
    "ensure_u2net_weights",
    "get_download_url",
    "get_model_path",
]
