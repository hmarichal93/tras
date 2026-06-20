"""TRAS - Tree Ring Analyzer Suite

Specialized tool for tree ring detection and dendrochronology analysis.
Integrates APD, CS-TRD, and DeepCS-TRD methods.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

__appname__ = "TRAS"

# Last-resort version if neither installed metadata nor pyproject.toml is available.
_FALLBACK_VERSION = "2.0.2"


def _read_version_from_pyproject() -> str | None:
    """Read the version from pyproject.toml (only present in a source checkout)."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    content = pyproject_path.read_text()
    # Simple regex to extract version (handles both quoted formats)
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def _get_version() -> str:
    """Resolve the running TRAS version.

    Prefer the installed package metadata (correct for wheel and source-tarball
    installs alike, and the only reliable source once installed). Fall back to
    pyproject.toml for an uninstalled source checkout, then to a hardcoded value.
    """
    try:
        return _package_version("tras")
    except PackageNotFoundError:
        pass
    except Exception:
        pass

    try:
        from_pyproject = _read_version_from_pyproject()
        if from_pyproject:
            return from_pyproject
    except Exception:
        pass

    return _FALLBACK_VERSION


__version__ = _get_version()

from tras import utils  # noqa: F401


__all__ = [
    "__appname__",
    "__version__",
    "utils",
]
