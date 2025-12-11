"""TRAS - Tree Ring Analyzer Suite

Specialized tool for tree ring detection and dendrochronology analysis.
Integrates APD, CS-TRD, and DeepCS-TRD methods.
"""

import re
from pathlib import Path

__appname__ = "TRAS"

# Read version from pyproject.toml dynamically
def _get_version():
    """Read version from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            # Simple regex to extract version (handles both quoted formats)
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
    except Exception:
        pass
    # Fallback to hardcoded version if reading fails
    return "2.0.2"

__version__ = _get_version()

from tras import utils  # noqa: F401


__all__ = [
    "__appname__",
    "__version__",
    "utils",
]
