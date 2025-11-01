"""TRAS - Tree Ring Analyzer Suite

Specialized tool for tree ring detection and dendrochronology analysis.
Integrates APD, CS-TRD, and DeepCS-TRD methods.
"""

__appname__ = "TRAS"
__version__ = "1.0.0"

from tras import utils  # noqa: F401


__all__ = [
    "__appname__",
    "__version__",
    "utils",
]
