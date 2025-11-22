"""Dependency validation utilities for TRAS.

This module provides functions to check if all required dependencies are available,
including Python packages, compiled libraries, model files, and system libraries.
"""

from __future__ import annotations

import copy
import importlib
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from loguru import logger


@lru_cache(maxsize=None)
def _check_python_package_cached(
    package_name: str, min_version: Optional[str] = None
) -> dict[str, Any]:
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", None)
        return {
            "installed": True,
            "version": version,
            "required_version": min_version,
            "error": None,
        }
    except ImportError as e:
        return {
            "installed": False,
            "version": None,
            "required_version": min_version,
            "error": str(e),
        }


def check_python_package(package_name: str, min_version: Optional[str] = None) -> dict[str, Any]:
    """
    Check if a Python package is installed and optionally verify version.
    """
    return copy.deepcopy(_check_python_package_cached(package_name, min_version))


@lru_cache(maxsize=None)
def _check_devernay_library_cached() -> dict[str, Any]:
    """
    Check if Devernay edge detector is compiled and available.

    Returns:
        Dictionary with keys: available (bool), path (str | None), platform_supported (bool), error (str | None)
    """
    system = platform.system().lower()
    platform_supported = system in ["linux", "darwin"]  # Not available on Windows

    if not platform_supported:
        return {
            "available": False,
            "path": None,
            "platform_supported": False,
            "error": "CS-TRD (Devernay) is not available on Windows. Use DeepCS-TRD instead.",
        }

    # Check for compiled executable
    devernay_path = (
        Path(__file__).parent.parent
        / "tree_ring_methods"
        / "cstrd"
        / "devernay"
        / "devernay"
    )

    if devernay_path.exists() and devernay_path.is_file():
        # Check if executable
        import os

        if os.access(devernay_path, os.X_OK):
            return {
                "available": True,
                "path": str(devernay_path),
                "platform_supported": True,
                "error": None,
            }
        else:
            return {
                "available": False,
                "path": str(devernay_path),
                "platform_supported": True,
                "error": f"Devernay executable exists but is not executable: {devernay_path}",
            }
    else:
        return {
            "available": False,
            "path": None,
            "platform_supported": True,
            "error": f"Devernay edge detector not compiled. Run: cd tras/tree_ring_methods/cstrd/devernay && make",
        }


def check_devernay_library() -> dict[str, Any]:
    """Public wrapper with cached results."""
    return copy.deepcopy(_check_devernay_library_cached())


@lru_cache(maxsize=None)
def _check_deepcstrd_models_cached() -> dict[str, Any]:
    """
    Check if DeepCS-TRD model files are available.

    Returns:
        Dictionary with keys: available (bool), models_found (list[str]), models_required (list[str]), path (str | None), error (str | None)
    """
    models_dir = (
        Path(__file__).parent.parent
        / "tree_ring_methods"
        / "deepcstrd"
        / "models"
        / "deep_cstrd"
    )

    required_models = ["0_all_1504.pth"]  # Minimum required: generic model
    optional_models = [
        "0_pinus_v1_1504.pth",
        "0_pinus_v2_1504.pth",
        "0_gleditsia_1504.pth",
        "0_salix_1504.pth",
    ]

    models_found = []
    if models_dir.exists():
        for model_file in models_dir.glob("*.pth"):
            models_found.append(model_file.name)

    # Check if at least required model exists
    required_found = any(model in models_found for model in required_models)

    if not required_found:
        return {
            "available": False,
            "models_found": models_found,
            "models_required": required_models,
            "path": str(models_dir),
            "error": f"DeepCS-TRD models not found. Download with: cd tras/tree_ring_methods/deepcstrd && ./download_models.sh",
        }

    return {
        "available": True,
        "models_found": models_found,
        "models_required": required_models,
        "path": str(models_dir),
        "error": None,
    }


def check_deepcstrd_models() -> dict[str, Any]:
    """Public wrapper with cached results."""
    return copy.deepcopy(_check_deepcstrd_models_cached())


@lru_cache(maxsize=None)
def _check_system_library_cached(library_name: str) -> dict[str, Any]:
    """
    Check if a system library is available (Qt5, OpenCV system libs).

    Args:
        library_name: Name of the library to check ("qt5" or "opencv")

    Returns:
        Dictionary with keys: available (bool), error (str | None)
    """
    if library_name == "qt5":
        try:
            import PyQt5.QtCore  # noqa: F401

            return {"available": True, "error": None}
        except ImportError as e:
            return {
                "available": False,
                "error": f"PyQt5 not available: {e}. Install with: pip install pyqt5",
            }
    elif library_name == "opencv":
        try:
            import cv2  # noqa: F401
            import numpy as np

            # Try to load a library function to verify system libs are available
            cv2.Canny(np.zeros((10, 10), dtype=np.uint8), 50, 150)
            return {"available": True, "error": None}
        except ImportError as e:
            return {
                "available": False,
                "error": f"OpenCV not available: {e}. Install with: pip install opencv-python",
            }
        except Exception as e:
            return {
                "available": False,
                "error": f"OpenCV system libraries may be missing: {e}",
            }
    else:
        return {
            "available": False,
            "error": f"Unknown system library: {library_name}",
        }


def check_system_library(library_name: str) -> dict[str, Any]:
    return copy.deepcopy(_check_system_library_cached(library_name))


def check_all_dependencies() -> dict[str, Any]:
    """
    Check all TRAS dependencies and return comprehensive status.

    Returns:
        Dictionary with dependency status for all components
    """

    # Check Python packages
    python_packages = {}
    packages_to_check = [
        ("numpy", None),
        ("cv2", "4.5.0"),  # opencv-python
        ("torch", "2.0.0"),
        ("PIL", None),  # Pillow
        ("scipy", "1.7.0"),
        ("shapely", "1.7.0"),
        ("skimage", None),  # scikit-image
        ("pandas", "1.3.0"),
        ("loguru", None),
    ]

    for pkg_name, min_ver in packages_to_check:
        python_packages[pkg_name] = check_python_package(pkg_name, min_ver)

    # Check uruDendro (special case - git dependency)
    try:
        import urudendro  # noqa: F401

        python_packages["urudendro"] = {
            "installed": True,
            "version": getattr(urudendro, "__version__", "unknown"),
            "required_version": None,
            "error": None,
        }
    except ImportError:
        python_packages["urudendro"] = {
            "installed": False,
            "version": None,
            "required_version": None,
            "error": "uruDendro not installed. Install with: pip install git+https://github.com/hmarichal93/uruDendro.git@main",
        }

    # Check compiled libraries
    compiled_libraries = {
        "devernay": check_devernay_library(),
    }

    # Check model files
    model_files = {
        "deepcstrd": check_deepcstrd_models(),
    }

    # Check system libraries
    system_libraries = {
        "qt5": check_system_library("qt5"),
        "opencv": check_system_library("opencv"),
    }

    # Platform info
    platform_info = {
        "os": platform.system().lower(),
        "python_version": sys.version.split()[0],
        "architecture": platform.machine(),
    }

    # Aggregate errors and warnings
    errors = []
    warnings = []

    # Check for critical errors
    for pkg_name, status in python_packages.items():
        if not status["installed"]:
            errors.append(f"Python package '{pkg_name}' not installed: {status.get('error', 'Unknown error')}")

    if not compiled_libraries["devernay"]["available"] and compiled_libraries["devernay"]["platform_supported"]:
        warnings.append(
            f"CS-TRD (Devernay) not available: {compiled_libraries['devernay']['error']}"
        )
    elif not compiled_libraries["devernay"]["platform_supported"]:
        warnings.append("CS-TRD (Devernay) not available on Windows. Use DeepCS-TRD instead.")

    if not model_files["deepcstrd"]["available"]:
        errors.append(f"DeepCS-TRD models not found: {model_files['deepcstrd']['error']}")

    for lib_name, status in system_libraries.items():
        if not status["available"]:
            warnings.append(f"System library '{lib_name}' issue: {status.get('error', 'Unknown error')}")

    # Determine overall status
    if errors:
        overall_status = "error"
    elif warnings:
        overall_status = "warning"
    else:
        overall_status = "ok"

    return {
        "python_packages": python_packages,
        "compiled_libraries": compiled_libraries,
        "model_files": model_files,
        "system_libraries": system_libraries,
        "platform": platform_info,
        "overall_status": overall_status,
        "errors": errors,
        "warnings": warnings,
    }


def validate_dependencies_for_method(method: str) -> tuple[bool, Optional[str]]:
    """
    Validate dependencies for a specific detection method.

    Args:
        method: Detection method name ("apd", "cstrd", "deepcstrd")

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if method == "apd":
        numpy_status = check_python_package("numpy")
        if not numpy_status.get("installed", False):
            return False, "NumPy is required for APD. Install with: pip install numpy"
        return True, None

    elif method == "cstrd":
        devernay_status = check_devernay_library()
        if not devernay_status["available"]:
            return False, devernay_status["error"]
        return True, None

    elif method == "deepcstrd":
        models_status = check_deepcstrd_models()
        if not models_status["available"]:
            return False, models_status["error"]

        # Also check PyTorch
        torch_status = check_python_package("torch", "2.0.0")
        if not torch_status["installed"]:
            return False, "PyTorch is required for DeepCS-TRD. Install with: pip install torch>=2.0.0"

        return True, None

    else:
        return False, f"Unknown detection method: {method}"
