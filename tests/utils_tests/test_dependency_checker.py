"""Tests for dependency_checker utilities."""

import pytest

from tras.utils.dependency_checker import (
    check_all_dependencies,
    check_deepcstrd_models,
    check_devernay_library,
    check_python_package,
    validate_dependencies_for_method,
)


def test_check_python_package_installed():
    """Test checking an installed Python package."""
    result = check_python_package("numpy")
    assert result["installed"] is True
    assert result["version"] is not None


def test_check_python_package_not_installed():
    """Test checking a non-installed Python package."""
    result = check_python_package("nonexistent_package_xyz_123")
    assert result["installed"] is False
    assert result["error"] is not None


def test_check_devernay_library():
    """Test checking Devernay library."""
    result = check_devernay_library()
    assert "available" in result
    assert "platform_supported" in result
    assert "error" in result


def test_check_deepcstrd_models():
    """Test checking DeepCS-TRD models."""
    result = check_deepcstrd_models()
    assert "available" in result
    assert "models_found" in result
    assert "path" in result


def test_check_all_dependencies():
    """Test checking all dependencies."""
    result = check_all_dependencies()
    assert "python_packages" in result
    assert "compiled_libraries" in result
    assert "model_files" in result
    assert "system_libraries" in result
    assert "platform" in result
    assert "overall_status" in result
    assert result["overall_status"] in ["ok", "warning", "error"]


def test_validate_dependencies_for_method_apd():
    """Test validating dependencies for APD method."""
    is_valid, error_msg = validate_dependencies_for_method("apd")
    # APD should work if numpy is available
    assert isinstance(is_valid, bool)
    assert error_msg is None or isinstance(error_msg, str)


def test_validate_dependencies_for_method_invalid():
    """Test validating dependencies for invalid method."""
    is_valid, error_msg = validate_dependencies_for_method("invalid_method")
    assert is_valid is False
    assert error_msg is not None




