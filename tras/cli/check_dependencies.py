"""CLI command to check TRAS dependencies.

This command validates all TRAS dependencies and reports their status.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import typer
from loguru import logger

from tras.utils.dependency_checker import check_all_dependencies

app = typer.Typer(
    name="check_dependencies",
    help="Check TRAS dependencies and report their status",
)


def _format_status(status: str) -> str:
    """Format status for display."""
    if status == "ok":
        return "✓ OK"
    elif status == "warning":
        return "⚠ WARNING"
    elif status == "error":
        return "✗ ERROR"
    return status


def _print_dependency_report(status: dict, verbose: bool = False) -> None:
    """Print human-readable dependency report."""
    print("\n" + "=" * 70)
    print("TRAS Dependency Check")
    print("=" * 70 + "\n")

    # Platform info
    platform_info = status["platform"]
    print(f"Platform: {platform_info['os']}, Python {platform_info['python_version']}")
    print(f"Architecture: {platform_info['architecture']}\n")

    # Python packages
    print("Python Packages:")
    print("-" * 70)
    for pkg_name, pkg_status in status["python_packages"].items():
        if pkg_status["installed"]:
            version_str = f" (v{pkg_status['version']})" if pkg_status["version"] else ""
            print(f"  ✓ {pkg_name}{version_str}")
            if verbose and pkg_status.get("required_version"):
                print(f"    Required: {pkg_status['required_version']}")
        else:
            print(f"  ✗ {pkg_name}: NOT INSTALLED")
            if verbose and pkg_status.get("error"):
                print(f"    Error: {pkg_status['error']}")

    # Compiled libraries
    print("\nCompiled Libraries:")
    print("-" * 70)
    for lib_name, lib_status in status["compiled_libraries"].items():
        if lib_status["available"]:
            print(f"  ✓ {lib_name}: Available")
            if verbose and lib_status.get("path"):
                print(f"    Path: {lib_status['path']}")
        else:
            status_icon = "⚠" if not lib_status["platform_supported"] else "✗"
            print(f"  {status_icon} {lib_name}: NOT AVAILABLE")
            if verbose and lib_status.get("error"):
                print(f"    Error: {lib_status['error']}")

    # Model files
    print("\nModel Files:")
    print("-" * 70)
    for model_name, model_status in status["model_files"].items():
        if model_status["available"]:
            models_found = len(model_status.get("models_found", []))
            print(f"  ✓ {model_name}: Available ({models_found} models found)")
            if verbose and model_status.get("path"):
                print(f"    Path: {model_status['path']}")
                if model_status.get("models_found"):
                    print(f"    Models: {', '.join(model_status['models_found'][:5])}")
                    if len(model_status["models_found"]) > 5:
                        print(f"    ... and {len(model_status['models_found']) - 5} more")
        else:
            print(f"  ✗ {model_name}: NOT AVAILABLE")
            if verbose and model_status.get("error"):
                print(f"    Error: {model_status['error']}")

    # System libraries
    print("\nSystem Libraries:")
    print("-" * 70)
    for lib_name, lib_status in status["system_libraries"].items():
        if lib_status["available"]:
            print(f"  ✓ {lib_name}: Available")
        else:
            print(f"  ⚠ {lib_name}: ISSUE DETECTED")
            if verbose and lib_status.get("error"):
                print(f"    Error: {lib_status['error']}")

    # Summary
    print("\n" + "=" * 70)
    overall_status = status["overall_status"]
    print(f"Overall Status: {_format_status(overall_status)}")
    print("=" * 70 + "\n")

    # Errors and warnings
    if status["errors"]:
        print("Errors (must be fixed):")
        for error in status["errors"]:
            print(f"  ✗ {error}")
        print()

    if status["warnings"]:
        print("Warnings (may affect functionality):")
        for warning in status["warnings"]:
            print(f"  ⚠ {warning}")
        print()

    # Actionable fixes
    if status["errors"] or status["warnings"]:
        print("Suggested Fixes:")
        print("-" * 70)
        for error in status["errors"]:
            if "Install with:" in error or "Run:" in error or "Download with:" in error:
                # Extract the command
                if "Install with:" in error:
                    cmd = error.split("Install with:")[-1].strip()
                elif "Run:" in error:
                    cmd = error.split("Run:")[-1].strip()
                elif "Download with:" in error:
                    cmd = error.split("Download with:")[-1].strip()
                else:
                    continue
                print(f"  {cmd}")
        print()


@app.command()
def main(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed information"),
    json_output: bool = typer.Option(False, "--json", help="Output results in JSON format"),
) -> None:
    """
    Check TRAS dependencies and report their status.

    This command validates:
    - Python package versions
    - Compiled libraries (Devernay edge detector)
    - Model files (DeepCS-TRD)
    - System libraries (Qt5, OpenCV)
    - Platform compatibility
    """
    try:
        status = check_all_dependencies()

        if json_output:
            # Output JSON format
            print(json.dumps(status, indent=2))
            sys.exit(0 if status["overall_status"] == "ok" else 1)
        else:
            # Output human-readable format
            _print_dependency_report(status, verbose=verbose)

            # Exit with appropriate code
            if status["overall_status"] == "error":
                sys.exit(1)
            elif status["overall_status"] == "warning":
                sys.exit(2)
            else:
                sys.exit(0)

    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        print(f"ERROR: Failed to check dependencies: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    app()




