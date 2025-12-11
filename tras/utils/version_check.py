"""Version checking utilities for update detection."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import NamedTuple

from loguru import logger


class VersionInfo(NamedTuple):
    """Version information result."""

    local_version: str
    latest_version: str | None
    is_up_to_date: bool
    error: str | None = None


def read_local_version() -> str:
    """Read the current version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a semantic version string into a tuple of integers for comparison."""
    # Remove 'v' prefix if present
    version_str = version_str.lstrip("v")
    # Remove any suffix after '-' (e.g., "2.0.2-beta" -> "2.0.2")
    if "-" in version_str:
        version_str = version_str.split("-")[0]
    # Split by dots and convert to integers
    parts = version_str.split(".")
    try:
        # Normalize to at least 3 parts for consistent comparison
        # e.g., "2.0" -> (2, 0, 0), "2.0.2" -> (2, 0, 2)
        normalized = [int(part) for part in parts]
        while len(normalized) < 3:
            normalized.append(0)
        return tuple(normalized)
    except ValueError:
        # If parsing fails, return (0, 0, 0) to make it sort last
        return (0, 0, 0)


def get_latest_remote_tag() -> str | None:
    """Fetch the latest git tag from origin."""
    try:
        # Fetch tags from origin
        result = subprocess.run(
            ["git", "ls-remote", "--tags", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("Failed to fetch tags from origin: {}", result.stderr)
            return None

        # Parse tags from output
        tags = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            # Format: <hash>	refs/tags/<tag>
            parts = line.split("\t")
            if len(parts) == 2 and parts[1].startswith("refs/tags/"):
                tag = parts[1].replace("refs/tags/", "").strip()
                # Remove ^{} suffix that git adds for annotated tags
                tag = tag.replace("^{}", "")
                if tag and tag not in tags:
                    tags.append(tag)

        if not tags:
            return None

        # Sort tags by semantic version (highest first)
        tags.sort(key=parse_version, reverse=True)
        return tags[0]

    except subprocess.TimeoutExpired:
        logger.warning("Timeout while fetching tags from origin")
        return None
    except FileNotFoundError:
        logger.warning("git command not found")
        return None
    except Exception as e:
        logger.warning("Error fetching tags: {}", e)
        return None


def check_version() -> VersionInfo:
    """Check if the local version matches the latest remote version."""
    try:
        local_version = read_local_version()
    except Exception as e:
        return VersionInfo(
            local_version="unknown",
            latest_version=None,
            is_up_to_date=False,
            error=f"Failed to read local version: {e}",
        )

    latest_version = get_latest_remote_tag()

    if latest_version is None:
        return VersionInfo(
            local_version=local_version,
            latest_version=None,
            is_up_to_date=False,
            error="Could not fetch latest version from remote",
        )

    # Compare versions
    local_parsed = parse_version(local_version)
    latest_parsed = parse_version(latest_version)

    is_up_to_date = local_parsed >= latest_parsed

    return VersionInfo(
        local_version=local_version,
        latest_version=latest_version,
        is_up_to_date=is_up_to_date,
        error=None,
    )

