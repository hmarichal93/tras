"""Version checking utilities for update detection."""

from __future__ import annotations

import re
import subprocess
import urllib.request
import json
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


def get_remote_url() -> str | None:
    """Get the remote URL from git config, trying common remote names."""
    # Try common remote names in order
    for remote_name in ["origin", "upstream", "github"]:
        try:
            result = subprocess.run(
                ["git", "config", "--get", f"remote.{remote_name}.url"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return remote_name
        except Exception:
            continue
    
    # If no named remote found, try to get the first remote
    try:
        result = subprocess.run(
            ["git", "remote"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            remotes = [r.strip() for r in result.stdout.strip().split("\n") if r.strip()]
            if remotes:
                return remotes[0]
    except Exception:
        pass
    
    return None


def get_latest_release_from_github_api() -> str | None:
    """Fetch the latest release tag from GitHub API."""
    try:
        url = "https://api.github.com/repos/hmarichal93/tras/releases/latest"
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status != 200:
                logger.debug("GitHub API returned status code: {}", response.status)
                return None
            data = json.loads(response.read().decode())
            tag_name = data.get("tag_name")
            if tag_name:
                return tag_name
            return None
    except urllib.error.URLError as e:
        logger.debug("Failed to fetch latest release from GitHub API: {}", e)
        return None
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse GitHub API response: {}", e)
        return None
    except Exception as e:
        logger.debug("Error fetching latest release from GitHub API: {}", e)
        return None


def get_latest_remote_tag() -> str | None:
    """Fetch the latest git tag from remote repository or GitHub API."""
    # First, try to use git if available
    try:
        # Get remote name
        remote_name = get_remote_url()
        if remote_name is not None:
            # Fetch tags from remote
            result = subprocess.run(
                ["git", "ls-remote", "--tags", remote_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
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

                if tags:
                    # Sort tags by semantic version (highest first)
                    tags.sort(key=parse_version, reverse=True)
                    return tags[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug("Git-based version check failed: {}", e)
    
    # Fallback to GitHub API if git is not available (e.g., installed from release)
    logger.debug("Falling back to GitHub API for version check")
    return get_latest_release_from_github_api()


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
        # Don't treat this as an error - it's normal when not in a git repo or remote unavailable
        return VersionInfo(
            local_version=local_version,
            latest_version=None,
            is_up_to_date=True,  # Assume up-to-date if we can't check
            error=None,  # Not an error, just unavailable
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

