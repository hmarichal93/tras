"""Helper functions for upgrading TRAS to a new version."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from loguru import logger


def upgrade_tras(new_version: str) -> tuple[bool, str]:
    """
    Upgrade TRAS to a new version.
    
    Args:
        new_version: The version to upgrade to (e.g., "2.0.3" or "v2.0.3")
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Strip "v" prefix if present (version might come as "v2.0.3" or "2.0.3")
        version_clean = new_version.lstrip("v")
        
        # Determine if we're in a git repository (development) or installed package
        repo_root = Path(__file__).parent.parent.parent
        is_git_repo = (repo_root / ".git").exists()
        
        if is_git_repo:
            # Development installation - use git to pull and update
            return _upgrade_via_git(version_clean)
        else:
            # Installed package - download and reinstall
            return _upgrade_via_pip(version_clean)
            
    except Exception as e:
        logger.error(f"Upgrade failed: {e}")
        return False, f"Upgrade failed: {str(e)}"


def _upgrade_via_git(new_version: str) -> tuple[bool, str]:
    """Upgrade by pulling latest changes from git."""
    try:
        repo_root = Path(__file__).parent.parent.parent
        
        # Check if we're on main branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        current_branch = result.stdout.strip()
        
        if current_branch != "main":
            return False, f"Not on main branch (currently on {current_branch}). Please switch to main branch first."
        
        # Fetch latest tags
        subprocess.run(
            ["git", "fetch", "--tags", "origin"],
            cwd=repo_root,
            capture_output=True,
            timeout=30,
            check=True,
        )
        
        # Checkout the new version tag
        tag_name = f"v{new_version}"
        subprocess.run(
            ["git", "checkout", tag_name],
            cwd=repo_root,
            capture_output=True,
            timeout=30,
            check=True,
        )
        
        # Reinstall in editable mode
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=repo_root,
            capture_output=True,
            timeout=120,
            check=True,
        )
        
        return True, f"Successfully upgraded to version {new_version}"
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        return False, f"Git upgrade failed: {error_msg}"
    except Exception as e:
        return False, f"Upgrade failed: {str(e)}"


def _upgrade_via_pip(new_version: str) -> tuple[bool, str]:
    """Upgrade by downloading from GitHub and installing via pip."""
    try:
        # Download the source tarball from GitHub
        repo_url = "https://github.com/hmarichal93/tras"
        tag_name = f"v{new_version}"
        tarball_url = f"{repo_url}/archive/refs/tags/{tag_name}.tar.gz"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tarball_path = tmp_path / f"tras-{new_version}.tar.gz"
            
            # Download tarball
            logger.info(f"Downloading {tarball_url}")
            urllib.request.urlretrieve(tarball_url, tarball_path)
            
            # Extract and install
            extract_dir = tmp_path / f"tras-{new_version}"
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(tmp_path)
            
            # Install
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", str(extract_dir)],
                capture_output=True,
                timeout=300,
                check=True,
            )
        
        return True, f"Successfully upgraded to version {new_version}"
        
    except Exception as e:
        logger.error(f"Pip upgrade failed: {e}")
        return False, f"Upgrade failed: {str(e)}"


def restart_application():
    """Restart the TRAS application."""
    import sys
    import subprocess
    
    # Get the current executable and arguments
    executable = sys.executable
    script_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    try:
        # Start new instance
        if sys.platform == "win32":
            # Windows: use CREATE_NEW_CONSOLE to start in new window
            subprocess.Popen(
                [executable, "-m", "tras"] + script_args,
                creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0,
                close_fds=False,
            )
        else:
            # Unix-like systems: use start_new_session
            subprocess.Popen(
                [executable, "-m", "tras"] + script_args,
                start_new_session=True,
                close_fds=False,
            )
        
        # Small delay to ensure new process starts
        import time
        time.sleep(0.5)
        
        return True
    except Exception as e:
        logger.error(f"Failed to restart application: {e}")
        return False

