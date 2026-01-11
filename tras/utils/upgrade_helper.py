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


def upgrade_tras(new_version: str, progress_callback=None) -> tuple[bool, str]:
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
            return _upgrade_via_git(version_clean, progress_callback)
        else:
            # Installed package - download and reinstall
            return _upgrade_via_pip(version_clean, progress_callback)
            
    except Exception as e:
        logger.error(f"Upgrade failed: {e}")
        return False, f"Upgrade failed: {str(e)}"


def _upgrade_via_git(new_version: str, progress_callback=None) -> tuple[bool, str]:
    """Upgrade by pulling latest changes from git."""
    try:
        repo_root = Path(__file__).parent.parent.parent
        
        # Get remote name (might not be "origin")
        from tras.utils.version_check import get_remote_url
        remote_name = get_remote_url()
        if remote_name is None:
            # Fallback to pip upgrade if no remote configured
            logger.info("No git remote found, falling back to pip upgrade")
            return _upgrade_via_pip(new_version, progress_callback)
        
        # Check if we're on main branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return False, f"Failed to determine current branch: {result.stderr}"
        
        current_branch = result.stdout.strip()
        
        if current_branch != "main":
            return False, f"Not on main branch (currently on {current_branch}). Please switch to main branch first."
        
        # Fetch latest tags
        if progress_callback:
            progress_callback(f"Fetching tags from {remote_name}...")
        logger.info(f"Fetching tags from {remote_name}...")
        result = subprocess.run(
            ["git", "fetch", "--tags", remote_name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return False, f"Failed to fetch tags: {result.stderr}"
        
        # Checkout the new version tag
        tag_name = f"v{new_version}"
        if progress_callback:
            progress_callback(f"Checking out tag {tag_name}...")
        logger.info(f"Checking out tag {tag_name}...")
        result = subprocess.run(
            ["git", "checkout", tag_name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return False, f"Failed to checkout tag {tag_name}: {result.stderr}"
        
        # Reinstall in editable mode
        if progress_callback:
            progress_callback("Reinstalling package...")
        logger.info("Reinstalling in editable mode...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        
        if result.returncode != 0:
            error_output = result.stderr or result.stdout
            return False, f"Reinstallation failed: {error_output[:200]}"
        
        return True, f"Successfully upgraded to version {new_version}"
        
    except subprocess.TimeoutExpired:
        return False, "Upgrade timed out"
    except Exception as e:
        logger.error(f"Git upgrade error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, f"Upgrade failed: {str(e)}"


def _upgrade_via_pip(new_version: str, progress_callback=None) -> tuple[bool, str]:
    """Upgrade by downloading from GitHub and installing via pip."""
    try:
        # Download the source tarball from GitHub
        repo_url = "https://github.com/hmarichal93/tras"
        tag_name = f"v{new_version}"
        tarball_url = f"{repo_url}/archive/refs/tags/{tag_name}.tar.gz"
        
        if progress_callback:
            progress_callback(f"Downloading release {tag_name} from GitHub...")
        logger.info(f"Downloading release {tag_name} from GitHub...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tarball_path = tmp_path / f"tras-{tag_name}.tar.gz"
            
            # Download tarball with progress logging
            try:
                urllib.request.urlretrieve(tarball_url, tarball_path)
                size_mb = tarball_path.stat().st_size / 1024 / 1024
                logger.info(f"Downloaded {size_mb:.2f} MB")
                if progress_callback:
                    progress_callback(f"Downloaded {size_mb:.2f} MB")
            except urllib.error.HTTPError as e:
                return False, f"Failed to download release: HTTP {e.code} - {e.reason}"
            except Exception as e:
                return False, f"Failed to download release: {str(e)}"
            
            # Extract tarball
            if progress_callback:
                progress_callback("Extracting archive...")
            logger.info("Extracting archive...")
            extract_dir = None
            try:
                with tarfile.open(tarball_path, "r:gz") as tar:
                    # Get the root directory name from the archive
                    members = tar.getmembers()
                    if not members:
                        return False, "Archive appears to be empty"
                    
                    # Find the root directory (first member's path)
                    root_dir = members[0].name.split('/')[0]
                    extract_dir = tmp_path / root_dir
                    
                    tar.extractall(tmp_path)
                    logger.info(f"Extracted to {extract_dir}")
            except Exception as e:
                return False, f"Failed to extract archive: {str(e)}"
            
            # Verify extraction directory exists and contains pyproject.toml
            if not extract_dir.exists():
                return False, f"Extraction directory not found: {extract_dir}"
            
            pyproject_path = extract_dir / "pyproject.toml"
            if not pyproject_path.exists():
                return False, f"pyproject.toml not found in extracted archive"
            
            # Install the package
            if progress_callback:
                progress_callback(f"Installing TRAS {new_version}...")
            logger.info(f"Installing TRAS {new_version}...")
            try:
                # Use --upgrade to upgrade existing installation, --force-reinstall to ensure clean install
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", str(extract_dir)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                
                if result.returncode != 0:
                    error_output = result.stderr or result.stdout
                    logger.error(f"pip install failed: {error_output}")
                    return False, f"Installation failed: {error_output[:200]}"
                
                logger.info("Installation completed successfully")
                
            except subprocess.TimeoutExpired:
                return False, "Installation timed out after 5 minutes"
            except Exception as e:
                return False, f"Installation error: {str(e)}"
        
        return True, f"Successfully upgraded to version {new_version}"
        
    except Exception as e:
        logger.error(f"Pip upgrade failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
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

