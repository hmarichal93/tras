#!/usr/bin/env python3
"""Download every asset from a GitHub release."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import typer


API_ROOT = "https://api.github.com"
GITHUB_HOSTS = {"github.com", "www.github.com"}
app = typer.Typer(add_completion=False, help="Download every asset from a GitHub release URL.")


def parse_release_url(url: str) -> Tuple[str, str, str]:
    """Return (owner, repo, slug) from a release URL."""
    parsed = urlparse(url)
    if parsed.netloc.lower() not in GITHUB_HOSTS:
        raise ValueError("The release URL must point to github.com")

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 5 or parts[2] != "releases" or parts[3] != "tag":
        raise ValueError("Release URL should look like https://github.com/<owner>/<repo>/releases/tag/<slug>")

    owner, repo, slug = parts[0], parts[1], parts[4]
    return owner, repo, slug


def github_request(url: str, token: Optional[str] = None, *, accept: str = "application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "tras-download-script"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    return urlopen(req)


def github_json(url: str, token: Optional[str] = None):
    with github_request(url, token=token) as resp:
        return json.load(resp)


def find_release(owner: str, repo: str, slug: str, token: Optional[str]) -> dict:
    # Try the direct tag endpoint first.
    try:
        return github_json(f"{API_ROOT}/repos/{owner}/{repo}/releases/tags/{slug}", token)
    except HTTPError as exc:
        if exc.code != 404:
            raise

    page = 1
    while True:
        releases: Iterable[dict] = github_json(
            f"{API_ROOT}/repos/{owner}/{repo}/releases?per_page=100&page={page}",
            token,
        )
        if not releases:
            break

        for release in releases:
            html_url = (release.get("html_url") or "").rstrip("/")
            if html_url.endswith(slug) or release.get("tag_name") == slug:
                return release
        page += 1

    raise RuntimeError(f"Release '{slug}' not found in {owner}/{repo}")


def download_asset(asset: dict, dest: Path, token: Optional[str], overwrite: bool) -> Path:
    filename = asset.get("name")
    if not filename:
        raise RuntimeError("Asset without a name cannot be downloaded.")

    dest_path = dest / filename
    if dest_path.exists() and not overwrite:
        print(f"[skip] {filename} already exists")
        return dest_path

    download_url = asset.get("browser_download_url")
    if not download_url:
        raise RuntimeError(f"No download URL for asset '{filename}'.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[downloading] {filename}")
    try:
        with github_request(download_url, token=token, accept="application/octet-stream") as resp, dest_path.open("wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    except (HTTPError, URLError) as exc:
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download '{filename}': {exc}") from exc

    return dest_path


def download_direct_url(url: str, dest_path: Path, overwrite: bool) -> Path:
    """Download a file directly from a URL to a specific destination path."""
    if dest_path.exists() and not overwrite:
        print(f"[skip] {dest_path.name} already exists")
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[downloading] {dest_path.name} from {url}")
    try:
        req = Request(url, headers={"User-Agent": "tras-download-script"})
        with urlopen(req) as resp, dest_path.open("wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    except (HTTPError, URLError) as exc:
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download '{dest_path.name}': {exc}") from exc

    return dest_path


@app.command()
def main(
    url: str = typer.Option("https://github.com/hmarichal93/tras/releases/tag/v2.0.2_models", "--url", "-u", help="Release URL such as https://github.com/<owner>/<repo>/releases/tag/<slug>"),
    dest: Path = typer.Option(Path("downloaded_assets"), "--dest", "-d", help="Destination directory (default: ./downloads)"),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        envvar="GITHUB_TOKEN",
        help="GitHub token. Defaults to $GITHUB_TOKEN when available.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", "-o", help="Overwrite existing files."),
) -> None:
    try:
        owner, repo, slug = parse_release_url(url)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--url") from exc

    try:
        release = find_release(owner, repo, slug, token)
    except HTTPError as exc:
        typer.secho(f"GitHub API error ({exc.code}): {exc.reason}", file=sys.stderr, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive
        typer.secho(f"Error: {exc}", file=sys.stderr, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    assets = release.get("assets") or []
    if not assets:
        typer.echo("No assets found in this release.")
        raise typer.Exit()

    dest_dir = Path(dest).expanduser()
    completed = 0

    for asset in assets:
        try:
            download_asset(asset, dest_dir, token, overwrite)
            completed += 1
        except RuntimeError as exc:
            typer.secho(str(exc), file=sys.stderr, fg=typer.colors.RED)
            raise typer.Exit(1) from exc

    # Download APD YOLO weights directly from GitHub releases
    apd_yolo_url = "https://github.com/hmarichal93/apd/releases/download/v1.0_icpr_2024_submission/all_best_yolov8.pt"
    apd_yolo_path = dest_dir / "apd" / "yolo" / "all_best_yolov8.pt"
    try:
        download_direct_url(apd_yolo_url, apd_yolo_path, overwrite)
        completed += 1
    except RuntimeError as exc:
        typer.secho(str(exc), file=sys.stderr, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    typer.echo(f"Downloaded {completed} asset(s) to {dest_dir.resolve()}")


if __name__ == "__main__":
    app()
