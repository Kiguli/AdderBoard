"""
Download and cache AdderBoard submissions from GitHub gists and repos.
"""

import json
import logging
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from .config import Submission, Category, LinkType, ALL_SUBMISSIONS

logger = logging.getLogger(__name__)

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
CACHE_MANIFEST = SUBMISSIONS_DIR / "manifest.json"


def _gist_raw_url(gist_url: str) -> Optional[str]:
    """Convert a gist URL to a raw content URL via the GitHub API."""
    match = re.search(r"gist\.github\.com/[\w-]+/([a-f0-9]+)", gist_url)
    if not match:
        return None
    gist_id = match.group(1)
    api_url = f"https://api.github.com/gists/{gist_id}"
    try:
        req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        # Find the first .py file in the gist
        for filename, file_info in data.get("files", {}).items():
            if filename.endswith(".py"):
                return file_info["raw_url"]
        # If no .py file, return the first file
        files = data.get("files", {})
        if files:
            first = next(iter(files.values()))
            return first["raw_url"]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to fetch gist API for %s: %s", gist_url, e)
    return None


def _repo_raw_url(repo_url: str) -> Optional[str]:
    """
    Convert a repo URL to a raw content URL.
    Handles:
      - Direct file links: github.com/user/repo/blob/main/file.py
      - Repo roots: github.com/user/repo (look for submission*.py)
      - Tree links: github.com/user/repo/tree/main/dir
    """
    # Direct file link (blob)
    blob_match = re.search(
        r"github\.com/([\w.-]+)/([\w.-]+)/blob/([\w.-]+)/(.*\.py)", repo_url
    )
    if blob_match:
        user, repo, branch, path = blob_match.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"

    # Tree link (directory) — look for submission*.py via API
    tree_match = re.search(
        r"github\.com/([\w.-]+)/([\w.-]+)/tree/([\w.-]+)/(.*)", repo_url
    )
    if tree_match:
        user, repo, branch, dir_path = tree_match.groups()
        return _find_submission_in_dir(user, repo, branch, dir_path)

    # Repo root — look for submission*.py or the main .py file
    root_match = re.search(r"github\.com/([\w.-]+)/([\w.-]+)/?$", repo_url)
    if root_match:
        user, repo = root_match.groups()
        return _find_submission_in_dir(user, repo, "main", "")

    return None


def _find_submission_in_dir(
    user: str, repo: str, branch: str, dir_path: str
) -> Optional[str]:
    """Search a GitHub directory for submission-like .py files."""
    dir_path = dir_path.rstrip("/")
    api_path = f"repos/{user}/{repo}/contents/{dir_path}" if dir_path else f"repos/{user}/{repo}/contents"
    api_url = f"https://api.github.com/{api_path}?ref={branch}"
    try:
        req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            entries = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.warning("Failed to list directory %s/%s: %s", user, repo, e)
        # Fall back to trying common filenames
        for name in ["submission.py", "adder.py", "model.py"]:
            raw = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{dir_path}/{name}" if dir_path else \
                  f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{name}"
            try:
                urllib.request.urlopen(urllib.request.Request(raw, method="HEAD"), timeout=10)
                return raw
            except urllib.error.URLError:
                continue
        return None

    if not isinstance(entries, list):
        return None

    py_files = [e for e in entries if e.get("name", "").endswith(".py")]

    # Prefer files named submission*.py
    for f in py_files:
        if f["name"].lower().startswith("submission"):
            return f.get("download_url")

    # Then adder*.py, model*.py
    for prefix in ("adder", "model", "main"):
        for f in py_files:
            if f["name"].lower().startswith(prefix):
                return f.get("download_url")

    # Fall back to first .py file
    if py_files:
        return py_files[0].get("download_url")

    return None


def _download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to local path."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AdderBoard-Verifier/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        logger.info("Downloaded %s -> %s", url, dest)
        return True
    except (urllib.error.URLError, OSError) as e:
        logger.error("Failed to download %s: %s", url, e)
        return False


def _load_manifest() -> dict:
    """Load the download manifest (tracks what we've fetched)."""
    if CACHE_MANIFEST.exists():
        return json.loads(CACHE_MANIFEST.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    """Save the download manifest."""
    CACHE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CACHE_MANIFEST.write_text(json.dumps(manifest, indent=2))


def fetch_submission(sub: Submission, force: bool = False) -> Optional[Path]:
    """
    Fetch a single submission and cache it locally.
    Returns the path to the cached .py file, or None on failure.
    """
    manifest = _load_manifest()
    subdir = "hand_coded" if sub.category == Category.HAND_CODED else "trained"
    dest = SUBMISSIONS_DIR / subdir / f"{sub.id}.py"

    if dest.exists() and not force and sub.id in manifest:
        logger.debug("Already cached: %s", sub.id)
        return dest

    # Resolve raw URL
    if sub.link_type == LinkType.GIST:
        raw_url = _gist_raw_url(sub.link_url)
    else:
        raw_url = _repo_raw_url(sub.link_url)

    if raw_url is None:
        logger.error("Could not resolve raw URL for %s (%s)", sub.id, sub.link_url)
        manifest[sub.id] = {"status": "UNFETCHABLE", "url": sub.link_url}
        _save_manifest(manifest)
        return None

    if _download_file(raw_url, dest):
        manifest[sub.id] = {
            "status": "OK",
            "url": sub.link_url,
            "raw_url": raw_url,
            "local_path": str(dest),
        }
        _save_manifest(manifest)
        return dest

    manifest[sub.id] = {"status": "DOWNLOAD_FAILED", "url": sub.link_url, "raw_url": raw_url}
    _save_manifest(manifest)
    return None


def fetch_all(force: bool = False) -> dict[str, Optional[Path]]:
    """Fetch all submissions. Returns {submission_id: local_path_or_None}."""
    results = {}
    for sub in ALL_SUBMISSIONS:
        path = fetch_submission(sub, force=force)
        results[sub.id] = path
        status = "OK" if path else "FAILED"
        logger.info("[%s] %s: %s", status, sub.id, path or "—")
    return results


def get_cached_path(sub: Submission) -> Optional[Path]:
    """Get the cached local path for a submission, if it exists."""
    subdir = "hand_coded" if sub.category == Category.HAND_CODED else "trained"
    dest = SUBMISSIONS_DIR / subdir / f"{sub.id}.py"
    return dest if dest.exists() else None
