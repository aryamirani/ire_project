"""Provenance for job reports: git commit, library versions, platform."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def git_commit() -> str:
    """Current commit hash, with '-dirty' if the working tree has changes; 'unknown' outside git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def library_versions(*dists: str) -> dict[str, str]:
    """Installed version of each distribution ('missing' if not installed), plus Python and platform."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for dist in dists:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = "missing"
    return out
