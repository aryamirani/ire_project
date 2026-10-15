"""Environment selection: reads `.env`, picks the `IRE_ENV` profile, resolves `IRE_DATA_DIR`.

This is the minimal loader needed by Step 0. Step 1 adds `configs/paths.yaml` and
`configs/experiment.yaml` on top of it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIG_DIR: Path = REPO_ROOT / "configs"
VALID_ENVS: tuple[str, ...] = ("local", "ada")


def load_dotenv_file() -> bool:
    """Load `<repo>/.env` into os.environ without overriding variables already set.

    Returns True if a `.env` file was found. Never prints values (HF_TOKEN lives there).
    """
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return False
    load_dotenv(env_file, override=False)
    return True


def get_env_name() -> str:
    """Return the active profile name from `IRE_ENV` (default `local`).

    Raises ValueError if `IRE_ENV` is set to something other than `local` or `ada`.
    """
    load_dotenv_file()
    name = os.environ.get("IRE_ENV", "local").strip().lower()
    if name not in VALID_ENVS:
        raise ValueError(f"IRE_ENV must be one of {VALID_ENVS}, got {name!r}")
    return name


def get_data_dir() -> Path:
    """Return `IRE_DATA_DIR` as an absolute Path (it must be set; it is never the repo)."""
    load_dotenv_file()
    raw = os.environ.get("IRE_DATA_DIR", "").strip()
    if not raw:
        raise RuntimeError("IRE_DATA_DIR is not set. Copy .env.example to .env and set it.")
    path = Path(raw).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise RuntimeError(f"IRE_DATA_DIR must be outside the repo, got {path}")
    return path


def load_env_profile(name: str | None = None) -> dict[str, Any]:
    """Load `configs/env_<name>.yaml` (default: the active `IRE_ENV`) as a dict."""
    name = name or get_env_name()
    path = CONFIG_DIR / f"env_{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing environment profile: {path}")
    with path.open("r", encoding="utf-8") as fh:
        profile = yaml.safe_load(fh)
    if profile.get("name") != name:
        raise ValueError(f"{path} has name={profile.get('name')!r}, expected {name!r}")
    return profile
