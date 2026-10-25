"""Merged project configuration: `experiment.yaml` + `env_<IRE_ENV>.yaml` + resolved `paths.yaml`.

    from ire.config import load_config
    cfg = load_config()                 # active IRE_ENV
    cfg["pool_size"]                    # 50
    cfg.get("cluster.tau_c")            # 0.85 (dotted lookup)
    cfg.path("corpus")                  # Path(<IRE_DATA_DIR>/data/corpus.jsonl)
    cfg.env                             # "local" | "ada"

Inputs: the YAML files in `configs/` and environment variables (loaded from `.env` by `ire.config.env`).
Every problem (missing file, missing key, unset variable) raises ConfigError with a message naming the
file and key.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from ire.config.env import CONFIG_DIR, REPO_ROOT, get_data_dir, get_env_name, load_env_profile

# Keys every paths.yaml must define (the code relies on them).
REQUIRED_PATH_KEYS: tuple[str, ...] = (
    "data_dir", "raw_dir", "corpus", "questions_all", "questions", "text_store", "dev_data_dir", "poison_dir", "indexes_dir",
    "vectors_dir", "cache_dir", "generation_cache", "perplexity_cache", "logs_dir", "reports_dir",
    "reports_repo", "perplexity_thresholds", "heldout_dir",
)
# Keys the merged experiment + env config must define.
REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "seed", "retrievers", "top_k_cache", "pool_size", "final_k", "rrf.k", "rrf.depth", "corpus.mode", "models.dense",
    "models.splade", "models.llm", "models.perplexity",
)
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MISSING = object()


class ConfigError(ValueError):
    """A configuration file is missing, incomplete or refers to an unset variable."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path.as_posix()}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path.as_posix()} must contain a mapping at the top level")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return base updated by override; nested dicts are merged, everything else is replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _lookup(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def resolve_path(raw: str, key: str = "?") -> Path:
    """Expand `${VAR}` from the environment; a path with no variable is taken relative to the repo root."""
    def substitute(match: re.Match[str]) -> str:
        var = match.group(1)
        if var == "IRE_DATA_DIR":
            try:
                return get_data_dir().as_posix()  # validated: set and outside the repo
            except RuntimeError as exc:
                raise ConfigError(f"paths.yaml key {key!r}: {exc}") from exc
        value = os.environ.get(var, "").strip()
        if not value:
            raise ConfigError(f"paths.yaml key {key!r} uses ${{{var}}}, but {var} is not set (check .env)")
        return value

    expanded = _VAR.sub(substitute, str(raw))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


class Config:
    """Merged configuration values plus resolved paths for one environment."""

    def __init__(self, env: str, values: dict[str, Any], paths: dict[str, Path]) -> None:
        self.env = env
        self.values = values
        self.paths = paths

    def __getitem__(self, key: str) -> Any:
        value = _lookup(self.values, key)
        if value is _MISSING:
            raise ConfigError(f"Config key {key!r} not found in configs/experiment.yaml or configs/env_{self.env}.yaml")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Dotted lookup (`"cluster.tau_c"`) with a default."""
        value = _lookup(self.values, key)
        return default if value is _MISSING else value

    def path(self, key: str) -> Path:
        """Resolved Path for a paths.yaml key. Unknown keys raise ConfigError listing the known ones."""
        if key not in self.paths:
            raise ConfigError(f"Path key {key!r} is not defined in configs/paths.yaml. "
                              f"Known keys: {', '.join(sorted(self.paths))}")
        return self.paths[key]

    def __repr__(self) -> str:
        return f"Config(env={self.env!r}, keys={sorted(self.values)}, paths={sorted(self.paths)})"


def load_config(env: str | None = None, config_dir: Path | None = None) -> Config:
    """Load and validate the merged config for `env` (default: `IRE_ENV`) from `config_dir` (default: configs/)."""
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    env = env or get_env_name()

    experiment = _read_yaml(config_dir / "experiment.yaml")
    try:
        profile = load_env_profile(env, config_dir)
    except FileNotFoundError as exc:
        raise ConfigError(str(exc)) from exc
    values = _deep_merge(experiment, profile)
    missing = [k for k in REQUIRED_CONFIG_KEYS if _lookup(values, k) is _MISSING]
    if missing:
        raise ConfigError(f"Missing required config key(s) {missing} in "
                          f"{(config_dir / 'experiment.yaml').as_posix()} + env_{env}.yaml")

    raw_paths = _read_yaml(config_dir / "paths.yaml")
    missing = [k for k in REQUIRED_PATH_KEYS if k not in raw_paths]
    if missing:
        raise ConfigError(f"{(config_dir / 'paths.yaml').as_posix()} is missing required path key(s): "
                          f"{', '.join(missing)}")
    paths = {key: resolve_path(raw, key) for key, raw in raw_paths.items()}
    return Config(env=env, values=values, paths=paths)
