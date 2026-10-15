"""Configuration. Importing this package loads `.env` (IRE_ENV, IRE_DATA_DIR, HF_HOME, HF_TOKEN).

`load_config()` returns the merged `experiment.yaml` + `env_<IRE_ENV>.yaml` values and the resolved
`paths.yaml` paths; see `ire.config.loader`.
"""

from ire.config.env import (
    CONFIG_DIR,
    REPO_ROOT,
    get_data_dir,
    get_env_name,
    load_dotenv_file,
    load_env_profile,
)
from ire.config.loader import Config, ConfigError, load_config, resolve_path

__all__ = [
    "CONFIG_DIR",
    "REPO_ROOT",
    "Config",
    "ConfigError",
    "get_data_dir",
    "get_env_name",
    "load_config",
    "load_dotenv_file",
    "load_env_profile",
    "resolve_path",
]
