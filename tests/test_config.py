"""Step 1: merged config loading, path resolution, and clear errors for missing keys/variables."""

import shutil
from pathlib import Path

import pytest
import yaml

from ire.config import CONFIG_DIR, REPO_ROOT, ConfigError, get_data_dir, load_config


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    """A writable copy of configs/ that a test can break."""
    dst = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, dst)
    return dst


def _edit_yaml(path: Path, fn) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.mark.parametrize("env", ["local", "ada"])
def test_load_config_merges_experiment_and_env(env: str) -> None:
    cfg = load_config(env)
    assert cfg.env == env
    assert cfg["seed"] == 13
    assert (cfg["top_k_cache"], cfg["pool_size"], cfg["final_k"], cfg["rrf_k"]) == (150, 50, 5, 60)
    assert cfg["retrievers"] == ["bm25", "dense", "splade", "rrf"]
    assert cfg.get("cluster.tau_c") == 0.85
    assert cfg.get("perplexity.alpha") == 0.025
    assert cfg["corpus.mode"] == ("random" if env == "local" else "full")


def test_models_differ_per_env() -> None:
    assert load_config("local")["models.dense"] == "intfloat/e5-small-v2"
    assert load_config("ada")["models.dense"] == "intfloat/e5-base-v2"


def test_data_paths_resolve_under_data_dir() -> None:
    cfg = load_config()
    data_dir = get_data_dir()
    for key in ("corpus", "questions", "poison_dir", "indexes_dir", "cache_dir", "logs_dir", "reports_dir"):
        path = cfg.path(key)
        assert path.is_absolute() and data_dir in path.parents, key
        assert REPO_ROOT not in path.parents, key
    assert cfg.path("data_dir") == data_dir
    assert cfg.path("corpus").name == "corpus.jsonl"


def test_repo_relative_paths_resolve_under_repo() -> None:
    cfg = load_config()
    assert cfg.path("perplexity_thresholds") == REPO_ROOT / "configs" / "perplexity_thresholds.json"
    assert cfg.path("reports_repo") == REPO_ROOT / "reports"


def test_unknown_path_key_has_clear_message() -> None:
    cfg = load_config()
    with pytest.raises(ConfigError, match=r"Path key 'corpuss' is not defined in configs/paths.yaml.*Known keys"):
        cfg.path("corpuss")


def test_unknown_config_key_has_clear_message() -> None:
    with pytest.raises(ConfigError, match="'cluster.nope' not found"):
        load_config()["cluster.nope"]
    assert load_config().get("cluster.nope", 7) == 7


def test_paths_yaml_missing_required_key_fails(config_copy: Path) -> None:
    _edit_yaml(config_copy / "paths.yaml", lambda d: d.pop("corpus"))
    with pytest.raises(ConfigError, match="paths.yaml is missing required path key\\(s\\): corpus"):
        load_config(config_dir=config_copy)


def test_experiment_missing_required_key_fails(config_copy: Path) -> None:
    _edit_yaml(config_copy / "experiment.yaml", lambda d: d.pop("pool_size"))
    with pytest.raises(ConfigError, match="pool_size"):
        load_config(config_dir=config_copy)


def test_unset_variable_in_paths_fails(config_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRE_NOT_SET_VAR", raising=False)
    _edit_yaml(config_copy / "paths.yaml", lambda d: d.update(extra="${IRE_NOT_SET_VAR}/x"))
    with pytest.raises(ConfigError, match="IRE_NOT_SET_VAR is not set"):
        load_config(config_dir=config_copy)


def test_missing_data_dir_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRE_DATA_DIR", "")
    with pytest.raises(ConfigError, match="IRE_DATA_DIR is not set"):
        load_config()


def test_missing_config_file_fails(config_copy: Path) -> None:
    (config_copy / "experiment.yaml").unlink()
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(config_dir=config_copy)


def test_dotenv_loaded_on_import() -> None:
    import os

    import ire.config  # noqa: F401  (import loads .env)

    assert os.environ.get("IRE_DATA_DIR"), "IRE_DATA_DIR should come from .env"
