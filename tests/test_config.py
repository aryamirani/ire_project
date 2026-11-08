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
    assert (cfg["top_k_cache"], cfg["pool_size"], cfg["final_k"]) == (150, 50, 5)
    assert (cfg["rrf.k"], cfg["rrf.depth"], cfg["rrf.components"]) == (60, 150, ["bm25", "dense"])
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
        assert REPO_ROOT not in path.parents or (REPO_ROOT / "ire_data_dir") in path.parents, key
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


# ---- threshold tables: the tracked files hold the Ada calibration; off Ada, calibrations stay in reports/

def _cfg_with(env: str, tmp_path: Path):
    from ire.config.loader import Config

    paths = {"reports_dir": tmp_path / "reports",
             "perplexity_thresholds": tmp_path / "configs" / "perplexity_thresholds.json",
             "query_phrase_thresholds": tmp_path / "configs" / "query_phrase_thresholds.json"}
    return Config(env, {"models": {"dense": "intfloat/e5-small-v2"}}, paths)


def test_thresholds_on_ada_use_the_tracked_file(tmp_path: Path) -> None:
    cfg = _cfg_with("ada", tmp_path)
    tracked = cfg.path("query_phrase_thresholds")
    assert cfg.thresholds_write_paths("query_phrase_thresholds") == [tracked]
    (tmp_path / "reports").mkdir()
    cfg.local_thresholds_path("query_phrase_thresholds").write_text("{}", encoding="utf-8")
    assert cfg.thresholds_path("query_phrase_thresholds") == tracked   # a stray local file is ignored on Ada


def test_thresholds_off_ada_never_overwrite_the_tracked_file(tmp_path: Path) -> None:
    cfg = _cfg_with("local", tmp_path)
    tracked, local = cfg.path("perplexity_thresholds"), cfg.local_thresholds_path("perplexity_thresholds")
    assert local == tmp_path / "reports" / "perplexity_thresholds_local.json"
    assert cfg.thresholds_write_paths("perplexity_thresholds") == [local]
    assert cfg.thresholds_write_paths("perplexity_thresholds", write_tracked=True) == [local, tracked]
    assert cfg.thresholds_path("perplexity_thresholds") == tracked     # no local calibration yet: the Ada table
    local.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    assert cfg.thresholds_path("perplexity_thresholds") == local       # then the local one
    with pytest.raises(ConfigError):
        cfg.local_thresholds_path("no_such_key")


def test_query_phrase_cutoffs_of_another_dense_model_are_refused(tmp_path: Path) -> None:
    import json

    from ire.defenses.query_phrase import load_thresholds

    cfg = _cfg_with("local", tmp_path)
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"dense_model": "intfloat/e5-base-v2", "cutoffs": {}}), encoding="utf-8")
    assert load_thresholds(path)["dense_model"] == "intfloat/e5-base-v2"   # without cfg: no check (old behaviour)
    with pytest.raises(ValueError, match="calibrate_query_phrase"):
        load_thresholds(path, cfg)
    path.write_text(json.dumps({"dense_model": "intfloat/e5-small-v2", "cutoffs": {}}), encoding="utf-8")
    assert load_thresholds(path, cfg)["cutoffs"] == {}
