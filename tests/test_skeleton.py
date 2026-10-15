"""Step 0: the package imports, both env profiles load, and IRE_DATA_DIR is outside the repo."""

import pytest

from ire.config import REPO_ROOT, get_data_dir, get_env_name, load_env_profile


def test_package_imports() -> None:
    import ire
    import ire.defenses.cluster  # noqa: F401
    import ire.defenses.perplexity  # noqa: F401

    assert ire.__version__


@pytest.mark.parametrize("name", ["local", "ada"])
def test_env_profiles_load(name: str) -> None:
    profile = load_env_profile(name)
    assert profile["name"] == name
    assert "seed" not in profile  # single source: configs/experiment.yaml
    assert set(profile["models"]) == {"dense", "splade", "llm", "perplexity"}
    assert profile["models"]["dense"].startswith("intfloat/e5-")


def test_active_env_is_valid() -> None:
    assert get_env_name() in {"local", "ada"}


def test_bad_env_name_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRE_ENV", "laptop")
    with pytest.raises(ValueError, match="IRE_ENV"):
        get_env_name()


def test_data_dir_outside_repo() -> None:
    data_dir = get_data_dir()
    assert data_dir != REPO_ROOT and REPO_ROOT not in data_dir.parents


def test_data_dir_inside_repo_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRE_DATA_DIR", str(REPO_ROOT / "data"))
    with pytest.raises(RuntimeError, match="outside the repo"):
        get_data_dir()
