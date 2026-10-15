"""Step 1: the logger writes to the console and to <logs_dir>/<name>.log with an env label."""

from pathlib import Path

import pytest

from ire.config import get_env_name
from ire.utils.log import close_logger, get_logger, log_file_path


def test_logger_writes_file_with_env_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("IRE_DATA_DIR", str(tmp_path))  # keep test logs out of the real data dir
    log = get_logger("ire_test_step1")
    try:
        assert get_logger("ire_test_step1") is log
        assert len(log.handlers) == 2  # no duplicates on the second call
        log.info("hello %d", 42)
        path = log_file_path(log)
        assert path == tmp_path / "logs" / "ire_test_step1.log"
    finally:
        close_logger(log)
    text = path.read_text(encoding="utf-8")
    assert "INFO" in text and f"env={get_env_name()}" in text and "ire_test_step1: hello 42" in text
    assert "hello 42" in capsys.readouterr().err
