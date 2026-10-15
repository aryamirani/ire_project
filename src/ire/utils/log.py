"""Logging to the console and to `$IRE_DATA_DIR/logs/<file>.log`, every line labelled with `env=<IRE_ENV>`.

    from ire.utils.log import get_logger
    log = get_logger("build_cache")          # console + <logs_dir>/build_cache.log
    log.info("wrote %d rows", n)
    # 2026-10-15 16:02:11 INFO    env=local build_cache: wrote 400 rows
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ire.config import load_config

_FORMAT = "%(asctime)s %(levelname)-7s env={env} %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_MARK = "_ire_handler"  # tags our handlers so repeated get_logger calls do not duplicate output


def get_logger(name: str = "ire", log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stderr and to `<logs_dir>/<log_file or name>.log` (UTF-8, appended).

    Calling it again with the same name returns the same logger without adding handlers.
    """
    logger = logging.getLogger(name)
    if any(getattr(h, _MARK, False) for h in logger.handlers):
        return logger

    cfg = load_config()
    logs_dir = cfg.path("logs_dir")
    logs_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT.format(env=cfg.env), datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(logs_dir / f"{log_file or name}.log", encoding="utf-8")
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        setattr(handler, _MARK, True)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_file_path(logger: logging.Logger) -> Path | None:
    """Path of the logger's file handler, if any."""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def close_logger(logger: logging.Logger) -> None:
    """Flush, close and detach this module's handlers (releases the log file, needed on Windows)."""
    for handler in [h for h in logger.handlers if getattr(h, _MARK, False)]:
        handler.close()
        logger.removeHandler(handler)
