"""
src/logging_config.py — پیکربندی سیستم لاگ‌گذاری

تنها نقطه پیکربندی logging در کل پروژه.
فراخوانی setup_logging() در ابتدای هر entry-point اجباری است.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False
_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATE_FORMAT,
) -> None:
    """
    Configure the root logger once.

    Parameters
    ----------
    level:
        Log level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.
    log_file:
        Path to the rotating log file. If *None*, no file handler is added.
    max_bytes:
        Maximum file size before rotation (bytes).
    backup_count:
        Number of rotated log files to keep.
    console:
        Whether to add a StreamHandler to *stderr*.
    fmt:
        Log record format string.
    datefmt:
        Datetime format string.
    """
    global _CONFIGURED  # noqa: PLW0603

    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handlers: list[logging.Handler] = []

    # ── Console handler ──────────────────────────────────────────────────────
    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(numeric_level)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    # ── Rotating file handler ────────────────────────────────────────────────
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Remove any previously installed handlers to avoid duplicates
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)

    # Silence overly verbose third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "feedparser", "apscheduler.executors"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root_logger.debug("Logging configured: level=%s, file=%s", level, log_file)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Usage::

        from src.logging_config import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def logging_config_from_dict(cfg: dict[str, Any]) -> None:
    """
    Helper to call :func:`setup_logging` from a config dictionary
    matching the ``logging`` section of ``config.yaml``.
    """
    logging_cfg = cfg.get("logging", {})
    setup_logging(
        level=logging_cfg.get("level", "INFO"),
        log_file=logging_cfg.get("file"),
        max_bytes=logging_cfg.get("max_bytes", 10 * 1024 * 1024),
        backup_count=logging_cfg.get("backup_count", 5),
        console=logging_cfg.get("console", True),
    )
