"""
src/config.py — بارگذاری و اعتبارسنجی پیکربندی

ترتیب اولویت:
  1. متغیرهای محیطی (یا فایل .env)
  2. config.yaml
  3. مقادیر پیش‌فرض
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.logging_config import get_logger
from src.utils import project_root

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Load .env into environment (idempotent)
# ---------------------------------------------------------------------------
_env_path = project_root() / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)
else:
    load_dotenv(override=False)  # still reads from OS environment


# ---------------------------------------------------------------------------
# Settings — pulled from environment variables
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    All secrets and sensitive settings loaded from environment variables.
    Non-secret tunables live in config.yaml.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini ───────────────────────────────────────────────────────────────
    gemini_api_key: str

    # ── Telegram ─────────────────────────────────────────────────────────────
    telegram_bot_token: str
    telegram_channel_id: str

    # ── Eitaa ────────────────────────────────────────────────────────────────
    eitaa_bot_token: str = ""
    eitaa_channel_id: str = ""

    # ── Optional overrides ───────────────────────────────────────────────────
    techbot_db_path: str = ""       # overrides database.path from config.yaml
    techbot_log_level: str = ""     # overrides logging.level from config.yaml

    @field_validator("gemini_api_key", "telegram_bot_token", "telegram_channel_id")
    @classmethod
    def must_not_be_empty(cls, v: str, info: Any) -> str:  # noqa: ANN401
        if not v or not v.strip():
            raise ValueError(f"Environment variable for '{info.field_name}' must not be empty.")
        return v.strip()

    @model_validator(mode="after")
    def warn_missing_eitaa(self) -> "Settings":
        if not self.eitaa_bot_token or not self.eitaa_channel_id:
            logger.warning(
                "EITAA_BOT_TOKEN or EITAA_CHANNEL_ID is not set — Eitaa publishing will be skipped."
            )
        return self


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and return parsed YAML from *path*."""
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"config.yaml must be a YAML mapping, got {type(data)}")
    return data


# ---------------------------------------------------------------------------
# Merged AppConfig — single source of truth at runtime
# ---------------------------------------------------------------------------

class AppConfig:
    """
    Combines YAML config + environment settings into a single object.

    Usage::

        cfg = get_config()
        cfg.settings.gemini_api_key
        cfg.yaml["gemini"]["model"]
        cfg.db_path
        cfg.log_level
    """

    def __init__(self, yaml_path: Path | None = None) -> None:
        _yaml_path = yaml_path or (project_root() / "config.yaml")
        if not _yaml_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {_yaml_path}")
        self.yaml: dict[str, Any] = _load_yaml(_yaml_path)
        self.settings: Settings = Settings()  # type: ignore[call-arg]
        self._resolve_overrides()

    def _resolve_overrides(self) -> None:
        """Apply environment-variable overrides on top of YAML values."""
        if self.settings.techbot_db_path:
            self.yaml.setdefault("database", {})["path"] = self.settings.techbot_db_path
        if self.settings.techbot_log_level:
            self.yaml.setdefault("logging", {})["level"] = self.settings.techbot_log_level

    # ── Convenience accessors ────────────────────────────────────────────────

    @property
    def db_path(self) -> Path:
        """Resolved path to the SQLite database file."""
        raw = self.yaml.get("database", {}).get("path", "data/techbot.sqlite")
        p = Path(raw)
        if not p.is_absolute():
            p = project_root() / p
        return p

    @property
    def log_level(self) -> str:
        return self.yaml.get("logging", {}).get("level", "INFO").upper()

    @property
    def log_file(self) -> Path | None:
        raw = self.yaml.get("logging", {}).get("file")
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = project_root() / p
        return p

    @property
    def timezone(self) -> str:
        return self.yaml.get("bot", {}).get("timezone", "Asia/Tehran")

    @property
    def gemini_model(self) -> str:
        return self.yaml.get("gemini", {}).get("model", "gemini-3.8-flash")

    @property
    def gemini_max_requests_per_day(self) -> int:
        return int(self.yaml.get("gemini", {}).get("max_requests_per_day", 1))

    @property
    def gemini_batch_size(self) -> int:
        return int(self.yaml.get("gemini", {}).get("batch_size", 12))

    @property
    def post_max_chars(self) -> int:
        return int(self.yaml.get("post", {}).get("max_chars", 1024))

    @property
    def post_footer(self) -> str:
        return self.yaml.get("post", {}).get("footer", "\n\n📡 @tech_news_110")

    @property
    def sources(self) -> list[dict[str, Any]]:
        return self.yaml.get("sources", [])

    @property
    def curriculum_path(self) -> Path:
        raw = self.yaml.get("curriculum", {}).get("path", "curriculum.json")
        p = Path(raw)
        if not p.is_absolute():
            p = project_root() / p
        return p

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.yaml.get("destinations", {}).get("telegram", {}).get("enabled", True))

    @property
    def eitaa_enabled(self) -> bool:
        return bool(self.yaml.get("destinations", {}).get("eitaa", {}).get("enabled", True))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AppConfig db={self.db_path} log_level={self.log_level} "
            f"gemini_model={self.gemini_model}>"
        )


# ---------------------------------------------------------------------------
# Module-level singleton factory (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config(yaml_path: str | None = None) -> AppConfig:
    """
    Return the singleton :class:`AppConfig` instance.

    The result is cached; pass *yaml_path* only in tests to use a fixture file.
    """
    path = Path(yaml_path) if yaml_path else None
    cfg = AppConfig(yaml_path=path)
    logger.info("Configuration loaded: db=%s, log_level=%s", cfg.db_path, cfg.log_level)
    return cfg
