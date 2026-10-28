"""
src/telegram.py — آداپتور Telegram Bot API

ویژگی‌ها:
  - ارسال پست با متن HTML + دکمه‌های inline (منبع، بیشتر بخوان)
  - retry محدود برای خطاهای موقتی (429، 5xx، شبکه)
  - ثبت نتیجه ارسال در SQLite برای idempotency
  - پشتیبانی از dry_run
"""
from __future__ import annotations

import json
import time
from typing import Optional

import httpx

from src.config import AppConfig
from src.content_builder import assemble_final_text
from src.logging_config import get_logger
from src.models import GeneratedPost
from src.state import StateManager
from src.utils import utcnow

logger = get_logger(__name__)

_TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramPublisher:
    """Telegram Bot API adapter for publishing posts to a channel."""

    def __init__(self, cfg: AppConfig, state: StateManager) -> None:
        self._cfg = cfg
        self._state = state
        tg_cfg = cfg.yaml.get("destinations", {}).get("telegram", {})
        self._max_retries: int = int(tg_cfg.get("max_retries", 3))
        self._retry_delay: float = float(tg_cfg.get("retry_delay_seconds", 5))
        self._parse_mode: str = tg_cfg.get("parse_mode", "HTML")
        self._token: str = cfg.settings.telegram_bot_token
        self._channel_id: str = cfg.settings.telegram_channel_id
        self._buttons_cfg: dict = cfg.yaml.get("post", {}).get("buttons", {})

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _api_url(self, method: str) -> str:
        return _TELEGRAM_API_BASE.format(token=self._token, method=method)

    def _build_inline_keyboard(self, post: GeneratedPost) -> Optional[dict]:
        """Build the inline keyboard JSON if a source URL is available."""
        if not post.source_url:
            return None
        source_label = self._buttons_cfg.get("source_label", "📎 منبع")
        read_more_label = self._buttons_cfg.get("read_more_label", "📖 بیشتر بخوان")
        return {
            "inline_keyboard": [
                [
                    {"text": source_label, "url": post.source_url},
                    {"text": read_more_label, "url": post.source_url},
                ]
            ]
        }

    def _is_transient(self, status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    # ── Public API ────────────────────────────────────────────────────────────

    def send_post(self, post: GeneratedPost, dry_run: bool = False) -> bool:
        """
        Send *post* to the configured Telegram channel.

        Returns True on success, False on permanent failure.
        Idempotent: returns True immediately if already sent.
        """
        destination = "telegram"

        # Idempotency guard
        if self._state.is_already_sent(post.id, destination):
            logger.info("Post %s already sent to Telegram — skipping.", post.id[:16])
            return True

        text = assemble_final_text(post, self._cfg)

        if dry_run:
            logger.info("[DRY-RUN] Telegram post:\n%s", text)
            self._state.record_delivery_attempt(
                post.id, destination, True, "dry_run_msg"
            )
            return True

        keyboard = self._build_inline_keyboard(post)
        payload: dict = {
            "chat_id": self._channel_id,
            "text": text,
            "parse_mode": self._parse_mode,
        }
        if keyboard:
            payload["reply_markup"] = json.dumps(keyboard)

        last_error: Optional[str] = None

        with httpx.Client(timeout=15.0) as client:
            for attempt in range(1, self._max_retries + 2):  # +1 for initial attempt
                try:
                    resp = client.post(self._api_url("sendMessage"), json=payload)
                    resp_json = resp.json()

                    if resp.status_code == 200 and resp_json.get("ok"):
                        result = resp_json.get("result", {})
                        message_id = str(result.get("message_id", ""))
                        self._state.record_delivery_attempt(
                            post.id, destination, True, message_id
                        )
                        logger.info(
                            "Telegram: post %s sent (message_id=%s).",
                            post.id[:16],
                            message_id,
                        )
                        return True

                    # Non-success response
                    error_desc = resp_json.get("description", f"HTTP {resp.status_code}")
                    logger.warning(
                        "Telegram API error (attempt %d/%d): %s",
                        attempt,
                        self._max_retries + 1,
                        error_desc,
                    )
                    last_error = error_desc

                    if resp.status_code == 429:
                        retry_after = int(
                            resp_json.get("parameters", {}).get("retry_after", self._retry_delay)
                        )
                        logger.info("Rate-limited by Telegram. Waiting %ds...", retry_after)
                        time.sleep(retry_after)
                    elif self._is_transient(resp.status_code):
                        time.sleep(self._retry_delay)
                    else:
                        # Non-transient error (4xx except 429) — do not retry
                        logger.error(
                            "Telegram: permanent error %d — not retrying. %s",
                            resp.status_code,
                            error_desc,
                        )
                        break

                except httpx.TimeoutException as exc:
                    last_error = f"Timeout: {exc}"
                    logger.warning(
                        "Telegram: timeout on attempt %d: %s", attempt, exc
                    )
                    time.sleep(self._retry_delay)
                except httpx.RequestError as exc:
                    last_error = f"NetworkError: {exc}"
                    logger.warning(
                        "Telegram: network error on attempt %d: %s", attempt, exc
                    )
                    time.sleep(self._retry_delay)

                if attempt > self._max_retries:
                    break

        # All attempts exhausted
        self._state.record_delivery_attempt(
            post.id, destination, False, error=last_error
        )
        logger.error("Telegram: failed to send post %s after %d attempts.", post.id[:16], self._max_retries + 1)
        return False
