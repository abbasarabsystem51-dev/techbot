"""
src/eitaa.py — آداپتور Eitaa HTTP API

ویژگی‌ها:
  - ارسال پست متنی ساده (بدون HTML — Eitaa HTML را پشتیبانی نمی‌کند)
  - دکمه inline برای منبع (با graceful fallback اگر پشتیبانی نشود)
  - retry محدود برای خطاهای موقتی (5xx، شبکه)
  - ثبت نتیجه ارسال در SQLite برای idempotency
  - پشتیبانی از dry_run
"""
from __future__ import annotations

import json
import re
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

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html_tags(text: str) -> str:
    """Remove all HTML tags from *text* (Eitaa does not support HTML)."""
    return _HTML_TAG_RE.sub("", text)


class EitaaPublisher:
    """Eitaa HTTP API adapter for publishing posts to a channel."""

    def __init__(self, cfg: AppConfig, state: StateManager) -> None:
        self._cfg = cfg
        self._state = state
        eitaa_cfg = cfg.yaml.get("destinations", {}).get("eitaa", {})
        self._base_url: str = eitaa_cfg.get("base_url", "https://eitaacloud.ir/bot")
        self._max_retries: int = int(eitaa_cfg.get("max_retries", 3))
        self._retry_delay: float = float(eitaa_cfg.get("retry_delay_seconds", 5))
        self._token: str = cfg.settings.eitaa_bot_token
        self._channel_id: str = cfg.settings.eitaa_channel_id
        self._buttons_cfg: dict = cfg.yaml.get("post", {}).get("buttons", {})

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _send_message_url(self) -> str:
        return f"{self._base_url}/{self._token}/sendMessage"

    def _build_inline_keyboard(self, post: GeneratedPost) -> Optional[list]:
        """Build Eitaa inline keyboard payload (if source URL is available)."""
        if not post.source_url:
            return None
        source_label = self._buttons_cfg.get("source_label", "📎 منبع")
        return [[{"text": source_label, "url": post.source_url}]]

    def _is_transient(self, status_code: int) -> bool:
        return status_code >= 500

    def _do_request(
        self,
        client: httpx.Client,
        payload: dict,
        attempt: int,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Execute a single HTTP POST request.

        Returns (success, message_id, error_description).
        """
        try:
            resp = client.post(self._send_message_url(), json=payload)
        except httpx.TimeoutException as exc:
            return False, None, f"Timeout: {exc}"
        except httpx.RequestError as exc:
            return False, None, f"NetworkError: {exc}"

        if resp.status_code == 200:
            try:
                body = resp.json()
            except Exception:
                body = {}
            # Eitaa API returns {"ok": true, "result": {"message_id": ...}} on success
            if body.get("ok") or body.get("result") is not None or "message_id" in body:
                result = body.get("result", body)
                msg_id = str(result.get("message_id", "")) if isinstance(result, dict) else ""
                return True, msg_id or None, None
            error = body.get("description", body.get("error", "Unknown error"))
            return False, None, str(error)

        if self._is_transient(resp.status_code):
            return False, None, f"HTTP {resp.status_code}"

        # Non-transient client error
        return False, None, f"HTTP {resp.status_code} (permanent)"

    # ── Public API ────────────────────────────────────────────────────────────

    def send_post(self, post: GeneratedPost, dry_run: bool = False) -> bool:
        """
        Send *post* to the configured Eitaa channel.

        Returns True on success, False on permanent failure.
        Idempotent: returns True immediately if already sent.
        """
        destination = "eitaa"

        # Idempotency guard
        if self._state.is_already_sent(post.id, destination):
            logger.info("Post %s already sent to Eitaa — skipping.", post.id[:16])
            return True

        # Eitaa does not support HTML — strip all tags
        raw_text = assemble_final_text(post, self._cfg)
        plain_text = _strip_html_tags(raw_text)

        if dry_run:
            logger.info("[DRY-RUN] Eitaa post:\n%s", plain_text)
            self._state.record_delivery_attempt(
                post.id, destination, True, "dry_run_msg"
            )
            return True

        base_payload: dict = {
            "chat_id": self._channel_id,
            "text": plain_text,
        }

        # Attempt with inline buttons first, then fall back without
        keyboard = self._build_inline_keyboard(post)
        use_keyboard = keyboard is not None

        last_error: Optional[str] = None

        with httpx.Client(timeout=15.0) as client:
            for attempt in range(1, self._max_retries + 2):
                payload = dict(base_payload)
                if use_keyboard:
                    payload["inline_keyboard"] = json.dumps(keyboard)

                success, message_id, error = self._do_request(client, payload, attempt)

                if success:
                    self._state.record_delivery_attempt(
                        post.id, destination, True, message_id
                    )
                    logger.info(
                        "Eitaa: post %s sent (message_id=%s).",
                        post.id[:16],
                        message_id,
                    )
                    return True

                last_error = error
                logger.warning(
                    "Eitaa API error (attempt %d/%d): %s",
                    attempt,
                    self._max_retries + 1,
                    error,
                )

                # If inline keyboard might be causing the error, disable and retry once
                if use_keyboard and "inline" in (error or "").lower():
                    logger.info("Retrying without inline keyboard (Eitaa may not support it).")
                    use_keyboard = False
                    continue

                if error and ("permanent" in error or "name or service not known" in error.lower() or "nodename nor servname" in error.lower()):
                    logger.warning("Eitaa host unreachable or permanent error: %s — skipping further retries.", error)
                    break

                if attempt <= self._max_retries:
                    time.sleep(self._retry_delay)

        # All attempts exhausted
        self._state.record_delivery_attempt(
            post.id, destination, False, error=last_error
        )
        logger.error(
            "Eitaa: failed to send post %s after %d attempts.",
            post.id[:16],
            self._max_retries + 1,
        )
        return False
