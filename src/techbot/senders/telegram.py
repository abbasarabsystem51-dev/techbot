"""
src/techbot/senders/telegram.py — Telegram Channel Publisher.

Supports:
  1. `sendPhoto` with photo URL, HTML caption, and inline button `🔗 منبع / لینک مستقیم`.
  2. Retry with FALLBACK_IMAGE_URL if photo URL fails.
  3. `sendMessage` fallback if photo send fails completely ("The post must always go out").
  4. `sendPoll` native anonymous quiz for lesson posts.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

from src.techbot.http import HttpClient, create_http_client
from src.techbot.images import FALLBACK_IMAGE_URL
from src.techbot.senders.base import BaseSender, DeliveryResult

logger = logging.getLogger(__name__)


class TelegramSender(BaseSender):
    """
    Sends posts, graphic cards, and quizzes to the Telegram channel.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None,
        enabled: bool = True,
        timeout: float = 15.0,
    ) -> None:
        super().__init__("telegram", enabled=enabled)
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN") or ""
        self.channel_id = channel_id or os.environ.get("TELEGRAM_CHANNEL_ID") or "@tech_news_110"
        self.timeout = timeout
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def _build_inline_keyboard(self, source_url: Optional[str]) -> Optional[str]:
        """Construct JSON for inline keyboard source button."""
        if not source_url:
            return None
        return json.dumps({
            "inline_keyboard": [
                [{"text": "🔗 منبع / لینک مستقیم", "url": source_url}]
            ]
        })

    async def _send_photo(
        self,
        photo_url: str,
        caption: str,
        reply_markup: Optional[str],
    ) -> Optional[str]:
        """Send a photo by URL to Telegram channel."""
        url = f"{self.base_url}/sendPhoto"
        data: dict[str, Any] = {
            "chat_id": self.channel_id,
            "photo": photo_url,
            "caption": caption,
            "parse_mode": "HTML",
        }
        if reply_markup:
            data["reply_markup"] = reply_markup

        async with httpx.AsyncClient(timeout=self.timeout) as raw_client:
            resp = await raw_client.post(url, data=data)
            resp.raise_for_status()
            res_json = resp.json()
            return str(res_json.get("result", {}).get("message_id", ""))

    async def _send_message(
        self,
        text: str,
        reply_markup: Optional[str],
    ) -> Optional[str]:
        """Send text-only message to Telegram channel."""
        url = f"{self.base_url}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self.channel_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        }
        if reply_markup:
            payload["reply_markup"] = json.loads(reply_markup)

        async with httpx.AsyncClient(timeout=self.timeout) as raw_client:
            resp = await raw_client.post(url, json=payload)
            resp.raise_for_status()
            res_json = resp.json()
            return str(res_json.get("result", {}).get("message_id", ""))

    async def send_post(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        source_url: Optional[str] = None,
        quiz: Optional[dict[str, Any]] = None,
    ) -> DeliveryResult:
        """
        Send formatted HTML message or photo to Telegram channel.
        Resolution:
          1. Send photo by URL (image_url or FALLBACK_IMAGE_URL).
          2. If photo send fails, retry once with FALLBACK_IMAGE_URL.
          3. If that also fails, publish text-only message.
        The post must always go out.
        """
        if not self.enabled:
            return DeliveryResult("telegram", success=True, status="skipped")

        if not self.bot_token:
            logger.warning("Telegram Bot Token is missing. Skipping send.")
            return DeliveryResult("telegram", success=False, status="failed", error="Missing TELEGRAM_BOT_TOKEN")

        reply_markup = self._build_inline_keyboard(source_url)
        message_id: Optional[str] = None
        target_img = image_url or FALLBACK_IMAGE_URL
        published = False

        # Attempt 1: sendPhoto by URL
        if target_img:
            try:
                message_id = await self._send_photo(target_img, text, reply_markup)
                published = True
            except Exception as exc:
                logger.warning("sendPhoto failed with image URL %s: %s", target_img, exc)
                published = False

        # Attempt 2: Retry with FALLBACK_IMAGE_URL if different
        if not published and target_img != FALLBACK_IMAGE_URL:
            try:
                logger.info("Retrying sendPhoto with FALLBACK_IMAGE_URL: %s", FALLBACK_IMAGE_URL)
                message_id = await self._send_photo(FALLBACK_IMAGE_URL, text, reply_markup)
                published = True
            except Exception as exc:
                logger.warning("sendPhoto retry with FALLBACK_IMAGE_URL failed: %s", exc)
                published = False

        # Attempt 3: Publish text-only message ("The post must always go out")
        if not published:
            try:
                logger.info("Falling back to text-only sendMessage...")
                message_id = await self._send_message(text, reply_markup)
                published = True
            except Exception as exc:
                logger.error("Text-only sendMessage failed: %s", exc)
                return DeliveryResult("telegram", success=False, status="failed", error=str(exc))

        # Optional: Send quiz poll if included
        if quiz and isinstance(quiz, dict):
            try:
                await self._send_quiz_poll(quiz)
            except Exception as exc:
                logger.warning("Failed sending Telegram quiz poll: %s", exc)

        return DeliveryResult("telegram", success=True, message_id=message_id, status="sent")

    async def _send_quiz_poll(self, quiz: dict[str, Any]) -> None:
        """Publish a native anonymous quiz poll on Telegram."""
        question = quiz.get("q") or quiz.get("question") or ""
        options = quiz.get("options") or []
        correct_id = quiz.get("correct") or quiz.get("correct_index", 0)
        explanation = quiz.get("explanation") or ""

        if not question or not options:
            return

        url = f"{self.base_url}/sendPoll"
        payload = {
            "chat_id": self.channel_id,
            "question": question[:250],
            "options": json.dumps([str(opt)[:95] for opt in options]),
            "type": "quiz",
            "is_anonymous": True,
            "correct_option_id": int(correct_id),
            "explanation": explanation[:190],
            "explanation_parse_mode": "HTML",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as raw_client:
                await raw_client.post(url, data=payload)
        except Exception as exc:
            logger.warning("Failed sending Telegram quiz poll: %s", exc)
