"""
src/techbot/senders/telegram.py — Telegram Channel Publisher.

Supports:
  1. `sendPhoto` with HTML caption and inline button `🔗 منبع / لینک مستقیم`.
  2. `sendMessage` fallback if no photo is available.
  3. `sendPoll` native anonymous quiz for lesson posts.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

from src.techbot.http import HttpClient, create_http_client
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
        """
        if not self.enabled:
            return DeliveryResult("telegram", success=True, status="skipped")

        if not self.bot_token:
            logger.warning("Telegram Bot Token is missing. Skipping send.")
            return DeliveryResult("telegram", success=False, status="failed", error="Missing TELEGRAM_BOT_TOKEN")

        client = create_http_client(timeout=self.timeout)
        reply_markup = self._build_inline_keyboard(source_url)

        try:
            message_id: Optional[str] = None

            # 1. Send Photo if binary or URL image provided
            if image_bytes:
                url = f"{self.base_url}/sendPhoto"
                files = {"photo": ("card.png", image_bytes, "image/png")}
                data = {
                    "chat_id": self.channel_id,
                    "caption": text,
                    "parse_mode": "HTML",
                }
                if reply_markup:
                    data["reply_markup"] = reply_markup

                async with httpx.AsyncClient(timeout=self.timeout) as raw_client:
                    resp = await raw_client.post(url, data=data, files=files)
                    resp.raise_for_status()
                    res_json = resp.json()
                    message_id = str(res_json.get("result", {}).get("message_id", ""))

            elif image_url:
                url = f"{self.base_url}/sendPhoto"
                data = {
                    "chat_id": self.channel_id,
                    "photo": image_url,
                    "caption": text,
                    "parse_mode": "HTML",
                }
                if reply_markup:
                    data["reply_markup"] = reply_markup

                async with httpx.AsyncClient(timeout=self.timeout) as raw_client:
                    resp = await raw_client.post(url, data=data)
                    resp.raise_for_status()
                    res_json = resp.json()
                    message_id = str(res_json.get("result", {}).get("message_id", ""))

            else:
                # Text-only sendMessage
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
                    message_id = str(res_json.get("result", {}).get("message_id", ""))

            # 2. If post includes a Quiz, publish native Telegram Quiz Poll
            if quiz and isinstance(quiz, dict):
                await self._send_quiz_poll(quiz)

            return DeliveryResult("telegram", success=True, message_id=message_id, status="sent")

        except Exception as exc:
            logger.error("Failed sending Telegram post: %s", exc)
            return DeliveryResult("telegram", success=False, status="failed", error=str(exc))
        finally:
            await client.aclose()

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
