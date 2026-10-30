"""
src/techbot/alerts.py — Administrative monitoring and failure alert system.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Sends administrative notifications when feeds fail repeatedly or critical errors occur.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        admin_chat_id: Optional[str] = None,
    ) -> None:
        self.bot_token = bot_token or os.environ.get("ADMIN_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN") or ""
        self.admin_chat_id = admin_chat_id or os.environ.get("ADMIN_CHAT_ID") or ""

    async def send_alert(self, title: str, message: str, level: str = "WARNING") -> bool:
        """Send alert message to admin channel or log as fallback."""
        formatted = f"🚨 <b>[{level}] TechBot Alert: {title}</b>\n\n{message}"
        logger.warning("ALERT [%s] %s: %s", level, title, message)

        if not self.bot_token or not self.admin_chat_id:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.admin_chat_id,
            "text": formatted,
            "parse_mode": "HTML",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                return resp.status_code == 200
        except Exception as exc:
            logger.error("Failed sending admin alert: %s", exc)
            return False
