"""
src/techbot/senders/eitaa.py — Eitaa Messenger Channel Publisher.

Uses Eitaayar Open API:
  - `https://eitaayar.ir/api/{TOKEN}/sendFile`
  - `https://eitaayar.ir/api/{TOKEN}/sendMessage`
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from src.techbot.senders.base import BaseSender, DeliveryResult

logger = logging.getLogger(__name__)

EITAA_DEFAULT_BASE_URL = "https://eitaayar.ir/api"


class EitaaSender(BaseSender):
    """
    Sends posts and cards to Eitaa channel with clean plain text formatting.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None,
        base_url: str = EITAA_DEFAULT_BASE_URL,
        enabled: bool = True,
        timeout: float = 15.0,
    ) -> None:
        super().__init__("eitaa", enabled=enabled)
        self.bot_token = bot_token or os.environ.get("EITAA_BOT_TOKEN") or ""
        self.channel_id = channel_id or os.environ.get("EITAA_CHANNEL_ID") or "tech_news_110"
        self.base_url = (base_url or EITAA_DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    async def send_post(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        source_url: Optional[str] = None,
        quiz: Optional[dict[str, Any]] = None,
    ) -> DeliveryResult:
        """
        Send plain-text post and optional image file to Eitaa.
        """
        if not self.enabled:
            return DeliveryResult("eitaa", success=True, status="skipped")

        if not self.bot_token:
            logger.warning("Eitaa Bot Token is missing. Skipping send.")
            return DeliveryResult("eitaa", success=False, status="failed", error="Missing EITAA_BOT_TOKEN")

        # If source_url is present, append link to end of text
        final_text = text
        if source_url and source_url not in final_text:
            final_text = f"{final_text}\n\n🔗 منبع: {source_url}"

        try:
            message_id: Optional[str] = None

            if image_bytes:
                url = f"{self.base_url}/{self.bot_token}/sendFile"
                files = {"file": ("card.png", image_bytes, "image/png")}
                data = {
                    "chat_id": self.channel_id,
                    "caption": final_text,
                }
                async with httpx.AsyncClient(timeout=self.timeout) as raw_client:
                    resp = await raw_client.post(url, data=data, files=files)
                    resp.raise_for_status()
                    res_json = resp.json()
                    message_id = str(res_json.get("result", {}).get("message_id", ""))

            else:
                url = f"{self.base_url}/{self.bot_token}/sendMessage"
                data = {
                    "chat_id": self.channel_id,
                    "text": final_text,
                }
                async with httpx.AsyncClient(timeout=self.timeout) as raw_client:
                    resp = await raw_client.post(url, data=data)
                    resp.raise_for_status()
                    res_json = resp.json()
                    message_id = str(res_json.get("result", {}).get("message_id", ""))

            return DeliveryResult("eitaa", success=True, message_id=message_id, status="sent")

        except Exception as exc:
            logger.error("Failed sending Eitaa post: %s", exc)
            # Never let Eitaa error crash the general pipeline
            return DeliveryResult("eitaa", success=False, status="failed", error=str(exc))
