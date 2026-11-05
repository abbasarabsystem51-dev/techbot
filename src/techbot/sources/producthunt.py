"""
src/techbot/sources/producthunt.py — Product Hunt feed fetcher for tools (tool_ai & tool_tech).

Fetches newly launched tech tools, applications, and AI products from Product Hunt feeds,
assigning them to tool_ai or tool_tech baskets for practical demonstration posts.
"""
from __future__ import annotations

import logging
from typing import Optional

import feedparser

from src.models import Item, PostBasket, PostCategory
from src.techbot.http import HttpClient, create_http_client
from src.techbot.sources.base import Source
from src.techbot.sources.rss import _extract_entry_image, _extract_published_at

logger = logging.getLogger(__name__)

PRODUCT_HUNT_FEED_URL = "https://www.producthunt.com/feed"
AI_KEYWORDS: tuple[str, ...] = (
    "ai",
    "gpt",
    "llm",
    "gemini",
    "claude",
    "agent",
    "prompt",
    "bot",
    "copilot",
    "assistant",
    "diffusion",
    "automation",
    "machine learning",
    "generative",
)


class ProductHuntSource(Source):
    """
    Product Hunt source for discovering trending AI and developer tools.
    """

    def __init__(
        self,
        source_id: str = "product_hunt",
        name: str = "Product Hunt",
        url: str = PRODUCT_HUNT_FEED_URL,
        weight: float = 1.2,
        enabled: bool = True,
        timeout: float = 10.0,
        max_items: int = 25,
    ) -> None:
        super().__init__(
            source_id=source_id,
            name=name,
            category=PostCategory.TOOL,
            basket=PostBasket.TOOL_AI,
            weight=weight,
            enabled=enabled,
            timeout=timeout,
        )
        self.url = url
        self.max_items = max_items

    def _classify_basket(self, title: str, summary: str) -> tuple[PostCategory, PostBasket]:
        """Classify tool into tool_ai or tool_tech based on keywords."""
        content = f"{title} {summary}".lower()
        tokens = set(content.replace("-", " ").replace("_", " ").split())
        for kw in AI_KEYWORDS:
            if " " in kw:
                if kw in content:
                    return PostCategory.AI, PostBasket.TOOL_AI
            elif kw in tokens:
                return PostCategory.AI, PostBasket.TOOL_AI
        return PostCategory.TECH, PostBasket.TOOL_TECH

    async def fetch(self, client: Optional[HttpClient] = None) -> list[Item]:
        """Fetch Product Hunt RSS/Atom feed and map to tool items."""
        http_client = client or create_http_client(timeout=self.timeout)
        content = await http_client.get_text(self.url)

        parsed = feedparser.parse(content)
        entries = getattr(parsed, "entries", [])

        items: list[Item] = []
        for entry in entries[: self.max_items]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            if not title or not link:
                continue

            summary = (
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
            )
            published_at = _extract_published_at(entry)
            image_url = _extract_entry_image(entry)

            category, basket = self._classify_basket(title, summary)

            item = self.make_item(
                title=title,
                url=link,
                summary=summary,
                published_at=published_at,
                category=category,
                basket=basket,
                image_url=image_url,
                raw_data={"feed": "product_hunt"},
            )
            items.append(item)

        return items
