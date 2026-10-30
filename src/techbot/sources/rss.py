"""
src/techbot/sources/rss.py — RSS/Atom feed fetcher.

Fetches and normalizes articles from RSS and Atom XML feeds (TechCrunch, The Verge,
Ars Technica, OpenAI Blog, Google AI Blog, Hacker News RSS, etc.).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import feedparser

from src.models import Item, PostBasket, PostCategory
from src.techbot.http import HttpClient, create_http_client
from src.techbot.sources.base import Source

logger = logging.getLogger(__name__)


def _struct_time_to_utc(st: time.struct_time) -> Optional[datetime]:
    """Convert feedparser's struct_time to a timezone-aware UTC datetime."""
    try:
        ts = time.mktime(st)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def _extract_published_at(entry: Any) -> Optional[datetime]:
    """Extract publication datetime from feedparser entry fields."""
    for field in ("published_parsed", "updated_parsed", "created_parsed"):
        st = getattr(entry, field, None)
        if st is not None:
            dt = _struct_time_to_utc(st)
            if dt is not None:
                return dt
    return None


def _extract_entry_image(entry: Any) -> Optional[str]:
    """Extract thumbnail or enclosure image URL from entry if available."""
    # Check media_thumbnail or media_content
    thumbnails = getattr(entry, "media_thumbnail", None) or []
    if thumbnails and isinstance(thumbnails, list) and isinstance(thumbnails[0], dict):
        url = thumbnails[0].get("url")
        if url:
            return str(url).strip()

    contents = getattr(entry, "media_content", None) or []
    if contents and isinstance(contents, list) and isinstance(contents[0], dict):
        for c in contents:
            if "image" in c.get("medium", "") or "image" in c.get("type", ""):
                url = c.get("url")
                if url:
                    return str(url).strip()

    enclosures = getattr(entry, "enclosures", None) or []
    if enclosures and isinstance(enclosures, list) and isinstance(enclosures[0], dict):
        for enc in enclosures:
            if "image" in enc.get("type", ""):
                url = enc.get("url") or enc.get("href")
                if url:
                    return str(url).strip()

    return None


class RSSSource(Source):
    """
    Modular RSS/Atom feed collector.
    """

    def __init__(
        self,
        source_id: str,
        name: str,
        url: str,
        category: PostCategory = PostCategory.AI,
        basket: Optional[PostBasket] = None,
        weight: float = 1.0,
        enabled: bool = True,
        timeout: float = 10.0,
        max_items: int = 20,
    ) -> None:
        super().__init__(
            source_id=source_id,
            name=name,
            category=category,
            basket=basket or (PostBasket.NEWS_AI if category == PostCategory.AI else PostBasket.NEWS_TECH),
            weight=weight,
            enabled=enabled,
            timeout=timeout,
        )
        self.url = url
        self.max_items = max_items

    async def fetch(self, client: Optional[HttpClient] = None) -> list[Item]:
        """Fetch XML content via HttpClient and parse with feedparser."""
        http_client = client or create_http_client(timeout=self.timeout)
        xml_content = await http_client.get_text(self.url)

        parsed = feedparser.parse(xml_content)
        entries = getattr(parsed, "entries", [])
        if not entries and getattr(parsed, "bozo", 0) and parsed.bozo_exception:
            logger.warning("Feed '%s' parsing warning/error: %s", self.name, parsed.bozo_exception)

        items: list[Item] = []
        for entry in entries[: self.max_items]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            if not title or not link:
                continue

            # Extract best summary / content
            raw_summary = (
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
            )
            if not raw_summary:
                contents = getattr(entry, "content", [])
                if contents and isinstance(contents, list) and isinstance(contents[0], dict):
                    raw_summary = contents[0].get("value", "")

            published_at = _extract_published_at(entry)
            image_url = _extract_entry_image(entry)

            item = self.make_item(
                title=title,
                url=link,
                summary=str(raw_summary),
                published_at=published_at,
                image_url=image_url,
                raw_data={"author": getattr(entry, "author", "")},
            )
            items.append(item)

        return items
