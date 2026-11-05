"""
src/techbot/sources/base.py — Abstract Base Class for modular news and tool sources.

Guarantees:
  1. Every source implements `async def fetch() -> list[Item]`.
  2. Isolated execution via `safe_fetch`: no individual source failure can crash the pipeline.
  3. Consecutive failures are tracked in state to detect stale or broken feeds.
  4. Automatic canonicalization, hashing, and Pydantic validation for all items.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from src.models import Item, PostBasket, PostCategory
from src.techbot.http import HttpClient, create_http_client
from src.techbot.state.atomic import reset_feed_failure, track_feed_failure
from src.utils import canonical_url, content_hash, normalize_whitespace, sha256_hex, strip_html

logger = logging.getLogger(__name__)


class Source(ABC):
    """
    Abstract Base Class for all content collection sources.
    """

    def __init__(
        self,
        source_id: str,
        name: str,
        category: PostCategory = PostCategory.AI,
        basket: Optional[PostBasket] = None,
        weight: float = 1.0,
        enabled: bool = True,
        timeout: float = 10.0,
    ) -> None:
        self.source_id = source_id
        self.name = name
        self.category = category
        self.basket = basket
        self.weight = weight
        self.enabled = enabled
        self.timeout = timeout

    @abstractmethod
    async def fetch(self, client: Optional[HttpClient] = None) -> list[Item]:
        """
        Fetch candidate items from the external source.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def safe_fetch(
        self,
        client: Optional[HttpClient] = None,
        state: Optional[dict[str, Any]] = None,
    ) -> list[Item]:
        """
        Execute fetch inside an isolated try/except block.
        Never allows an exception to bubble up and crash the caller.
        Tracks consecutive failures and successes in the provided state dictionary.
        """
        if not self.enabled:
            logger.info("Source '%s' (%s) is disabled. Skipping.", self.name, self.source_id)
            return []

        http_client = client or create_http_client(timeout=self.timeout)
        try:
            items = await self.fetch(client=http_client)
            logger.info(
                "Source '%s' (%s) successfully fetched %d items.",
                self.name,
                self.source_id,
                len(items),
            )
            if state is not None:
                reset_feed_failure(state, self.source_id)
            return items
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Source '%s' (%s) failed during fetch: %s",
                self.name,
                self.source_id,
                exc,
                exc_info=True,
            )
            if state is not None:
                track_feed_failure(state, self.source_id, error=str(exc))
            return []

    def make_item(
        self,
        title: str,
        url: str,
        summary: str = "",
        published_at: Optional[datetime] = None,
        raw_data: Optional[dict[str, Any]] = None,
        image_url: Optional[str] = None,
        basket: Optional[PostBasket] = None,
        category: Optional[PostCategory] = None,
        score: float = 0.0,
    ) -> Item:
        """
        Factory helper to create a standardized, validated `Item` instance.
        """
        clean_title = normalize_whitespace(strip_html(title))
        canon_url = canonical_url(url.strip())
        c_hash = content_hash(canon_url, clean_title, summary)
        item_id = sha256_hex(canon_url + "|" + clean_title)

        return Item(
            id=item_id,
            title=clean_title,
            url=url.strip(),
            canonical_url=canon_url,
            content_hash=c_hash,
            summary=strip_html(summary)[:8000],
            source_name=self.name,
            category=category or self.category,
            basket=basket or self.basket,
            published_at=published_at,
            fetched_at=datetime.now(timezone.utc),
            score=score,
            image_url=image_url,
            raw_data=raw_data or {},
        )
