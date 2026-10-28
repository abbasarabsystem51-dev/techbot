"""
src/sources.py — واکشی خوراک‌های RSS

ویژگی‌های کلیدی:
  - واکشی موازی با asyncio + httpx (timeout=10s)
  - تحمل‌پذیری مستقل هر منبع (خرابی یک منبع بقیه را متوقف نمی‌کند)
  - حذف HTML، نرمال‌سازی URL، محاسبه SHA-256
  - تشخیص تکراری با content_hash قبل از ذخیره
  - ذخیره خودکار در SQLite از طریق StateManager
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Optional

import feedparser
import httpx

from src.config import AppConfig
from src.logging_config import get_logger
from src.models import Article, PostCategory
from src.state import StateManager
from src.utils import (
    canonical_url as make_canonical,
    content_hash as make_content_hash,
    is_valid_url,
    normalize_whitespace,
    sha256_hex,
    strip_html,
    utcnow,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRESHNESS_HOURS_DEFAULT = 48


def _struct_time_to_utc(st: time.struct_time) -> Optional[datetime]:
    """Convert a feedparser struct_time (UTC) to a timezone-aware datetime."""
    try:
        ts = time.mktime(st)
        return datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _extract_published_at(entry: Any) -> Optional[datetime]:
    """Extract publication datetime from a feedparser entry."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        st = getattr(entry, attr, None)
        if st is not None:
            dt = _struct_time_to_utc(st)
            if dt is not None:
                return dt
    return None


def _entry_url(entry: Any) -> str:
    """Extract the best available URL from a feedparser entry."""
    url = getattr(entry, "link", "") or ""
    if not url:
        # Try enclosures
        enclosures = getattr(entry, "enclosures", [])
        if enclosures:
            url = enclosures[0].get("url", "")
    return url.strip()


def _entry_summary(entry: Any) -> str:
    """Extract and clean the summary/description from a feedparser entry."""
    raw = (
        getattr(entry, "summary", "")
        or getattr(entry, "description", "")
        or getattr(entry, "content", [{"value": ""}])[0].get("value", "")
    )
    return strip_html(str(raw))[:4096]


# ---------------------------------------------------------------------------
# Per-source async fetch
# ---------------------------------------------------------------------------

async def fetch_source(
    source_cfg: dict[str, Any],
    client: httpx.AsyncClient,
    cfg: AppConfig,
) -> list[Article]:
    """
    Fetch a single RSS source and return parsed Article objects.

    Failures are caught and logged; an empty list is returned on error.
    """
    source_name: str = source_cfg["name"]
    url: str = source_cfg["url"]
    category_str: str = source_cfg.get("category", "ai")
    category = PostCategory.AI if category_str == "ai" else PostCategory.TECH

    fetching_cfg: dict[str, Any] = cfg.yaml.get("fetching", {})
    max_items: int = int(fetching_cfg.get("max_items_per_source", 20))
    freshness_hours: int = int(fetching_cfg.get("freshness_hours", _FRESHNESS_HOURS_DEFAULT))

    try:
        logger.debug("Fetching source: %s (%s)", source_name, url)
        response = await client.get(url)
        response.raise_for_status()
        raw_content = response.content
        logger.debug("Fetched %d bytes from %s", len(raw_content), source_name)
    except httpx.TimeoutException:
        logger.warning("Timeout fetching source: %s", source_name)
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %d for source %s: %s", exc.response.status_code, source_name, url)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error fetching source %s: %s", source_name, exc)
        return []

    # Parse with feedparser (accepts raw bytes)
    try:
        feed = feedparser.parse(raw_content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("feedparser failed for %s: %s", source_name, exc)
        return []

    if feed.bozo and not feed.entries:
        logger.warning("feedparser bozo error for %s: %s", source_name, getattr(feed, "bozo_exception", "unknown"))
        return []

    now_utc = utcnow()
    articles: list[Article] = []

    for entry in feed.entries[:max_items]:
        try:
            raw_url = _entry_url(entry)
            if not raw_url or not is_valid_url(raw_url):
                logger.debug("Skipping entry with invalid URL in %s: %r", source_name, raw_url)
                continue

            title = normalize_whitespace(strip_html(getattr(entry, "title", "") or ""))
            if not title:
                logger.debug("Skipping entry with empty title in %s", source_name)
                continue

            canon_url = make_canonical(raw_url)
            summary = _entry_summary(entry)
            published_at = _extract_published_at(entry)

            # Freshness check
            if published_at is not None:
                age_hours = (now_utc - published_at).total_seconds() / 3600
                if age_hours > freshness_hours:
                    logger.debug(
                        "Skipping stale article (%.1fh old): %s", age_hours, title[:60]
                    )
                    continue

            # Deduplication key
            chash = make_content_hash(canon_url, title, summary)

            article = Article(
                id=sha256_hex(canon_url + title),
                title=title,
                url=raw_url,
                canonical_url=canon_url,
                content_hash=chash,
                summary=summary,
                source_name=source_name,
                category=category,
                published_at=published_at,
                fetched_at=now_utc,
                score=0.0,
                is_duplicate=False,
            )
            articles.append(article)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Error parsing entry in %s: %s", source_name, exc)
            continue

    logger.info("Parsed %d articles from %s", len(articles), source_name)
    return articles


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

async def fetch_all_sources(
    state: StateManager,
    cfg: AppConfig,
) -> list[Article]:
    """
    Fetch all enabled RSS sources concurrently.

    Returns only newly-inserted (non-duplicate) articles.
    One source's failure does not affect others.
    """
    fetching_cfg = cfg.yaml.get("fetching", {})
    timeout = float(fetching_cfg.get("timeout_seconds", 10))
    user_agent = fetching_cfg.get("user_agent", "TechBot/1.0")

    enabled_sources = [s for s in cfg.sources if s.get("enabled", True)]
    if not enabled_sources:
        logger.warning("No enabled RSS sources configured.")
        return []

    logger.info("Fetching %d RSS sources (timeout=%.0fs)...", len(enabled_sources), timeout)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        headers={"User-Agent": user_agent},
        follow_redirects=True,
    ) as client:
        tasks = [fetch_source(src, client, cfg) for src in enabled_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles: list[Article] = []
    new_articles: list[Article] = []

    for source_cfg, result in zip(enabled_sources, results):
        if isinstance(result, Exception):
            logger.error(
                "Unexpected exception from source %s: %s",
                source_cfg["name"],
                result,
            )
            continue
        all_articles.extend(result)

    # Deduplicate and persist
    for article in all_articles:
        if state.is_duplicate_article(article.content_hash):
            logger.debug("Duplicate skipped: %s", article.content_hash[:16])
            continue
        inserted = state.save_article(article)
        if inserted:
            new_articles.append(article)

    logger.info(
        "Fetch complete: %d total parsed, %d new articles saved.",
        len(all_articles),
        len(new_articles),
    )
    return new_articles
