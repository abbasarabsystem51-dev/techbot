"""
src/techbot/sources/fulltext.py — Full-text article extraction using trafilatura with BeautifulSoup fallback.

Applied strictly to top-ranked candidates to avoid unnecessary bandwidth and CPU consumption.
Truncates output to 6000 characters per SPEC requirements.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from src.models import Item
from src.techbot.http import HttpClient, create_http_client
from src.utils import normalize_whitespace, truncate

logger = logging.getLogger(__name__)

MAX_FULLTEXT_CHARS = 6000

# Try importing trafilatura if installed; otherwise fallback seamlessly to BeautifulSoup
try:
    import trafilatura
except ImportError:
    trafilatura = None


def _extract_with_bs4(html: str) -> str:
    """Extract readable text from HTML using BeautifulSoup as fallback."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "form", "iframe"]):
        tag.decompose()

    # Look for common article containers
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(r"(post|article|entry|content)-body", re.I))
        or soup.find("div", class_=re.compile(r"(content|article|story)", re.I))
        or soup.body
    )

    if container is None:
        return ""

    elements = container.find_all(["h1", "h2", "h3", "h4", "p", "li"])
    text_parts = [el.get_text(separator=" ", strip=True) for el in elements if len(el.get_text(strip=True)) > 15]

    if not text_parts:
        text_parts = [container.get_text(separator=" ", strip=True)]

    raw_text = "\n\n".join(text_parts)
    return normalize_whitespace(raw_text)


def extract_fulltext_from_html(html: str) -> Optional[str]:
    """
    Extract readable article text from HTML content.
    Prioritizes trafilatura; falls back to BeautifulSoup.
    Truncates to at most 6000 characters.
    """
    if not html or not html.strip():
        return None

    text: Optional[str] = None
    if trafilatura is not None:
        try:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
        except Exception as exc:
            logger.debug("trafilatura extraction failed: %s", exc)

    if not text or len(text.strip()) < 50:
        text = _extract_with_bs4(html)

    if not text or not text.strip():
        return None

    cleaned = text.strip()
    if len(cleaned) > MAX_FULLTEXT_CHARS:
        cleaned = truncate(cleaned, MAX_FULLTEXT_CHARS, ellipsis="…")

    return cleaned


async def extract_fulltext(
    html_or_url: str,
    client: Optional[HttpClient] = None,
) -> Optional[str]:
    """
    Extract full text from either raw HTML or a web URL.
    """
    if not html_or_url:
        return None

    target = html_or_url.strip()
    if target.startswith("http://") or target.startswith("https://"):
        http_client = client or create_http_client(timeout=10.0)
        try:
            html = await http_client.get_text(target)
            return extract_fulltext_from_html(html)
        except Exception as exc:
            logger.warning("Failed to fetch fulltext from URL %s: %s", target, exc)
            return None

    return extract_fulltext_from_html(target)


async def enrich_top_candidates(
    items: list[Item],
    max_items: int = 5,
    client: Optional[HttpClient] = None,
) -> list[Item]:
    """
    Asynchronously fetch and attach full_text to the top `max_items` candidates.
    Items beyond `max_items` are returned unmodified.
    """
    if not items:
        return []

    http_client = client or create_http_client(timeout=10.0)
    top_candidates = items[:max_items]
    remaining = items[max_items:]

    async def _enrich_one(item: Item) -> None:
        if item.full_text:
            return
        extracted = await extract_fulltext(item.url, client=http_client)
        if extracted:
            item.full_text = extracted

    tasks = [_enrich_one(item) for item in top_candidates]
    await asyncio.gather(*tasks, return_exceptions=True)

    return top_candidates + remaining
