"""
src/techbot/images.py — Lightweight URL-based image resolution for TechNews Bot.

Resolution order:
  1. Image from the RSS entry (media_content, media_thumbnail, image-type enclosures).
  2. Else fetch article page (timeout 5s, stream=True, read <= 200 KB) and extract og:image / twitter:image.
  3. Validate candidate URL (https only, HEAD/ranged GET returns 2xx with Content-Type image/*, reject icons/logos).
  4. Else return FALLBACK_IMAGE_URL.

Total image-resolution budget <= 10 seconds.
The function get_image_url never raises and always returns a string URL.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FALLBACK_IMAGE_URL = os.environ.get(
    "FALLBACK_IMAGE_URL",
    "https://raw.githubusercontent.com/abbasarabsystem51-dev/techbot/main/assets/fallback.jpg",
)

CHROME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

REJECTED_KEYWORDS: frozenset[str] = frozenset({
    "favicon",
    "logo",
    "1x1",
    "avatar",
    "icon",
    "badge",
    "pixel",
    "spinner",
})


def validate_image_url(url: str, timeout: float = 5.0) -> bool:
    """
    Validate an image URL:
    - https only
    - Reject obvious icons/logos ("favicon", "logo", "1x1", etc.)
    - HEAD (or ranged GET) must return 2xx with Content-Type starting with image/ (timeout 5s).
    """
    if not url or not isinstance(url, str):
        return False

    clean_url = url.strip()
    if not clean_url.lower().startswith("https://"):
        return False

    url_lower = clean_url.lower()
    for kw in REJECTED_KEYWORDS:
        if kw in url_lower:
            return False

    headers = {"User-Agent": CHROME_USER_AGENT}
    try:
        resp = requests.head(clean_url, headers=headers, timeout=timeout, allow_redirects=True)
        ct = resp.headers.get("Content-Type", "").lower()
        if resp.status_code == 200 and ct.startswith("image/"):
            return True

        # If server rejects HEAD (e.g. 403, 405) or didn't provide Content-Type, try ranged GET
        if resp.status_code in (400, 403, 405) or not ct.startswith("image/"):
            get_resp = requests.get(
                clean_url,
                headers={"User-Agent": CHROME_USER_AGENT, "Range": "bytes=0-1024"},
                timeout=timeout,
                stream=True,
            )
            g_ct = get_resp.headers.get("Content-Type", "").lower()
            if get_resp.status_code in (200, 206) and g_ct.startswith("image/"):
                return True
    except Exception as exc:
        logger.debug("Validation failed for %s: %s", clean_url, exc)
        return False

    return False


def extract_rss_image(article: Any) -> Optional[str]:
    """
    Extract candidate image from RSS entry:
    - media_content
    - media_thumbnail
    - image-type enclosures
    - direct image_url attribute
    """
    # Direct attribute or dict field
    if hasattr(article, "image_url") and article.image_url:
        return str(article.image_url).strip()
    if isinstance(article, dict) and article.get("image_url"):
        return str(article["image_url"]).strip()

    raw = getattr(article, "raw_data", None) or (article if isinstance(article, dict) else {})
    if not isinstance(raw, dict):
        return None

    # media_content
    media_content = raw.get("media_content", [])
    if isinstance(media_content, list):
        for mc in media_content:
            if isinstance(mc, dict) and mc.get("url"):
                return str(mc["url"]).strip()

    # media_thumbnail
    media_thumbnail = raw.get("media_thumbnail", [])
    if isinstance(media_thumbnail, list):
        for mt in media_thumbnail:
            if isinstance(mt, dict) and mt.get("url"):
                return str(mt["url"]).strip()

    # enclosures
    enclosures = raw.get("enclosures", [])
    if isinstance(enclosures, list):
        for enc in enclosures:
            if isinstance(enc, dict) and enc.get("url"):
                enc_type = str(enc.get("type", "")).lower()
                if not enc_type or enc_type.startswith("image/"):
                    return str(enc["url"]).strip()

    return None


def extract_meta_image(html_text: str, base_url: str = "") -> Optional[str]:
    """Extract og:image or twitter:image from HTML."""
    if not html_text:
        return None

    try:
        soup = BeautifulSoup(html_text, "html.parser")
        meta_candidates = [
            ("meta", {"property": "og:image"}),
            ("meta", {"property": "og:image:url"}),
            ("meta", {"name": "twitter:image"}),
            ("meta", {"name": "twitter:image:src"}),
        ]
        for tag, attrs in meta_candidates:
            found = soup.find(tag, attrs=attrs)
            if found and found.get("content"):
                url = found["content"].strip()
                if base_url:
                    url = urljoin(base_url, url)
                return url
    except Exception as exc:
        logger.debug("Failed extracting meta tags: %s", exc)

    # Regex fallback if soup fails
    m = re.search(r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\'][^>]+content=["\']([^"\']+)["\']', html_text, re.IGNORECASE)
    if m:
        url = m.group(1).strip()
        if base_url:
            url = urljoin(base_url, url)
        return url

    return None


def get_image_url(article: Any) -> str:
    """
    Resolve image URL for an article in <= 10 seconds:
    1. Image from RSS entry (media_content, media_thumbnail, enclosures).
    2. Else fetch article page (timeout 5s, stream=True, read <= 200 KB) and extract og:image / twitter:image.
    3. Validate candidate URL (https only, HEAD/ranged GET 2xx with image/* Content-Type).
    4. Else return FALLBACK_IMAGE_URL.

    Never raises an exception and always returns a string URL.
    """
    start_time = time.monotonic()
    max_budget = 10.0

    try:
        # 1. Candidate from RSS entry
        candidate = extract_rss_image(article)
        if candidate:
            elapsed = time.monotonic() - start_time
            val_timeout = min(5.0, max(1.0, max_budget - elapsed))
            if validate_image_url(candidate, timeout=val_timeout):
                return candidate

        # 2. Fetch article page and extract og:image / twitter:image
        elapsed = time.monotonic() - start_time
        if elapsed < max_budget:
            article_url = getattr(article, "url", None) or (article.get("url") if isinstance(article, dict) else None)
            if article_url and str(article_url).startswith("http"):
                fetch_timeout = min(5.0, max(1.0, max_budget - elapsed))
                try:
                    resp = requests.get(
                        str(article_url),
                        headers={"User-Agent": CHROME_USER_AGENT},
                        timeout=fetch_timeout,
                        stream=True,
                    )
                    chunks: list[bytes] = []
                    total_bytes = 0
                    for chunk in resp.iter_content(chunk_size=8192):
                        chunks.append(chunk)
                        total_bytes += len(chunk)
                        if total_bytes >= 200 * 1024:
                            break
                    html_text = b"".join(chunks).decode("utf-8", errors="replace")
                    meta_img = extract_meta_image(html_text, base_url=str(article_url))
                    if meta_img:
                        rem_val = min(5.0, max(1.0, max_budget - (time.monotonic() - start_time)))
                        if validate_image_url(meta_img, timeout=rem_val):
                            return meta_img
                except Exception as fetch_exc:
                    logger.debug("Failed fetching page %s for og:image: %s", article_url, fetch_exc)

    except Exception as exc:
        logger.debug("Unexpected error in get_image_url: %s", exc)

    return FALLBACK_IMAGE_URL
