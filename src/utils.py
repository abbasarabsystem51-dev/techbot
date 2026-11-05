"""
src/utils.py — توابع کمکی عمومی

توابع اینجا هیچ وابستگی درونی به بقیه ماژول‌های src ندارند.
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """Remove all HTML tags from *text* and unescape HTML entities."""
    # Remove tags
    cleaned = re.sub(r"<[^>]+>", "", text)
    # Unescape HTML entities (e.g. &amp; → &)
    cleaned = html.unescape(cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def truncate(text: str, max_chars: int, ellipsis: str = "…") -> str:
    """Truncate *text* to at most *max_chars* characters, appending *ellipsis* if cut."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ellipsis)].rstrip() + ellipsis


def normalize_whitespace(text: str) -> str:
    """Collapse all consecutive whitespace (including newlines) into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def count_chars(text: str) -> int:
    """Return the number of Unicode characters (code points) in *text*."""
    return len(text)


def persian_digit(n: int) -> str:
    """Convert an integer to its Persian/Arabic numeral string representation."""
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    return "".join(persian_digits[int(d)] for d in str(n))


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form (recommended for Persian text)."""
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------

def canonical_url(url: str) -> str:
    """
    Return a canonical form of *url*:
    - scheme and host lowercased
    - fragment removed
    - trailing slash on path normalised
    """
    try:
        parsed = urlparse(url.strip())
        # Lowercase scheme and netloc
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        # Remove fragment
        canonical = urlunparse((scheme, netloc, parsed.path, parsed.params, parsed.query, ""))
        return canonical
    except Exception:
        return url.strip()


def is_valid_url(url: str) -> bool:
    """Return True if *url* is a syntactically valid HTTP/HTTPS URL."""
    try:
        result = urlparse(url)
        return result.scheme in {"http", "https"} and bool(result.netloc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of *text* (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(url: str, title: str, summary: str = "") -> str:
    """
    Produce a stable deduplication hash from (canonical_url, title, summary).
    Uses SHA-256 of the concatenated canonical url + normalised title.
    """
    key = canonical_url(url) + "|" + normalize_whitespace(title) + "|" + normalize_whitespace(summary)
    return sha256_hex(key)


# ---------------------------------------------------------------------------
# Date / time utilities
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def to_iso8601(dt: datetime) -> str:
    """Return *dt* as an ISO-8601 string with 'Z' UTC suffix."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso8601(s: str) -> datetime:
    """Parse an ISO-8601 string (with or without 'Z') to a timezone-aware datetime."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def read_text(path: Path | str, encoding: str = "utf-8") -> str:
    """Read and return the contents of a text file."""
    return Path(path).read_text(encoding=encoding)


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and any parents) if it does not exist. Return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the project root directory (two levels above this file: src/../)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely navigate nested dict *d* using *keys*.
    Returns *default* if any key is missing or the value is None.
    """
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* between *lo* and *hi* (inclusive)."""
    return max(lo, min(hi, value))
