"""
src/techbot/scoring.py — Local candidate scoring and deduplication engine.

Implements scoring rules from SPEC.md (Section 4 & 6) and scoring_weights.json:
  1. Freshness (0-3.0 pts): Age-based exponential decay (half-life 18h, max 48h).
  2. Topic Relevance (0-3.0 pts): High, medium, and penalized keyword matching with AI basket boost.
  3. Source Authority (0-2.0 pts): Authority weights per source.
  4. Recency & Uniqueness (0-2.0 pts): URL normalization and title-similarity deduplication against sent_links and recent_titles.
"""
from __future__ import annotations

import difflib
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.models import Item, PostBasket, PostCategory
from src.utils import clamp, normalize_whitespace, utcnow

logger = logging.getLogger(__name__)

# Tracking query parameters to strip during URL normalization
TRACKING_PARAMS: frozenset[str] = frozenset({
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "ref",
    "source",
    "fbclid",
    "gclid",
    "ncid",
    "sr_share",
    "mc_cid",
    "mc_eid",
    "guccounter",
    "guce_referrer",
    "guce_referrer_sig",
})


# ---------------------------------------------------------------------------
# URL Normalization & Title Similarity
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """
    Produce a canonical, clean URL stripped of tracking parameters, fragments,
    and extraneous trailing slashes.
    """
    if not url or not url.strip():
        return ""

    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()

        # Remove www. prefix for consistent comparison
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # Strip tracking parameters
        clean_query_pairs = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in TRACKING_PARAMS
        ]
        # Sort query params for deterministic string representation
        clean_query_pairs.sort(key=lambda pair: pair[0])
        clean_query = urlencode(clean_query_pairs)

        # Remove trailing slash from path unless it's just '/'
        path = parsed.path
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")

        return urlunparse((scheme, netloc, path, "", clean_query, ""))
    except Exception:
        return url.strip()


def calculate_title_similarity(t1: str, t2: str) -> float:
    """
    Compute similarity between two titles using a combination of
    token overlap (Jaccard) and sequence character matcher.
    Returns float in range [0.0, 1.0].
    """
    s1 = normalize_whitespace(re.sub(r"[^\w\s]", "", t1.lower()))
    s2 = normalize_whitespace(re.sub(r"[^\w\s]", "", t2.lower()))

    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    # Sequence matcher ratio
    seq_ratio = difflib.SequenceMatcher(None, s1, s2).ratio()

    # Token-level Jaccard similarity
    words1 = set(s1.split())
    words2 = set(s2.split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    jaccard_ratio = len(intersection) / len(union) if union else 0.0

    return max(seq_ratio, jaccard_ratio)


# ---------------------------------------------------------------------------
# Scorer Class
# ---------------------------------------------------------------------------

class Scorer:
    """
    Local scoring and ranking engine for candidate items.
    """

    def __init__(self, weights_path: Optional[Path | str] = None) -> None:
        self.weights = self._load_weights(weights_path)

        # Keyword sets
        kw_cfg = self.weights.get("keyword_weights", {})
        self.high_priority_kws = [k.lower() for k in kw_cfg.get("high_priority", {}).get("keywords", [])]
        self.medium_priority_kws = [k.lower() for k in kw_cfg.get("medium_priority", {}).get("keywords", [])]
        self.penalized_kws = [k.lower() for k in kw_cfg.get("penalized_keywords", {}).get("keywords", [])]

        # Source authorities
        src_auth = self.weights.get("item_selection", {}).get("criteria", {}).get("source_authority", {})
        self.source_authorities: dict[str, float] = src_auth.get("sources", {
            "OpenAI Blog": 2.0,
            "Google AI Blog": 2.0,
            "TechCrunch AI": 1.8,
            "The Verge AI": 1.6,
            "Ars Technica": 1.5,
            "Hacker News AI": 1.3,
        })
        self.default_source_weight: float = src_auth.get("default_source_weight", 1.0)

    def _load_weights(self, path: Optional[Path | str]) -> dict[str, Any]:
        """Load scoring weights from JSON file or fallback to defaults."""
        target_path = Path(path) if path else Path(__file__).resolve().parents[2] / "scoring_weights.json"
        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not read %s (%s). Using defaults.", target_path, exc)

        return {
            "item_selection": {
                "max_score": 10.0,
                "criteria": {
                    "freshness": {"weight": 3.0, "max_age_hours": 48, "half_life_hours": 18},
                    "topic_relevance": {"weight": 3.0, "ai_boost": 1.5},
                    "source_authority": {"weight": 2.0, "default_source_weight": 1.0},
                    "recency_uniqueness": {"weight": 2.0, "penalty_for_duplicate": -10.0},
                },
            },
            "keyword_weights": {
                "high_priority": {"multiplier": 1.5, "keywords": ["ai", "llm", "gpt", "gemini", "claude", "openai", "agent", "rag"]},
                "medium_priority": {"multiplier": 1.0, "keywords": ["machine learning", "neural", "gpu", "robotics", "open source"]},
                "penalized_keywords": {"multiplier": -2.0, "keywords": ["crypto", "memecoin", "giveaway", "rumor"]},
            },
        }

    # 1. Freshness Score (0.0 to 3.0)
    def calculate_freshness(self, published_at: Optional[datetime]) -> float:
        """
        Calculates freshness score up to 3.0 points.
        Items under 6 hours old receive 3.0 points.
        Between 6h and 48h: exponential decay based on 18h half-life.
        Older than 48h: 0.0 points.
        """
        if published_at is None:
            return 1.5  # Neutral default for undated articles

        now = utcnow()
        age_hours = (now - published_at).total_seconds() / 3600.0
        if age_hours <= 0:
            return 3.0
        if age_hours <= 6.0:
            return 3.0
        if age_hours > 48.0:
            return 0.0

        # Decay: 3.0 * (0.5 ** ((age - 6) / 18))
        decay_factor = 0.5 ** ((age_hours - 6.0) / 18.0)
        return clamp(3.0 * decay_factor, 0.0, 3.0)

    # 2. Topic Relevance Score (0.0 to 3.0)
    def calculate_relevance(
        self,
        title: str,
        summary: str,
        basket: Optional[PostBasket | str] = None,
        category: Optional[PostCategory | str] = None,
    ) -> float:
        """
        Calculates keyword relevance score up to 3.0 points.
        High priority (+1.5 per unique match), medium (+1.0), penalized (-2.0).
        Applies AI boost for AI-specific baskets or categories.
        """
        text = f"{title} {summary}".lower()

        # Count occurrences of keywords
        high_hits = sum(1 for kw in self.high_priority_kws if kw in text)
        med_hits = sum(1 for kw in self.medium_priority_kws if kw in text)
        pen_hits = sum(1 for kw in self.penalized_kws if kw in text)

        score = (high_hits * 1.5) + (med_hits * 1.0) - (pen_hits * 2.0)

        # Basket/Category AI boost
        is_ai = False
        basket_val = str(basket).lower() if basket else ""
        category_val = str(category).lower() if category else ""
        if "ai" in basket_val or "ai" in category_val or category == PostCategory.AI:
            is_ai = True

        if is_ai and score > 0:
            score *= 1.5

        # Normalize into [0.0, 3.0]
        return clamp(score, 0.0, 3.0)

    # 3. Source Authority Score (0.0 to 2.0)
    def calculate_authority(self, source_name: str) -> float:
        """Returns source authority score up to 2.0 points."""
        weight = self.source_authorities.get(source_name, self.default_source_weight)
        return clamp(weight, 0.0, 2.0)

    # 4. Recency & Uniqueness Check (0.0 to 2.0)
    def check_uniqueness(
        self,
        item: Item,
        sent_links: Optional[set[str]] = None,
        recent_titles: Optional[Sequence[str]] = None,
        similarity_threshold: float = 0.8,
    ) -> tuple[float, bool, Optional[str]]:
        """
        Checks item against previously published links and titles.
        Returns: (uniqueness_score, is_duplicate, duplicate_reason)
        """
        canon_url = normalize_url(item.url)

        # 1. Exact URL match
        if sent_links and (canon_url in sent_links or item.canonical_url in sent_links):
            return -10.0, True, "duplicate_url"

        # 2. Title similarity check
        if recent_titles:
            for past_title in recent_titles:
                sim = calculate_title_similarity(item.title, past_title)
                if sim >= similarity_threshold:
                    return -10.0, True, f"title_similarity ({sim:.2f} with '{past_title[:30]}')"
                if sim >= 0.60:
                    # Near duplicate gets penalized but not outright banned
                    return 0.5, False, None

        return 2.0, False, None

    # Total score calculation
    def score_item(
        self,
        item: Item,
        basket: Optional[PostBasket | str] = None,
        sent_links: Optional[set[str]] = None,
        recent_titles: Optional[Sequence[str]] = None,
    ) -> float:
        """
        Computes composite score in range [0.0, 10.0] and marks duplicate status on the item.
        """
        freshness = self.calculate_freshness(item.published_at)
        relevance = self.calculate_relevance(item.title, item.summary, basket=basket, category=item.category)
        authority = self.calculate_authority(item.source_name)
        uniqueness, is_dup, reason = self.check_uniqueness(item, sent_links, recent_titles)

        if is_dup:
            item.is_duplicate = True
            item.duplicate_reason = reason
            item.score = 0.0
            return 0.0

        total = freshness + relevance + authority + uniqueness
        item.score = round(clamp(total, 0.0, 10.0), 2)
        return item.score

    def rank_candidates(
        self,
        items: list[Item],
        basket: PostBasket | str,
        sent_links: Optional[set[str]] = None,
        recent_titles: Optional[Sequence[str]] = None,
        limit: int = 5,
    ) -> list[Item]:
        """
        Scores all candidates for the target basket, filters out duplicates,
        and returns the top `limit` items sorted by score descending.
        """
        scored_items: list[Item] = []
        for item in items:
            self.score_item(item, basket=basket, sent_links=sent_links, recent_titles=recent_titles)
            if not item.is_duplicate:
                scored_items.append(item)

        scored_items.sort(key=lambda it: it.score, reverse=True)
        return scored_items[:limit]
