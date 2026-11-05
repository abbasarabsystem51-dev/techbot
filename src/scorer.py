"""
src/scorer.py — موتور امتیازدهی مقالات

سه بُعد امتیاز:
  1. تازگی (Freshness)  — وزن ۰.۴ — کاهش نمایی بر پایه ۷ روز
  2. ارتباط با AI       — وزن ۰.۴ — تطابق کلیدواژه + پاداش دسته‌بندی
  3. اعتبار منبع        — وزن ۰.۲ — نرمال‌شده از weight در config.yaml
"""
from __future__ import annotations

import math
from typing import Any, Optional

from src.config import AppConfig
from src.logging_config import get_logger
from src.models import Article, PostCategory
from src.utils import clamp, utcnow

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_source_weight(source_name: str, sources: list[dict[str, Any]]) -> float:
    """Return the configured weight for *source_name*, defaulting to 1.0."""
    for src in sources:
        if src.get("name") == source_name:
            return float(src.get("weight", 1.0))
    return 1.0


def _freshness_score(article: Article) -> float:
    """
    Exponential decay freshness score.

    Half-decay period = 7 days (168 hours).
    Returns 1.0 for brand-new articles, approaching 0 for old ones.
    If no published_at, returns a neutral 0.5.
    """
    if article.published_at is None:
        return 0.5

    now = utcnow()
    try:
        age_hours = (now - article.published_at).total_seconds() / 3600.0
        if age_hours < 0:
            age_hours = 0.0
        # decay constant: λ = ln(2) / (7 * 24) so that score halves every 7 days
        decay_lambda = math.log(2) / (7.0 * 24.0)
        return clamp(math.exp(-decay_lambda * age_hours), 0.0, 1.0)
    except Exception:  # noqa: BLE001
        return 0.5


def _relevance_score(article: Article, ai_keywords: list[str]) -> float:
    """
    Keyword-based AI relevance score.

    Counts how many keywords appear in (title + summary), normalized by 3 hits.
    AI-category articles get a +0.15 bonus.
    """
    search_text = (article.title + " " + article.summary).lower()
    hits = sum(1 for kw in ai_keywords if kw.lower() in search_text)
    score = min(1.0, hits / 3.0)

    # Bonus for explicitly AI-categorized sources
    if article.category == PostCategory.AI:
        score = clamp(score + 0.15, 0.0, 1.0)

    return score


def _authority_score(article: Article, sources: list[dict[str, Any]]) -> float:
    """
    Source authority score normalized to [0, 1].

    Max configured weight is 1.5 (OpenAI Blog), so we divide by 1.5.
    """
    weight = _get_source_weight(article.source_name, sources)
    max_weight = 1.5  # as configured in config.yaml
    return clamp(weight / max_weight, 0.0, 1.0)


# Re-export Part 2 Scorer & Deduplication
from src.techbot.scoring import (
    Scorer,
    calculate_title_similarity,
    normalize_url,
)

__all__ = [
    "score_article",
    "rank_articles",
    "Scorer",
    "normalize_url",
    "calculate_title_similarity",
]
# ---------------------------------------------------------------------------

def score_article(article: Article, cfg: AppConfig) -> float:
    """
    Compute a composite relevance+freshness score for *article*.

    Returns a float in [0.0, 1.0].
    """
    scoring_cfg = cfg.yaml.get("scoring", {})
    ai_keywords: list[str] = scoring_cfg.get("ai_keywords", [])

    freshness = _freshness_score(article)
    relevance = _relevance_score(article, ai_keywords)
    authority = _authority_score(article, cfg.sources)

    score = clamp(
        0.4 * freshness + 0.4 * relevance + 0.2 * authority,
        0.0,
        1.0,
    )

    logger.debug(
        "Score for '%s': freshness=%.3f relevance=%.3f authority=%.3f → %.3f",
        article.title[:50],
        freshness,
        relevance,
        authority,
        score,
    )
    return score


def score_articles(articles: list[Article], cfg: AppConfig) -> list[Article]:
    """
    Score all *articles* and return a new list sorted by score descending.

    Each article is updated via `model_copy` (immutable Pydantic pattern).
    """
    scored: list[Article] = []
    for article in articles:
        new_score = score_article(article, cfg)
        scored.append(article.model_copy(update={"score": new_score}))

    scored.sort(key=lambda a: a.score, reverse=True)
    logger.info(
        "Scored %d articles. Top score: %.3f, Bottom: %.3f",
        len(scored),
        scored[0].score if scored else 0.0,
        scored[-1].score if scored else 0.0,
    )
    return scored
