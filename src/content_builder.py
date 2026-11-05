"""
src/content_builder.py — مونتاژ نهایی پست‌های قابل انتشار

مسئولیت‌ها:
  - ساخت GeneratedPost از پاسخ Gemini
  - افزودن فوتر فارسی، آیدی کانال @tech_news_110 و هشتگ‌های موضوعی
  - اعمال سقف ۱۰۲۴ کاراکتری بدون از دست دادن انسجام متنی

نکته مهم: فوتر، هشتگ، و آیدی کانال توسط این ماژول اضافه می‌شوند —
نه توسط مدل هوش مصنوعی.
"""
from __future__ import annotations

import re
from typing import Optional

from src.config import AppConfig
from src.logging_config import get_logger
from src.models import Article, GeminiNewsResponse, GeneratedPost, PostCategory, PostStatus
from src.utils import sha256_hex, utcnow

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?؟])\s+")


def _trim_to_limit(text: str, max_chars: int) -> str:
    """
    Trim *text* to at most *max_chars* characters.

    Strategy:
      1. If text already fits, return as-is.
      2. Try to cut at the last sentence boundary before the limit.
      3. If no boundary found, hard-cut and append ellipsis.
    """
    if len(text) <= max_chars:
        return text

    # Try sentence boundary trim
    truncated = text[:max_chars]
    sentences = _SENTENCE_BOUNDARY_RE.split(truncated)
    if len(sentences) > 1:
        # Keep all sentences except the last (incomplete) one
        candidate = " ".join(sentences[:-1]).rstrip()
        if candidate and len(candidate) <= max_chars:
            return candidate

    # Hard cut with ellipsis
    ellipsis = "…"
    return text[: max_chars - len(ellipsis)].rstrip() + ellipsis


def _get_hashtags(category: PostCategory, cfg: AppConfig) -> str:
    """Return the hashtag string for *category* from config."""
    hashtags_map: dict[str, str] = cfg.yaml.get("post", {}).get("hashtags", {})
    return hashtags_map.get(category.value, "")


def _find_article(article_id: str, articles: list[Article]) -> Optional[Article]:
    """Find an article by id in the list."""
    for a in articles:
        if a.id == article_id:
            return a
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_final_text(post: GeneratedPost, cfg: AppConfig) -> str:
    """
    Assemble the complete publishable post text.

    Structure (programmatically constructed — NOT by the AI model):
      [post.text]
      [footer: cfg.post_footer]
      [hashtags for category]

    Total must be <= cfg.post_max_chars (1024 chars).
    If it exceeds the limit, post.text is trimmed from the last
    sentence boundary outward until the total fits.

    Returns the fully assembled string ready for publishing.
    """
    footer: str = cfg.post_footer  # e.g. "\n\n📡 @tech_news_110"
    hashtags: str = _get_hashtags(post.category, cfg)
    suffix = footer + ("\n" + hashtags if hashtags else "")

    max_chars: int = cfg.post_max_chars  # 1024
    max_body = max_chars - len(suffix)

    if max_body <= 0:
        # Pathological config — return only suffix
        logger.error("post_max_chars is too small to fit even the footer.")
        return suffix.lstrip("\n")

    body = _trim_to_limit(post.text, max_body)
    assembled = body + suffix

    if len(assembled) > max_chars:
        # Safety trim — should not normally reach here
        logger.warning(
            "Post still exceeds limit after trim (%d > %d). Hard-cutting.",
            len(assembled),
            max_chars,
        )
        body = body[: max_body - 1].rstrip() + "…"
        assembled = body + suffix

    return assembled


def build_posts(
    response: GeminiNewsResponse,
    articles: list[Article],
    cfg: AppConfig,
) -> list[GeneratedPost]:
    """
    Convert Gemini API response items into validated GeneratedPost instances.

    For each item:
      - Constructs a GeneratedPost.
      - Validates that assemble_final_text() fits within the 1024-char limit.
      - Trims post.text if needed.

    Returns the list of posts ready for scheduling.
    """
    posts: list[GeneratedPost] = []

    for item in response.posts:
        # Ensure text fits inside the assembled limit before we finalize
        # Build a scratch post to test length
        post_id = sha256_hex(item.article_id + item.text[:50])

        # Look up matching article for context (optional, not strictly required)
        matched_article = _find_article(item.article_id, articles)
        source_name = matched_article.source_name if matched_article else item.source_title

        draft_post = GeneratedPost(
            id=post_id,
            article_id=item.article_id,
            text=item.text,
            category=item.category,
            source_url=item.source_url,
            source_title=item.source_title,
            status=PostStatus.PENDING,
            created_at=utcnow(),
        )

        # Enforce total character limit
        footer: str = cfg.post_footer
        hashtags: str = _get_hashtags(draft_post.category, cfg)
        suffix = footer + ("\n" + hashtags if hashtags else "")
        max_body = cfg.post_max_chars - len(suffix)

        if max_body > 0 and len(draft_post.text) > max_body:
            trimmed_text = _trim_to_limit(draft_post.text, max_body)
            logger.info(
                "Post %s trimmed: %d → %d chars (limit=%d after footer).",
                post_id[:16],
                len(draft_post.text),
                len(trimmed_text),
                max_body,
            )
            draft_post = draft_post.model_copy(update={"text": trimmed_text})

        # Final sanity check
        final_text = assemble_final_text(draft_post, cfg)
        if len(final_text) > cfg.post_max_chars:
            logger.warning(
                "Post %s final text %d chars exceeds %d — skipping.",
                post_id[:16],
                len(final_text),
                cfg.post_max_chars,
            )
            continue

        posts.append(draft_post)
        logger.debug(
            "Built post %s: category=%s, text=%d chars, final=%d chars",
            post_id[:16],
            draft_post.category.value,
            len(draft_post.text),
            len(final_text),
        )

    logger.info("Content builder: %d posts built from %d Gemini items.", len(posts), len(response.posts))
    return posts
