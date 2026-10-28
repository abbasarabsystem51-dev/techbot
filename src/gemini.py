"""
src/gemini.py — کلاینت مقاوم و بهینه Gemini API
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai

from src.config import AppConfig
from src.logging_config import get_logger
from src.models import Article, GeminiNewsResponse
from src.state import StateManager
from src.utils import project_root

logger = get_logger(__name__)


class GeminiAPIError(RuntimeError):
    """خطای غیرمنتظره در فراخوانی Gemini API."""


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    """Fault-tolerant JSON parser with automatic partial-batch recovery."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^`{1,3}(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*`{1,3}$", "", cleaned)
    cleaned = cleaned.strip()

    # Attempt 1: Standard non-strict parse
    try:
        data = json.loads(cleaned, strict=False)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"posts": data}
    except Exception:
        pass

    # Attempt 2: Stream-extract all completed post objects if truncated
    decoder = json.JSONDecoder(strict=False)
    posts: list[dict[str, Any]] = []
    arr_match = re.search(r'"posts"\s*:\s*\[', cleaned)
    idx = arr_match.end() if arr_match else 0

    while idx < len(cleaned):
        brace_idx = cleaned.find("{", idx)
        if brace_idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(cleaned, brace_idx)
            if isinstance(obj, dict) and ("article_id" in obj or "text" in obj):
                posts.append(obj)
            idx = end_idx
        except Exception:
            idx = brace_idx + 1

    if posts:
        logger.warning("Recovered %d complete posts from truncated Gemini JSON.", len(posts))
        return {"posts": posts}

    # Attempt 3: Auto-close single truncated post
    for suffix in ['"}]}', '"}]', '"}']:
        try:
            repaired = cleaned + suffix
            data = json.loads(repaired, strict=False)
            if isinstance(data, dict) and data.get("posts"):
                return data
        except Exception:
            continue

    return json.loads(cleaned, strict=False)


def _sanitize_gemini_payload(
    data: dict[str, Any], articles_batch: list[Article]
) -> dict[str, Any]:
    """Sanitize fields so a minor length overflow never aborts the whole batch."""
    raw_posts = data.get("posts", [])
    if not isinstance(raw_posts, list):
        return {"posts": []}

    article_map = {a.id: a for a in articles_batch}
    fallback_article = articles_batch[0] if articles_batch else None
    clean_posts: list[dict[str, Any]] = []

    for item in raw_posts:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if len(text) > 800:
            text = text[:797].rsplit(" ", 1)[0] + "..."

        art_id = str(item.get("article_id") or "").strip()
        matched_art = article_map.get(art_id, fallback_article)
        if not art_id and matched_art:
            art_id = matched_art.id

        cat = str(item.get("category") or "").lower().strip()
        if cat not in ("ai", "tech"):
            cat = matched_art.category.value if matched_art else "ai"

        src_url = str(item.get("source_url") or "").strip()
        if not src_url.startswith("http") and matched_art:
            src_url = matched_art.canonical_url

        src_title = str(item.get("source_title") or "").strip()
        if not src_title and matched_art:
            src_title = matched_art.title[:80]
        if len(src_title) > 80:
            src_title = src_title[:77] + "..."

        clean_posts.append(
            {
                "article_id": art_id,
                "text": text,
                "category": cat,
                "source_url": src_url,
                "source_title": src_title,
            }
        )

    return {"posts": clean_posts}


def _read_prompt(name: str) -> str:
    path: Path = project_root() / "prompts" / name
    if not path.exists():
        logger.warning("Prompt file not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8").strip()


def _articles_to_json(articles: list[Article]) -> str:
    items: list[dict[str, Any]] = []
    for a in articles:
        items.append(
            {
                "id": a.id,
                "title": a.title,
                "url": a.canonical_url,
                "summary": a.summary[:450],
                "source_name": a.source_name,
                "category": a.category.value,
                "score": round(a.score, 4),
            }
        )
    return json.dumps(items, ensure_ascii=False, indent=2)


def _build_batch_prompt(articles: list[Article], system_prompt: str, news_prompt: str) -> str:
    schema_description = """{
  "posts": [
    {
      "article_id": "<same id as in the input articles array>",
      "text": "<Farsi post text, max 700 characters, HTML tags allowed: <b>, <i>, <code>, <a href='...'>. NO footer, NO hashtags, NO @channel handle.>",
      "category": "<'ai' or 'tech'>",
      "source_url": "<original article URL>",
      "source_title": "<short descriptive title in Farsi, max 70 chars>"
    }
  ]
}"""

    articles_json = _articles_to_json(articles)

    prompt_parts = [
        system_prompt,
        "",
        "---",
        news_prompt if news_prompt else "Write engaging Farsi tech news posts.",
        "",
        "---",
        "## دستورالعمل خروجی",
        "ورودی: آرایه‌ای از مقالات RSS به زبان انگلیسی.",
        "خروجی: فقط یک JSON معتبر طبق ساختار زیر و بدون هیچ توضیح اضافی:",
        schema_description,
        "",
        "قوانین الزامی:",
        "- برای هر مقاله یک post فارسی روان، دقیق و خلاصه (حداکثر ۶۵۰ کاراکتر) بنویس.",
        "- فقط از تگ‌های HTML مجاز تلگرام استفاده کن: <b>, <i>, <code>",
        "- هیچ فوتر، هشتگ یا آیدی کانال اضافه نکن.",
        "- article_id باید دقیقاً همان id ورودی باشد.",
        "",
        "## مقالات ورودی",
        articles_json,
    ]
    return "\n".join(prompt_parts)


def call_gemini_batch(
    articles: list[Article],
    state: StateManager,
    cfg: AppConfig,
) -> GeminiNewsResponse:
    if not articles:
        raise GeminiAPIError("No articles provided for batch processing.")

    batch_size = cfg.gemini_batch_size
    articles_batch = articles[:batch_size]

    logger.info(
        "Calling Gemini batch API: model=%s, articles=%d",
        cfg.gemini_model,
        len(articles_batch),
    )

    genai.configure(api_key=cfg.settings.gemini_api_key)

    system_prompt = _read_prompt("system.md")
    news_prompt = _read_prompt("news.md")
    full_prompt = _build_batch_prompt(articles_batch, system_prompt, news_prompt)

    gemini_cfg = cfg.yaml.get("gemini", {})
    temperature = float(gemini_cfg.get("temperature", 0.4))
    max_output_tokens = int(gemini_cfg.get("max_output_tokens", 32768))
    response_mime_type = gemini_cfg.get("response_mime_type", "application/json")

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=response_mime_type,
    )

    with state.gemini_quota_context():
        last_exc: Exception | None = None
        response = None
        for attempt in range(1, 4):
            try:
                model = genai.GenerativeModel(cfg.gemini_model)
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )
                break
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if ("429" in err_str or "503" in err_str or "quota" in err_str.lower()) and attempt < 3:
                    wait_sec = 20 * attempt
                    logger.warning(
                        "Transient Gemini error (attempt %d/3): %s. Retrying in %ds...",
                        attempt,
                        exc,
                        wait_sec,
                    )
                    time.sleep(wait_sec)
                else:
                    logger.error("Gemini API call failed: %s", exc)
                    raise GeminiAPIError(f"Gemini API call failed: {exc}") from exc

        if response is None:
            raise GeminiAPIError(f"Gemini API call failed after retries: {last_exc}")

        raw_text = response.text
        logger.debug("Gemini raw response (%d chars): %s...", len(raw_text), raw_text[:200])

        try:
            raw_data = _safe_json_loads(raw_text)
            sanitized_data = _sanitize_gemini_payload(raw_data, articles_batch)
        except Exception as exc:
            logger.error("Failed to parse Gemini JSON response: %s\nRaw: %s", exc, raw_text[:600])
            raise GeminiAPIError(f"Gemini returned invalid JSON: {exc}") from exc

        try:
            validated = GeminiNewsResponse.model_validate(sanitized_data)
        except Exception as exc:
            logger.error("Pydantic validation failed: %s\nData: %s", exc, sanitized_data)
            raise GeminiAPIError(f"Gemini response failed Pydantic validation: {exc}") from exc

    logger.info("Gemini batch complete: %d posts returned.", len(validated.posts))
    return validated
