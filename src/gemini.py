"""
src/gemini.py — کلاینت Gemini API

الزام حیاتی: هر فراخوانی API باید درون gemini_quota_context() از state.py باشد.
این ماژول فقط یک درخواست دسته‌ای روزانه ارسال می‌کند.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import google.generativeai as genai

from src.config import AppConfig
from src.logging_config import get_logger
from src.models import Article, GeminiNewsResponse
from src.state import StateManager
from src.utils import project_root

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GeminiAPIError(RuntimeError):
    """خطای غیرمنتظره در فراخوانی Gemini API."""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _read_prompt(name: str) -> str:
    """Read a prompt file from the prompts/ directory."""
    path: Path = project_root() / "prompts" / name
    if not path.exists():
        logger.warning("Prompt file not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8").strip()


def _articles_to_json(articles: list[Article]) -> str:
    """Serialize articles to a compact JSON array for the prompt."""
    items: list[dict[str, Any]] = []
    for a in articles:
        items.append(
            {
                "id": a.id,
                "title": a.title,
                "url": a.canonical_url,
                "summary": a.summary[:500],
                "source_name": a.source_name,
                "category": a.category.value,
                "score": round(a.score, 4),
            }
        )
    return json.dumps(items, ensure_ascii=False, indent=2)


def _build_batch_prompt(articles: list[Article], system_prompt: str, news_prompt: str) -> str:
    """
    Construct the full batch prompt for Gemini.

    Instructs the model to select, summarize, and output valid JSON conforming
    to GeminiNewsResponse schema.
    """
    schema_description = """{
  "posts": [
    {
      "article_id": "<same id as in the input articles array>",
      "text": "<Farsi post text, max 800 characters, HTML tags allowed: <b>, <i>, <code>, <a href='...'>. NO footer, NO hashtags, NO @channel handle.>",
      "category": "<'ai' or 'tech'>",
      "source_url": "<original article URL>",
      "source_title": "<short descriptive title in Farsi, max 80 chars>"
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
        "",
        "ورودی: آرایه‌ای از مقالات RSS به زبان انگلیسی.",
        "خروجی: یک JSON object معتبر با schema زیر (و فقط همین JSON، بدون هیچ متن اضافه):",
        "",
        schema_description,
        "",
        "قوانین مهم:",
        "- برای هر مقاله‌ای که انتخاب می‌کنی یک post بنویس.",
        "- متن post باید به فارسی روان باشد.",
        "- از تگ‌های HTML مجاز تلگرام استفاده کن: <b>, <i>, <code>, <a href='...'>",
        "- هیچ عبارت ممنوعه‌ای ننویس: «در دنیای امروز»، «بی‌شک»، «انقلابی»، «بازی را عوض کرد»، «شگفت‌انگیز»، «متحول می‌کند»",
        "- هیچ فوتر، هشتگ، یا آیدی کانال اضافه نکن — کد پایتون این‌ها را اضافه می‌کند.",
        "- هیچ عدد، قیمت یا تاریخ توهمی ننویس — فقط اطلاعاتی که در متن منبع وجود دارد.",
        "- متن هر post حداکثر ۸۰۰ کاراکتر باشد.",
        "- article_id باید دقیقاً همان id از آرایه ورودی باشد.",
        "",
        "## مقالات ورودی",
        "",
        articles_json,
    ]
    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_gemini_batch(
    articles: list[Article],
    state: StateManager,
    cfg: AppConfig,
) -> GeminiNewsResponse:
    """
    Call Gemini API with a batch of articles.

    CRITICAL: This function MUST be called (and IS called) within
    state.gemini_quota_context() to enforce the 1-request-per-day limit.

    Raises:
        GeminiAPIError: on JSON parse failure, Pydantic validation failure,
                        or any unrecoverable API error.
        GeminiQuotaExhaustedError: propagated from state.gemini_quota_context()
                                   if quota is already consumed.
    """
    if not articles:
        raise GeminiAPIError("No articles provided for batch processing.")

    # Enforce batch size limit
    batch_size = cfg.gemini_batch_size
    articles_batch = articles[:batch_size]

    logger.info(
        "Calling Gemini batch API: model=%s, articles=%d",
        cfg.gemini_model,
        len(articles_batch),
    )

    # Configure the SDK
    genai.configure(api_key=cfg.settings.gemini_api_key)

    # Read prompts from disk
    system_prompt = _read_prompt("system.md")
    news_prompt = _read_prompt("news.md")

    # Build the combined prompt
    full_prompt = _build_batch_prompt(articles_batch, system_prompt, news_prompt)

    gemini_cfg = cfg.yaml.get("gemini", {})
    temperature = float(gemini_cfg.get("temperature", 0.7))
    max_output_tokens = int(gemini_cfg.get("max_output_tokens", 8192))
    response_mime_type = gemini_cfg.get("response_mime_type", "application/json")

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=response_mime_type,
    )

    # --- All Gemini API interaction wrapped in quota context ---
    with state.gemini_quota_context():
        try:
            model = genai.GenerativeModel(cfg.gemini_model)
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise GeminiAPIError(f"Gemini API call failed: {exc}") from exc

        raw_text = response.text
        logger.debug("Gemini raw response (%d chars): %s...", len(raw_text), raw_text[:200])

        # Parse JSON
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON response: %s\nRaw: %s", exc, raw_text[:500])
            raise GeminiAPIError(f"Gemini returned invalid JSON: {exc}") from exc

        # Validate with Pydantic
        try:
            validated = GeminiNewsResponse.model_validate(data)
        except Exception as exc:
            logger.error("Pydantic validation of Gemini response failed: %s\nData: %s", exc, data)
            raise GeminiAPIError(f"Gemini response failed Pydantic validation: {exc}") from exc

    logger.info(
        "Gemini batch complete: %d posts returned.", len(validated.posts)
    )
    return validated
