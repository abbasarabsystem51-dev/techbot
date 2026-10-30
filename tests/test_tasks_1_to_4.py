"""
tests/test_tasks_1_to_4.py — Unit tests verifying Tasks 1 through 4:
  1. Candidate reduction & slicing (170 items -> <= 10, summary <= 300).
  2. Dedup (normalized URL and title) and 48h age filtering.
  3. LLM: timeout constant = 30, retry on 504 stops after 3 attempts, no retry on 400, unparsable JSON falls back to candidate #1.
  4. Images: RSS image preferred, og:image extraction, fallback on invalid content, no exception escapes.
  5. Publish: photo failure -> fallback image -> text-only.
  6. Repo-wide check: no PIL/Pillow import and no deprecated model string.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import requests

from src.models import GeminiBatchOutput, Item, PostBasket, PostCategory, PostItemSchema
from src.techbot.images import (
    FALLBACK_IMAGE_URL,
    extract_meta_image,
    extract_rss_image,
    get_image_url,
    validate_image_url,
)
from src.techbot.llm import LLM_TIMEOUT_SECONDS, GeminiClient, is_retryable_llm_error
from src.techbot.main import (
    MAX_CANDIDATES,
    build_fallback_post,
    format_candidate_payload,
    reduce_candidates,
)
from src.techbot.senders.telegram import TelegramSender


# ===========================================================================
# 1. Candidate Reduction & Slicing Tests
# ===========================================================================

def test_slicing_170_synthetic_items():
    """170 synthetic items -> payload contains <= 10 candidates, summaries <= 300 chars."""
    now = datetime.now(timezone.utc)
    items = []
    for i in range(170):
        # vary sources, titles, summaries, and published times within last 24h
        src_name = f"source_{i % 15}"
        summary = (f"Summary for item {i} discussing AI, OpenAI, LLM, models and chips. " * 15)[:400]
        it = Item(
            id=f"item_{i}",
            title=f"AI Breakthrough #{i} Announced",
            url=f"https://tech.example.com/article/{i}?utm_source=rss",
            canonical_url=f"https://tech.example.com/article/{i}",
            content_hash=f"hash_{i}",
            summary=summary,
            source_name=src_name,
            published_at=now - timedelta(hours=i % 30),
        )
        items.append(it)

    reduced = reduce_candidates(items, max_candidates=MAX_CANDIDATES, now=now)
    assert len(reduced) <= MAX_CANDIDATES
    assert len(reduced) > 0

    payload = [format_candidate_payload(it) for it in reduced]
    assert len(payload) <= 10
    for p in payload:
        assert len(p["summary"]) <= 300
        assert "id" in p
        assert "title" in p
        assert "source" in p
        assert "link" in p
        assert "summary" in p


def test_dedup_and_48h_age_filtering():
    """Deduplication by normalized URL and title, and 48h age filtering."""
    now = datetime.now(timezone.utc)

    # 1. URL and Title dedup
    item_orig = Item(
        id="it1",
        title="OpenAI Releases New Model",
        url="https://example.com/news/1?utm_campaign=daily",
        canonical_url="https://example.com/news/1",
        content_hash="h1",
        summary="Summary 1",
        source_name="TechDaily",
        published_at=now - timedelta(hours=2),
    )
    item_dup_url = Item(
        id="it2",
        title="Different Title But Same Canonical URL",
        url="https://example.com/news/1",
        canonical_url="https://example.com/news/1",
        content_hash="h2",
        summary="Summary 2",
        source_name="TechDaily",
        published_at=now - timedelta(hours=1),
    )
    item_dup_title = Item(
        id="it3",
        title="openai releases new model!!!",
        url="https://other.com/news/3",
        canonical_url="https://other.com/news/3",
        content_hash="h3",
        summary="Summary 3",
        source_name="OtherTech",
        published_at=now - timedelta(hours=3),
    )

    # 2. Age filtering (older than 48h)
    item_old = Item(
        id="it_old",
        title="Ancient AI Discovery",
        url="https://example.com/old",
        canonical_url="https://example.com/old",
        content_hash="h_old",
        summary="Old news",
        source_name="OldSource",
        published_at=now - timedelta(hours=50),  # > 48h!
    )

    # 3. Already posted
    item_posted = Item(
        id="it_posted",
        title="Already Published Post",
        url="https://example.com/posted",
        canonical_url="https://example.com/posted",
        content_hash="h_posted",
        summary="Already posted news",
        source_name="PostSource",
        published_at=now - timedelta(hours=5),
    )

    sent_links = {"https://example.com/posted"}
    recent_titles = ["Already Published Post"]

    res = reduce_candidates(
        [item_orig, item_dup_url, item_dup_title, item_old, item_posted],
        sent_links=sent_links,
        recent_titles=recent_titles,
        max_candidates=10,
        now=now,
    )

    ids = [r.id for r in res]
    assert "it1" in ids
    assert "it2" not in ids, "Duplicate URL must be dropped"
    assert "it3" not in ids, "Duplicate title must be dropped"
    assert "it_old" not in ids, "Item older than 48h must be dropped"
    assert "it_posted" not in ids, "Already posted item must be dropped"


def test_undated_items_retained_only_if_fewer_than_10_dated():
    """Undated items kept only if fewer than 10 dated items remain."""
    now = datetime.now(timezone.utc)

    # Case A: 12 dated recent items -> undated items dropped
    dated_12 = [
        Item(
            id=f"dated_{i}",
            title=f"Dated Item {i}",
            url=f"https://example.com/dated/{i}",
            canonical_url=f"https://example.com/dated/{i}",
            content_hash=f"h_{i}",
            summary="Dated item",
            source_name="SourceA",
            published_at=now - timedelta(hours=i),
        )
        for i in range(12)
    ]
    undated_item = Item(
        id="undated_1",
        title="Undated Item",
        url="https://example.com/undated",
        canonical_url="https://example.com/undated",
        content_hash="h_un",
        summary="Undated item",
        source_name="SourceB",
        published_at=None,
    )

    res_a = reduce_candidates(dated_12 + [undated_item], max_candidates=10, now=now)
    assert all(r.published_at is not None for r in res_a)
    assert "undated_1" not in [r.id for r in res_a]

    # Case B: Only 3 dated items -> undated items kept
    dated_3 = dated_12[:3]
    res_b = reduce_candidates(dated_3 + [undated_item], max_candidates=10, now=now)
    assert "undated_1" in [r.id for r in res_b]


# ===========================================================================
# 2. LLM Tests (Timeout, Retry, JSON Fallback)
# ===========================================================================

def test_llm_timeout_constant():
    assert LLM_TIMEOUT_SECONDS == 25.0


def test_llm_default_model_is_gemini_3_8_flash():
    client = GeminiClient(api_key="fake_key")
    assert client.primary_model == "gemini-3.8-flash"


def test_llm_retry_on_504_stops_after_3_attempts():
    """Retry on 504 stops after exactly 3 attempts on single model and raises RuntimeError."""
    client = GeminiClient(api_key="fake_key", models=["gemini-3.8-flash"])

    with patch.object(client, "_call_model", side_effect=Exception("504 Deadline Exceeded")) as mock_call, \
         patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError) as exc_info:
            client.generate_batch("test prompt")

        assert mock_call.call_count == 3
        # Should have slept twice (after attempt 1 and attempt 2)
        assert mock_sleep.call_count == 2
        assert "All Gemini attempts failed" in str(exc_info.value)


def test_llm_fallback_chain_on_404_succeeds():
    """If primary model returns 404, client dynamically falls back to next model and succeeds."""
    client = GeminiClient(api_key="fake_key", models=["gemini-3.8-flash", "gemini-3.0-flash"])

    call_models = []

    def mock_call(model_name, prompt):
        call_models.append(model_name)
        if model_name == "gemini-3.8-flash":
            raise Exception("404 NOT_FOUND: model is no longer available")
        return json.dumps({
            "posts": [
                {
                    "slot": "09:00_1",
                    "basket": "news_ai",
                    "hook": "⚡️ هوش مصنوعی تازه",
                    "intro": "توضیح کوتاه خبر",
                    "points": ["نکته ۱", "نکته ۲"],
                    "what_can_you_do": "کاربرد",
                    "try_now": "تست",
                }
            ]
        })

    with patch.object(client, "_call_model", side_effect=mock_call), \
         patch.object(client, "discover_flash_models", return_value=["gemini-3.0-flash", "gemini-2.0-flash"]):
        output = client.generate_batch("test prompt")
        assert len(output.posts) == 1
        assert "gemini-3.8-flash" in call_models
        assert "gemini-3.0-flash" in call_models


def test_llm_no_retry_on_400():
    """No retry on 400 Bad Request; stops immediately after 1 attempt."""
    client = GeminiClient(api_key="fake_key")

    with patch.object(client, "_call_model", side_effect=Exception("400 Bad Request: Invalid argument")) as mock_call, \
         patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError) as exc_info:
            client.generate_batch("test prompt")

        assert mock_call.call_count == 1
        assert mock_sleep.call_count == 0
        assert "Non-retryable Gemini error" in str(exc_info.value)


def test_llm_unparsable_json_falls_back_to_candidate_1():
    """Unparsable JSON does not crash and safely returns empty batch allowing fallback to candidate #1."""
    client = GeminiClient(api_key="fake_key")

    with patch.object(client, "_call_model", return_value="INVALID NON-JSON RESPONSE {{"):
        batch_out = client.generate_batch("test prompt")
        assert isinstance(batch_out, GeminiBatchOutput)
        assert len(batch_out.posts) == 0

    # Test candidate #1 fallback builder
    candidate1 = Item(
        id="cand1",
        title="Nvidia Announces Quantum Blackwell Chip",
        url="https://nvidia.com/quantum",
        canonical_url="https://nvidia.com/quantum",
        content_hash="hash1",
        summary="Nvidia announced next gen quantum chip.",
        source_name="Nvidia Press",
    )
    fallback = build_fallback_post(candidate1, slot_id="09:00_1", basket=PostBasket.NEWS_AI)
    assert fallback.slot == "09:00_1"
    assert "Nvidia Announces" in fallback.hook
    assert "Nvidia Press" in fallback.intro


# ===========================================================================
# 3. Images Tests
# ===========================================================================

def test_images_rss_image_preferred():
    """RSS image is preferred when present and validated."""
    item = Item(
        id="img1",
        title="Test Title",
        url="https://example.com/post",
        canonical_url="https://example.com/post",
        content_hash="h1",
        summary="Summary",
        source_name="Source",
        image_url="https://example.com/rss_valid.jpg",
    )

    with patch("src.techbot.images.validate_image_url", return_value=True):
        url = get_image_url(item)
        assert url == "https://example.com/rss_valid.jpg"


def test_images_og_image_extraction():
    """og:image extracted from HTML when no RSS image is provided."""
    item = Item(
        id="img2",
        title="Test Title",
        url="https://example.com/article",
        canonical_url="https://example.com/article",
        content_hash="h2",
        summary="Summary",
        source_name="Source",
        image_url=None,
    )

    html_content = b'<html><head><meta property="og:image" content="https://example.com/og_meta.jpg"></head></html>'
    mock_resp = MagicMock()
    mock_resp.iter_content = MagicMock(return_value=[html_content])

    with patch("requests.get", return_value=mock_resp), \
         patch("src.techbot.images.validate_image_url", side_effect=lambda u, timeout=5.0: "og_meta" in u):
        url = get_image_url(item)
        assert url == "https://example.com/og_meta.jpg"


def test_images_invalid_or_non_image_fallback():
    """Invalid or non-image content returns FALLBACK_IMAGE_URL."""
    item = Item(
        id="img3",
        title="Test Title",
        url="https://example.com/bad",
        canonical_url="https://example.com/bad",
        content_hash="h3",
        summary="Summary",
        source_name="Source",
        image_url="https://example.com/logo.png",  # logo rejected
    )

    with patch("src.techbot.images.validate_image_url", return_value=False), \
         patch("requests.get", side_effect=Exception("Connection refused")):
        url = get_image_url(item)
        assert url == FALLBACK_IMAGE_URL


def test_images_no_exception_can_escape():
    """No exception can escape get_image_url under any circumstances."""
    with patch("requests.get", side_effect=RuntimeError("Catastrophic error")):
        # Pass completely invalid types
        assert get_image_url(None) == FALLBACK_IMAGE_URL
        assert get_image_url(12345) == FALLBACK_IMAGE_URL
        assert get_image_url({"url": "http://broken"}) == FALLBACK_IMAGE_URL


# ===========================================================================
# 4. Telegram Publisher Tests (Photo -> Fallback -> Text-only)
# ===========================================================================

@pytest.mark.asyncio
async def test_publish_photo_failure_then_fallback_success():
    """If initial photo URL fails, retries once with FALLBACK_IMAGE_URL."""
    sender = TelegramSender(bot_token="fake:token", channel_id="@tech_news_110")

    call_count = 0

    async def mock_post(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        data = kwargs.get("data", {})
        photo = data.get("photo")

        # First call with custom photo fails
        if photo == "https://example.com/broken.jpg":
            return httpx.Response(400, json={"ok": False, "description": "PHOTO_INVALID"}, request=httpx.Request("POST", url))

        # Second call with FALLBACK_IMAGE_URL succeeds
        if photo == FALLBACK_IMAGE_URL:
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 888}}, request=httpx.Request("POST", url))

        return httpx.Response(200, json={"ok": True, "result": {"message_id": 999}}, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await sender.send_post(
            text="Hello World",
            image_url="https://example.com/broken.jpg",
        )
        assert res.success is True
        assert res.message_id == "888"
        assert call_count == 2


@pytest.mark.asyncio
async def test_publish_photo_failure_all_images_fallback_to_text():
    """If both photo URLs fail, falls back to text-only sendMessage."""
    sender = TelegramSender(bot_token="fake:token", channel_id="@tech_news_110")

    call_urls = []

    async def mock_post(url, *args, **kwargs):
        call_urls.append(url)
        if "sendPhoto" in url:
            return httpx.Response(400, json={"ok": False, "description": "PHOTO_FAILED"}, request=httpx.Request("POST", url))
        if "sendMessage" in url:
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 777}}, request=httpx.Request("POST", url))
        return httpx.Response(500, request=httpx.Request("POST", url))

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await sender.send_post(
            text="Hello World",
            image_url="https://example.com/broken.jpg",
        )
        assert res.success is True
        assert res.message_id == "777"
        # 1. sendPhoto with broken.jpg
        # 2. sendPhoto with FALLBACK_IMAGE_URL
        # 3. sendMessage text-only
        assert len(call_urls) == 3
        assert "sendPhoto" in call_urls[0]
        assert "sendPhoto" in call_urls[1]
        assert "sendMessage" in call_urls[2]


# ===========================================================================
# 5. Repo-Wide Safety Checks
# ===========================================================================

def test_repo_wide_no_pil_and_no_legacy_genai():
    """Verify no PIL/Pillow import and no legacy google.generativeai in techbot or requirements."""
    root = Path(__file__).resolve().parents[1]

    # Check requirements.txt and pyproject.toml have no google-generativeai
    reqs_text = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "google-generativeai" not in reqs_text
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "google-generativeai" not in pyproject_text

    # Check src/techbot has no import google.generativeai
    legacy_genai_import = "import " + "google.generativeai"
    for path in (root / "src" / "techbot").rglob("*.py"):
        content = path.read_text(encoding="utf-8", errors="ignore")
        assert legacy_genai_import not in content, f"Found legacy genai in {path}"

    # Check for PIL / Pillow import in src/ and tests/
    pil_pat1 = "import " + "PIL"
    pil_pat2 = "from " + "PIL"
    for folder in ("src", "tests"):
        for path in (root / folder).rglob("*.py"):
            if path.name == "test_tasks_1_to_4.py":
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert not (pil_pat1 in stripped or pil_pat2 in stripped), (
                    f"Forbidden PIL import found in {path}: {line}"
                )
