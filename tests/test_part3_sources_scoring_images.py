"""
tests/test_part3_sources_scoring_images.py — Unit tests for Part 2:
HTTP Client, Modular Sources, Local Scoring, Deduplication, and Images.
"""
from __future__ import annotations

import io
import struct
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.models import Item, PostBasket, PostCategory
from src.techbot.healthcheck import check_source_health
from src.techbot.http import HttpClient, create_http_client
from src.techbot.images import (
    ImageFinder,
    LessonCardGenerator,
    parse_image_dimensions,
    shape_persian_text,
)
from src.techbot.scoring import (
    Scorer,
    calculate_title_similarity,
    normalize_url,
)
from src.techbot.sources.base import Source
from src.techbot.sources.fulltext import (
    enrich_top_candidates,
    extract_fulltext_from_html,
)
from src.techbot.sources.github_search import GitHubSearchSource
from src.techbot.sources.hackernews import HackerNewsSource
from src.techbot.sources.huggingface import HuggingFaceSource
from src.techbot.sources.producthunt import ProductHuntSource
from src.techbot.sources.rss import RSSSource
from src.techbot.state.atomic import get_feed_failures, track_feed_failure


# ---------------------------------------------------------------------------
# Helpers to construct minimal valid binary image headers
# ---------------------------------------------------------------------------

def create_mock_png_bytes(width: int, height: int) -> bytes:
    """Generate minimal valid PNG header with specified width and height."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = b"\x00\x00\x00\x00"
    return signature + ihdr_len + ihdr_type + ihdr_data + ihdr_crc + b"\x00" * 32


# ===========================================================================
# 1. HTTP Client Tests
# ===========================================================================

def test_http_client_defaults():
    client = create_http_client()
    assert client.timeout == 10.0, "SPEC requires 10s network timeout"
    assert "TechNewsBot/1.0" in client.user_agent
    assert client.headers["User-Agent"] == client.user_agent


@pytest.mark.asyncio
async def test_http_client_get_text_success():
    client = HttpClient(timeout=5.0)

    mock_resp = httpx.Response(200, text="Sample News Content", request=httpx.Request("GET", "https://example.com"))
    with patch.object(client.async_client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        text = await client.get_text("https://example.com/test")

        assert text == "Sample News Content"
        mock_get.assert_called_once()

    await client.aclose()


@pytest.mark.asyncio
async def test_http_client_retries_on_503():
    client = HttpClient(timeout=5.0, max_retries=2, initial_delay=0.01)

    req = httpx.Request("GET", "https://example.com/error")
    resp_503 = httpx.Response(503, text="Service Unavailable", request=req)
    resp_200 = httpx.Response(200, text="Recovered!", request=req)

    with patch.object(client.async_client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [resp_503, resp_200]
        text = await client.get_text("https://example.com/error")

        assert text == "Recovered!"
        assert mock_get.call_count == 2

    await client.aclose()


# ===========================================================================
# 2. Modular Sources Tests
# ===========================================================================

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>TechCrunch AI</title>
    <link>https://techcrunch.com</link>
    <description>AI News</description>
    <item>
      <title>Anthropic announces Claude 3.7 Sonnet with hybrid reasoning</title>
      <link>https://techcrunch.com/2026/03/anthropic-claude-3-7/?utm_source=feed</link>
      <description>&lt;p&gt;Anthropic has released a new frontier model with hybrid reasoning.&lt;/p&gt;</description>
      <pubDate>Thu, 29 Oct 2026 12:00:00 GMT</pubDate>
      <media:thumbnail url="https://techcrunch.com/wp-content/uploads/claude.jpg" />
    </item>
  </channel>
</rss>
"""


@pytest.mark.asyncio
async def test_rss_source_fetch():
    source = RSSSource(
        source_id="test_tc_ai",
        name="TechCrunch AI",
        url="https://techcrunch.com/feed",
        category=PostCategory.AI,
        basket=PostBasket.NEWS_AI,
    )

    mock_client = MagicMock(spec=HttpClient)
    mock_client.get_text = AsyncMock(return_value=SAMPLE_RSS_XML)

    items = await source.fetch(client=mock_client)
    assert len(items) == 1
    item = items[0]

    assert "Claude 3.7" in item.title
    assert "techcrunch.com" in item.url
    assert "Anthropic has released" in item.summary
    assert item.category == PostCategory.AI
    assert item.basket == PostBasket.NEWS_AI
    assert item.image_url == "https://techcrunch.com/wp-content/uploads/claude.jpg"
    assert item.id
    assert item.canonical_url


@pytest.mark.asyncio
async def test_hackernews_source_fetch():
    hn = HackerNewsSource(source_id="hn_test", name="Hacker News", story_limit=2, min_score=10)

    top_ids = [101, 102]
    story_101 = {
        "id": 101,
        "type": "story",
        "title": "OpenAI releases new open weights reasoning LLM",
        "url": "https://openai.com/news/open-weights",
        "score": 450,
        "descendants": 120,
        "time": 1793280000,
        "by": "sam_altman",
    }
    story_102 = {
        "id": 102,
        "type": "story",
        "title": "Show HN: Fast SQLite replication in Rust",
        "url": "https://github.com/fast/sqlite-rep",
        "score": 85,
        "descendants": 22,
        "time": 1793280000,
        "by": "rustacean",
    }

    mock_client = MagicMock(spec=HttpClient)

    async def mock_get_json(url, **kwargs):
        if "topstories.json" in url:
            return top_ids
        if "101.json" in url:
            return story_101
        if "102.json" in url:
            return story_102
        return None

    mock_client.get_json = AsyncMock(side_effect=mock_get_json)

    items = await hn.fetch(client=mock_client)
    assert len(items) == 2

    # Item 1 has "LLM" so it should be categorized as AI
    assert items[0].category == PostCategory.AI
    assert items[0].basket == PostBasket.NEWS_AI
    assert items[0].raw_data["score"] == 450

    # Item 2 is general tech
    assert items[1].category == PostCategory.TECH
    assert items[1].basket == PostBasket.NEWS_TECH


@pytest.mark.asyncio
async def test_producthunt_source_fetch():
    ph = ProductHuntSource(source_id="ph_test", name="Product Hunt")

    sample_ph_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Product Hunt</title>
        <item>
          <title>PromptForge AI: Generate production prompts</title>
          <link>https://www.producthunt.com/posts/promptforge-ai</link>
          <description>An AI tool to craft perfect prompts for Claude and GPT</description>
        </item>
      </channel>
    </rss>
    """
    mock_client = MagicMock(spec=HttpClient)
    mock_client.get_text = AsyncMock(return_value=sample_ph_xml)

    items = await ph.fetch(client=mock_client)
    assert len(items) == 1
    assert items[0].basket == PostBasket.TOOL_AI
    assert items[0].category == PostCategory.AI


@pytest.mark.asyncio
async def test_github_search_source_fetch():
    gh = GitHubSearchSource(source_id="gh_test", name="GitHub AI", limit=2)

    mock_data = {
        "items": [
            {
                "full_name": "vllm-project/vllm",
                "html_url": "https://github.com/vllm-project/vllm",
                "description": "A high-throughput and memory-efficient LLM serving engine",
                "stargazers_count": 32000,
                "forks_count": 4800,
                "language": "Python",
                "topics": ["llm", "inference", "deep-learning"],
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    }
    mock_client = MagicMock(spec=HttpClient)
    mock_client.get_json = AsyncMock(return_value=mock_data)

    items = await gh.fetch(client=mock_client)
    assert len(items) == 1
    assert items[0].basket == PostBasket.TOOL_AI
    assert items[0].raw_data["stars"] == 32000


@pytest.mark.asyncio
async def test_huggingface_source_fetch():
    hf = HuggingFaceSource(source_id="hf_test", name="Hugging Face Daily", limit=2)

    mock_papers = [
        {
            "paper": {
                "id": "2603.12345",
                "title": "Scaling Test-Time Compute in Multimodal Agents",
                "summary": "We study test-time scaling for visual grounding and reasoning.",
                "upvotes": 142,
                "publishedAt": "2026-10-29T10:00:00Z",
            }
        }
    ]
    mock_client = MagicMock(spec=HttpClient)
    mock_client.get_json = AsyncMock(return_value=mock_papers)

    items = await hf.fetch(client=mock_client)
    assert len(items) == 1
    assert "Multimodal Agents" in items[0].title
    assert items[0].raw_data["upvotes"] == 142


@pytest.mark.asyncio
async def test_source_isolation_and_failure_tracking():
    """Verify that a failing source never crashes the pipeline and updates state."""
    class BrokenSource(Source):
        async def fetch(self, client=None):
            raise ConnectionError("Upstream server timeout")

    src = BrokenSource(source_id="broken_feed", name="Broken Feed")
    state = {}

    items = await src.safe_fetch(state=state)
    assert items == []
    assert get_feed_failures(state, "broken_feed") == 1

    # Second failure increments counter
    await src.safe_fetch(state=state)
    assert get_feed_failures(state, "broken_feed") == 2


@pytest.mark.asyncio
async def test_fulltext_extraction_and_enrichment():
    html = """
    <html>
      <head><title>Test Article</title></head>
      <body>
        <nav><a href="/">Home</a></nav>
        <article>
          <h1>Deep Learning Advances in 2026</h1>
          <p>Researchers have discovered novel architectures that optimize context memory.</p>
          <p>This allows models to process millions of tokens with sub-quadratic complexity.</p>
        </article>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    extracted = extract_fulltext_from_html(html)
    assert extracted is not None
    assert "Deep Learning Advances" in extracted
    assert "Copyright" not in extracted
    assert len(extracted) <= 6000

    item = Item(
        id="12345" * 12 + "1234",
        title="Test Article",
        url="https://example.com/art1",
        canonical_url="https://example.com/art1",
        content_hash="hash1",
        summary="short summary",
        source_name="Blog",
    )

    mock_client = MagicMock(spec=HttpClient)
    mock_client.get_text = AsyncMock(return_value=html)

    enriched = await enrich_top_candidates([item], max_items=1, client=mock_client)
    assert enriched[0].full_text is not None
    assert "Deep Learning Advances" in enriched[0].full_text


@pytest.mark.asyncio
async def test_healthcheck_diagnostics():
    class DummySource(Source):
        async def fetch(self, client=None):
            return [MagicMock()]

    res = await check_source_health(DummySource("dummy", "Dummy"))
    assert res["status"] == "OK"
    assert res["items"] == 1
    assert res["latency_ms"] >= 0


# ===========================================================================
# 3. Local Scoring & Deduplication Tests
# ===========================================================================

def test_normalize_url():
    dirty = "https://techcrunch.com/2026/03/claude-update/?utm_source=newsletter&utm_medium=email&fbclid=XYZ#comments"
    clean = normalize_url(dirty)
    assert clean == "https://techcrunch.com/2026/03/claude-update"

    # Preserves essential functional query params
    query_url = "https://example.com/search?q=machine+learning&utm_source=twitter"
    assert normalize_url(query_url) == "https://example.com/search?q=machine+learning"


def test_title_similarity():
    t1 = "Anthropic releases Claude 3.7 with hybrid reasoning"
    t2 = "Anthropic Releases Claude 3.7 Sonnet With Hybrid Reasoning!"
    sim = calculate_title_similarity(t1, t2)
    assert sim >= 0.85

    t3 = "SpaceX launches Starship Flight 8 into orbit"
    sim_unrelated = calculate_title_similarity(t1, t3)
    assert sim_unrelated < 0.35


def test_scorer_freshness_decay():
    scorer = Scorer()
    now = datetime.now(timezone.utc)

    # Under 6 hours: full 3.0 points
    score_fresh = scorer.calculate_freshness(now - timedelta(hours=2))
    assert score_fresh == 3.0

    # 24 hours: decayed
    score_24h = scorer.calculate_freshness(now - timedelta(hours=24))
    assert 0.0 < score_24h < 3.0

    # Over 48 hours: 0.0 points
    score_old = scorer.calculate_freshness(now - timedelta(hours=50))
    assert score_old == 0.0


def test_scorer_relevance_and_boost():
    scorer = Scorer()

    # AI article with high priority keywords
    ai_score = scorer.calculate_relevance(
        title="OpenAI and Google DeepMind collaborate on LLM reasoning",
        summary="Novel transformer architecture for RAG agents",
        basket=PostBasket.NEWS_AI,
    )
    assert ai_score == 3.0

    # Penalized keywords
    penalized_score = scorer.calculate_relevance(
        title="New crypto memecoin giveaway rumor",
        summary="Get free tokens",
        basket=PostBasket.NEWS_TECH,
    )
    assert penalized_score == 0.0


def test_scorer_deduplication():
    scorer = Scorer()
    item = Item(
        id="a" * 64,
        title="Major Gemini 2.5 Flash Announcement",
        url="https://googleblog.com/gemini-flash?utm_source=feed",
        canonical_url="https://googleblog.com/gemini-flash",
        content_hash="h1",
        summary="Summary text",
        source_name="Google AI Blog",
        published_at=datetime.now(timezone.utc),
    )

    sent_links = {"https://googleblog.com/gemini-flash"}
    score = scorer.score_item(item, basket=PostBasket.NEWS_AI, sent_links=sent_links)

    assert score == 0.0
    assert item.is_duplicate is True
    assert item.duplicate_reason == "duplicate_url"


def test_rank_candidates():
    scorer = Scorer()
    now = datetime.now(timezone.utc)

    item1 = Item(
        id="1" * 64,
        title="Gemini 2.5 Flash released with RAG agent improvements",
        url="https://blog.google/gemini",
        canonical_url="https://blog.google/gemini",
        content_hash="h1",
        summary="High priority LLM announcement",
        source_name="Google AI Blog",
        published_at=now - timedelta(hours=1),
    )
    item2 = Item(
        id="2" * 64,
        title="New keyboard switch released",
        url="https://arstechnica.com/keyboard",
        canonical_url="https://arstechnica.com/keyboard",
        content_hash="h2",
        summary="Mechanical switch review",
        source_name="Ars Technica",
        published_at=now - timedelta(hours=1),
    )

    ranked = scorer.rank_candidates([item2, item1], basket=PostBasket.NEWS_AI, limit=2)
    assert len(ranked) == 2
    assert ranked[0].id == item1.id
    assert ranked[0].score > ranked[1].score


# ===========================================================================
# 4. Images & Graphic Cards Tests
# ===========================================================================

def test_image_dimensions_binary_parser():
    png_bytes = create_mock_png_bytes(1200, 675)
    dims = parse_image_dimensions(png_bytes)
    assert dims == (1200, 675)


def test_image_finder_og_extraction_and_filtering():
    finder = ImageFinder(min_width=800)
    html = """
    <html>
      <head>
        <meta property="og:image" content="https://example.com/images/hero-article.jpg" />
      </head>
    </html>
    """
    og_url = finder.extract_og_image(html, base_url="https://example.com")
    assert og_url == "https://example.com/images/hero-article.jpg"

    # Rejection of icons / logos / avatars
    assert finder.is_valid_image_url("https://example.com/hero.jpg") is True
    assert finder.is_valid_image_url("https://example.com/site_logo.png") is False
    assert finder.is_valid_image_url("https://example.com/user_avatar.jpg") is False
    assert finder.is_valid_image_url("https://example.com/icon.svg") is False


def test_image_finder_dimension_validation():
    finder = ImageFinder(min_width=800)

    # 1200x675 HD image passes
    valid_bytes = create_mock_png_bytes(1200, 675)
    assert finder.validate_image_dimensions(valid_bytes) is True

    # 400x300 image fails (too small)
    small_bytes = create_mock_png_bytes(400, 300)
    assert finder.validate_image_dimensions(small_bytes) is False

    # 500x500 square avatar fails
    square_bytes = create_mock_png_bytes(500, 500)
    assert finder.validate_image_dimensions(square_bytes) is False


def test_persian_text_shaping():
    raw_text = "درس ۱ از ۳۰"
    shaped = shape_persian_text(raw_text)
    assert shaped != ""
    assert len(shaped) > 0


def test_lesson_card_generator():
    gen = LessonCardGenerator()
    card_bytes = gen.generate_card(
        lesson_number=5,
        lesson_title="مفاهیم پردازش تصویر و بینایی ماشین",
        total_lessons=30,
        channel_handle="@tech_news_110",
    )

    assert isinstance(card_bytes, bytes)
    assert len(card_bytes) > 200

    # If Pillow is available, check PNG signature; else SVG XML
    is_png = card_bytes.startswith(b"\x89PNG")
    is_svg = b"<svg" in card_bytes
    assert is_png or is_svg, "Card output must be valid PNG or SVG bytes"
