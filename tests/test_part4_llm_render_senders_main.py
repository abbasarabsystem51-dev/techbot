"""
tests/test_part4_llm_render_senders_main.py — Unit tests for Part 3:
LLM Client, Fact Checking, Quality Review, Emoji Rendering, Senders, and Master Orchestration Pipeline.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.models import (
    GeminiBatchOutput,
    Item,
    PostBasket,
    PostItemSchema,
)
from src.techbot.facts_check import (
    extract_numeric_tokens,
    normalize_digits_and_symbols,
    verify_facts,
    verify_post_item,
)
from src.techbot.images import FALLBACK_IMAGE_URL, get_image_url
from src.techbot.llm import GeminiClient, clean_json_markdown
from src.techbot.main import TechBotOrchestrator
from src.techbot.render import BRAND_FOOTER, PostRenderer, html_to_plain_text
from src.techbot.review import QualityReviewer
from src.techbot.senders.eitaa import EitaaSender
from src.techbot.senders.telegram import TelegramSender
from src.techbot.state.atomic import StateManager


# ===========================================================================
# 1. LLM Client & Markdown Stripping Tests
# ===========================================================================

def test_clean_json_markdown():
    wrapped = "```json\n{\"posts\": []}\n```"
    assert clean_json_markdown(wrapped) == "{\"posts\": []}"

    raw = "{\"posts\": []}"
    assert clean_json_markdown(raw) == "{\"posts\": []}"


def test_gemini_client_generate_batch_mocked():
    sample_json = json.dumps({
        "posts": [
            {
                "slot": "09:00_1",
                "basket": "news_ai",
                "hook": "⚡️ معرفی رسمی مدل جدید هوش مصنوعی",
                "intro": "گوگل مدل سبک و بسیار سریع خود را در دسترس عموم قرار داد.",
                "points": ["کاهش ۵۰ درصدی تاخیر پاسخ", "پشتیبانی از پردازش تصویر"],
                "what_can_you_do": "می‌تونی روی سرورهای ابری با کمترین هزینه اجراش کنی.",
                "try_now": "نسخه دمو را در گوگل استودیو تست کن.",
                "terms": ["Latency", "API"],
            }
        ]
    })

    client = GeminiClient(api_key="mock_key")
    with patch.object(client, "_call_model", return_value=sample_json):
        output = client.generate_batch("prompt")
        assert isinstance(output, GeminiBatchOutput)
        assert len(output.posts) == 1
        assert output.posts[0].basket == PostBasket.NEWS_AI


# ===========================================================================
# 2. Fact Checking Tests
# ===========================================================================

def test_normalize_digits_and_symbols():
    text = "قیمت ۱۰ دلار با تخفیف ۵۰٪ و نسخه ۲.۵"
    norm = normalize_digits_and_symbols(text)
    assert "10" in norm
    assert "50%" in norm
    assert "2.5" in norm


def test_extract_numeric_tokens():
    text = "مدل GPT-4o با هزینه $15 برای هر 1M توکن و افزایش سرعت 99.5% عرضه شد."
    tokens = extract_numeric_tokens(text)
    assert "$15" in tokens or "15" in tokens
    assert "1m" in tokens
    assert "99.5%" in tokens or "99.5" in tokens


def test_verify_facts_success():
    post_text = "مدل جدید با پنجره زمینه 1M توکن و ۵۰٪ کاهش قیمت عرضه شد."
    source_text = "The new architecture supports 1M context window with 50% price reduction."
    is_valid, unverified = verify_facts(post_text, source_text)
    assert is_valid is True
    assert len(unverified) == 0


def test_verify_facts_hallucination_detected():
    post_text = "مدل با پنجره زمینه 10M توکن و ۹۵٪ دقت بالا معرفی شد."
    source_text = "The model was released with a 1M token window."
    is_valid, unverified = verify_facts(post_text, source_text)
    assert is_valid is False
    assert any("95" in tok or "10m" in tok for tok in unverified)


# ===========================================================================
# 3. Quality Review Tests
# ===========================================================================

def test_quality_reviewer_valid_post():
    reviewer = QualityReviewer(passing_threshold=70)
    post = PostItemSchema(
        slot="09:00_1",
        basket=PostBasket.NEWS_AI,
        hook="⚡️ گوگل از قابلیت پردازش هم‌زمان رونمایی کرد",
        intro="معماری جدید پردازش تصویر با مصرف حافظه بهینه عرضه شد.",
        points=["سرعت ۲ برابری", "کاهش هزینه توکن"],
        try_now="همین الان نمونه کد پایتون را اجرا کن.",
    )
    result = reviewer.review(post)
    assert result.passed is True
    assert result.score >= 70


def test_quality_reviewer_forbidden_cliche_fails():
    reviewer = QualityReviewer(passing_threshold=70)
    post = PostItemSchema(
        slot="09:00_1",
        basket=PostBasket.NEWS_AI,
        hook="⚡️ رونمایی از قابلیت جدید",
        intro="مدل جدید برای کارهای مهندسی منتشر شد.",
        points=["نکته ۱", "نکته ۲"],
        try_now="همین الان تست کن",
    )
    dirty_rendered = "⚡️ رونمایی از قابلیت جدید\nبی‌شک همه چیز را متحول می‌کند."
    result = reviewer.review(post, rendered_text=dirty_rendered)
    assert any("کلیشه‌ای" in f for f in result.feedback)


def test_quality_reviewer_html_tags_validation():
    reviewer = QualityReviewer()
    valid_html = "<b>تست</b> و <code>کد پایتون</code>"
    is_valid, issues = reviewer.validate_html(valid_html)
    assert is_valid is True

    invalid_html = "<b>تگ باز بدون بسته شدن"
    is_valid, issues = reviewer.validate_html(invalid_html)
    assert is_valid is False


# ===========================================================================
# 4. Visual & Emoji Rendering Tests
# ===========================================================================

def test_post_renderer_news_structure():
    renderer = PostRenderer()
    post = PostItemSchema(
        slot="09:00_1",
        basket=PostBasket.NEWS_AI,
        hook="مدل جدید دیپ‌سیک در دسترس قرار گرفت",
        intro="نسخه سبک برای سیستم‌های شخصی منتشر شد.",
        points=["مصرف ۶ گیگابایت VRAM", "پشتیبانی از پایتون"],
        analysis="رقابت روی مدل‌های محلی فشرده‌تر شده است.",
        try_now="مدل را با کتابخانه Ollama روی سیستم اجرا کن.",
        terms=["VRAM", "LLM"],
        topic_tags=["هوش_مصنوعی", "مدل_محلی"],
    )

    rendered = renderer.render_telegram_html(post)

    # Check eye-catching emojis
    assert "⚡️" in rendered or "🔥" in rendered or "🚀" in rendered
    assert "🔹" in rendered
    assert "🎯 <b>تحلیل کوتاه:</b>" in rendered
    assert "🧪 <b>همین الان تست کن:</b>" in rendered

    # Check footer and channel ID
    assert "جلوتر بمون، تِک‌نیوز رو دنبال کن 🚀" in rendered
    assert "@tech_news_110" in rendered
    assert len(rendered) <= 1024


def test_post_renderer_tool_structure():
    renderer = PostRenderer()
    post = PostItemSchema(
        slot="12:00_1",
        basket=PostBasket.TOOL_AI,
        hook="ابزار Cursor نسخه جدید را عرضه کرد",
        intro="کدنویسی هوشمندتر با کمک هوش مصنوعی برای توسعه‌دهندگان.",
        what_can_you_do="تولید خودکار کامپوننت‌های فرانت‌اند و رفع باگ‌های تست.",
        steps=["نصب افزونه", "افزودن کلید API", "فشردن کلید ترکیبی Ctrl+K"],
        tip="همیشه فایل‌های کانفیگ را قبل از ساخت مجدد بررسی کن.",
        prompt_to_copy="Generate a clean FastAPI CRUD router with Pydantic schemas",
        try_now="یک تابع در ادیتور بنویس و کلید Ctrl+K را بزن.",
    )

    rendered = renderer.render_telegram_html(post)

    assert "🛠" in rendered
    assert "🚀 <b>باهاش چیکار میشه کرد؟</b>" in rendered
    assert "1️⃣" in rendered and "2️⃣" in rendered and "3️⃣" in rendered
    assert "💡 <b>قِلق کار:</b>" in rendered
    assert "📋 <b>پرامپت آماده کپی:</b>" in rendered
    assert "<code>Generate a clean FastAPI" in rendered
    assert len(rendered) <= 1024


def test_post_renderer_lesson_structure():
    renderer = PostRenderer()
    post = PostItemSchema(
        slot="09:00_2",
        basket=PostBasket.LESSON,
        hook="مفهوم RAG و حافظه معنایی",
        intro="تفاوت جستجوی هوشمند بر اساس مفهوم کلمات به جای تطابق کلمه‌به‌کلمه.",
        steps=["تبدیل متن به بردار عددی", "جستجو در وکتور دیتابیس", "ارسال به مدل زبانی"],
        tip="چانک‌های متن را خیلی بزرگ نگیرید تا تمرکز مدل حفظ شود.",
        try_now="یک داکیومنت در گوگل درایو سرچ کن و تطابق معنایی را بررسی کن.",
        prompt_to_copy="What is the difference between keyword search and vector embedding?",
    )

    rendered = renderer.render_telegram_html(post, lesson_number=3, total_lessons=30)

    assert "🎓 <b>درس ۳ از ۳۰" in rendered
    assert "▫️" in rendered
    assert "🧪 <b>تمرین ۲ دقیقه‌ای:</b>" in rendered
    assert len(rendered) <= 1024


def test_render_eitaa_plain_text():
    html_sample = "<b>تیتر خبر</b>\n🔹 نکته اول\n<code>import torch</code>\n\n@tech_news_110"
    plain = html_to_plain_text(html_sample)
    assert "<b>" not in plain
    assert "<code>" not in plain
    assert "تیتر خبر" in plain
    assert "🔹 نکته اول" in plain


def test_post_renderer_strict_1024_char_limit():
    renderer = PostRenderer()
    post = PostItemSchema(
        slot="09:00_1",
        basket=PostBasket.NEWS_AI,
        hook="⚡️ هوک تستی بسیار پرانرژی و جذاب هوش مصنوعی",
        intro="مقدمه کامل برای تست عملکرد رندرینگ و ارزیابی سقف مجاز کاراکترها در کانال تک نیوز.",
        points=[
            "نکته اول درباره افزایش چشمگیر کارایی مدل‌های زبانی در مقیاس‌های بسیار بزرگ",
            "نکته دوم درباره کاهش مصرف حافظه گرافیکی و بهینه‌سازی دسترسی به پردازنده‌ها",
            "نکته سوم درباره قابلیت یکپارچه‌سازی ابزارهای خارجی و پردازش بلادرنگ",
        ],
        what_can_you_do="می‌توانید تمام کارهای تکراری روزمره برنامه‌نویسی را به صورت خودکار به مدل بسپارید.",
        analysis="رقابت روی ابزارهای توسعه‌دهنده به اوج رسیده و هزینه‌های اجرایی به شدت در حال کاهش است.",
        try_now="همین الان نمونه کد تست را در محیط تستی اجرا کن و خروجی را ارزیابی کن.",
        terms=["LLM", "RAG", "Agent", "Fine-tuning", "Context Window"],
        topic_tags=["هوش_مصنوعی", "تک_نیوز"],
    )

    rendered = renderer.render_telegram_html(post)
    assert len(rendered) <= 1024
    assert BRAND_FOOTER in rendered
    assert "⚡️" in rendered


# ===========================================================================
# 5. Senders Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_telegram_sender_photo_mocked():
    sender = TelegramSender(bot_token="123456:ABC-DEF", channel_id="@tech_news_110")

    mock_resp = httpx.Response(
        200,
        json={"ok": True, "result": {"message_id": 999}},
        request=httpx.Request("POST", "https://api.telegram.org"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await sender.send_post(
            text="<b>سلام دنیا</b>",
            image_url="https://example.com/photo.png",
            source_url="https://example.com/source",
        )

        assert res.success is True
        assert res.message_id == "999"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_sender_poll_mocked():
    sender = TelegramSender(bot_token="123456:ABC-DEF", channel_id="@tech_news_110")

    mock_resp = httpx.Response(
        200,
        json={"ok": True, "result": {"message_id": 1001}},
        request=httpx.Request("POST", "https://api.telegram.org"),
    )

    quiz = {
        "q": "مفهوم RAG چیست؟",
        "options": ["گزینه ۱", "گزینه ۲", "گزینه ۳"],
        "correct": 0,
        "explanation": "توضیح کوتاه",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await sender.send_post(text="درس جدید", quiz=quiz)
        assert res.success is True
        # Called for sendMessage and sendPoll
        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_eitaa_sender_mocked():
    sender = EitaaSender(bot_token="mock_eitaa_token", channel_id="tech_news_110")

    mock_resp = httpx.Response(
        200,
        json={"status": "ok", "result": {"message_id": 555}},
        request=httpx.Request("POST", "https://eitaayar.ir"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await sender.send_post(text="متن تست ایتا")
        assert res.success is True
        assert res.message_id == "555"


# ===========================================================================
# 6. Master Orchestration Dry-Run Pipeline Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_orchestrator_dry_run_pipeline(tmp_path):
    state_file = tmp_path / "state.json"
    orchestrator = TechBotOrchestrator(state_file=state_file)

    # Mock sources so unit tests run completely offline and ultra-fast
    mock_item = Item(
        id="1" * 64,
        title="مدل هوش مصنوعی جدید معرفی شد",
        url="https://example.com/ai-news",
        canonical_url="https://example.com/ai-news",
        content_hash="hash1",
        summary="خلاصه خبر هوش مصنوعی برای تست",
        full_text="متن کامل خبر هوش مصنوعی جهت جلوگیری از واکشی شبکه",
        image_url="https://example.com/img.png",
        source_name="Test Source",
        basket=PostBasket.NEWS_AI,
    )
    mock_src = MagicMock()
    mock_src.safe_fetch = AsyncMock(return_value=[mock_item])

    with patch("src.techbot.main.load_configured_sources", return_value=[mock_src]), \
         patch("src.techbot.main.get_image_url", return_value="https://example.com/img.png"):
        # Run index 1 dry-run execution
        success = await orchestrator.run(
            run_index=1,
            dry_run=True,
            target_date="2026-10-30",
        )
        assert success is True

        # Second execution on the same date should be completely idempotent (skip completed slots)
        success_idempotent = await orchestrator.run(
            run_index=1,
            dry_run=True,
            target_date="2026-10-30",
        )
        assert success_idempotent is True


def test_lesson_renderer_omits_duplicate_prompt_to_copy():
    """Verify that 📋 is omitted if prompt_to_copy is equal to try_now or empty."""
    renderer = PostRenderer()
    # Case 1: Identical try_now and prompt_to_copy
    post_dup = PostItemSchema(
        slot="09:00_2",
        basket=PostBasket.LESSON,
        hook="مفهوم هوش مصنوعی چیست؟",
        intro="یک سیستم هوشمند برای یادگیری از داده‌ها به جای اجرای خط به خط.",
        steps=["جمع‌آوری داده", "آموزش مدل", "ارزیابی دقت"],
        tip="همیشه کیفیت داده مهم‌تر از حجم مدل است.",
        try_now="از مدل بخواهید تفاوت را در ۲ خط بگوید.",
        prompt_to_copy="از مدل بخواهید تفاوت را در ۲ خط بگوید.",
    )
    rendered_dup = renderer.render_telegram_html(post_dup, lesson_number=1)
    assert "🧪 <b>تمرین ۲ دقیقه‌ای:</b>" in rendered_dup
    assert "📋" not in rendered_dup, "Duplicate 📋 line must be omitted"

    # Case 2: Distinct prompt_to_copy
    post_distinct = PostItemSchema(
        slot="09:00_2",
        basket=PostBasket.LESSON,
        hook="مفهوم هوش مصنوعی چیست؟",
        intro="یک سیستم هوشمند برای یادگیری از داده‌ها به جای اجرای خط به خط.",
        steps=["جمع‌آوری داده", "آموزش مدل", "ارزیابی دقت"],
        tip="همیشه کیفیت داده مهم‌تر از حجم مدل است.",
        try_now="تمرین را در مرورگر انجام بده.",
        prompt_to_copy="Explain the difference between AI and traditional software in 2 lines.",
    )
    rendered_distinct = renderer.render_telegram_html(post_distinct, lesson_number=1)
    assert "🧪 <b>تمرین ۲ دقیقه‌ای:</b>" in rendered_distinct
    assert "📋" in rendered_distinct, "Distinct 📋 prompt snippet must be rendered"



