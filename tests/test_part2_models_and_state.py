"""
tests/test_part2_models_and_state.py — Tests for Pydantic models, schemas, atomic state, and idempotency.
"""
import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.techbot.models import (
    PostBasket,
    PostCategory,
    QuizSchema,
    PostItemSchema,
    GeminiBatchOutput,
    Post,
    validate_editorial_phrases,
    FORBIDDEN_PHRASES,
)
from src.techbot.state.atomic import (
    read_state,
    write_state_atomic,
    build_idempotency_key,
    is_run_completed,
    mark_run_completed,
    already_processed,
    mark_processed,
    StateManager,
)


# ---------------------------------------------------------------------------
# 1. Models & Validation Tests
# ---------------------------------------------------------------------------

def test_quiz_schema_valid():
    quiz = QuizSchema(
        q="مفهوم RAG در مدل‌های زبانی چیست؟",
        options=[
            "بازیابی اسناد و تولید پاسخ",
            "کاهش مستقیم دمای مدل",
            "افزایش سرعت دانلود دیتا"
        ],
        correct=0,
        explanation="این روش اسناد معتبر را بازیابی کرده و به مدل می‌دهد."
    )
    assert quiz.correct == 0
    assert len(quiz.options) == 3


def test_quiz_schema_invalid_index():
    with pytest.raises(ValidationError):
        QuizSchema(
            q="سؤال تستی",
            options=["الف", "ب", "ج"],
            correct=5,  # Out of range!
            explanation="توضیح"
        )


def test_forbidden_phrases_validation():
    for phrase in FORBIDDEN_PHRASES[:5]:
        with pytest.raises(ValueError, match="عبارت ممنوعه"):
            validate_editorial_phrases(f"این مدل {phrase} است.")


def test_post_item_schema_valid_news():
    post = PostItemSchema(
        slot="09:00_1",
        basket=PostBasket.NEWS_AI,
        hook="🤖 عرضه نسخه جدید مدل جمینای",
        intro="گوگل امروز نسخه جدید را با پنجره زمینه ۱ میلیون توکن معرفی کرد.",
        points=[
            "پنجره زمینه به ۱M توکن رسید",
            "کاهش ۵۰ درصدی قیمت پردازش توکن‌ها",
            "پشتیبانی از پردازش هم‌زمان ویدئو"
        ],
        what_can_you_do="می‌توانی یک ویدیوی ۱ ساعته آموزشی را برای تحلیل آپلود کنی.",
        try_now="یک ویدیو در Google AI Studio آپلود کن و خلاصه بخواه.",
        analysis="رقابت روی پردازش هم‌زمان ورودی‌های سنگین شدیدتر شده است.",
        topic_tags=["AI", "LLM"],
        terms=["Context Window", "Multimodal"]
    )
    assert post.basket == PostBasket.NEWS_AI
    assert len(post.points) == 3
    assert not post.skip_reason


def test_post_item_schema_forbidden_cliche_rejected():
    with pytest.raises(ValidationError):
        PostItemSchema(
            slot="1",
            basket=PostBasket.NEWS_AI,
            hook="🚀 این یک ابزار انقلابی است!",  # Forbidden word: انقلابی
            intro="توضیحات ابزار",
            points=["نکته ۱"],
            try_now="امتحان کن"
        )


def test_gemini_batch_output_filtering():
    raw_json = {
        "posts": [
            {
                "slot": "1",
                "basket": "news_ai",
                "hook": "📰 خبر اول هوش مصنوعی",
                "intro": "معرفی مدل جدید با امکانات کاربردی.",
                "points": ["نکته یک", "نکته دو"],
                "try_now": "پرامپت را تست کن"
            },
            {
                "slot": "2",
                "basket": "tool_ai",
                "skip_reason": "منبع ارائه شده اطلاعات کافی برای معرفی ابزار نداشت."
            }
        ]
    }
    batch = GeminiBatchOutput.model_validate(raw_json)
    assert len(batch.posts) == 2
    assert len(batch.valid_posts()) == 1
    assert len(batch.skipped_posts()) == 1
    assert batch.skipped_posts()[0].slot == "2"


# ---------------------------------------------------------------------------
# 2. Atomic State & Idempotency Tests
# ---------------------------------------------------------------------------

def test_read_missing_state_returns_empty(tmp_path):
    assert read_state(tmp_path / "non_existent.json") == {}


def test_write_state_atomic_roundtrip(tmp_path):
    state_file = tmp_path / "state.json"
    data = {
        "processed_ids": ["hash1", "hash2"],
        "channel": "@tech_news_110",
        "تست_فارسی": "کامپایل و ذخیره بدون نقص"
    }

    write_state_atomic(state_file, data)
    assert state_file.exists()

    loaded = read_state(state_file)
    assert loaded == data
    assert loaded["تست_فارسی"] == "کامپایل و ذخیره بدون نقص"


def test_write_state_atomic_is_idempotent(tmp_path):
    state_file = tmp_path / "state.json"
    data = {"key": "value", "runs": [1, 2, 3]}

    write_state_atomic(state_file, data)
    content1 = state_file.read_text(encoding="utf-8")

    write_state_atomic(state_file, data)
    content2 = state_file.read_text(encoding="utf-8")

    assert content1 == content2


def test_idempotency_key_generation():
    key = build_idempotency_key("2026-10-29", 1, "news_ai")
    assert key == "2026-10-29#1#news_ai"

    # Case insensitivity for basket and stripping
    key2 = build_idempotency_key("2026-10-29 ", " 1 ", "NEWS_AI")
    assert key2 == "2026-10-29#1#news_ai"


def test_idempotency_check_and_mark():
    state = {}
    date = "2026-10-29"
    run_idx = 2
    basket = "tool_ai"

    assert not is_run_completed(state, date, run_idx, basket)

    state = mark_run_completed(state, date, run_idx, basket, details={"post_count": 2})

    assert is_run_completed(state, date, run_idx, basket)
    assert "2026-10-29#2#tool_ai" in state["idempotency_keys"]
    assert "2026-10-29#2#tool_ai" in state["completed_runs"]

    # Marking twice should not duplicate
    state = mark_run_completed(state, date, run_idx, basket)
    assert state["idempotency_keys"].count("2026-10-29#2#tool_ai") == 1


def test_state_manager_transaction(tmp_path):
    state_file = tmp_path / "sub" / "state.json"
    manager = StateManager(state_file)

    assert not manager.is_run_completed("2026-10-29", 1, "news_ai")

    manager.record_completed_run("2026-10-29", 1, "news_ai", details={"author": "bot"})

    assert manager.is_run_completed("2026-10-29", 1, "news_ai")
    assert not manager.is_run_completed("2026-10-29", 2, "news_ai")

    # Verify persistent file content
    loaded = read_state(state_file)
    assert "2026-10-29#1#news_ai" in loaded["idempotency_keys"]
