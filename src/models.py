"""
src/models.py — مدل‌های داده با Pydantic v2

تمام ساختارهای داده‌ای پروژه اینجا تعریف می‌شوند.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PostCategory(str, Enum):
    """دسته‌بندی پست."""
    AI = "ai"
    TECH = "tech"
    LESSON = "lesson"
    TOOL = "tool"
    WEEKLY_QA = "weekly_qa"


class PostStatus(str, Enum):
    """وضعیت یک پست در چرخه عمرش."""
    PENDING = "pending"         # تولیدشده، منتظر انتشار
    SCHEDULED = "scheduled"     # زمان‌بندی‌شده
    PUBLISHED = "published"     # منتشرشده با موفقیت
    FAILED = "failed"           # انتشار ناموفق (پس از حداکثر retry)
    SKIPPED = "skipped"         # رد شده (مثلاً تکراری)


class DeliveryStatus(str, Enum):
    """وضعیت ارسال به هر مقصد."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SKIPPED = "skipped"


class LessonStatus(str, Enum):
    """وضعیت هر درس در برنامه آموزشی."""
    PENDING = "pending"
    PUBLISHED = "published"


class GeminiQuotaStatus(str, Enum):
    """وضعیت سهمیه روزانه Gemini API."""
    AVAILABLE = "available"     # هنوز استفاده نشده
    LOCKED = "locked"           # در حال استفاده (atomic lock)
    CONSUMED = "consumed"       # مصرف شده امروز


# ---------------------------------------------------------------------------
# Article — خبر خام از RSS
# ---------------------------------------------------------------------------

class Article(BaseModel):
    """یک خبر خام استخراج‌شده از یک فید RSS."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(
        ...,
        description="شناسه یکتا: SHA-256 از (canonical_url + title)",
        min_length=64,
        max_length=64,
    )
    title: str = Field(..., min_length=1, max_length=512)
    url: str = Field(..., description="آدرس URL اصلی (canonical)")
    canonical_url: str = Field(..., description="URL پس از نرمال‌سازی")
    content_hash: str = Field(..., description="SHA-256 از (url|title|summary)")
    summary: str = Field(default="", max_length=4096)
    source_name: str = Field(..., description="نام منبع RSS")
    category: PostCategory = Field(default=PostCategory.AI)
    published_at: Optional[datetime] = Field(
        default=None,
        description="زمان انتشار اصلی خبر (UTC)",
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="زمان fetch شدن توسط ربات (UTC)",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="امتیاز ارتباط و تازگی (۰ تا ۱)",
    )
    is_duplicate: bool = Field(default=False)

    @field_validator("url", "canonical_url")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL must start with http(s)://: {v!r}")
        return v


# ---------------------------------------------------------------------------
# GeneratedPost — پست تولیدشده توسط Gemini
# ---------------------------------------------------------------------------

class GeneratedPost(BaseModel):
    """پست فارسی تولیدشده توسط مدل Gemini."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., description="شناسه یکتا (UUID4 یا SHA-256)")
    article_id: Optional[str] = Field(
        default=None,
        description="شناسه خبر مرجع (از Article.id)",
    )
    text: str = Field(..., min_length=1, max_length=800, description="متن پست بدون فوتر/هشتگ")
    category: PostCategory
    source_url: Optional[str] = Field(default=None)
    source_title: Optional[str] = Field(default=None, max_length=256)
    lesson_number: Optional[int] = Field(default=None, ge=1, le=30)
    status: PostStatus = Field(default=PostStatus.PENDING)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
    )
    scheduled_at: Optional[datetime] = Field(default=None)
    published_at: Optional[datetime] = Field(default=None)

    # ── Quiz fields (درس‌ها) ───────────────────────────────────────────────
    quiz_question: Optional[str] = Field(default=None, max_length=512)
    quiz_options: Optional[dict[str, str]] = Field(default=None)
    correct_answer: Optional[str] = Field(default=None)

    @field_validator("text")
    @classmethod
    def no_forbidden_phrases(cls, v: str) -> str:
        forbidden = [
            "در دنیای امروز",
            "بی‌شک",
            "انقلابی",
            "بازی را عوض کرد",
            "شگفت‌انگیز",
            "متحول می‌کند",
        ]
        for phrase in forbidden:
            if phrase in v:
                raise ValueError(f"متن شامل عبارت ممنوعه است: «{phrase}»")
        return v


# ---------------------------------------------------------------------------
# DeliveryRecord — سابقه ارسال به هر مقصد
# ---------------------------------------------------------------------------

class DeliveryRecord(BaseModel):
    """سابقه ارسال یک پست به یک مقصد (Telegram یا Eitaa)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., description="شناسه یکتا رکورد ارسال")
    post_id: str = Field(..., description="شناسه پست (از GeneratedPost.id)")
    destination: str = Field(..., description="«telegram» یا «eitaa»")
    status: DeliveryStatus = Field(default=DeliveryStatus.PENDING)
    message_id: Optional[str] = Field(
        default=None,
        description="message_id برگشت‌داده‌شده توسط API پیام‌رسان",
    )
    attempts: int = Field(default=0, ge=0)
    last_attempt_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None, max_length=1024)
    sent_at: Optional[datetime] = Field(default=None)


# ---------------------------------------------------------------------------
# GeminiDailyQuota — ردیابی سهمیه روزانه Gemini
# ---------------------------------------------------------------------------

class GeminiDailyQuota(BaseModel):
    """ردیابی مصرف سهمیه روزانه Gemini API."""

    model_config = ConfigDict(str_strip_whitespace=True)

    date: str = Field(
        ...,
        description="تاریخ روز به فرمت YYYY-MM-DD (به‌وقت Asia/Tehran)",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    requests_used: int = Field(default=0, ge=0)
    max_requests: int = Field(default=1, ge=1)
    status: GeminiQuotaStatus = Field(default=GeminiQuotaStatus.AVAILABLE)
    lock_acquired_at: Optional[datetime] = Field(default=None)
    last_request_at: Optional[datetime] = Field(default=None)

    @model_validator(mode="after")
    def validate_quota_consistency(self) -> "GeminiDailyQuota":
        if self.requests_used > self.max_requests and self.status == GeminiQuotaStatus.AVAILABLE:
            raise ValueError("requests_used cannot exceed max_requests when status is AVAILABLE")
        return self

    @property
    def is_available(self) -> bool:
        """آیا هنوز سهمیه برای امروز وجود دارد؟"""
        return (
            self.requests_used < self.max_requests
            and self.status == GeminiQuotaStatus.AVAILABLE
        )

    @property
    def is_exhausted(self) -> bool:
        return self.requests_used >= self.max_requests or self.status == GeminiQuotaStatus.CONSUMED


# ---------------------------------------------------------------------------
# LessonProgress — وضعیت دوره آموزشی
# ---------------------------------------------------------------------------

class LessonProgress(BaseModel):
    """وضعیت هر درس در دوره ۳۰‌درسی."""

    model_config = ConfigDict(str_strip_whitespace=True)

    lesson_number: int = Field(..., ge=1, le=30)
    title: str = Field(..., max_length=256)
    status: LessonStatus = Field(default=LessonStatus.PENDING)
    published_at: Optional[datetime] = Field(default=None)
    post_id: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# AudienceQuestion — سؤال مخاطب
# ---------------------------------------------------------------------------

class AudienceQuestion(BaseModel):
    """سؤال دریافت‌شده از مخاطبان کانال."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., description="شناسه یکتا سؤال")
    text: str = Field(..., min_length=5, max_length=1024)
    source: str = Field(default="telegram", description="منبع: telegram یا eitaa")
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
    )
    is_answered: bool = Field(default=False)
    answered_in_post_id: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Gemini API response schemas (parsed with Pydantic)
# ---------------------------------------------------------------------------

class GeminiNewsPostItem(BaseModel):
    """یک پست خبری در پاسخ JSON از Gemini."""

    article_id: str
    text: str = Field(..., min_length=1, max_length=800)
    category: PostCategory
    source_url: str
    source_title: str = Field(default="")


class GeminiNewsResponse(BaseModel):
    """پاسخ JSON کامل Gemini برای درخواست دسته‌ای اخبار."""

    posts: list[GeminiNewsPostItem]


class GeminiLessonResponse(BaseModel):
    """پاسخ JSON از Gemini برای تولید پست درس."""

    text: str = Field(..., min_length=1, max_length=800)
    lesson_number: int = Field(..., ge=1, le=30)
    lesson_title: str
    quiz_question: str
    quiz_options: dict[str, str]
    correct_answer: str


class GeminiToolResponse(BaseModel):
    """پاسخ JSON از Gemini برای معرفی ابزار."""

    text: str = Field(..., min_length=1, max_length=800)
    tool_name: str
    tool_url: str
    is_free: bool
    copyable_prompt: Optional[str] = None


class GeminiWeeklyQAResponse(BaseModel):
    """پاسخ JSON از Gemini برای پست هفتگی پرسش‌وپاسخ."""

    text: str = Field(..., min_length=1, max_length=800)
    week_number: int
    questions_answered: int
    skipped_questions: list[str] = Field(default_factory=list)
