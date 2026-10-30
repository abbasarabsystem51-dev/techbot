"""
src/techbot/models.py — Data models and Pydantic schemas for TechNews Bot.

Includes:
  1. Enums for baskets, categories, lifecycle and quota statuses
  2. Pydantic schemas for Gemini structured output (PostItemSchema, QuizSchema, GeminiBatchOutput)
  3. Validation rules enforcing editorial guidelines (character limits, no forbidden clichés)
  4. RunPlan and Slot configuration schemas
  5. Domain models for articles, generated posts, and state records
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PostBasket(str, Enum):
    """سبدهای موضوعی پست‌های ربات بر اساس برنامه روزانه و پرامپت‌ها."""
    NEWS_AI = "news_ai"
    NEWS_TECH = "news_tech"
    TOOL_AI = "tool_ai"
    TOOL_TECH = "tool_tech"
    LESSON = "lesson"
    WEEKLY_QA = "weekly_qa"


class PostCategory(str, Enum):
    """دسته‌بندی کلی جهت گزارش‌گیری و ذخیره در دیتابیس."""
    AI = "ai"
    TECH = "tech"
    LESSON = "lesson"
    TOOL = "tool"
    WEEKLY_QA = "weekly_qa"


class PostStatus(str, Enum):
    """چرخه حیات یک پست از تولید تا انتشار."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"
    SKIPPED = "skipped"


class DeliveryStatus(str, Enum):
    """وضعیت ارسال به کانال‌های مقصد (تلگرام، ایتا)."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SKIPPED = "skipped"


class LessonStatus(str, Enum):
    """وضعیت پیشرفت ۳۰ درس آموزشی."""
    PENDING = "pending"
    PUBLISHED = "published"


class GeminiQuotaStatus(str, Enum):
    """وضعیت مصرف سهمیه ۱ درخواست در روز Gemini."""
    AVAILABLE = "available"
    LOCKED = "locked"
    CONSUMED = "consumed"


# ---------------------------------------------------------------------------
# Forbidden Phrases from editorial.md Section 4
# ---------------------------------------------------------------------------

FORBIDDEN_PHRASES: tuple[str, ...] = (
    "در این مقاله",
    "در این پست",
    "به بررسی",
    "در دنیای امروز",
    "در عصر حاضر",
    "لازم به ذکر است",
    "شایان ذکر است",
    "همان‌طور که می‌دانیم",
    "بی‌شک",
    "بدون شک",
    "در نهایت",
    "نتیجه‌گیری",
    "به گزارش",
    "بیایید با هم",
    "امیدواریم مفید بوده باشد",
    "در ادامه خواهیم دید",
    "انقلابی",
    "شگفت‌انگیز",
    "بی‌نظیر",
    "فوق‌العاده",
    "همه‌چیز را تغییر می‌دهد",
    "بازی را عوض کرد",
    "متحول می‌کند",
)


def validate_editorial_phrases(text: str, field_name: str = "text") -> str:
    """Validate that text does not contain forbidden editorial clichés."""
    if not text:
        return text
    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            raise ValueError(
                f"فیلد '{field_name}' شامل عبارت ممنوعه سردبیری است: «{phrase}»"
            )
    return text


# ---------------------------------------------------------------------------
# Pydantic Schemas for Gemini Structured Output
# ---------------------------------------------------------------------------

class QuizSchema(BaseModel):
    """
    اسکیمای آزمون کوتاه چندگزینه‌ای انتهای درس‌های آموزشی.
    طبق دستورالعمل lesson.md و system.md: ۳ گزینه هم‌طول و یک توضیح کوتاه.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    q: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="صورت سؤال کوییز (زیر ۲۰۰ کاراکتر)",
    )
    options: list[str] = Field(
        ...,
        min_length=3,
        max_length=4,
        description="گزینه‌های پاسخ (حداقل ۳ گزینه، ترجیحاً هم‌طول)",
    )
    correct: int = Field(
        ...,
        ge=0,
        description="شاخص گزینه صحیح بر مبنای صفر (0-based)",
    )
    explanation: str = Field(
        ...,
        min_length=5,
        max_length=180,
        description="توضیح کوتاه ۱ یا ۲ خطی برای درک دلیل درستی گزینه",
    )

    @field_validator("options")
    @classmethod
    def validate_options(cls, opts: list[str]) -> list[str]:
        for opt in opts:
            if len(opt) > 80:
                raise ValueError(f"طول گزینه کوییز بیش از ۸۰ کاراکتر است: '{opt[:30]}...'")
            validate_editorial_phrases(opt, "quiz.option")
        return opts

    @field_validator("q", "explanation")
    @classmethod
    def validate_text(cls, v: str) -> str:
        return validate_editorial_phrases(v, "quiz")

    @model_validator(mode="after")
    def validate_correct_index(self) -> "QuizSchema":
        if self.correct >= len(self.options):
            raise ValueError(
                f"شاخص گزینه صحیح ({self.correct}) خارج از محدوده گزینه‌ها (تعداد: {len(self.options)}) است."
            )
        return self


class PostItemSchema(BaseModel):
    """
    اسکیمای کامل ساختار خروجی JSON مدل Gemini برای هر پست.
    دقیقاً مطابق فیلدهای تعریف‌شده در system.md و editorial.md.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    slot: str = Field(
        ...,
        description="شناسه یا شماره بلوک اسلات (مثلاً '1', '2' یا '09:00_1')",
    )
    basket: PostBasket = Field(
        ...,
        description="سبد موضوعی محتوا: news_ai | news_tech | tool_ai | tool_tech | lesson | weekly_qa",
    )
    skip_reason: Optional[str] = Field(
        default=None,
        description="در صورت ناکافی بودن یا تکراری بودن منبع، توضیح کوتاه دلیل رد شدن",
    )
    hook: str = Field(
        default="",
        max_length=80,
        description="قلاب خط اول پست با یک ایموجی در آغاز (حداکثر ۸۰ کاراکتر)",
    )
    intro: str = Field(
        default="",
        max_length=160,
        description="مقدمه ۱ تا ۲ جمله‌ای (حداکثر ۱۶۰ کاراکتر)",
    )
    points: list[str] = Field(
        default_factory=list,
        description="۳ نکته مشخص خبری (فقط برای سبدهای خبری news_*؛ حداکثر ۹۰ کاراکتر هر نکته)",
    )
    what_can_you_do: str = Field(
        default="",
        max_length=150,
        description="«چه کاری می‌شه باهاش کرد؟» کاربرد ملموس (مخصوص اخبار)",
    )
    steps: list[str] = Field(
        default_factory=list,
        description="۳ قدم عملی (برای ابزار و درس)",
    )
    tip: str = Field(
        default="",
        max_length=120,
        description="نکته و ترفند مفید (برای ابزار و درس)",
    )
    prompt_to_copy: str = Field(
        default="",
        max_length=300,
        description="پرامپت نمونه قابل کپی برای ابزار یا درس (حداکثر ۳۰۰ کاراکتر)",
    )
    try_now: str = Field(
        default="",
        max_length=130,
        description="اقدام مشخص ۲ دقیقه‌ای؛ پیشوند «همین الان امتحان کن» توسط کد اضافه می‌شود",
    )
    analysis: str = ""
    topic_tags: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="حداکثر ۲ برچسب موضوعی مجاز",
    )
    terms: list[str] = Field(
        default_factory=list,
        description="اصطلاحات فنی استفاده‌شده به زبان انگلیسی (لاتین)",
    )
    quiz: Optional[QuizSchema] = Field(
        default=None,
        description="کوییز ۳ گزینه‌ای (مخصوص درس‌های آموزشی)",
    )

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator("hook", "intro", "what_can_you_do", "tip", "try_now", "analysis")
    @classmethod
    def check_cliches(cls, v: str) -> str:
        return validate_editorial_phrases(v, "post_field")

    @field_validator("points", "steps")
    @classmethod
    def check_list_cliches(cls, items: list[str]) -> list[str]:
        for item in items:
            validate_editorial_phrases(item, "bullet_item")
        return items

    @field_validator("try_now")
    @classmethod
    def check_try_now_prefix(cls, v: str) -> str:
        """کد اسمبلر خودش پیشوند 'همین الان امتحان کن' را اضافه می‌کند."""
        cleaned = v.strip()
        if cleaned.startswith("همین الان امتحان کن"):
            cleaned = cleaned.replace("همین الان امتحان کن:", "").replace("همین الان امتحان کن", "").strip()
        return cleaned

    @model_validator(mode="after")
    def validate_basket_specific_rules(self) -> "PostItemSchema":
        """اعتبارسنجی قوانین اختصاصی هر سبد طبق editorial.md."""
        if self.skip_reason:
            return self

        # در صورت نبود skip_reason، داشتن hook و intro الزامی است
        if not self.hook:
            raise ValueError(f"پست اسلات {self.slot} باید دارای قلاب (hook) باشد.")

        # سبدهای خبری نیازمند حداقل ۱ تا ۳ نکته مشخص هستند
        if self.basket in (PostBasket.NEWS_AI, PostBasket.NEWS_TECH):
            if not self.points:
                raise ValueError(f"پست خبری اسلات {self.slot} باید دارای فیلد points باشد.")

        # سبدهای ابزار و درس نیازمند steps هستند
        if self.basket in (PostBasket.TOOL_AI, PostBasket.TOOL_TECH, PostBasket.LESSON):
            if not self.steps and not self.intro:
                raise ValueError(f"پست ابزار یا درس اسلات {self.slot} نیازمند فیلد steps یا intro است.")

        return self


class GeminiBatchOutput(BaseModel):
    """
    پاسخ جامع و کامل دریافت شده از Gemini API در درخواست Batch روزانه.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    posts: list[PostItemSchema] = Field(
        ...,
        description="لیست پست‌های تولیدشده برای تمام اسلات‌های برنامه‌ریزی‌شده",
    )

    @field_validator("posts", mode="before")
    @classmethod
    def coerce_posts_dict_to_list(cls, v: Any) -> Any:
        if isinstance(v, dict):
            res = []
            for slot_id, post_data in v.items():
                if isinstance(post_data, dict):
                    if "slot" not in post_data:
                        post_data["slot"] = str(slot_id)
                    res.append(post_data)
                else:
                    res.append(post_data)
            return res
        return v

    def valid_posts(self) -> list[PostItemSchema]:
        """پست‌های معتبر و رد نشده را برمی‌گرداند."""
        return [p for p in self.posts if not p.skip_reason]

    def skipped_posts(self) -> list[PostItemSchema]:
        """پست‌هایی که به دلیل کمبود دیتا رد شده‌اند."""
        return [p for p in self.posts if p.skip_reason]


# ---------------------------------------------------------------------------
# Domain Models (Backward-compatible Post representation)
# ---------------------------------------------------------------------------

class Post(BaseModel):
    """
    مدل پست داخلی مورداستفاده در موتور ساخت محتوا (content_builder).
    با کلاس قبلی dataclass و سیستم ثبت وضعیت کاملاً سازگار است.
    """
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    slot: str
    basket: str
    hook: str = ""
    intro: str = ""
    points: Optional[list[str]] = None
    what_can_you_do: str = ""
    steps: Optional[list[str]] = None
    tip: str = ""
    prompt_to_copy: str = ""
    try_now: str = ""
    analysis: str = ""
    topic_tags: Optional[list[str]] = None
    terms: Optional[list[str]] = None
    quiz: Optional[dict[str, Any]] = None
    skip_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Feed & Modular Source Item Models
# ---------------------------------------------------------------------------

class Item(BaseModel):
    """
    مدل استاندارد و یکپارچه آیتم خام استخراج‌شده از منابع مختلف
    (فیدهای RSS، Hacker News، Product Hunt، GitHub و Hugging Face).
    """
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    id: str = Field(..., description="هش یکتا SHA-256")
    title: str = Field(..., min_length=1, max_length=512)
    url: str
    canonical_url: str
    content_hash: str
    summary: str = Field(default="", max_length=8192)
    full_text: Optional[str] = Field(default=None, max_length=10000, description="متن کامل مقاله استخراج‌شده")
    source_name: str
    category: PostCategory = Field(default=PostCategory.AI)
    basket: Optional[PostBasket] = Field(default=None, description="سبد موضوعی هدف (news_ai, tool_ai, ...)")
    published_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    score: float = Field(default=0.0, ge=0.0, le=10.0)
    image_url: Optional[str] = None
    raw_data: dict[str, Any] = Field(default_factory=dict, description="سیگنال‌های خام مانند ستاره‌های گیت‌هاب یا امتیاز HN")
    is_duplicate: bool = False
    duplicate_reason: Optional[str] = None


class Article(Item):
    """مدل متوافق با گذشته برای خبرهای RSS."""
    pass


# ---------------------------------------------------------------------------
# Run Plan & Scheduling Schemas
# ---------------------------------------------------------------------------

class RunSlotConfig(BaseModel):
    slot_id: str
    slot_number: int
    basket: PostBasket
    category: PostCategory
    description: str = ""
    gap_minutes: int = 8


class DailyRunConfig(BaseModel):
    run_index: int = Field(..., ge=1, le=5)
    time: str = Field(..., pattern=r"^\d{2}:\d{2}$")
    label: str = ""
    slots: list[RunSlotConfig]


class RunPlanConfig(BaseModel):
    total_daily_runs: int = 5
    total_daily_posts: int = 10
    ai_posts_count: int = 7
    tech_posts_count: int = 3
    runs: list[DailyRunConfig]


# ---------------------------------------------------------------------------
# Idempotency and State Models
# ---------------------------------------------------------------------------

class IdempotencyRecord(BaseModel):
    """رکورد اجرای موفق یک اسلات با کلید date+run_index+basket."""
    key: str = Field(..., description="کلید یکتا: YYYY-MM-DD#run_index#basket")
    date: str
    run_index: int
    basket: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    post_id: Optional[str] = None
    status: str = "completed"
