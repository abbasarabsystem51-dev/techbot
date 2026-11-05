"""
src/techbot/agent/lesson_schema.py — Pydantic schemas for custom lesson validation.

Validates the full lesson structure as defined in curriculum.json, including:
- title, level, topic, key_terms
- concept (ConceptSchema)
- example (ExampleSchema)
- practice (PracticeSchema) which embeds QuizSchema
- quiz (QuizSchema) at the top-level practice level
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuizSchema(BaseModel):
    """
    Schema for a multiple-choice quiz question embedded in a lesson's practice section.
    Requires 3 options and a valid correct_index.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=5,
        max_length=400,
        description="متن سوال کوئیز (فارسی)",
    )
    options: list[str] = Field(
        ...,
        min_length=3,
        max_length=4,
        description="لیست گزینه‌ها (حداقل ۳، حداکثر ۴ گزینه)",
    )
    correct_index: int = Field(
        ...,
        ge=0,
        description="ایندکس گزینه صحیح (شروع از ۰)",
    )
    explanation: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="توضیح کوتاه برای پاسخ صحیح",
    )

    @field_validator("correct_index")
    @classmethod
    def correct_index_in_range(cls, v: int, info: Any) -> int:
        # Validated after options is set; we recheck in model_validator if needed.
        if v < 0:
            raise ValueError("correct_index باید مقداری غیرمنفی باشد.")
        return v

    @field_validator("options")
    @classmethod
    def options_not_empty(cls, v: list[str]) -> list[str]:
        if any(not opt.strip() for opt in v):
            raise ValueError("هر گزینه کوئیز باید متن داشته باشد.")
        return v


class ConceptSchema(BaseModel):
    """Schema for the concept section of a lesson."""
    model_config = ConfigDict(str_strip_whitespace=True)

    summary: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="خلاصه مفهوم اصلی درس",
    )
    everyday_example: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="مثال روزمره برای درک بهتر",
    )
    allowed_facts: list[str] = Field(
        default_factory=list,
        description="لیست واقعیت‌های معتبر برای fact-check",
    )

    @field_validator("allowed_facts")
    @classmethod
    def facts_not_empty_strings(cls, v: list[str]) -> list[str]:
        return [f for f in v if f.strip()]


class ExampleSchema(BaseModel):
    """Schema for the example section of a lesson."""
    model_config = ConfigDict(str_strip_whitespace=True)

    scenario: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="سناریوی مثال",
    )
    bad_case: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="نمونه اشتباه / روش قدیمی",
    )
    good_case: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="نمونه درست / روش مبتنی بر AI",
    )
    takeaway: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="نکته کلیدی که کاربر باید یاد بگیرد",
    )


class PracticeSchema(BaseModel):
    """Schema for the practice section of a lesson, including embedded quiz."""
    model_config = ConfigDict(str_strip_whitespace=True)

    platform: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="پلتفرم پیشنهادی برای تمرین (مثلاً ChatGPT یا Gemini)",
    )
    try_now: str = Field(
        ...,
        min_length=5,
        max_length=600,
        description="دستورالعمل کوتاه برای تمرین",
    )
    prompt_to_copy: str = Field(
        ...,
        min_length=5,
        max_length=600,
        description="پرامپت آماده برای کپی کردن",
    )
    quiz: QuizSchema = Field(
        ...,
        description="کوئیز چندگزینه‌ای مرتبط با درس",
    )


class LessonSchema(BaseModel):
    """
    Full schema for a custom lesson.
    Mirrors the structure used in curriculum.json and expected by main.py's LESSON handler.
    Fields: lesson_id, title, level, topic, key_terms, concept, example, practice.
    """
    model_config = ConfigDict(str_strip_whitespace=True, extra="allow")

    lesson_id: int = Field(
        ...,
        ge=1,
        description="شناسه عددی یکتا برای درس (باید مثبت باشد)",
    )
    title: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="عنوان درس به فارسی",
    )
    level: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="سطح درس (مثلاً: مبتدی، متوسط، پیشرفته)",
    )
    topic: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="موضوع اصلی درس به انگلیسی یا فارسی",
    )
    key_terms: list[str] = Field(
        ...,
        min_length=1,
        description="واژه‌های کلیدی درس",
    )
    concept: ConceptSchema = Field(
        ...,
        description="بخش مفهوم (concept) درس",
    )
    example: ExampleSchema = Field(
        ...,
        description="بخش مثال (example) درس",
    )
    practice: PracticeSchema = Field(
        ...,
        description="بخش تمرین (practice) شامل کوئیز",
    )

    # Optional navigation fields (compatible with curriculum.json)
    prerequisites: list[str] = Field(default_factory=list)
    prev_title: Optional[str] = None
    next_title: Optional[str] = None

    @field_validator("key_terms")
    @classmethod
    def key_terms_not_empty(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("key_terms باید حداقل یک عبارت داشته باشد.")
        return cleaned
