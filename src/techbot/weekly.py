"""
src/techbot/weekly.py — Weekly interactive Q&A and poll generator for Run 5 slot 9.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.models import PostItemSchema, PostBasket
from src.techbot.inbox import InboxManager

logger = logging.getLogger(__name__)

WEEKLY_DEFAULT_CHALLENGES = (
    "یک پرامپت بنویس که یک کد پایتون پیچیده را مرحله‌به‌مرحله به یک بچه ۱۰ ساله توضیح دهد.",
    "با ابزار ChatGPT یا Claude یک سناریوی تست واحد برای یک تابع دیتابیس بنویس و خروجی را بررسی کن.",
    "از یک مدل زبانی بخواه تفاوت دقیق RAG و Fine-tuning را در یک جدول مقایسه‌ای به تو بدهد.",
)


class WeeklyContentBuilder:
    """
    Builds content for the weekly Q&A, challenges, and polls.
    """

    def __init__(self, inbox_manager: Optional[InboxManager] = None) -> None:
        self.inbox_manager = inbox_manager

    def build_weekly_post(self, week_number: int = 1) -> PostItemSchema:
        """Construct PostItemSchema for slot 9 weekly_qa."""
        pending_q = []
        if self.inbox_manager:
            pending_q = self.inbox_manager.get_pending_questions(limit=3)

        challenge = WEEKLY_DEFAULT_CHALLENGES[(week_number - 1) % len(WEEKLY_DEFAULT_CHALLENGES)]

        if pending_q:
            points = [f"❓ {q['text'][:80]}" for q in pending_q]
            intro = "رفقا، این هفته سوالات خیلی خوبی از شما رسید. نکات کلیدی زیر پاسخ پرتکرارترین دغدغه‌هاست:"
        else:
            points = [
                "آیا ابزارهای هوش مصنوعی حافظه دارند؟ (تفاوت Session و Persistent Storage)",
                "چطور هزینه مصرف توکن‌های API را ۵۰٪ کاهش دهیم؟",
                "بهترین راه شروع ساخت ایجنت شخصی چیه؟",
            ]
            intro = "جمعه‌ها زمان جمع‌بندی، رفع اشکال و مرور مباحث داغ این هفته در تک‌نیوز است."

        return PostItemSchema(
            slot="21:00_1",
            basket=PostBasket.WEEKLY_QA,
            hook="🔥 جمع‌بندی هفتگی و پاسخ به سوالات شما",
            intro=intro,
            points=points,
            what_can_you_do="این تمرین هفتگی بهت دید عمیقی درباره نحوه کارکرد منطقی مدل‌ها میده.",
            try_now=f"چالش آخر هفته: {challenge}",
            topic_tags=["پرسش_پاسخ", "چالش_هفتگی"],
            terms=["Agent", "Tokens", "RAG"],
        )
