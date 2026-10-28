"""
src/state.py — مدیریت وضعیت پایدار و قفل‌گذاری

این ماژول یک لایه انتزاعی بالای Database برای مدیریت:
  1. سهمیه روزانه Gemini با قفل اتمی مقاوم در برابر restart
  2. وضعیت پست‌ها در چرخه انتشار
  3. پیگیری وضعیت دوره آموزشی
  4. context manager برای دریافت ایمن سهمیه Gemini

اصل طراحی: هر تغییر وضعیت باید در SQLite ثبت شود قبل از انجام عملیات
خارجی (ارسال به API)، تا restart در هر نقطه‌ای ایمن باشد.
"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from src.database import Database
from src.logging_config import get_logger
from src.models import (
    Article,
    AudienceQuestion,
    DeliveryRecord,
    DeliveryStatus,
    GeminiDailyQuota,
    GeneratedPost,
    LessonProgress,
    PostCategory,
    PostStatus,
)
from src.utils import utcnow

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """
    High-level state management for the TechBot.

    Wraps :class:`~src.database.Database` with business-logic methods.
    One instance per process (singleton pattern via :func:`get_state_manager`).
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def checkpoint(self) -> None:
        """Flush SQLite WAL to main database file."""
        self._db.checkpoint()

    # =========================================================================
    # Gemini Quota — core concurrency-safety mechanism
    # =========================================================================

    @contextmanager
    def gemini_quota_context(self) -> Generator[None, None, None]:
        """
        Context manager for safe Gemini API usage.

        Usage::

            with state.gemini_quota_context():
                response = call_gemini_api(...)
                # quota is consumed after the with-block exits normally

        Behaviour:
          - On entry: atomically acquires the quota lock in SQLite.
            If the lock cannot be acquired (quota exhausted or already locked),
            raises :exc:`GeminiQuotaExhaustedError`.
          - On successful exit: marks quota as consumed.
          - On exception: releases the lock without consuming quota,
            so a future run can still try once.

        This pattern survives process restarts:
          - If the process is killed AFTER lock acquisition but BEFORE consumed:
            the lock remains in ``locked`` state. On next startup, the DB will
            have status='locked' from the same calendar day. The
            :meth:`repair_stale_locks` method handles this case.
        """
        acquired = self._db.try_acquire_gemini_quota()
        if not acquired:
            quota = self._db.get_gemini_quota()
            raise GeminiQuotaExhaustedError(
                f"Gemini API quota exhausted for {quota.date}: "
                f"{quota.requests_used}/{quota.max_requests} used, status={quota.status.value}"
            )
        try:
            yield
        except Exception:
            # Request failed — release lock so a future run can try
            self._db.release_gemini_quota_lock()
            raise
        else:
            # Success — mark consumed
            self._db.mark_gemini_quota_consumed()

    def get_gemini_quota(self) -> GeminiDailyQuota:
        """Return today's quota status."""
        return self._db.get_gemini_quota()

    def is_gemini_available(self) -> bool:
        """Return True if the Gemini API quota is available for today."""
        return self._db.get_gemini_quota().is_available

    def repair_stale_locks(self) -> None:
        """
        Release any quota lock that was acquired on a PREVIOUS calendar day
        and is stuck in 'locked' status (process crashed before consuming).

        Call this once on startup.
        """
        quota = self._db.get_gemini_quota()
        today = self._db._today_date_str()  # noqa: SLF001

        # If there is a lock from a previous day still in 'locked' state,
        # we release it (the new day gets a fresh row, so this only matters
        # if for some reason the date matches today and the lock is stale).
        if quota.status.value == "locked" and quota.date == today:
            # Check if lock was acquired more than 30 minutes ago —
            # that's a sign of a crashed process
            if quota.lock_acquired_at:
                age_seconds = (utcnow() - quota.lock_acquired_at).total_seconds()
                if age_seconds > 1800:  # 30 minutes
                    logger.warning(
                        "Stale Gemini quota lock detected (age=%.0fs). Releasing.", age_seconds
                    )
                    self._db.release_gemini_quota_lock()
                else:
                    logger.debug(
                        "Gemini quota lock age=%.0fs — not stale yet.", age_seconds
                    )
            else:
                # Lock without timestamp — release unconditionally
                logger.warning("Gemini quota lock has no timestamp. Releasing.")
                self._db.release_gemini_quota_lock()

    # =========================================================================
    # Articles
    # =========================================================================

    def save_article(self, article: Article) -> bool:
        """
        Persist *article* to the database.

        Returns True if newly inserted, False if duplicate.
        """
        return self._db.insert_article(article)

    def is_duplicate_article(self, content_hash: str) -> bool:
        return self._db.is_duplicate_article(content_hash)

    def get_top_articles(
        self,
        limit: int = 20,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[Article]:
        return self._db.get_top_articles(limit=limit, category=category, min_score=min_score)

    # =========================================================================
    # Generated Posts
    # =========================================================================

    def save_post(self, post: GeneratedPost) -> None:
        """Persist a generated post."""
        self._db.insert_post(post)

    def get_pending_posts(self, limit: int = 50) -> list[GeneratedPost]:
        return self._db.get_pending_posts(limit=limit)

    def mark_post_published(self, post_id: str) -> None:
        self._db.update_post_status(post_id, PostStatus.PUBLISHED, published_at=utcnow())
        logger.info("Post %s marked as published.", post_id)

    def mark_post_failed(self, post_id: str) -> None:
        self._db.update_post_status(post_id, PostStatus.FAILED)
        logger.warning("Post %s marked as failed.", post_id)

    def mark_post_scheduled(self, post_id: str, scheduled_at: datetime) -> None:
        self._db.update_post_status(post_id, PostStatus.SCHEDULED)
        # Update scheduled_at via direct SQL (convenience — avoids a new public DB method)
        with self._db._transaction() as conn:  # noqa: SLF001
            conn.execute(
                "UPDATE generated_posts SET scheduled_at = ? WHERE id = ?",
                (scheduled_at.isoformat(), post_id),
            )

    # =========================================================================
    # Delivery Records
    # =========================================================================

    def record_delivery_attempt(
        self,
        post_id: str,
        destination: str,
        success: bool,
        message_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> DeliveryRecord:
        """
        Record a delivery attempt for *post_id* to *destination*.

        Returns the updated :class:`DeliveryRecord`.
        """
        existing = self._db.get_delivery_record(post_id, destination)
        if existing:
            rec = existing.model_copy(
                update={
                    "attempts": existing.attempts + 1,
                    "last_attempt_at": utcnow(),
                    "status": DeliveryStatus.SENT if success else DeliveryStatus.FAILED,
                    "message_id": message_id or existing.message_id,
                    "error_message": error,
                    "sent_at": utcnow() if success else existing.sent_at,
                }
            )
        else:
            rec = DeliveryRecord(
                id=str(uuid.uuid4()),
                post_id=post_id,
                destination=destination,
                status=DeliveryStatus.SENT if success else DeliveryStatus.FAILED,
                message_id=message_id,
                attempts=1,
                last_attempt_at=utcnow(),
                error_message=error,
                sent_at=utcnow() if success else None,
            )
        self._db.upsert_delivery_record(rec)
        return rec

    def is_already_sent(self, post_id: str, destination: str) -> bool:
        """Return True if *post_id* was already successfully sent to *destination*."""
        return self._db.is_already_sent(post_id, destination)

    # =========================================================================
    # Lessons
    # =========================================================================

    def initialize_curriculum(self, lessons: list[dict[str, Any]]) -> None:
        """
        Ensure the lesson_progress table is populated from *lessons*.
        Safe to call multiple times (idempotent).
        """
        self._db.initialize_lessons(lessons)

    def get_next_lesson(self) -> Optional[LessonProgress]:
        """Return the next unpublished lesson."""
        return self._db.get_next_lesson()

    def mark_lesson_published(self, lesson_number: int, post_id: str) -> None:
        self._db.mark_lesson_published(lesson_number, post_id)
        logger.info("Lesson %d marked as published (post_id=%s).", lesson_number, post_id)

    def get_curriculum_status(self) -> list[LessonProgress]:
        """Return all lessons with their current status."""
        return self._db.get_all_lessons()

    def lessons_completed_count(self) -> int:
        lessons = self._db.get_all_lessons()
        return sum(1 for l in lessons if l.status == LessonStatus.PUBLISHED)

    # =========================================================================
    # Audience Questions
    # =========================================================================

    def save_question(self, question: AudienceQuestion) -> bool:
        return self._db.insert_question(question)

    def get_unanswered_questions(self, limit: int = 10) -> list[AudienceQuestion]:
        return self._db.get_unanswered_questions(limit=limit)

    def mark_questions_answered(self, question_ids: list[str], post_id: str) -> None:
        self._db.mark_questions_answered(question_ids, post_id)

    # =========================================================================
    # Status / diagnostics
    # =========================================================================

    def get_full_status(self) -> dict[str, Any]:
        """Return a complete status snapshot for the CLI `status` command."""
        stats = self._db.get_stats()
        stats["gemini_available"] = self.is_gemini_available()
        return stats

    def format_status_report(self) -> str:
        """Return a human-readable status report string (Persian)."""
        s = self.get_full_status()
        quota = s["gemini_quota"]
        lines = [
            "═══════════ وضعیت ربات تک‌نیوز ═══════════",
            f"📰 مقالات ذخیره‌شده:     {s['total_articles']}",
            f"📝 کل پست‌های تولیدشده: {s['total_posts']}",
            f"✅ پست‌های منتشرشده:    {s['published_posts']}",
            f"⏳ پست‌های در انتظار:   {s['pending_posts']}",
            f"📚 درس‌های منتشرشده:    {s['lessons_published']}/30",
            f"❓ سؤالات بی‌پاسخ:      {s['unanswered_questions']}",
            "─────────────────────────────────────────",
            f"🤖 Gemini — تاریخ:       {quota['date']}",
            f"🤖 Gemini — درخواست:     {quota['requests_used']}/{quota['max_requests']}",
            f"🤖 Gemini — وضعیت:       {quota['status']}",
            f"🤖 Gemini — در دسترس:   {'بله ✅' if s['gemini_available'] else 'خیر ❌'}",
            "═══════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GeminiQuotaExhaustedError(RuntimeError):
    """Raised when the daily Gemini API quota is exhausted or locked."""


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_state_manager: Optional[StateManager] = None


def get_state_manager(db: Optional[Database] = None) -> StateManager:
    """
    Return the singleton :class:`StateManager` instance.

    On first call, *db* must be provided.
    Subsequent calls return the existing instance (ignoring *db*).
    """
    global _state_manager  # noqa: PLW0603
    if _state_manager is None:
        if db is None:
            raise RuntimeError(
                "StateManager has not been initialised yet. "
                "Call get_state_manager(db=<Database instance>) on first use."
            )
        _state_manager = StateManager(db)
        logger.info("StateManager initialised.")
    return _state_manager


def reset_state_manager() -> None:
    """
    Reset the singleton (for testing purposes only).
    Do NOT call in production code.
    """
    global _state_manager  # noqa: PLW0603
    _state_manager = None
