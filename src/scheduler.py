"""
src/scheduler.py — زمان‌بندی انتشار با timezone Asia/Tehran

ساختار روزانه:
  - ۵ نوبت (۰۹:۰۰، ۱۲:۰۰، ۱۵:۰۰، ۱۸:۰۰، ۲۱:۰۰) × ۲ پست = ۱۰ پست روزانه
  - هر نوبت ۲ پست با فاصله ۸ دقیقه
  - یک job روزانه در ۰۷:۰۰ برای واکشی، امتیازدهی، فراخوانی Gemini و ساخت پست‌ها
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import AppConfig
from src.content_builder import build_posts
from src.gemini import GeminiAPIError, call_gemini_batch
from src.logging_config import get_logger
from src.models import GeneratedPost, PostStatus
from src.publisher import Publisher
from src.scorer import score_articles
from src.sources import fetch_all_sources
from src.state import GeminiQuotaExhaustedError, StateManager
from src.utils import utcnow

logger = get_logger(__name__)

_TEHRAN_TZ = pytz.timezone("Asia/Tehran")


def _parse_slot_time(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' into (hour, minute) integers."""
    h, m = time_str.split(":")
    return int(h), int(m)


def _today_slot_datetime(hour: int, minute: int) -> datetime:
    """Return today's datetime at (hour, minute) in Tehran timezone (UTC-aware)."""
    now_tehran = datetime.now(tz=_TEHRAN_TZ)
    slot = now_tehran.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return slot


class TechBotScheduler:
    """
    APScheduler-based timezone-aware scheduler for TechBot.

    Distributes 10 posts across 5 daily slots (2 posts each).
    Runs the daily content generation pipeline at 07:00 Tehran time.
    """

    def __init__(
        self,
        cfg: AppConfig,
        state: StateManager,
        publisher: Publisher,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._publisher = publisher
        self._scheduler = BlockingScheduler(timezone=_TEHRAN_TZ)

    # =========================================================================
    # Daily pipeline
    # =========================================================================

    def run_once(self, dry_run: bool = False) -> None:
        """
        Execute the full daily content generation and scheduling pipeline.

        Steps:
          1. Fetch articles from all RSS sources.
          2. Score and filter articles.
          3. Call Gemini batch API (single daily request).
          4. Build posts from Gemini response.
          5. Persist posts and assign scheduled slots.
        """
        logger.info("=== Daily pipeline starting (dry_run=%s) ===", dry_run)

        # 1. Fetch articles
        try:
            articles = asyncio.run(fetch_all_sources(self._state, self._cfg))
        except Exception as exc:  # noqa: BLE001
            logger.error("Article fetch failed: %s", exc)
            articles = []

        if not articles:
            logger.warning("No new articles fetched today. Pipeline aborted.")
            return

        # 2. Score and filter
        scored = score_articles(articles, self._cfg)
        min_score = float(self._cfg.yaml.get("scoring", {}).get("min_score", 0.3))
        eligible = [a for a in scored if a.score >= min_score]

        if not eligible:
            logger.warning(
                "No articles passed min_score=%.2f threshold. Pipeline aborted.", min_score
            )
            return

        # Select top N for Gemini batch
        batch_size = self._cfg.gemini_batch_size
        top_articles = eligible[:batch_size]
        logger.info(
            "Scored %d articles, %d eligible, sending %d to Gemini.",
            len(scored),
            len(eligible),
            len(top_articles),
        )

        # 3. Call Gemini
        try:
            gemini_response = call_gemini_batch(top_articles, self._state, self._cfg)
        except GeminiQuotaExhaustedError as exc:
            logger.warning("Gemini quota exhausted — skipping content generation: %s", exc)
            return
        except GeminiAPIError as exc:
            logger.error("Gemini API error — pipeline aborted: %s", exc)
            return

        # 4. Build posts
        posts = build_posts(gemini_response, articles, self._cfg)
        if not posts:
            logger.warning("No posts built from Gemini response.")
            return

        # 5. Persist and schedule
        slots_cfg: list[dict[str, Any]] = self._cfg.yaml.get("schedule", {}).get("slots", [])
        scheduled_times = self._compute_slot_datetimes(slots_cfg)

        for idx, post in enumerate(posts):
            try:
                self._state.save_post(post)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to save post %s: %s", post.id[:16], exc)
                continue

            if idx < len(scheduled_times):
                slot_time = scheduled_times[idx]
                try:
                    self._state.mark_post_scheduled(post.id, slot_time)
                    logger.debug(
                        "Post %s scheduled for %s",
                        post.id[:16],
                        slot_time.isoformat(),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not mark post %s as scheduled: %s", post.id[:16], exc)

        logger.info(
            "Pipeline complete: %d posts built, %d scheduled.",
            len(posts),
            min(len(posts), len(scheduled_times)),
        )

    def _compute_slot_datetimes(
        self, slots_cfg: list[dict[str, Any]]
    ) -> list[datetime]:
        """
        Compute all scheduled datetimes for today's slots.

        Each slot generates TWO datetimes: slot_time and slot_time + gap_minutes.
        Returns a flat list of UTC-aware datetimes.
        """
        result: list[datetime] = []
        for slot in slots_cfg:
            time_str: str = slot.get("time", "09:00")
            gap: int = int(slot.get("gap_minutes", 8))
            posts_per_slot: int = int(slot.get("posts", 2))

            hour, minute = _parse_slot_time(time_str)
            base_dt = _today_slot_datetime(hour, minute)

            for i in range(posts_per_slot):
                result.append(base_dt + timedelta(minutes=gap * i))

        return result

    # =========================================================================
    # Slot publishing jobs
    # =========================================================================

    def _publish_slot_post(self, dry_run: bool = False) -> None:
        """
        APScheduler job: publish the next pending post.

        Gets the first pending/scheduled post and publishes it.
        """
        posts = self._state.get_pending_posts(limit=1)
        if not posts:
            logger.info("No pending posts to publish at this slot.")
            return

        post = posts[0]
        logger.info(
            "Slot job: publishing post %s (category=%s).",
            post.id[:16],
            post.category.value,
        )
        self._publisher.publish(post, dry_run=dry_run)

    def schedule_daily_run(self, dry_run: bool = False) -> None:
        """
        Register APScheduler jobs for all daily time slots.

        Each slot spawns two jobs separated by gap_minutes.
        """
        slots_cfg: list[dict[str, Any]] = self._cfg.yaml.get("schedule", {}).get("slots", [])

        for slot in slots_cfg:
            time_str: str = slot.get("time", "09:00")
            gap: int = int(slot.get("gap_minutes", 8))
            posts_per_slot: int = int(slot.get("posts", 2))
            hour, minute = _parse_slot_time(time_str)

            for i in range(posts_per_slot):
                slot_minute = minute + gap * i
                slot_hour = hour + slot_minute // 60
                slot_minute = slot_minute % 60

                trigger = CronTrigger(
                    hour=slot_hour,
                    minute=slot_minute,
                    timezone=_TEHRAN_TZ,
                )
                job_id = f"slot_{hour:02d}{minute:02d}_post{i + 1}"
                self._scheduler.add_job(
                    self._publish_slot_post,
                    trigger=trigger,
                    id=job_id,
                    kwargs={"dry_run": dry_run},
                    replace_existing=True,
                    name=f"Publish slot {time_str} post {i + 1}",
                )
                logger.debug(
                    "Scheduled slot job: %s at %02d:%02d Tehran",
                    job_id,
                    slot_hour,
                    slot_minute,
                )

    # =========================================================================
    # Serve
    # =========================================================================

    def serve(self, dry_run: bool = False) -> None:
        """
        Start the blocking scheduler.

        Schedule:
          - 07:00 Tehran: daily pipeline (fetch → score → Gemini → build → schedule)
          - 09:00, 09:08, 12:00, 12:08, 15:00, 15:08, 18:00, 18:08, 21:00, 21:08: publish slots
        """
        logger.info("Starting TechBot scheduler (timezone=Asia/Tehran)...")

        # Daily pipeline job at 07:00 Tehran
        self._scheduler.add_job(
            self.run_once,
            trigger=CronTrigger(hour=7, minute=0, timezone=_TEHRAN_TZ),
            id="daily_pipeline",
            kwargs={"dry_run": dry_run},
            replace_existing=True,
            name="Daily pipeline (fetch → Gemini → build)",
        )
        logger.info("Daily pipeline job registered at 07:00 Tehran.")

        # Slot publishing jobs
        self.schedule_daily_run(dry_run=dry_run)

        logger.info("TechBot scheduler started. Press Ctrl+C to stop.")
        try:
            self._scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user — shutting down.")
            self._scheduler.shutdown(wait=False)
