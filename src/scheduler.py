"""
src/scheduler.py — زمان‌بندی و اجرای پایپ‌لاین انتشار
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import AppConfig
from src.content_builder import build_posts
from src.gemini import GeminiAPIError, call_gemini_batch
from src.logging_config import get_logger
from src.publisher import Publisher
from src.scorer import score_articles
from src.sources import fetch_all_sources
from src.state import GeminiQuotaExhaustedError, StateManager

logger = get_logger(__name__)

_TEHRAN_TZ = pytz.timezone("Asia/Tehran")


def _parse_slot_time(time_str: str) -> tuple[int, int]:
    h, m = time_str.split(":")
    return int(h), int(m)


def _today_slot_datetime(hour: int, minute: int) -> datetime:
    now_tehran = datetime.now(tz=_TEHRAN_TZ)
    return now_tehran.replace(hour=hour, minute=minute, second=0, microsecond=0)


class TechBotScheduler:
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

    def run_once(self, dry_run: bool = False) -> None:
        """
        Execute full content generation AND immediate publication for one-shot CI/CD runs.
        """
        logger.info("=== Daily pipeline starting (dry_run=%s) ===", dry_run)

        # 1. Fetch articles
        try:
            articles = asyncio.run(fetch_all_sources(self._state, self._cfg))
        except Exception as exc:
            logger.error("Article fetch failed: %s", exc)
            articles = []

        if not articles:
            logger.warning("No new articles fetched today.")
            self._publish_all_pending(dry_run=dry_run)
            return

        # 2. Score and filter
        scored = score_articles(articles, self._cfg)
        min_score = float(self._cfg.yaml.get("scoring", {}).get("min_score", 0.3))
        eligible = [a for a in scored if a.score >= min_score]

        if not eligible:
            logger.warning("No articles passed min_score=%.2f threshold.", min_score)
            self._publish_all_pending(dry_run=dry_run)
            return

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
            logger.warning("Gemini quota exhausted — publishing any existing pending posts: %s", exc)
            self._publish_all_pending(dry_run=dry_run)
            return
        except GeminiAPIError as exc:
            logger.error("Gemini API error — pipeline aborted: %s", exc)
            self._publish_all_pending(dry_run=dry_run)
            return

        # 4. Build posts
        posts = build_posts(gemini_response, articles, self._cfg)
        if not posts:
            logger.warning("No posts built from Gemini response.")
            return

        # 5. Persist posts to database
        slots_cfg: list[dict[str, Any]] = self._cfg.yaml.get("schedule", {}).get("slots", [])
        scheduled_times = self._compute_slot_datetimes(slots_cfg)

        for idx, post in enumerate(posts):
            try:
                self._state.save_post(post)
                if idx < len(scheduled_times):
                    self._state.mark_post_scheduled(post.id, scheduled_times[idx])
            except Exception as exc:
                logger.error("Failed to save post %s: %s", post.id[:16], exc)

        logger.info("Built and saved %d posts. Starting publication to channels...", len(posts))

        # 6. Publish generated posts immediately (Essential for GitHub Actions one-shot mode)
        published_count = 0
        for idx, post in enumerate(posts, start=1):
            try:
                logger.info(
                    "Publishing post %d/%d (id=%s, category=%s)...",
                    idx,
                    len(posts),
                    post.id[:12],
                    post.category.value,
                )
                ok = self._publisher.publish(post, dry_run=dry_run)
                if ok:
                    published_count += 1
                if idx < len(posts):
                    time.sleep(3)
            except Exception as exc:
                logger.error("Unexpected error publishing post %s: %s", post.id[:12], exc)

        logger.info(
            "=== Pipeline complete: %d/%d posts successfully published! ===",
            published_count,
            len(posts),
        )

    def _publish_all_pending(self, dry_run: bool = False) -> None:
        """Publish any pending posts remaining in the database."""
        pending = self._state.get_pending_posts(limit=10)
        if not pending:
            logger.info("No pending posts in database to publish.")
            return
        logger.info("Publishing %d pending posts from database...", len(pending))
        for post in pending:
            try:
                self._publisher.publish(post, dry_run=dry_run)
                time.sleep(3)
            except Exception as exc:
                logger.error("Failed to publish pending post %s: %s", post.id[:12], exc)

    def _compute_slot_datetimes(
        self, slots_cfg: list[dict[str, Any]]
    ) -> list[datetime]:
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

    def _publish_slot_post(self, dry_run: bool = False) -> None:
        posts = self._state.get_pending_posts(limit=1)
        if not posts:
            logger.info("No pending posts to publish at this slot.")
            return

        post = posts[0]
        logger.info("Slot job: publishing post %s (category=%s).", post.id[:16], post.category.value)
        self._publisher.publish(post, dry_run=dry_run)

    def schedule_daily_run(self, dry_run: bool = False) -> None:
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

    def serve(self, dry_run: bool = False) -> None:
        logger.info("Starting TechBot scheduler (timezone=Asia/Tehran)...")
        self._scheduler.add_job(
            self.run_once,
            trigger=CronTrigger(hour=7, minute=0, timezone=_TEHRAN_TZ),
            id="daily_pipeline",
            kwargs={"dry_run": dry_run},
            replace_existing=True,
            name="Daily pipeline (fetch -> Gemini -> build)",
        )
        self.schedule_daily_run(dry_run=dry_run)
        try:
            self._scheduler.start()
        except KeyboardInterrupt:
            self._scheduler.shutdown(wait=False)
