"""
src/publisher.py — لایه انتشار — ارسال به همه مقصدها

این ماژول از TelegramPublisher و EitaaPublisher استفاده می‌کند و
وضعیت نهایی پست را در SQLite ثبت می‌کند.
"""
from __future__ import annotations

import time
from typing import Optional

from src.config import AppConfig
from src.eitaa import EitaaPublisher
from src.logging_config import get_logger
from src.models import GeneratedPost
from src.state import StateManager
from src.telegram import TelegramPublisher
from src.utils import utcnow

logger = get_logger(__name__)


class Publisher:
    """
    Orchestrates publishing a :class:`GeneratedPost` to all enabled destinations.

    Destinations:
      - Telegram (via :class:`TelegramPublisher`)
      - Eitaa    (via :class:`EitaaPublisher`)

    Each destination is tried independently.
    Delivery outcomes are recorded in SQLite by the destination adapters.
    """

    def __init__(self, cfg: AppConfig, state: StateManager) -> None:
        self._cfg = cfg
        self._state = state
        self._telegram: Optional[TelegramPublisher] = (
            TelegramPublisher(cfg, state) if cfg.telegram_enabled else None
        )
        self._eitaa: Optional[EitaaPublisher] = (
            EitaaPublisher(cfg, state) if cfg.eitaa_enabled else None
        )
        self._inter_post_delay: float = float(
            cfg.yaml.get("destinations", {})
            .get("telegram", {})
            .get("inter_post_delay_seconds", 2)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def publish(
        self, post: GeneratedPost, dry_run: bool = False
    ) -> dict[str, bool]:
        """
        Publish *post* to all enabled destinations.

        Behaviour:
          1. Checks each enabled destination individually for prior sends (idempotency).
          2. Publishes to Telegram (if enabled).
          3. Publishes to Eitaa (if enabled).
          4. Updates post status in DB:
             - PUBLISHED if all enabled destinations succeeded.
             - FAILED if any enabled destination failed.

        Returns a dict mapping destination name → success bool.
        """
        results: dict[str, bool] = {}
        enabled_destinations: list[str] = []

        if self._telegram is not None:
            enabled_destinations.append("telegram")
        if self._eitaa is not None:
            enabled_destinations.append("eitaa")

        if not enabled_destinations:
            logger.warning("No destinations are enabled — post %s not published.", post.id[:16])
            return results

        # Check if ALL destinations already have this post sent (full idempotency)
        all_already_sent = all(
            self._state.is_already_sent(post.id, dest)
            for dest in enabled_destinations
        )
        if all_already_sent:
            logger.info(
                "Post %s already delivered to all destinations — skipping.", post.id[:16]
            )
            return {dest: True for dest in enabled_destinations}

        # Telegram
        if self._telegram is not None:
            try:
                tg_ok = self._telegram.send_post(post, dry_run=dry_run)
                results["telegram"] = tg_ok
                logger.debug(
                    "Telegram delivery for post %s: %s",
                    post.id[:16],
                    "OK" if tg_ok else "FAIL",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error publishing to Telegram: %s", exc)
                results["telegram"] = False

        # Eitaa
        if self._eitaa is not None:
            try:
                eitaa_ok = self._eitaa.send_post(post, dry_run=dry_run)
                results["eitaa"] = eitaa_ok
                logger.debug(
                    "Eitaa delivery for post %s: %s",
                    post.id[:16],
                    "OK" if eitaa_ok else "FAIL",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error publishing to Eitaa: %s", exc)
                results["eitaa"] = False

        # Update post-level status in DB
        all_succeeded = all(results.get(d, False) for d in enabled_destinations)
        any_failed = any(not results.get(d, True) for d in enabled_destinations)

        if all_succeeded:
            self._state.mark_post_published(post.id)
            logger.info(
                "Post %s: PUBLISHED to %s.", post.id[:16], ", ".join(enabled_destinations)
            )
        elif any_failed:
            # Only mark failed if it wasn't already published from a previous partial run
            self._state.mark_post_failed(post.id)
            logger.warning(
                "Post %s: FAILED on some destinations. Results: %s",
                post.id[:16],
                results,
            )

        return results

    def publish_many(
        self,
        posts: list[GeneratedPost],
        dry_run: bool = False,
    ) -> list[dict[str, bool]]:
        """
        Publish multiple posts sequentially with inter-post delay.

        Returns a list of result dicts, one per post.
        """
        all_results: list[dict[str, bool]] = []
        total = len(posts)

        for idx, post in enumerate(posts):
            logger.info(
                "Publishing post %d/%d (id=%s, category=%s)...",
                idx + 1,
                total,
                post.id[:16],
                post.category.value,
            )
            result = self.publish(post, dry_run=dry_run)
            all_results.append(result)

            # Sleep between posts (not after the last one)
            if idx < total - 1 and self._inter_post_delay > 0:
                logger.debug("Sleeping %.1fs between posts...", self._inter_post_delay)
                time.sleep(self._inter_post_delay)

        logger.info(
            "publish_many complete: %d/%d succeeded.",
            sum(1 for r in all_results if all(r.values())),
            total,
        )
        return all_results
