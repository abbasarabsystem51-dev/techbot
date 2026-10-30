"""
src/techbot/state package — Atomic persistence and idempotency module.
"""
from src.techbot.state.atomic import (
    StateManager,
    already_processed,
    build_idempotency_key,
    get_feed_failures,
    get_recent_titles,
    get_sent_links,
    is_run_completed,
    mark_processed,
    mark_run_completed,
    read_state,
    record_recent_title,
    record_sent_link,
    reset_feed_failure,
    track_feed_failure,
    write_state_atomic,
)

__all__ = [
    "StateManager",
    "already_processed",
    "build_idempotency_key",
    "get_feed_failures",
    "get_recent_titles",
    "get_sent_links",
    "is_run_completed",
    "mark_processed",
    "mark_run_completed",
    "read_state",
    "record_recent_title",
    "record_sent_link",
    "reset_feed_failure",
    "track_feed_failure",
    "write_state_atomic",
]
