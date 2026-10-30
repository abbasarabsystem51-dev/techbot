"""
src/techbot/state/atomic.py — Atomic state management and idempotency guarantees.

Features:
  1. Atomic file writes via temp file creation, flush + fsync, and atomic rename (os.replace).
  2. Idempotency check mechanism using composite key: `date+run_index+basket`.
  3. Safe JSON read with UTF-8 / UTF-8-SIG fallback and missing file handling.
  4. StateManager class with thread-safe atomic transaction context manager.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional


# ---------------------------------------------------------------------------
# Key Generator for Idempotency
# ---------------------------------------------------------------------------

def build_idempotency_key(date: str, run_index: int | str, basket: str) -> str:
    """
    Generate standard idempotency key in the format: `date#run_index#basket`.

    Example:
        >>> build_idempotency_key("2026-10-29", 1, "news_ai")
        '2026-10-29#1#news_ai'
    """
    clean_date = str(date).strip()
    clean_run = str(run_index).strip()
    clean_basket = str(basket).strip().lower()
    return f"{clean_date}#{clean_run}#{clean_basket}"


# ---------------------------------------------------------------------------
# Atomic File I/O
# ---------------------------------------------------------------------------

def read_state(path: str | Path) -> dict[str, Any]:
    """
    Read the persistent state file safely.

    Returns:
        dict[str, Any]: Parsed JSON content, or an empty dict if the file does not exist.

    Raises:
        ValueError: If file content is not a valid JSON object.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}

    # utf-8-sig automatically strips BOM if present
    with file_path.open("r", encoding="utf-8-sig") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as err:
            raise ValueError(f"Corrupted state JSON in {file_path}: {err}") from err

    if not isinstance(data, dict):
        raise ValueError(f"State file {file_path} must contain a JSON object, got {type(data).__name__}")

    return data


def write_state_atomic(path: str | Path, state: dict[str, Any]) -> None:
    """
    Atomically write the state dictionary to disk.

    Process:
      1. Ensure parent directory exists.
      2. Write formatted JSON to a temporary file in the *same* directory.
      3. Flush application buffer and force OS-level disk flush with os.fsync.
      4. Atomically swap the temp file to destination with os.replace.
      5. Guarantee temp file deletion on any error.
    """
    if not isinstance(state, dict):
        raise ValueError(f"State must be a dictionary, got {type(state).__name__}")

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Creating temp file in the SAME directory ensures it lives on the same filesystem/drive,
    # which is required for os.replace to be an atomic rename operation.
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.",
        suffix=".tmp",
        dir=str(file_path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Atomic replacement of destination file
        os.replace(temp_name, str(file_path))
    except Exception:
        # Clean up temporary file on failure
        try:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Idempotency Checks and Mutations
# ---------------------------------------------------------------------------

def is_run_completed(
    state: dict[str, Any],
    date: str,
    run_index: int | str,
    basket: str,
) -> bool:
    """
    Check if a specific run slot (`date+run_index+basket`) has already completed.
    """
    key = build_idempotency_key(date, run_index, basket)
    keys_list = state.get("idempotency_keys", [])
    if key in keys_list:
        return True

    completed_runs = state.get("completed_runs", {})
    return key in completed_runs


def mark_run_completed(
    state: dict[str, Any],
    date: str,
    run_index: int | str,
    basket: str,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Mark a run slot as completed with key `date+run_index+basket`.
    Returns a new copy of the updated state dict.
    """
    updated = dict(state)
    key = build_idempotency_key(date, run_index, basket)

    # 1. Update list of keys
    keys_list = list(updated.get("idempotency_keys", []))
    if key not in keys_list:
        keys_list.append(key)
    updated["idempotency_keys"] = keys_list

    # 2. Update map of completed run records
    completed_runs = dict(updated.get("completed_runs", {}))
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "key": key,
        "date": str(date),
        "run_index": int(run_index) if str(run_index).isdigit() else run_index,
        "basket": str(basket),
        "completed_at": now_iso,
    }
    if details:
        record["details"] = details
    completed_runs[key] = record
    updated["completed_runs"] = completed_runs

    return updated


# Backward-compatible feed item deduplication
def already_processed(item_id: str, state: dict[str, Any]) -> bool:
    """Return True if item_id is present in state['processed_ids']."""
    processed = state.get("processed_ids", [])
    return item_id in processed


def mark_processed(item_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """Append item_id to state['processed_ids'] without duplicates."""
    updated = dict(state)
    ids = list(updated.get("processed_ids", []))
    if item_id not in ids:
        ids.append(item_id)
    updated["processed_ids"] = ids
    return updated


# ---------------------------------------------------------------------------
# Feed Failure & Health Tracking
# ---------------------------------------------------------------------------

def track_feed_failure(state: dict[str, Any], source_id: str, error: str = "") -> dict[str, Any]:
    """Increment consecutive failures for a source and record the last error timestamp."""
    failures = dict(state.get("feed_failures", {}))
    current_count = failures.get(source_id, 0)
    failures[source_id] = current_count + 1
    state["feed_failures"] = failures

    feed_errors = dict(state.get("feed_last_errors", {}))
    feed_errors[source_id] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": str(error)[:300],
    }
    state["feed_last_errors"] = feed_errors
    return state


def reset_feed_failure(state: dict[str, Any], source_id: str) -> dict[str, Any]:
    """Reset consecutive failure counter for a source upon successful fetch."""
    failures = dict(state.get("feed_failures", {}))
    if source_id in failures:
        failures[source_id] = 0
        state["feed_failures"] = failures
    return state


def get_feed_failures(state: dict[str, Any], source_id: str) -> int:
    """Return consecutive failure count for a source."""
    return state.get("feed_failures", {}).get(source_id, 0)


# ---------------------------------------------------------------------------
# Sent Links & Recent Titles Deduplication State
# ---------------------------------------------------------------------------

def record_sent_link(state: dict[str, Any], canonical_url: str) -> dict[str, Any]:
    """Add a normalized/canonical URL to the sent_links set."""
    links = list(state.get("sent_links", []))
    cleaned = canonical_url.strip()
    if cleaned and cleaned not in links:
        links.append(cleaned)
    state["sent_links"] = links
    return state


def get_sent_links(state: dict[str, Any]) -> set[str]:
    """Return all previously sent canonical URLs as a set."""
    return set(state.get("sent_links", []))


def record_recent_title(state: dict[str, Any], title: str, max_entries: int = 150) -> dict[str, Any]:
    """Add a published title to recent_titles with rolling window limit."""
    titles = list(state.get("recent_titles", []))
    cleaned = title.strip()
    if cleaned and cleaned not in titles:
        titles.append(cleaned)
    if len(titles) > max_entries:
        titles = titles[-max_entries:]
    state["recent_titles"] = titles
    return state


def get_recent_titles(state: dict[str, Any]) -> list[str]:
    """Return recent published titles for similarity checks."""
    return list(state.get("recent_titles", []))


# ---------------------------------------------------------------------------
# StateManager Class
# ---------------------------------------------------------------------------

class StateManager:
    """
    Thread-safe state manager providing transaction-based atomic file updates.
    """

    def __init__(self, state_path: str | Path) -> None:
        self.path = Path(state_path)
        self._lock = threading.Lock()

    def get_state(self) -> dict[str, Any]:
        """Read and return current state from file."""
        with self._lock:
            return read_state(self.path)

    def save_state(self, state: dict[str, Any]) -> None:
        """Atomically persist state dict to file."""
        with self._lock:
            write_state_atomic(self.path, state)

    @contextmanager
    def transaction(self) -> Generator[dict[str, Any], None, None]:
        """
        Transactional context manager.
        Loads state, yields it for mutation, and writes it atomically upon normal exit.

        Usage:
            >>> manager = StateManager("data/state.json")
            >>> with manager.transaction() as state:
            ...     mark_run_completed(state, "2026-10-29", 1, "news_ai")
        """
        with self._lock:
            state = read_state(self.path)
            yield state
            write_state_atomic(self.path, state)

    def is_run_completed(self, date: str, run_index: int | str, basket: str) -> bool:
        """Check if date+run_index+basket has already completed."""
        state = self.get_state()
        return is_run_completed(state, date, run_index, basket)

    def record_completed_run(
        self,
        date: str,
        run_index: int | str,
        basket: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Mark date+run_index+basket as completed atomically."""
        with self.transaction() as state:
            updated = mark_run_completed(state, date, run_index, basket, details)
            state.clear()
            state.update(updated)

    def is_item_processed(self, item_id: str) -> bool:
        """Check if an article/feed item has already been processed."""
        state = self.get_state()
        return already_processed(item_id, state)

    def record_processed_item(self, item_id: str) -> None:
        """Mark an article/feed item as processed atomically."""
        with self.transaction() as state:
            updated = mark_processed(item_id, state)
            state.clear()
            state.update(updated)

    def track_feed_failure(self, source_id: str, error: str = "") -> None:
        """Increment failure counter for a source atomically."""
        with self.transaction() as state:
            updated = track_feed_failure(state, source_id, error)
            state.clear()
            state.update(updated)

    def reset_feed_failure(self, source_id: str) -> None:
        """Reset failure counter for a source atomically."""
        with self.transaction() as state:
            updated = reset_feed_failure(state, source_id)
            state.clear()
            state.update(updated)

    def get_feed_failures(self, source_id: str) -> int:
        """Get consecutive failures for a source."""
        state = self.get_state()
        return get_feed_failures(state, source_id)

    def record_sent_link(self, canonical_url: str) -> None:
        """Record sent link atomically."""
        with self.transaction() as state:
            updated = record_sent_link(state, canonical_url)
            state.clear()
            state.update(updated)

    def get_sent_links(self) -> set[str]:
        """Get all sent canonical links."""
        state = self.get_state()
        return get_sent_links(state)

    def record_recent_title(self, title: str) -> None:
        """Record recent title atomically."""
        with self.transaction() as state:
            updated = record_recent_title(state, title)
            state.clear()
            state.update(updated)

    def get_recent_titles(self) -> list[str]:
        """Get list of recent titles."""
        state = self.get_state()
        return get_recent_titles(state)
