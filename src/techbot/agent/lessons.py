"""
src/techbot/agent/lessons.py — Custom lesson persistence layer.

Provides functions to load, save, validate, and retrieve custom lessons stored
in data/custom_lessons.json. Custom lessons override curriculum.json entries
when a matching lesson_id is found.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from src.techbot.agent.lesson_schema import LessonSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _get_custom_lessons_path() -> Path:
    """
    Resolve the path to data/custom_lessons.json relative to the project root.
    The project root is 3 levels up from this file:
      src/techbot/agent/lessons.py -> src/techbot/agent -> src/techbot -> src -> root
    """
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "custom_lessons.json"


# ---------------------------------------------------------------------------
# Low-level JSON I/O (atomic writes)
# ---------------------------------------------------------------------------

def _load_raw() -> list[dict[str, Any]]:
    """
    Load raw lesson list from data/custom_lessons.json.
    Returns an empty list if the file does not exist or is malformed.
    """
    path = _get_custom_lessons_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        logger.warning("custom_lessons.json is not a list; resetting to empty.")
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read custom_lessons.json: %s", exc)
        return []


def _save_raw(lessons: list[dict[str, Any]]) -> None:
    """
    Atomically write the lesson list to data/custom_lessons.json.
    Uses a temp file + os.replace for safety (same strategy as state/atomic.py).
    """
    path = _get_custom_lessons_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(lessons, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_custom_lessons() -> list[dict[str, Any]]:
    """
    Return the full list of custom lessons from data/custom_lessons.json.
    Each item is a raw dict (not a Pydantic model).
    """
    return _load_raw()


def get_lesson(lesson_id: int) -> Optional[dict[str, Any]]:
    """
    Look up a custom lesson by its integer lesson_id.

    Returns:
        dict if found, None otherwise.
    """
    lessons = _load_raw()
    for lesson in lessons:
        if lesson.get("lesson_id") == lesson_id:
            return lesson
    return None


def validate_lesson(data: dict[str, Any]) -> LessonSchema:
    """
    Validate raw lesson data against LessonSchema.

    Returns:
        A validated LessonSchema instance.

    Raises:
        pydantic.ValidationError: if the data does not conform to LessonSchema.
    """
    return LessonSchema.model_validate(data)


def save_lesson(data: dict[str, Any]) -> LessonSchema:
    """
    Validate and persist a lesson to data/custom_lessons.json.

    If a lesson with the same lesson_id already exists it is replaced (upsert).
    Invalid data raises pydantic.ValidationError before any file is written.

    Args:
        data: Raw lesson dict (from JSON parsed admin input).

    Returns:
        The validated LessonSchema instance.

    Raises:
        pydantic.ValidationError: if validation fails.
        OSError: if writing to disk fails.
    """
    lesson = validate_lesson(data)
    lesson_dict = lesson.model_dump(mode="json")

    lessons = _load_raw()

    # Upsert: replace existing entry with same lesson_id, or append.
    replaced = False
    for i, existing in enumerate(lessons):
        if existing.get("lesson_id") == lesson.lesson_id:
            lessons[i] = lesson_dict
            replaced = True
            break

    if not replaced:
        lessons.append(lesson_dict)

    _save_raw(lessons)

    action = "جایگزین" if replaced else "اضافه"
    logger.info(
        "Custom lesson %d '%s' %s شد و در custom_lessons.json ذخیره شد.",
        lesson.lesson_id,
        lesson.title,
        action,
    )
    return lesson


def delete_lesson(lesson_id: int) -> bool:
    """
    Remove a custom lesson by lesson_id.

    Returns:
        True if a lesson was found and removed, False otherwise.
    """
    lessons = _load_raw()
    filtered = [l for l in lessons if l.get("lesson_id") != lesson_id]
    if len(filtered) == len(lessons):
        return False
    _save_raw(filtered)
    logger.info("Custom lesson %d حذف شد.", lesson_id)
    return True
