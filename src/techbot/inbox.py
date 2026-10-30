"""
src/techbot/inbox.py — Audience Question & Feedback Manager for Weekly Q&A.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.techbot.state.atomic import StateManager

logger = logging.getLogger(__name__)


class InboxManager:
    """
    Manages incoming audience questions and topics for the weekly Q&A slot.
    """

    def __init__(self, state_manager: StateManager) -> None:
        self.state_manager = state_manager

    def add_question(
        self,
        question_text: str,
        user_id: Optional[str] = None,
        topic: str = "ai",
    ) -> str:
        """Store a new audience question into persistent state."""
        q_id = str(uuid.uuid4())[:8]
        record = {
            "id": q_id,
            "text": question_text.strip(),
            "user_id": user_id,
            "topic": topic,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }

        with self.state_manager.transaction() as state:
            questions = list(state.get("audience_questions", []))
            questions.append(record)
            state["audience_questions"] = questions

        logger.info("New audience question registered [%s]: %s", q_id, question_text[:50])
        return q_id

    def get_pending_questions(self, limit: int = 5) -> list[dict[str, Any]]:
        """Fetch unanswered pending questions."""
        state = self.state_manager.get_state()
        all_q = state.get("audience_questions", [])
        pending = [q for q in all_q if q.get("status") == "pending"]
        return pending[:limit]

    def mark_answered(self, question_ids: list[str]) -> None:
        """Mark processed questions as answered."""
        id_set = set(question_ids)
        with self.state_manager.transaction() as state:
            questions = list(state.get("audience_questions", []))
            for q in questions:
                if q.get("id") in id_set:
                    q["status"] = "answered"
                    q["answered_at"] = datetime.now(timezone.utc).isoformat()
            state["audience_questions"] = questions
