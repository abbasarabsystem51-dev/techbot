"""
src/database.py — لایه داده SQLite

ویژگی‌های کلیدی:
  - WAL mode + busy timeout برای دسترسی هم‌زمان ایمن
  - تمام نوشتن‌ها درون تراکنش اتمی
  - تشخیص تکراری (deduplication) با content_hash
  - قفل اتمی سهمیه روزانه Gemini API (بدون امکان race condition)
  - عدم تزریق SQL (پارامترهای bind)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from src.logging_config import get_logger
from src.models import (
    Article,
    AudienceQuestion,
    DeliveryRecord,
    DeliveryStatus,
    GeminiDailyQuota,
    GeminiQuotaStatus,
    GeneratedPost,
    LessonProgress,
    LessonStatus,
    PostStatus,
)
from src.utils import utcnow

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL — Schema definition
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS articles (
    id             TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    url            TEXT NOT NULL,
    canonical_url  TEXT NOT NULL,
    content_hash   TEXT NOT NULL UNIQUE,
    summary        TEXT DEFAULT '',
    source_name    TEXT NOT NULL,
    category       TEXT NOT NULL DEFAULT 'ai',
    published_at   TEXT,
    fetched_at     TEXT NOT NULL,
    score          REAL NOT NULL DEFAULT 0.0,
    is_duplicate   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_fetched_at   ON articles(fetched_at);
CREATE INDEX IF NOT EXISTS idx_articles_score        ON articles(score DESC);

CREATE TABLE IF NOT EXISTS generated_posts (
    id             TEXT PRIMARY KEY,
    article_id     TEXT REFERENCES articles(id) ON DELETE SET NULL,
    text           TEXT NOT NULL,
    category       TEXT NOT NULL,
    source_url     TEXT,
    source_title   TEXT,
    lesson_number  INTEGER,
    status         TEXT NOT NULL DEFAULT 'pending',
    created_at     TEXT NOT NULL,
    scheduled_at   TEXT,
    published_at   TEXT,
    quiz_question  TEXT,
    quiz_options   TEXT,   -- JSON encoded dict
    correct_answer TEXT
);

CREATE INDEX IF NOT EXISTS idx_posts_status       ON generated_posts(status);
CREATE INDEX IF NOT EXISTS idx_posts_scheduled_at ON generated_posts(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_posts_category     ON generated_posts(category);

CREATE TABLE IF NOT EXISTS delivery_records (
    id               TEXT PRIMARY KEY,
    post_id          TEXT NOT NULL REFERENCES generated_posts(id) ON DELETE CASCADE,
    destination      TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    message_id       TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_attempt_at  TEXT,
    error_message    TEXT,
    sent_at          TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_delivery_post_dest
    ON delivery_records(post_id, destination);
CREATE INDEX IF NOT EXISTS idx_delivery_status ON delivery_records(status);

CREATE TABLE IF NOT EXISTS gemini_quota (
    date             TEXT PRIMARY KEY,
    requests_used    INTEGER NOT NULL DEFAULT 0,
    max_requests     INTEGER NOT NULL DEFAULT 1,
    status           TEXT NOT NULL DEFAULT 'available',
    lock_acquired_at TEXT,
    last_request_at  TEXT
);

CREATE TABLE IF NOT EXISTS lesson_progress (
    lesson_number INTEGER PRIMARY KEY,
    title         TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    published_at  TEXT,
    post_id       TEXT REFERENCES generated_posts(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audience_questions (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'telegram',
    received_at     TEXT NOT NULL,
    is_answered     INTEGER NOT NULL DEFAULT 0,
    answered_in_post_id TEXT REFERENCES generated_posts(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_answered ON audience_questions(is_answered);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """
    Thread-safe SQLite database wrapper.

    One :class:`Database` instance per process is recommended.
    Uses a per-thread connection cache (``check_same_thread=False`` with
    explicit locking via :pyattr:`_lock`).
    """

    def __init__(self, db_path: Path | str, max_requests_per_day: int = 1) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_requests_per_day = max_requests_per_day
        self._lock = threading.Lock()
        self._local = threading.local()
        self._initialize()
        logger.info("Database ready: %s", self._path)

    # ── Connection management ─────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Return (and cache) a per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                isolation_level=None,  # autocommit off; we manage transactions
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._local.conn = conn
        return self._local.conn  # type: ignore[return-value]

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that wraps a SQLite transaction with a thread lock."""
        with self._lock:
            conn = self._connect()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _initialize(self) -> None:
        """Create schema (idempotent)."""
        with self._lock:
            conn = self._connect()
            conn.executescript(_DDL)
            conn.commit()
        logger.debug("Database schema initialised.")

    def checkpoint(self) -> None:
        """Run WAL checkpoint to flush changes to disk."""
        try:
            with self._lock:
                conn = self._connect()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.commit()
            logger.debug("Database WAL checkpoint completed.")
        except Exception as exc:
            logger.warning("WAL checkpoint error: %s", exc)

    def close(self) -> None:
        """Close the per-thread connection if open."""
        self.checkpoint()
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # ── Helper: row → dict ────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    # =========================================================================
    # Articles
    # =========================================================================

    def insert_article(self, article: Article) -> bool:
        """
        Insert *article* into the DB.

        Returns ``True`` if inserted, ``False`` if skipped (duplicate content_hash).
        """
        sql = """
        INSERT OR IGNORE INTO articles
            (id, title, url, canonical_url, content_hash, summary,
             source_name, category, published_at, fetched_at, score, is_duplicate)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            article.id,
            article.title,
            article.url,
            article.canonical_url,
            article.content_hash,
            article.summary,
            article.source_name,
            article.category.value,
            article.published_at.isoformat() if article.published_at else None,
            article.fetched_at.isoformat(),
            article.score,
            int(article.is_duplicate),
        )
        with self._transaction() as conn:
            cursor = conn.execute(sql, params)
            inserted = cursor.rowcount > 0

        if not inserted:
            logger.debug("Duplicate article skipped: %s", article.content_hash[:16])
        return inserted

    def is_duplicate_article(self, content_hash: str) -> bool:
        """Return True if an article with this *content_hash* already exists."""
        with self._lock:
            row = self._connect().execute(
                "SELECT 1 FROM articles WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
        return row is not None

    def get_article(self, article_id: str) -> Optional[Article]:
        """Fetch a single Article by its primary key."""
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM articles WHERE id = ? LIMIT 1",
                (article_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_article(dict(row))

    def get_top_articles(
        self, limit: int = 20, category: Optional[str] = None, min_score: float = 0.0
    ) -> list[Article]:
        """Return up to *limit* non-duplicate articles ordered by score DESC."""
        sql = """
        SELECT * FROM articles
        WHERE is_duplicate = 0 AND score >= ?
        {cat_filter}
        ORDER BY score DESC, fetched_at DESC
        LIMIT ?
        """.format(
            cat_filter="AND category = ?" if category else ""
        )
        params: tuple[Any, ...] = (min_score, category, limit) if category else (min_score, limit)
        with self._lock:
            rows = self._connect().execute(sql, params).fetchall()
        return [self._row_to_article(dict(r)) for r in rows]

    @staticmethod
    def _row_to_article(row: dict[str, Any]) -> Article:
        return Article(
            id=row["id"],
            title=row["title"],
            url=row["url"],
            canonical_url=row["canonical_url"],
            content_hash=row["content_hash"],
            summary=row.get("summary", ""),
            source_name=row["source_name"],
            category=row["category"],
            published_at=datetime.fromisoformat(row["published_at"]) if row.get("published_at") else None,
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            score=row["score"],
            is_duplicate=bool(row["is_duplicate"]),
        )

    # =========================================================================
    # Generated Posts
    # =========================================================================

    def insert_post(self, post: GeneratedPost) -> None:
        """Insert a generated post (raises on duplicate id)."""
        sql = """
        INSERT INTO generated_posts
            (id, article_id, text, category, source_url, source_title,
             lesson_number, status, created_at, scheduled_at, published_at,
             quiz_question, quiz_options, correct_answer)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            post.id,
            post.article_id,
            post.text,
            post.category.value,
            post.source_url,
            post.source_title,
            post.lesson_number,
            post.status.value,
            post.created_at.isoformat(),
            post.scheduled_at.isoformat() if post.scheduled_at else None,
            post.published_at.isoformat() if post.published_at else None,
            post.quiz_question,
            json.dumps(post.quiz_options, ensure_ascii=False) if post.quiz_options else None,
            post.correct_answer,
        )
        with self._transaction() as conn:
            conn.execute(sql, params)
        logger.debug("Post inserted: %s (category=%s)", post.id, post.category.value)

    def update_post_status(
        self,
        post_id: str,
        status: PostStatus,
        published_at: Optional[datetime] = None,
    ) -> None:
        """Update the status (and optionally published_at) of a post."""
        sql = """
        UPDATE generated_posts
        SET status = ?, published_at = ?
        WHERE id = ?
        """
        with self._transaction() as conn:
            conn.execute(
                sql,
                (
                    status.value,
                    published_at.isoformat() if published_at else None,
                    post_id,
                ),
            )

    def get_pending_posts(self, limit: int = 50) -> list[GeneratedPost]:
        """Return pending posts ordered by scheduled_at ASC."""
        sql = """
        SELECT * FROM generated_posts
        WHERE status IN ('pending', 'scheduled')
        ORDER BY scheduled_at ASC NULLS LAST, created_at ASC
        LIMIT ?
        """
        with self._lock:
            rows = self._connect().execute(sql, (limit,)).fetchall()
        return [self._row_to_post(dict(r)) for r in rows]

    def get_post(self, post_id: str) -> Optional[GeneratedPost]:
        """Fetch a single GeneratedPost by its primary key."""
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM generated_posts WHERE id = ? LIMIT 1",
                (post_id,),
            ).fetchone()
        return self._row_to_post(dict(row)) if row else None

    @staticmethod
    def _row_to_post(row: dict[str, Any]) -> GeneratedPost:
        quiz_options: Optional[dict[str, str]] = None
        if row.get("quiz_options"):
            quiz_options = json.loads(row["quiz_options"])
        return GeneratedPost(
            id=row["id"],
            article_id=row.get("article_id"),
            text=row["text"],
            category=row["category"],
            source_url=row.get("source_url"),
            source_title=row.get("source_title"),
            lesson_number=row.get("lesson_number"),
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            scheduled_at=datetime.fromisoformat(row["scheduled_at"]) if row.get("scheduled_at") else None,
            published_at=datetime.fromisoformat(row["published_at"]) if row.get("published_at") else None,
            quiz_question=row.get("quiz_question"),
            quiz_options=quiz_options,
            correct_answer=row.get("correct_answer"),
        )

    # =========================================================================
    # Delivery Records
    # =========================================================================

    def upsert_delivery_record(self, record: DeliveryRecord) -> None:
        """Insert or update a delivery record."""
        sql = """
        INSERT INTO delivery_records
            (id, post_id, destination, status, message_id, attempts,
             last_attempt_at, error_message, sent_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(post_id, destination) DO UPDATE SET
            status          = excluded.status,
            message_id      = excluded.message_id,
            attempts        = excluded.attempts,
            last_attempt_at = excluded.last_attempt_at,
            error_message   = excluded.error_message,
            sent_at         = excluded.sent_at
        """
        params = (
            record.id,
            record.post_id,
            record.destination,
            record.status.value,
            record.message_id,
            record.attempts,
            record.last_attempt_at.isoformat() if record.last_attempt_at else None,
            record.error_message,
            record.sent_at.isoformat() if record.sent_at else None,
        )
        with self._transaction() as conn:
            conn.execute(sql, params)

    def get_delivery_record(
        self, post_id: str, destination: str
    ) -> Optional[DeliveryRecord]:
        """Fetch delivery record for a (post_id, destination) pair."""
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM delivery_records WHERE post_id = ? AND destination = ? LIMIT 1",
                (post_id, destination),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_delivery(dict(row))

    def is_already_sent(self, post_id: str, destination: str) -> bool:
        """Return True if this post was already successfully sent to *destination*."""
        rec = self.get_delivery_record(post_id, destination)
        return rec is not None and rec.status == DeliveryStatus.SENT

    @staticmethod
    def _row_to_delivery(row: dict[str, Any]) -> DeliveryRecord:
        return DeliveryRecord(
            id=row["id"],
            post_id=row["post_id"],
            destination=row["destination"],
            status=row["status"],
            message_id=row.get("message_id"),
            attempts=row["attempts"],
            last_attempt_at=datetime.fromisoformat(row["last_attempt_at"]) if row.get("last_attempt_at") else None,
            error_message=row.get("error_message"),
            sent_at=datetime.fromisoformat(row["sent_at"]) if row.get("sent_at") else None,
        )

    # =========================================================================
    # Gemini Quota — atomic locking (CRITICAL)
    # =========================================================================

    def _today_date_str(self) -> str:
        """Return today's date as YYYY-MM-DD in Asia/Tehran time."""
        try:
            import pytz  # type: ignore[import-untyped]
            tz = pytz.timezone("Asia/Tehran")
            return datetime.now(tz=tz).strftime("%Y-%m-%d")
        except ImportError:
            # Fallback: UTC (acceptable if pytz unavailable during testing)
            return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    def _ensure_quota_row(self, conn: sqlite3.Connection, date_str: str) -> None:
        """Ensure a quota row exists for *date_str* (no-op if already present)."""
        conn.execute(
            """
            INSERT OR IGNORE INTO gemini_quota (date, requests_used, max_requests, status)
            VALUES (?, 0, ?, 'available')
            """,
            (date_str, self._max_requests_per_day),
        )

    def try_acquire_gemini_quota(self) -> bool:
        """
        Atomically attempt to acquire the Gemini quota for today.

        Algorithm (CRITICAL — race-condition-free):
          1. BEGIN IMMEDIATE (blocks all concurrent writers)
          2. Ensure today's row exists (INSERT OR IGNORE)
          3. Read the row
          4. If requests_used >= max_requests OR status != 'available': ROLLBACK → return False
          5. SET status = 'locked', lock_acquired_at = now
          6. COMMIT → return True

        This method must be called before every Gemini API request.
        Call :meth:`mark_gemini_quota_consumed` after a successful request.
        Call :meth:`release_gemini_quota_lock` on failure to reset status.
        """
        date_str = self._today_date_str()
        with self._lock:
            conn = self._connect()
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_quota_row(conn, date_str)
                row = conn.execute(
                    "SELECT requests_used, max_requests, status FROM gemini_quota WHERE date = ?",
                    (date_str,),
                ).fetchone()

                if row is None:
                    conn.execute("ROLLBACK")
                    logger.error("Quota row missing after INSERT OR IGNORE — unexpected.")
                    return False

                requests_used = row["requests_used"]
                max_requests = row["max_requests"]
                status = row["status"]

                if requests_used >= max_requests or status in ("consumed", "locked"):
                    conn.execute("ROLLBACK")
                    logger.warning(
                        "Gemini quota not available for %s: used=%d/%d status=%s",
                        date_str,
                        requests_used,
                        max_requests,
                        status,
                    )
                    return False

                # Acquire the lock
                now_iso = utcnow().isoformat()
                conn.execute(
                    """
                    UPDATE gemini_quota
                    SET status = 'locked', lock_acquired_at = ?
                    WHERE date = ?
                    """,
                    (now_iso, date_str),
                )
                conn.execute("COMMIT")
                logger.info("Gemini quota lock acquired for %s.", date_str)
                return True

            except Exception:
                conn.execute("ROLLBACK")
                raise

    def mark_gemini_quota_consumed(self) -> None:
        """
        Mark today's Gemini quota as consumed.
        Must be called immediately after a successful API request.
        """
        date_str = self._today_date_str()
        now_iso = utcnow().isoformat()
        with self._transaction() as conn:
            self._ensure_quota_row(conn, date_str)
            conn.execute(
                """
                UPDATE gemini_quota
                SET requests_used = requests_used + 1,
                    status = 'consumed',
                    last_request_at = ?
                WHERE date = ?
                """,
                (now_iso, date_str),
            )
        logger.info("Gemini quota marked as consumed for %s.", date_str)

    def release_gemini_quota_lock(self) -> None:
        """
        Release the quota lock WITHOUT consuming it.
        Call this if a Gemini API request fails (e.g. network error) so that
        the lock is not permanently stuck, and the next run can retry.

        NOTE: Per spec, there is no automatic retry on quota exhaustion.
        This is only for cases where the lock was acquired but the request
        itself failed before completion.
        """
        date_str = self._today_date_str()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE gemini_quota
                SET status = 'available', lock_acquired_at = NULL
                WHERE date = ? AND status = 'locked'
                """,
                (date_str,),
            )
        logger.warning("Gemini quota lock released (request failed) for %s.", date_str)

    def get_gemini_quota(self) -> GeminiDailyQuota:
        """Return today's :class:`GeminiDailyQuota` (creates row if absent)."""
        date_str = self._today_date_str()
        with self._lock:
            conn = self._connect()
            # Ensure row exists (read-only path, use a lightweight write)
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_quota_row(conn, date_str)
            row = conn.execute(
                "SELECT * FROM gemini_quota WHERE date = ?", (date_str,)
            ).fetchone()
            conn.execute("COMMIT")

        r = dict(row)
        return GeminiDailyQuota(
            date=r["date"],
            requests_used=r["requests_used"],
            max_requests=r["max_requests"],
            status=r["status"],
            lock_acquired_at=datetime.fromisoformat(r["lock_acquired_at"]) if r.get("lock_acquired_at") else None,
            last_request_at=datetime.fromisoformat(r["last_request_at"]) if r.get("last_request_at") else None,
        )

    # =========================================================================
    # Lesson Progress
    # =========================================================================

    def initialize_lessons(self, lessons: list[dict[str, Any]]) -> None:
        """
        Populate lesson_progress table from curriculum.json data.
        Uses INSERT OR IGNORE so re-runs are idempotent.
        """
        sql = "INSERT OR IGNORE INTO lesson_progress (lesson_number, title, status) VALUES (?, ?, 'pending')"
        with self._transaction() as conn:
            for lesson in lessons:
                conn.execute(sql, (lesson["lesson_number"], lesson["title"]))
        logger.info("Lesson progress initialized (%d lessons).", len(lessons))

    def get_next_lesson(self) -> Optional[LessonProgress]:
        """Return the next pending lesson, or None if all are done."""
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM lesson_progress WHERE status = 'pending' ORDER BY lesson_number ASC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return self._row_to_lesson(dict(row))

    def mark_lesson_published(self, lesson_number: int, post_id: str) -> None:
        """Mark a lesson as published."""
        now_iso = utcnow().isoformat()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE lesson_progress
                SET status = 'published', published_at = ?, post_id = ?
                WHERE lesson_number = ?
                """,
                (now_iso, post_id, lesson_number),
            )

    def get_all_lessons(self) -> list[LessonProgress]:
        """Return all lessons ordered by lesson_number."""
        with self._lock:
            rows = self._connect().execute(
                "SELECT * FROM lesson_progress ORDER BY lesson_number ASC"
            ).fetchall()
        return [self._row_to_lesson(dict(r)) for r in rows]

    @staticmethod
    def _row_to_lesson(row: dict[str, Any]) -> LessonProgress:
        return LessonProgress(
            lesson_number=row["lesson_number"],
            title=row["title"],
            status=row["status"],
            published_at=datetime.fromisoformat(row["published_at"]) if row.get("published_at") else None,
            post_id=row.get("post_id"),
        )

    # =========================================================================
    # Audience Questions
    # =========================================================================

    def insert_question(self, question: AudienceQuestion) -> bool:
        """Insert an audience question. Returns True if inserted, False if duplicate id."""
        sql = """
        INSERT OR IGNORE INTO audience_questions
            (id, text, source, received_at, is_answered, answered_in_post_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            question.id,
            question.text,
            question.source,
            question.received_at.isoformat(),
            int(question.is_answered),
            question.answered_in_post_id,
        )
        with self._transaction() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount > 0

    def get_unanswered_questions(self, limit: int = 10) -> list[AudienceQuestion]:
        """Return unanswered questions ordered by received_at ASC."""
        with self._lock:
            rows = self._connect().execute(
                """
                SELECT * FROM audience_questions
                WHERE is_answered = 0
                ORDER BY received_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_question(dict(r)) for r in rows]

    def mark_questions_answered(self, question_ids: list[str], post_id: str) -> None:
        """Mark a list of questions as answered."""
        if not question_ids:
            return
        placeholders = ",".join("?" * len(question_ids))
        now_iso = utcnow().isoformat()
        with self._transaction() as conn:
            conn.execute(
                f"""
                UPDATE audience_questions
                SET is_answered = 1, answered_in_post_id = ?
                WHERE id IN ({placeholders})
                """,
                [post_id, *question_ids],
            )

    @staticmethod
    def _row_to_question(row: dict[str, Any]) -> AudienceQuestion:
        return AudienceQuestion(
            id=row["id"],
            text=row["text"],
            source=row["source"],
            received_at=datetime.fromisoformat(row["received_at"]),
            is_answered=bool(row["is_answered"]),
            answered_in_post_id=row.get("answered_in_post_id"),
        )

    # =========================================================================
    # Stats / Status
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Return a dictionary of key database statistics."""
        with self._lock:
            conn = self._connect()
            total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            total_posts = conn.execute("SELECT COUNT(*) FROM generated_posts").fetchone()[0]
            published_posts = conn.execute(
                "SELECT COUNT(*) FROM generated_posts WHERE status = 'published'"
            ).fetchone()[0]
            pending_posts = conn.execute(
                "SELECT COUNT(*) FROM generated_posts WHERE status IN ('pending','scheduled')"
            ).fetchone()[0]
            lessons_done = conn.execute(
                "SELECT COUNT(*) FROM lesson_progress WHERE status = 'published'"
            ).fetchone()[0]
            unanswered_q = conn.execute(
                "SELECT COUNT(*) FROM audience_questions WHERE is_answered = 0"
            ).fetchone()[0]

        quota = self.get_gemini_quota()
        return {
            "total_articles": total_articles,
            "total_posts": total_posts,
            "published_posts": published_posts,
            "pending_posts": pending_posts,
            "lessons_published": lessons_done,
            "unanswered_questions": unanswered_q,
            "gemini_quota": {
                "date": quota.date,
                "requests_used": quota.requests_used,
                "max_requests": quota.max_requests,
                "status": quota.status.value,
            },
        }
