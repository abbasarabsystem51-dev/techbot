"""
src/techbot/main.py — Master Orchestration Pipeline for TechNews Bot (@tech_news_110).

Usage:
  python -m techbot.main --run-index 1 --dry-run
  python main.py --run-index 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import yaml
from zoneinfo import ZoneInfo

# Ensure both project root and src/ are in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CURRENT_DIR.parent
_ROOT_DIR = _SRC_DIR.parent
for _p in (_ROOT_DIR, _SRC_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT_DIR / ".env")
except ImportError:
    pass

from src.models import (
    GeminiBatchOutput,
    Item,
    PostBasket,
    PostCategory,
    PostItemSchema,
)
from src.techbot.agent.admin_jobs import drain_updates, send_to_admin, load_user_inbox
from src.techbot.agent.lessons import get_lesson
from src.techbot.facts_check import verify_post_item
from src.techbot.healthcheck import load_configured_sources
from src.techbot.http import create_http_client
from src.techbot.inbox import InboxManager
from src.techbot.llm import LLM_TIMEOUT_SECONDS, GeminiClient
from src.techbot.render import PostRenderer
from src.techbot.review import QualityReviewer
from src.techbot.scoring import normalize_url
from src.techbot.senders.eitaa import EitaaSender
from src.techbot.senders.telegram import TelegramSender
from src.techbot.state.atomic import StateManager
from src.techbot.weekly import WeeklyContentBuilder
from src.utils import strip_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("techbot.main")

MAX_CANDIDATES = 8


def get_tehran_date() -> str:
    """Get current date formatted as YYYY-MM-DD in Asia/Tehran timezone."""
    try:
        tz = ZoneInfo("Asia/Tehran")
    except Exception:
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_print(text: str) -> None:
    """Print text safely across consoles with explicit flush."""
    try:
        print(text, flush=True)
    except Exception:
        try:
            sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()
        except Exception:
            pass


def normalize_title_for_dedup(title: str) -> str:
    """Produce clean normalized string for title deduplication."""
    if not title:
        return ""
    clean = re.sub(r"[^\w\s]", "", title.lower())
    return " ".join(clean.split())


def reduce_candidates(
    items: list[Item],
    sent_links: Optional[set[str]] = None,
    recent_titles: Optional[list[str]] = None,
    max_candidates: int = MAX_CANDIDATES,
    now: Optional[datetime] = None,
) -> list[Item]:
    """
    Candidate reduction pipeline:
    a. Deduplicate by normalized URL and normalized title.
    b. Drop items older than 48h (keep undated items only if fewer than 10 would remain otherwise).
    c. Drop items already posted, if a state store exists.
    d. Score: keyword relevance + recency boost - source-diversity penalty (max 3 items per source).
    e. Sort by score descending, slice [:MAX_CANDIDATES], with MAX_CANDIDATES = 8.
    """
    if not items:
        return []

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    # a. Deduplicate by normalized URL and normalized title
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[Item] = []

    for item in items:
        raw_url = item.url or item.canonical_url or ""
        norm_url = normalize_url(raw_url)
        norm_title = normalize_title_for_dedup(item.title)

        if norm_url and norm_url in seen_urls:
            continue
        if norm_title and norm_title in seen_titles:
            continue

        if norm_url:
            seen_urls.add(norm_url)
        if norm_title:
            seen_titles.add(norm_title)
        deduped.append(item)

    # b. Drop items older than 48h (keep undated items only if fewer than 10 would remain otherwise)
    cutoff = current_time - timedelta(hours=48)
    dated_recent: list[Item] = []
    undated: list[Item] = []

    for item in deduped:
        if item.published_at:
            pub = item.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub >= cutoff:
                dated_recent.append(item)
            # Items older than 48h are dropped
        else:
            undated.append(item)

    if len(dated_recent) >= max_candidates:
        age_filtered = dated_recent
    else:
        age_filtered = dated_recent + undated

    # c. Drop items already posted
    sent = sent_links or set()
    recent_set = {normalize_title_for_dedup(t) for t in (recent_titles or []) if t}
    unposted: list[Item] = []

    for item in age_filtered:
        raw_u = item.url or ""
        norm_u = normalize_url(raw_u)
        canon_u = normalize_url(item.canonical_url or "")

        if (norm_u and norm_u in sent) or (canon_u and canon_u in sent) or (raw_u and raw_u in sent):
            continue

        norm_t = normalize_title_for_dedup(item.title)
        if norm_t and norm_t in recent_set:
            continue

        unposted.append(item)

    # d. Score: keyword relevance + recency boost - source diversity (max 3 items per source)
    keywords = [
        "ai", "llm", "openai", "google", "apple", "nvidia", "microsoft",
        "chips", "security", "startup", "robotics", "open-source", "open source",
        "deepmind", "anthropic", "meta", "gpt", "agent", "agents", "machine learning",
    ]

    scored_items: list[tuple[float, Item]] = []
    for item in unposted:
        text = f"{item.title} {item.summary}".lower()
        kw_score = 0.0
        for kw in keywords:
            if kw in text:
                kw_score += 1.5

        recency_boost = 0.0
        if item.published_at:
            pub = item.published_at if item.published_at.tzinfo else item.published_at.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (current_time - pub).total_seconds() / 3600.0)
            if age_hours < 48:
                recency_boost = 3.0 * (1.0 - (age_hours / 48.0))

        total_score = kw_score + recency_boost
        scored_items.append((total_score, item))

    # Sort descending
    scored_items.sort(key=lambda x: x[0], reverse=True)

    # Source diversity penalty: max 3 items per source
    source_counts: dict[str, int] = {}
    selected: list[Item] = []
    for score, item in scored_items:
        src = (item.source_name or "unknown").strip().lower()
        if source_counts.get(src, 0) >= 3:
            continue
        source_counts[src] = source_counts.get(src, 0) + 1
        selected.append(item)
        if len(selected) >= max_candidates:
            break

    # e. Slice top MAX_CANDIDATES
    return selected[:max_candidates]


def format_candidate_payload(item: Item) -> dict[str, str]:
    """
    Format item for LLM candidate payload:
    {id, title, source, link, summary};
    Title + summary capped at 200 characters max, HTML stripped.
    """
    clean_title = strip_html(item.title or "")
    clean_title = " ".join(clean_title.split())
    if len(clean_title) > 90:
        clean_title = clean_title[:87] + "..."

    clean_summary = strip_html(item.summary or "")
    clean_summary = " ".join(clean_summary.split())

    max_summary_len = max(0, 200 - len(clean_title))
    if len(clean_summary) > max_summary_len:
        if max_summary_len > 3:
            clean_summary = clean_summary[:max_summary_len - 3] + "..."
        else:
            clean_summary = clean_summary[:max_summary_len]

    return {
        "id": item.id,
        "title": clean_title,
        "source": item.source_name,
        "link": item.url,
        "summary": clean_summary,
    }


class TechBotOrchestrator:
    """
    Master pipeline controller.
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        state_file: Optional[Path | str] = None,
    ) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[2]
        self.config = self._load_config()
        self.curriculum = self._load_curriculum()

        # State Manager
        if state_file:
            state_path = Path(state_file)
        else:
            state_path = self.root_dir / self.config.get("storage", {}).get("state_file_path", "data/state.json")
        self.state_manager = StateManager(state_path)

        # Core Engines
        self.renderer = PostRenderer()
        self.reviewer = QualityReviewer(passing_threshold=70)
        self.inbox_manager = InboxManager(self.state_manager)
        self.weekly_builder = WeeklyContentBuilder(self.inbox_manager)

        models_cfg = self.config.get("llm", {}).get("models", None)
        self.llm = GeminiClient(models=models_cfg) if models_cfg else GeminiClient()

    def _load_config(self) -> dict[str, Any]:
        cfg_path = self.root_dir / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_curriculum(self) -> dict[str, Any]:
        cur_path = self.root_dir / "curriculum.json"
        if cur_path.exists():
            with open(cur_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        return {}

    async def run(
        self,
        run_index: int = 1,
        dry_run: bool = True,
        target_date: Optional[str] = None,
    ) -> bool:
        """
        Execute a scheduled daily run (1 through 5) following the 5 Persian stage logs.
        """
        date_str = target_date or get_tehran_date()
        logger.info("Starting Run %d for date %s (dry_run=%s)", run_index, date_str, dry_run)

        # Ensure weekly-analysis state keys exist
        with self.state_manager.transaction() as _st:
            if "awaiting_weekly_analysis_prompt" not in _st:
                _st["awaiting_weekly_analysis_prompt"] = False
            if "pending_weekly_messages" not in _st:
                _st["pending_weekly_messages"] = []

        if run_index > 0:
            send_to_admin(f"🚀 ران {run_index} شروع شد.")

        if run_index == 0:
            logger.info("Running in INBOX-ONLY mode (run_index=0).")
            
            drain_updates(self.state_manager)
            pending_jobs = self.state_manager.get_state().get("admin_jobs", [])
            if not pending_jobs:
                logger.info("No pending admin jobs.")
                return True
                
            senders = [TelegramSender(), EitaaSender()]
            
            try:
                for job in pending_jobs:
                    if job["type"] == "raw":
                        text = job["content"]
                        for s in senders:
                            await s.send_post(text=text)
                        send_to_admin("✅ پست متنی شما با موفقیت در کانال منتشر شد.")
                    elif job["type"] == "new":
                        topic = job["content"]
                        prompt = f"Write a short, engaging post in Persian about this topic: {topic}. Only output the post text."
                        try:
                            batch = self.llm.generate_batch(prompt)
                            if batch.posts:
                                text = batch.posts[0].hook + "\n\n" + batch.posts[0].intro
                                for s in senders:
                                    await s.send_post(text=text)
                                send_to_admin("✅ پست سفارشی با هوش مصنوعی ساخته و منتشر شد.")
                            else:
                                send_to_admin("❌ هوش مصنوعی پاسخی نداد.")
                        except Exception as e:
                            send_to_admin(f"❌ خطا در تولید پست: {e}")
                    elif job["type"] == "analyze":
                        admin_prompt = job["content"]
                        try:
                            user_messages = load_user_inbox()
                            if not user_messages:
                                send_to_admin("📭 صندوق پیام‌ها خالی است.")
                            else:
                                messages_text = "\n".join(
                                    f"[{m.get('date', '')}] {m.get('username', 'ناشناس')} "
                                    f"(id:{m.get('user_id', '')}): {m.get('text', '')}"
                                    for m in user_messages
                                )
                                analysis_prompt = (
                                    f"اینجا لیست پیام‌های کاربران است:\n{messages_text}\n\n"
                                    f"دستور ادمین: {admin_prompt}\n\n"
                                    "لطفاً بر اساس دستور ادمین، یک تحلیل جامع و مفید به زبان فارسی ارائه بده."
                                )
                                try:
                                    import httpx as _httpx
                                    import os as _os
                                    _api_key = (
                                        _os.environ.get("GEMINI_KEY")
                                        or _os.environ.get("GEMINI_API_KEY")
                                        or ""
                                    )
                                    _model = _os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
                                    _url = (
                                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                                        f"{_model}:generateContent?key={_api_key}"
                                    )
                                    _payload = {
                                        "contents": [{"parts": [{"text": analysis_prompt}]}],
                                        "generationConfig": {"temperature": 0.4},
                                    }
                                    with _httpx.Client(timeout=60.0) as _client:
                                        _resp = _client.post(_url, json=_payload)
                                    _data = _resp.json()
                                    _candidates = _data.get("candidates", [])
                                    if _candidates:
                                        _parts = _candidates[0].get("content", {}).get("parts", [])
                                        analysis_result = _parts[0].get("text", "") if _parts else ""
                                    else:
                                        analysis_result = ""
                                    if analysis_result:
                                        send_to_admin(f"📊 نتیجه تحلیل:\n\n{analysis_result}")
                                    else:
                                        send_to_admin("❌ هوش مصنوعی پاسخی برای تحلیل نداد.")
                                except Exception as llm_exc:
                                    send_to_admin(f"❌ خطا در تحلیل پیام‌ها با هوش مصنوعی: {llm_exc}")
                        except Exception as analyze_exc:
                            send_to_admin(f"❌ خطا در پردازش دستور /analyze: {analyze_exc}")
                            logger.error("Analyze job failed: %s", analyze_exc)
            except Exception as admin_exc:
                logger.error("Admin jobs processing failed: %s", admin_exc)
                with self.state_manager.transaction() as st:
                    st["admin_jobs"] = []
            
            # Clear processed jobs
            with self.state_manager.transaction() as st:
                st["admin_jobs"] = []
            
            # FORCE SAVE STATE TO PREVENT DUPLICATE POSTS
            self.state_manager.save_state(self.state_manager.get_state())
            return True

        # Locate Run Configuration
        runs = self.config.get("run_plan", {}).get("runs", [])
        selected_run = next((r for r in runs if r.get("run_index") == run_index), None)
        if not selected_run:
            logger.error("Run index %d not found in config.yaml run_plan.", run_index)
            return False

        slots = selected_run.get("slots", [])

        # Check Idempotency for slots
        slots_to_process = []
        for slot in slots:
            slot_id = slot.get("slot_id")
            basket = slot.get("basket")
            if self.state_manager.is_run_completed(date_str, run_index, basket):
                logger.info("Slot %s (%s) already completed for %s. Skipping.", slot_id, basket, date_str)
            else:
                slots_to_process.append(slot)

        if not slots_to_process:
            logger.info("All slots for Run %d on %s are already completed. Exiting.", run_index, date_str)
            return True

        raw_state = self.state_manager.get_state()
        sent_links = self.state_manager.get_sent_links()
        recent_titles = self.state_manager.get_recent_titles()

        # =======================================================================
        # [۱/۵] دریافت خبرها از فیدها...
        # =======================================================================
        t_stage1_start = time.monotonic()
        all_candidates: list[Item] = []
        try:
            sources = load_configured_sources(self.root_dir / "config.yaml")
            fetch_cfg = self.config.get("fetching", {})
            timeout_sec = float(fetch_cfg.get("timeout_seconds", 10.0))
            http_client = create_http_client(timeout=timeout_sec, max_retries=1)

            fetch_tasks = [s.safe_fetch(client=http_client, state=raw_state) for s in sources]
            fetch_results = await asyncio.gather(*fetch_tasks)
            for batch in fetch_results:
                all_candidates.extend(batch)
            await http_client.aclose()
        except Exception as exc:
            safe_print(f"❌ [۱/۵] خطا در دریافت خبرها از فیدها: {exc}")
            logger.error("Stage 1 fetch error: %s", exc)

        t_stage1_elapsed = time.monotonic() - t_stage1_start
        safe_print(f"[۱/۵] دریافت خبرها از فیدها... (⏱ {t_stage1_elapsed:.1f}s)")

        # =======================================================================
        # [۲/۵] فیلتر و انتخاب ۱۰ خبر برتر...
        # =======================================================================
        t_stage2_start = time.monotonic()
        top_candidates = reduce_candidates(
            all_candidates,
            sent_links=sent_links,
            recent_titles=recent_titles,
            max_candidates=MAX_CANDIDATES,
        )
        t_stage2_elapsed = time.monotonic() - t_stage2_start
        safe_print(f"[۲/۵] فیلتر و انتخاب ۱۰ خبر برتر... (⏱ {t_stage2_elapsed:.1f}s)")

        # =======================================================================
        # [۳/۵] تحلیل و تولید متن با هوش مصنوعی...
        # =======================================================================
        t_stage3_start = time.monotonic()
        llm_posts: dict[str, PostItemSchema] = {}
        slot_contexts: dict[str, dict[str, Any]] = {}

        candidate_payloads = [format_candidate_payload(c) for c in top_candidates]

        for slot in slots_to_process:
            slot_id = slot.get("slot_id", f"{run_index}_1")
            basket_str = slot.get("basket", "news_ai")
            basket = PostBasket(basket_str) if basket_str in PostBasket._value2member_map_ else PostBasket.NEWS_AI

            if basket == PostBasket.LESSON:
                current_lesson_id = raw_state.get("current_lesson_id", 1)

                # Try custom lesson first (data/custom_lessons.json), fallback to curriculum.json
                lesson_data = get_lesson(current_lesson_id)
                if lesson_data is not None:
                    logger.info(
                        "Using custom lesson id=%d from custom_lessons.json", current_lesson_id
                    )
                else:
                    lessons_list = self.curriculum.get("lessons", [])
                    lesson_data = next(
                        (l for l in lessons_list if l.get("lesson_id") == current_lesson_id), None
                    )
                    if not lesson_data and lessons_list:
                        lesson_data = lessons_list[0]

                lesson_title = lesson_data.get("title", f"مفهوم {current_lesson_id}") if lesson_data else "یادگیری عمیق"
                slot_contexts[slot_id] = {
                    "basket": basket,
                    "lesson_id": current_lesson_id,
                    "lesson_title": lesson_title,
                    "lesson_data": lesson_data,
                }
            elif basket == PostBasket.WEEKLY_QA:
                slot_contexts[slot_id] = {
                    "basket": basket,
                }
            else:
                candidate = top_candidates[0] if top_candidates else None
                slot_contexts[slot_id] = {
                    "basket": basket,
                    "candidate": candidate,
                    "source_url": candidate.url if candidate else None,
                }

        # Call Gemini if API key is configured and not in test
        if self.llm.api_key and candidate_payloads and not os.environ.get("PYTEST_CURRENT_TEST"):
            # Load prompts from files
            from pathlib import Path as _Path
            _prompts_dir = _Path(__file__).resolve().parents[2] / "prompts"

            def _read_prompt(name: str) -> str:
                p = _prompts_dir / name
                return p.read_text(encoding="utf-8").strip() if p.exists() else ""

            _system = _read_prompt("system.md")
            _news = _read_prompt("news.md")
            _tool = _read_prompt("tool.md")
            _lesson = _read_prompt("lesson.md")

            # Group slots by basket
            slots_desc = []
            for s in slots_to_process:
                sid = s.get("slot_id")
                basket = s.get("basket", "news_ai")
                slots_desc.append(f"- slot={sid} basket={basket}")

            prompt_parts = [
                _system,
                "",
                "## قوانین پست خبری (news_ai / news_tech):",
                _news,
                "",
                "## قوانین پست ابزار (tool_ai / tool_tech):",
                _tool,
                "",
                "## قوانین پست درس (lesson):",
                _lesson,
                "",
                "## اسلاتهای این ران:",
                "\n".join(slots_desc),
                "",
                "## کاندیداها:",
                json.dumps(candidate_payloads, ensure_ascii=False, indent=2),
                "",
                "## دستور نهایی:",
                "برای هر اسلات بالا دقیقاً یک پست بساز. فیلد `slot` باید عیناً برابر slot_id باشد. "
                "برای سبدهای news_* از قوانین خبر، برای tool_* از قوانین ابزار، برای lesson از قوانین درس پیروی کن. "
                "اگر برای اسلاتی منبع کافی نیست، فیلد `skip_reason` را پر کن و بقیه فیلدها را خالی بگذار.",
                "",
                "خروجی: یک JSON معتبر با کلید `posts`.",
            ]

            prompt = "\n".join(prompt_parts)
            try:
                batch_res = self.llm.generate_batch(prompt)
                for p in batch_res.posts:
                    llm_posts[p.slot] = p
            except Exception as exc:
                safe_print(f"❌ [۳/۵] خطا در LLM: {exc}")
                logger.error("Stage 3 Gemini batch call failed: %s", exc)
                send_to_admin("⚠️ سهمیه هوش مصنوعی تمام شد یا خطا داد. پست ساخته نشد.")

        # Resolve posts for slots (using candidate #1 fallback if needed)
        posts_ready: list[tuple[dict[str, Any], PostItemSchema, Optional[str]]] = []

        for slot in slots_to_process:
            slot_id = slot.get("slot_id", f"{run_index}_1")
            ctx = slot_contexts.get(slot_id)
            if not ctx:
                continue

            basket = ctx["basket"]
            post_item: Optional[PostItemSchema] = llm_posts.get(slot_id)

            if basket == PostBasket.LESSON:
                if post_item is None:
                    lesson_data = ctx.get("lesson_data", {})
                    lesson_title = ctx.get("lesson_title", "آموزش هوش مصنوعی")
                    concept = lesson_data.get("concept", {})
                    example = lesson_data.get("example", {})
                    practice = lesson_data.get("practice", {})
                    quiz_data = practice.get("quiz", {})

                    allowed_facts = concept.get("allowed_facts", [])
                    dyn_steps = allowed_facts[:3] if len(allowed_facts) >= 3 else [
                        concept.get("summary", ""),
                        example.get("scenario", ""),
                        example.get("takeaway", ""),
                    ]
                    post_item = PostItemSchema(
                        slot=slot_id,
                        basket=PostBasket.LESSON,
                        hook=f"💡 {lesson_title}",
                        intro=concept.get("everyday_example") or concept.get("summary", "مفهوم کاربردی هوش مصنوعی"),
                        steps=[s for s in dyn_steps if s],
                        tip=example.get("takeaway") or (allowed_facts[0] if allowed_facts else ""),
                        try_now=practice.get("try_now", "").strip(),
                        prompt_to_copy=practice.get("prompt_to_copy", "").strip(),
                        quiz={
                            "q": quiz_data.get("question", "صورت سوال"),
                            "options": [str(opt)[:80] for opt in quiz_data.get("options", ["گزینه ۱", "گزینه ۲", "گزینه ۳"])],
                            "correct": quiz_data.get("correct_index", 0),
                            "explanation": quiz_data.get("explanation", "توضیح کوتاه"),
                        } if quiz_data else None,
                    )

            elif basket == PostBasket.WEEKLY_QA:
                if post_item is None:
                    week_no = raw_state.get("completed_weeks", 1)
                    post_item = self.weekly_builder.build_weekly_post(week_number=week_no)

            else:
                # News / Tool: LLM did not return a valid post
                if post_item is None:
                    logger.error("Slot %s (%s): LLM did not return a valid post. Skipping — no fake content.", slot_id, basket.value)
                    send_to_admin(f"⚠️ ران: LLM برای {slot_id} پست نساخت. skip شد.")
                    post_item = None

            if post_item is None:
                continue

            posts_ready.append((slot, post_item, ctx.get("source_url")))

        t_stage3_elapsed = time.monotonic() - t_stage3_start
        safe_print(f"[۳/۵] تحلیل و تولید متن با هوش مصنوعی... (⏱ {t_stage3_elapsed:.1f}s)")

        # =======================================================================
        # [۴/۵] آماده‌سازی پست‌ها برای انتشار (بدون تصویر)...
        # =======================================================================
        t_stage4_start = time.monotonic()
        posts_with_images: list[tuple[dict[str, Any], PostItemSchema, Optional[str]]] = [
            (slot, post_item, source_url) for slot, post_item, source_url in posts_ready
        ]
        t_stage4_elapsed = time.monotonic() - t_stage4_start
        safe_print(f"[۴/۵] آماده‌سازی پست‌ها برای انتشار... (⏱ {t_stage4_elapsed:.1f}s)")

        # =======================================================================
        # [۵/۵] انتشار پست در کانال...
        # =======================================================================
        t_stage5_start = time.monotonic()
        senders = [
            TelegramSender(enabled=not dry_run),
            EitaaSender(enabled=not dry_run),
        ]

        used_ids: set[str] = set()

        def pick_for_slot(basket: PostBasket) -> Optional[Item]:
            for c in top_candidates:
                if c.id in used_ids:
                    continue
                if basket == PostBasket.NEWS_AI and c.category != PostCategory.AI:
                    continue
                if basket == PostBasket.NEWS_TECH and c.category != PostCategory.TECH:
                    continue
                if basket in (PostBasket.TOOL_AI, PostBasket.TOOL_TECH) \
                   and not any(k in (c.source_name or "").lower()
                               for k in ("product", "github", "hugging")):
                    continue
                used_ids.add(c.id)
                return c
            return None

        for slot, post_item, source_url in posts_with_images:
            slot_id = slot.get("slot_id")
            basket = post_item.basket

            ctx = slot_contexts.get(slot_id, {})
            if basket not in (PostBasket.LESSON, PostBasket.WEEKLY_QA):
                if ctx.get("candidate") is None:
                    cand = pick_for_slot(basket)
                    if cand:
                        ctx["candidate"] = cand
                        slot_contexts[slot_id] = ctx
                        source_url = cand.url

            lesson_num = raw_state.get("current_lesson_id", 1)
            rendered_tg = self.renderer.render_telegram_html(post_item, lesson_number=lesson_num)
            rendered_eitaa = self.renderer.render_eitaa_plain_text(post_item, lesson_number=lesson_num)

            if basket != PostBasket.LESSON:
                candidate = ctx.get("candidate")
                if candidate is None:
                    logger.warning("Slot %s: no source candidate -> skipping.", slot_id)
                    continue
                src_text = f"{candidate.summary} {(candidate.full_text or '')}"
                if not src_text.strip():
                    logger.warning("Slot %s: empty source text -> skipping.", slot_id)
                    continue
                ok, bad = verify_post_item(post_item, source_text=src_text)
                if not ok:
                    logger.warning("Slot %s fact-check fail: %s -> skip.", slot_id, bad[:3])
                    continue
            else:
                lesson_data = ctx.get("lesson_data") or {}
                facts = (lesson_data.get("concept") or {}).get("allowed_facts", [])
                if facts:
                    ok, bad = verify_post_item(post_item, source_text=" ".join(facts))
                    if not ok:
                        logger.warning("Lesson %s fact-check fail: %s -> skip.", slot_id, bad[:3])
                        continue

            review = self.reviewer.review(post_item, rendered_text=rendered_tg)
            if not review.passed:
                logger.warning("Slot %s review rejected (score=%d): %s",
                               slot_id, review.score, review.feedback[:2])
                continue

            if dry_run:
                safe_print(f"\n================ [DRY-RUN] Slot {slot_id} ({basket}) ================")
                safe_print(rendered_tg)
                safe_print("====================================================================\n")
                continue

            tg_ok = False
            for sender in senders:
                caption = rendered_tg if sender.destination_name == "telegram" else rendered_eitaa
                try:
                    res = await sender.send_post(
                        text=caption,
                        source_url=source_url,
                        quiz=post_item.quiz.model_dump() if post_item.quiz else None,
                    )
                    if sender.destination_name == "telegram" and res.success:
                        tg_ok = True
                    if not res.success:
                        safe_print(f"❌ [۵/۵] خطا در انتشار به {sender.destination_name}: {res.error}")
                except Exception as pub_exc:
                    safe_print(f"❌ [۵/۵] خطا در انتشار به {sender.destination_name}: {pub_exc}")

            if not tg_ok:
                logger.warning("Slot %s: Telegram failed → state NOT recorded.", slot_id)
                continue

            self.state_manager.record_completed_run(date_str, run_index, str(basket.value))
            if source_url:
                self.state_manager.record_sent_link(source_url)
            self.state_manager.record_recent_title(post_item.hook)

            if basket == PostBasket.LESSON:
                with self.state_manager.transaction() as state:
                    state["current_lesson_id"] = (lesson_num % 30) + 1

        t_stage5_elapsed = time.monotonic() - t_stage5_start
        safe_print(f"[۵/۵] انتشار پست در کانال... (⏱ {t_stage5_elapsed:.1f}s)")

        self.state_manager.save_state(self.state_manager.get_state())
        return True


def cli_entry() -> None:
    """CLI Entry point for python -m techbot.main."""
    parser = argparse.ArgumentParser(description="TechNews Bot Orchestrator")
    parser.add_argument("--run-index", type=int, default=1, choices=range(0, 6), help="Daily run index (0..5)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate without publishing")
    parser.add_argument("--date", type=str, default=None, help="Target execution date (YYYY-MM-DD)")
    parser.add_argument("--serve", action="store_true", help="Run continuously 24/7")

    args = parser.parse_args()
    orchestrator = TechBotOrchestrator()

    if args.serve:
        import datetime
        from zoneinfo import ZoneInfo
        logger.info("Starting TechNews Bot in 24/7 SERVE mode...")

        async def serve_loop():
            while True:
                try:
                    now = datetime.datetime.now(ZoneInfo("Asia/Tehran"))
                    hour = now.hour
                    minute = now.minute

                    # Determine run_index based on time
                    run_index = -1
                    if minute == 0:
                        if hour == 9: run_index = 1
                        elif hour == 12: run_index = 2
                        elif hour == 15: run_index = 3
                        elif hour == 18: run_index = 4
                        elif hour == 21: run_index = 5

                    # Check inbox every 15 minutes
                    if minute in (0, 15, 30, 45):
                        logger.info(f"[SERVE] Waking up for Inbox check at {hour}:{minute:02d}")
                        await orchestrator.run(run_index=0, dry_run=False)

                    # -------------------------------------------------------
                    # Thursday 21:00 — weekly inbox report to admin
                    # -------------------------------------------------------
                    if now.weekday() == 3 and hour == 21 and minute == 0:
                        _today_str = now.strftime("%Y-%m-%d")
                        _st_snap = orchestrator.state_manager.get_state()
                        if _st_snap.get("last_weekly_report_date") != _today_str:
                            from src.techbot.agent.admin_jobs import (
                                load_user_inbox as _load_inbox,
                                send_to_admin as _send_admin,
                            )
                            _inbox_messages = _load_inbox()
                            if _inbox_messages:
                                _messages_text = "\n".join(
                                    f"[{m.get('date', '')}] "
                                    f"{m.get('username', 'ناشناس')} "
                                    f"(id:{m.get('user_id', '')}): "
                                    f"{m.get('text', '')}"
                                    for m in _inbox_messages
                                )
                                _send_admin(
                                    f"🗂️ این هم لیست پیامهای این هفته کاربران:\n\n{_messages_text}"
                                )
                                _send_admin(
                                    "🤖 پیامها دریافت شد. حالا بگو بر اساس چه چیزی "
                                    "تحلیلشون کنم؟ (مثلاً: بر اساس میزان رضایت، یا موضوعات پرتکرار...)"
                                )
                                with orchestrator.state_manager.transaction() as _st2:
                                    _st2["awaiting_weekly_analysis_prompt"] = True
                                    _st2["pending_weekly_messages"] = _inbox_messages
                                    _st2["last_weekly_report_date"] = _today_str
                                logger.info("[SERVE] Thursday weekly inbox report sent to admin.")

                    # Generate content on the hour
                    if run_index > 0:
                        logger.info(f"[SERVE] Waking up for Content Run {run_index} at {hour}:{minute:02d}")
                        await orchestrator.run(run_index=run_index, dry_run=False)

                except Exception as e:
                    logger.error(f"[SERVE] Error during run: {e}", exc_info=True)

                # Sleep for 60 seconds before checking the time again
                await asyncio.sleep(10)

        try:
            asyncio.run(serve_loop())
        except KeyboardInterrupt:
            logger.info("Serve mode stopped by user.")
        sys.exit(0)

    success = asyncio.run(orchestrator.run(run_index=args.run_index, dry_run=args.dry_run, target_date=args.date))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    cli_entry()
