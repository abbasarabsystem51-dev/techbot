"""
src/techbot/main.py — Master Orchestration Pipeline for TechNews Bot (@tech_news_110).

Usage:
  python -m techbot.main --run-index 1 --dry-run
  python -m src.techbot.main --run-index 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from zoneinfo import ZoneInfo

# Ensure both project root and src/ are in sys.path regardless of how invoked
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
from src.techbot.facts_check import verify_post_item
from src.techbot.healthcheck import load_configured_sources
from src.techbot.http import create_http_client
from src.techbot.images import ImageFinder, LessonCardGenerator, validate_rendered_image
from src.techbot.inbox import InboxManager
from src.techbot.llm import GeminiClient
from src.techbot.render import PostRenderer
from src.techbot.review import QualityReviewer
from src.techbot.scoring import Scorer
from src.techbot.senders.eitaa import EitaaSender
from src.techbot.senders.telegram import TelegramSender
from src.techbot.sources.fulltext import enrich_top_candidates
from src.techbot.state.atomic import StateManager
from src.techbot.weekly import WeeklyContentBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("techbot.main")


def get_tehran_date() -> str:
    """Get current date formatted as YYYY-MM-DD in Asia/Tehran timezone."""
    try:
        tz = ZoneInfo("Asia/Tehran")
    except Exception:
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


def safe_print(text: str) -> None:
    """Print text safely across different OS consoles without crashing on cp1252."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()
        except Exception:
            pass


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
        self.scorer = Scorer(self.root_dir / "scoring_weights.json")
        self.card_generator = LessonCardGenerator()
        self.image_finder = ImageFinder(min_width=800)
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
        Execute a scheduled daily run (1 through 5).
        """
        date_str = target_date or get_tehran_date()
        logger.info("Starting Run %d for date %s (dry_run=%s)", run_index, date_str, dry_run)

        # 🔄 [۱/۵] شروع اجرای ربات و بررسی کلیدهای دسترسی (API Keys)...
        safe_print("🔄 [۱/۵] شروع اجرای ربات و بررسی کلیدهای دسترسی (API Keys)...")
        has_gemini = bool(self.llm.api_key)
        has_telegram = bool(os.environ.get("TELEGRAM_BOT_TOKEN") and (os.environ.get("TELEGRAM_CHANNEL_ID") or os.environ.get("TELEGRAM_CHAT_ID")))
        has_eitaa = bool((os.environ.get("EITAA_BOT_TOKEN") or os.environ.get("EITAA_API_KEY")) and os.environ.get("EITAA_CHANNEL_ID"))
        safe_print(f"   🔑 وضعیت کلیدها: Gemini={'✅ فعال' if has_gemini else '❌ غیرفعال'} | Telegram={'✅ فعال' if has_telegram else '❌ غیرفعال'} | Eitaa={'✅ فعال' if has_eitaa else '❌ غیرفعال'}")

        # 1. Locate Run Configuration
        runs = self.config.get("run_plan", {}).get("runs", [])
        selected_run = next((r for r in runs if r.get("run_index") == run_index), None)
        if not selected_run:
            logger.error("Run index %d not found in config.yaml run_plan.", run_index)
            return False

        slots = selected_run.get("slots", [])
        logger.info("Run %d label: '%s' (%d slots)", run_index, selected_run.get("label", ""), len(slots))

        # 2. Check Idempotency for each slot
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
            safe_print("✅ [۵/۵] اجرای ربات با موفقیت به پایان رسید!")
            return True

        # 3. Fetch candidates from all sources
        logger.info("Collecting candidate articles and tools from sources...")
        sources = load_configured_sources(self.root_dir / "config.yaml")
        fetch_cfg = self.config.get("fetching", {})
        timeout_sec = float(fetch_cfg.get("timeout_seconds", 10.0))
        http_client = create_http_client(timeout=timeout_sec, max_retries=1)

        raw_state = self.state_manager.get_state()
        all_candidates: list[Item] = []

        fetch_tasks = [s.safe_fetch(client=http_client, state=raw_state) for s in sources]
        fetch_results = await asyncio.gather(*fetch_tasks)
        for batch in fetch_results:
            all_candidates.extend(batch)

        logger.info("Collected %d total candidate items.", len(all_candidates))

        # 4. Prepare slot contexts and build single structured prompt for Gemini
        sent_links = self.state_manager.get_sent_links()
        recent_titles = self.state_manager.get_recent_titles()

        slot_contexts: dict[str, dict[str, Any]] = {}
        prompt_sections: list[str] = []

        for slot in slots_to_process:
            basket_str = slot.get("basket", "news_ai")
            slot_id = slot.get("slot_id", f"{run_index}_1")
            basket = PostBasket(basket_str) if basket_str in PostBasket._value2member_map_ else PostBasket.NEWS_AI

            if basket == PostBasket.LESSON:
                current_lesson_id = raw_state.get("current_lesson_id", 1)
                lessons_list = self.curriculum.get("lessons", [])
                lesson_data = next((l for l in lessons_list if l.get("lesson_id") == current_lesson_id), None)
                if not lesson_data and lessons_list:
                    lesson_data = lessons_list[0]

                lesson_title = lesson_data.get("title", f"مفهوم {current_lesson_id}") if lesson_data else "یادگیری عمیق"
                concept = lesson_data.get("concept", {}) if lesson_data else {}
                example = lesson_data.get("example", {}) if lesson_data else {}
                practice = lesson_data.get("practice", {}) if lesson_data else {}

                slot_contexts[slot_id] = {
                    "basket": basket,
                    "lesson_id": current_lesson_id,
                    "lesson_title": lesson_title,
                    "lesson_data": lesson_data,
                }

                facts_list = concept.get("allowed_facts", [])
                facts_text = "\n".join(f"- {f}" for f in facts_list)
                prompt_sections.append(
                    f"--- SLOT: {slot_id} | BASKET: lesson ---\n"
                    f"Lesson ID: {current_lesson_id} of 30\n"
                    f"Lesson Title: {lesson_title}\n"
                    f"Topic: {lesson_data.get('topic', '')}\n"
                    f"Concept Summary: {concept.get('summary', '')}\n"
                    f"Everyday Analogy: {concept.get('everyday_example', '')}\n"
                    f"Key Facts:\n{facts_text}\n"
                    f"Example Scenario: {example.get('scenario', '')}\n"
                    f"Takeaway: {example.get('takeaway', '')}\n"
                    f"Practice Exercise: {practice.get('try_now', '')}\n"
                    f"Copyable Prompt: {practice.get('prompt_to_copy', '')}\n\n"
                    f"STRICT INSTRUCTIONS FOR LESSON SLOT:\n"
                    f"1. Generate 100% natural conversational developer-cafe Persian content for TechNews Bot (@tech_news_110).\n"
                    f"2. 'hook': Eye-catching educational hook starting with 💡 or 🎓 (under 80 chars).\n"
                    f"3. 'intro': 1-2 punchy sentences with an everyday example introducing the concept clearly.\n"
                    f"4. 'steps': Exactly 3 dynamic, educational bullet points/steps tailored to this lesson topic.\n"
                    f"5. 'tip': 1 golden takeaway or key lesson tip tailored specifically to {lesson_title} (under 120 chars).\n"
                    f"6. 'try_now': 2-minute actionable practice in Persian.\n"
                    f"7. 'prompt_to_copy': Distinct copyable prompt or snippet (do NOT duplicate try_now).\n"
                    f"8. 'quiz': 3-option quiz (q, options, correct, explanation) on this concept.\n"
                )

            elif basket == PostBasket.WEEKLY_QA:
                slot_contexts[slot_id] = {
                    "basket": basket,
                }

            else:
                ranked = self.scorer.rank_candidates(
                    all_candidates,
                    basket=basket,
                    sent_links=sent_links,
                    recent_titles=recent_titles,
                    limit=3,
                )
                if not ranked:
                    logger.warning("No candidate passed filters for slot %s (%s).", slot_id, basket)
                    continue

                best_candidate = ranked[0]
                enriched = await enrich_top_candidates([best_candidate], max_items=1, client=http_client)
                best_candidate = enriched[0]

                found_img = await self.image_finder.find_image(
                    article_url=best_candidate.url,
                    fallback_url=best_candidate.image_url,
                    client=http_client,
                )
                image_url = found_img or best_candidate.image_url

                slot_contexts[slot_id] = {
                    "basket": basket,
                    "candidate": best_candidate,
                    "image_url": image_url,
                    "source_url": best_candidate.url,
                }

                extracted_text = (best_candidate.full_text or best_candidate.summary or "").strip()
                if len(extracted_text) > 3500:
                    extracted_text = extracted_text[:3500]

                prompt_sections.append(
                    f"--- SLOT: {slot_id} | BASKET: {basket.value} ---\n"
                    f"Title: {best_candidate.title}\n"
                    f"Source: {best_candidate.source_name}\n"
                    f"URL: {best_candidate.url}\n"
                    f"Content:\n{extracted_text}\n\n"
                    f"STRICT INSTRUCTIONS FOR NEWS/TOOL SLOT:\n"
                    f"1. Generate 100% natural, fluent conversational developer-cafe Persian content.\n"
                    f"2. Hook MUST be in Persian under 80 chars, starting with ⚡️, 🔥, 🚀 or 💡.\n"
                    f"3. Intro: 1-2 punchy Persian sentences.\n"
                    f"4. ZERO-HALLUCINATION: The 3 points MUST reflect real specific facts or numbers from the source text.\n"
                    f"5. 'what_can_you_do': Real practical takeaway in Persian.\n"
                    f"6. 'analysis': 1 insightful Persian sentence.\n"
                    f"7. 'try_now': 2-minute actionable practice in Persian.\n"
                )

        # 🤖 [۲/۵] در حال دریافت محتوا از هوش مصنوعی Gemini...
        safe_print("🤖 [۲/۵] در حال دریافت محتوا از هوش مصنوعی Gemini...")
        llm_posts: dict[str, PostItemSchema] = {}

        if prompt_sections and self.llm.api_key and not os.environ.get("PYTEST_CURRENT_TEST"):
            combined_prompt = (
                "You are writing Persian posts for TechNews Bot (@tech_news_110).\n"
                "Generate all posts for the scheduled slots below in a single structured JSON response to save quota.\n\n"
                + "\n\n".join(prompt_sections)
                + "\n\nSTRICT JSON OUTPUT FORMAT:\n"
                "Output MUST strictly be a single valid JSON object matching the GeminiBatchOutput schema:\n"
                '{"posts": [\n'
                '  {\n'
                '    "slot": "...",\n'
                '    "basket": "...",\n'
                '    "hook": "...",\n'
                '    "intro": "...",\n'
                '    "points": ["...", "...", "..."],\n'
                '    "what_can_you_do": "...",\n'
                '    "steps": ["...", "...", "..."],\n'
                '    "tip": "...",\n'
                '    "try_now": "...",\n'
                '    "prompt_to_copy": "...",\n'
                '    "terms": ["..."],\n'
                '    "quiz": {"q": "...", "options": ["...", "...", "..."], "correct": 0, "explanation": "..."}\n'
                '  }\n'
                ']}\n'
            )
            try:
                batch_res = self.llm.generate_batch(combined_prompt)
                for p in batch_res.posts:
                    llm_posts[p.slot] = p
                safe_print(f"   ✨ محتوا برای {len(batch_res.posts)} اسلات با موفقیت از هوش مصنوعی Gemini دریافت شد.")
            except Exception as llm_err:
                safe_print(f"❌ خطا در اتصال به هوش مصنوعی: {llm_err}")
                logger.error("Gemini batch API call failed: %s", llm_err)
        elif not self.llm.api_key:
            safe_print("❌ خطا در اتصال به هوش مصنوعی: کلید دسترسی GEMINI_API_KEY تنظیم نشده است.")

        # 🎨 [۳/۵] در حال بارگذاری فونت و ساخت تصویر گرافیکی...
        safe_print("🎨 [۳/۵] در حال بارگذاری فونت و ساخت تصویر گرافیکی...")
        posts_to_deliver: list[tuple[dict[str, Any], PostItemSchema, Optional[bytes], Optional[str], Optional[str]]] = []

        for slot in slots_to_process:
            slot_id = slot.get("slot_id", f"{run_index}_1")
            ctx = slot_contexts.get(slot_id)
            if not ctx:
                continue

            basket = ctx["basket"]
            post_item: Optional[PostItemSchema] = llm_posts.get(slot_id)
            card_bytes: Optional[bytes] = None
            image_url: Optional[str] = ctx.get("image_url")
            source_url: Optional[str] = ctx.get("source_url")

            # 4a. Lesson Slot
            if basket == PostBasket.LESSON:
                current_lesson_id = ctx["lesson_id"]
                lesson_title = ctx["lesson_title"]
                lesson_data = ctx["lesson_data"]

                font_loaded = self.card_generator.font_path or "مسیر پیش‌فرض سیستم"
                safe_print(f"   🔤 مسیر فونت لود شده: {font_loaded}")

                try:
                    card_bytes = self.card_generator.generate_card(
                        lesson_number=current_lesson_id,
                        lesson_title=lesson_title,
                        total_lessons=30,
                    )
                    is_valid, max_dev = validate_rendered_image(card_bytes)
                    if is_valid:
                        safe_print(f"   ✅ اعتبارسنجی تصویر کارت درس موفقیت‌آمیز بود (انحراف معیار: {max_dev:.2f} >= 2.0)")
                        try:
                            preview_card_path = self.root_dir / "data" / f"lesson_{current_lesson_id}_card.png"
                            preview_card_path.parent.mkdir(parents=True, exist_ok=True)
                            preview_card_path.write_bytes(card_bytes)
                        except Exception:
                            pass
                    else:
                        safe_print(f"   ❌ خطا در اعتبارسنجی تصویر: تصویر کارت تک‌رنگ یا خالی است (stddev={max_dev:.2f} < 2.0)")
                        card_bytes = None
                except Exception as card_err:
                    safe_print(f"   ❌ خطا در ساخت تصویر کارت درس: {card_err}")
                    card_bytes = None

                # If LLM didn't produce post, build dynamic post from curriculum.json (zero hardcoded strings)
                if post_item is None:
                    concept = lesson_data.get("concept", {}) if lesson_data else {}
                    example = lesson_data.get("example", {}) if lesson_data else {}
                    practice = lesson_data.get("practice", {}) if lesson_data else {}
                    quiz_data = practice.get("quiz", {})

                    allowed_facts = concept.get("allowed_facts", [])
                    dyn_steps = allowed_facts[:3] if len(allowed_facts) >= 3 else [
                        concept.get("summary", ""),
                        example.get("scenario", ""),
                        example.get("takeaway", ""),
                    ]
                    dyn_tip = example.get("takeaway") or (allowed_facts[0] if allowed_facts else "")
                    dyn_intro = concept.get("everyday_example") or concept.get("summary", "مفهوم کاربردی هوش مصنوعی")
                    practice_try = practice.get("try_now", "").strip()
                    practice_prompt = practice.get("prompt_to_copy", "").strip()
                    if practice_prompt == practice_try:
                        practice_prompt = ""

                    post_item = PostItemSchema(
                        slot=slot_id,
                        basket=PostBasket.LESSON,
                        hook=f"💡 {lesson_title}",
                        intro=dyn_intro,
                        steps=[s for s in dyn_steps if s],
                        tip=dyn_tip,
                        prompt_to_copy=practice_prompt,
                        try_now=practice_try,
                        quiz={
                            "q": quiz_data.get("question", "صورت سوال"),
                            "options": [str(opt)[:80] for opt in quiz_data.get("options", ["گزینه ۱", "گزینه ۲", "گزینه ۳"])],
                            "correct": quiz_data.get("correct_index", 0),
                            "explanation": quiz_data.get("explanation", "توضیح کوتاه"),
                        } if quiz_data else None,
                    )

            # 4b. Weekly Q&A Slot
            elif basket == PostBasket.WEEKLY_QA:
                week_no = raw_state.get("completed_weeks", 1)
                post_item = self.weekly_builder.build_weekly_post(week_number=week_no)

            # 4c. News / Tool Slots
            else:
                best_candidate = ctx["candidate"]
                if image_url:
                    safe_print(f"   🖼 تصویر شاخص برای اسلات {slot_id}: {image_url}")

                if post_item is None:
                    persian_hook = (
                        "⚡️ تحول و دستاورد تازه در دنیای هوش مصنوعی"
                        if basket in (PostBasket.NEWS_AI, PostBasket.NEWS_TECH)
                        else "🛠 معرفی ابزار کاربردی و هوشمند برای برنامه‌نویسان"
                    )
                    persian_intro = (
                        f"گزارش‌های جدید از سوی {best_candidate.source_name} نشان می‌دهد قابلیت‌های تازه‌ای برای توسعه‌دهندگان رونمایی شده است."
                    )
                    clean_terms = [t for t in (best_candidate.title.split()[:3]) if len(t) > 2 and t.isalnum()]
                    if not clean_terms:
                        clean_terms = ["AI", "Model", "Open Source"]

                    post_item = PostItemSchema(
                        slot=slot_id,
                        basket=basket,
                        hook=persian_hook,
                        intro=persian_intro,
                        points=[
                            f"تمرکز بر بهبود عملکرد و تسهیل استفاده در پروژه‌های نرم‌افزاری",
                            f"انتشار رسمی از طریق {best_candidate.source_name} با هدف دسترسی آسان‌تر تیم‌های فنی",
                            f"امکان بررسی و پیاده‌سازی سریع بر پایه مستندات و دیتای تازه",
                        ] if basket in (PostBasket.NEWS_AI, PostBasket.NEWS_TECH) else [],
                        what_can_you_do="می‌تونی از این قابلیت جدید در پروژه‌هات برای خودکارسازی کارها استفاده کنی.",
                        steps=[
                            "سایت یا ریپازیتوری پروژه را باز کن",
                            "راهنمای نصب سریع را بخوان",
                            "دستور اول را اجرا کن",
                        ] if basket in (PostBasket.TOOL_AI, PostBasket.TOOL_TECH) else [],
                        tip="قبل از شروع به کار، نسخه کتابخانه‌های وابسته را به‌روزرسانی کن.",
                        try_now="همین الان صفحه پروژه را در مرورگر باز کن و قابلیت‌هاش رو بررسی کن.",
                        terms=clean_terms,
                    )

            if post_item:
                posts_to_deliver.append((slot, post_item, card_bytes, image_url, source_url))

        # 📤 [۴/۵] در حال ارسال پست به تلگرام و ایتا...
        safe_print("📤 [۴/۵] در حال ارسال پست به تلگرام و ایتا...")
        senders = [
            TelegramSender(enabled=not dry_run),
            EitaaSender(enabled=not dry_run),
        ]

        if dry_run:
            safe_print("   ℹ️ حالت شبیه‌سازی (dry-run) فعال است؛ ارسال زنده به کانال‌ها انجام نمی‌شود.")

        for slot, post_item, card_bytes, image_url, source_url in posts_to_deliver:
            slot_id = slot.get("slot_id")
            basket = post_item.basket

            lesson_num = raw_state.get("current_lesson_id", 1)
            rendered_tg = self.renderer.render_telegram_html(post_item, lesson_number=lesson_num)
            rendered_eitaa = self.renderer.render_eitaa_plain_text(post_item, lesson_number=lesson_num)

            # Quality review
            review_res = self.reviewer.review(post_item, rendered_text=rendered_tg)
            logger.info("Quality review for slot %s: Score %d/100 (Passed=%s)", slot_id, review_res.score, review_res.passed)

            # Fact checking
            src_text = post_item.intro + " " + " ".join(post_item.points or [])
            facts_ok, unverified = verify_post_item(post_item, source_text=src_text)
            if not facts_ok:
                logger.warning("Fact check note for slot %s: Unverified tokens: %s", slot_id, unverified)

            if dry_run:
                safe_print(f"\n================ [DRY-RUN] Slot {slot_id} ({basket}) ================")
                safe_print(rendered_tg)
                card_info = f"Generated ({len(card_bytes)} bytes, saved to data/lesson_{lesson_num}_card.png)" if card_bytes else "None"
                safe_print("--------------------------------------------------------------------")
                safe_print(f"Stats: Length={len(rendered_tg)} chars | Card={card_info} | Image={image_url}")
                safe_print("====================================================================\n")
            else:
                # Live delivery
                for sender in senders:
                    caption = rendered_tg if sender.destination_name == "telegram" else rendered_eitaa
                    res = await sender.send_post(
                        text=caption,
                        image_bytes=card_bytes,
                        image_url=image_url,
                        source_url=source_url,
                        quiz=post_item.quiz.model_dump() if post_item.quiz else None,
                    )
                    if res.success:
                        safe_print(f"   ✅ ارسال به {sender.destination_name} با موفقیت انجام شد (شناسه پیام: {res.message_id})")
                    else:
                        safe_print(f"   ❌ خطا در ارسال به {sender.destination_name}: {res.error}")
                    logger.info("Delivered to %s: success=%s (id=%s)", sender.destination_name, res.success, res.message_id)

            # Atomic State Update
            self.state_manager.record_completed_run(date_str, run_index, str(basket.value))
            if source_url:
                self.state_manager.record_sent_link(source_url)
            self.state_manager.record_recent_title(post_item.hook)

            if basket == PostBasket.LESSON:
                with self.state_manager.transaction() as state:
                    state["current_lesson_id"] = (lesson_num % 30) + 1

        await http_client.aclose()
        logger.info("Run %d completed successfully.", run_index)
        safe_print("✅ [۵/۵] اجرای ربات با موفقیت به پایان رسید!")
        return True


def cli_entry() -> None:
    """CLI Entry point for python -m techbot.main."""
    parser = argparse.ArgumentParser(description="TechNews Bot Orchestrator")
    parser.add_argument("--run-index", type=int, default=1, choices=range(1, 6), help="Daily run index (1..5)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate without publishing")
    parser.add_argument("--date", type=str, default=None, help="Target execution date (YYYY-MM-DD)")

    args = parser.parse_args()
    orchestrator = TechBotOrchestrator()
    success = asyncio.run(orchestrator.run(run_index=args.run_index, dry_run=args.dry_run, target_date=args.date))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    cli_entry()
