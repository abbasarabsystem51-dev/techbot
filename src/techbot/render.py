"""
src/techbot/render.py — Visual & Emoji-Rich Post Renderer for Telegram (HTML) and Eitaa (Plain Text).

Features:
  1. Basket-specific layouts with modern developer-cafe emojis:
     - News: ⚡️ Hook, 🔹 Bullets, 🎯 Analysis, 🧪 Try-Now
     - Tool: 🛠 Hook, 🚀 What you can do, 1️⃣2️⃣3️⃣ Steps, 💡 Pro tip, 📋 Copyable prompt
     - Lesson: 🎓 Badge, 💡 Hook, ▫️ Concept points, 🧪 2-min practice
  2. Glossary injection: 📖 <b>واژه‌نامه:</b> <code>{term}</code> یعنی {short_def}
  3. Whitelisted hashtags from hashtags.json.
  4. Exact fixed brand footer:
     جلوتر بمون، تِک‌نیوز رو دنبال کن 🚀
     ┄┅═✧☫تِک نیوز☫✧═┅┄
     🆔 @tech_news_110
  5. Strict <= 1024 character enforcement with smart body trimming.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.models import PostBasket, PostItemSchema
from src.utils import persian_digit, strip_html

logger = logging.getLogger(__name__)

MAX_CAPTION_CHARS = 1024

BRAND_FOOTER = """جلوتر بمون، تِک‌نیوز رو دنبال کن 🚀
┄┅═✧☫تِک نیوز☫✧═┅┄
🆔 @tech_news_110"""


def html_to_plain_text(html: str) -> str:
    """Convert HTML Telegram caption to clean plain text for Eitaa."""
    text = re.sub(r"</?(b|strong|i|em|s|u)>", "", html)
    text = re.sub(r"</?code>", "", text)
    text = re.sub(r"</?pre>", "", text)
    text = re.sub(r'<a\s+href="([^"]+)">([^<]+)</a>', r"\2 (\1)", text)
    return text.strip()


class PostRenderer:
    """
    Renders structured PostItemSchema objects into beautiful captions for Telegram and Eitaa.
    """

    def __init__(
        self,
        glossary_path: Optional[Path | str] = None,
        hashtags_path: Optional[Path | str] = None,
    ) -> None:
        self.glossary = self._load_glossary(glossary_path)
        self.hashtags = self._load_hashtags(hashtags_path)

    def _load_glossary(self, path: Optional[Path | str]) -> dict[str, str]:
        root = Path(__file__).resolve().parents[2]
        target = Path(path) if path else root / "glossary.json"
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {k.lower(): v.get("short_def", v.get("definition", str(v))) for k, v in data.items()}
            except Exception as exc:
                logger.debug("Failed loading glossary.json: %s", exc)
        return {
            "rag": "ترکیب بازیابی اطلاعات با مدل زبانی برای پاسخ‌های دقیق‌تر",
            "llm": "مدل‌های زبانی بزرگ که متن تولید و درک می‌کنند",
            "fine-tuning": "تنظیم و آموزش مجدد مدل برای کارهای خاص",
            "agent": "هوش مصنوعی با توانایی تصمیم‌گیری و استفاده از ابزارها",
            "context window": "میزان متنی که مدل در یک لحظه می‌تواند در حافظه نگه دارد",
        }

    def _load_hashtags(self, path: Optional[Path | str]) -> list[str]:
        root = Path(__file__).resolve().parents[2]
        target = Path(path) if path else root / "hashtags.json"
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data.get("default", ["#هوش_مصنوعی", "#تک_نیوز"])
                    if isinstance(data, list):
                        return data
            except Exception as exc:
                logger.debug("Failed loading hashtags.json: %s", exc)
        return ["#هوش_مصنوعی", "#تک_نیوز"]

    def _format_hook(self, hook: str, default_emoji: str = "⚡️") -> str:
        """Ensure hook is bold and starts with a sharp emoji."""
        clean = hook.strip()
        # If clean already starts with an emoji, don't duplicate
        if not re.match(r"^[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff]", clean):
            clean = f"{default_emoji} {clean}"
        return f"<b>{clean}</b>"

    def _render_news(self, post: PostItemSchema) -> list[str]:
        """Format news_ai / news_tech post blocks."""
        blocks: list[str] = []
        blocks.append(self._format_hook(post.hook, "⚡️"))

        if post.intro:
            blocks.append(post.intro.strip())

        if post.points:
            point_lines = [f"🔹 {p.strip()}" for p in post.points if p.strip()]
            if point_lines:
                blocks.append("\n".join(point_lines))

        if post.what_can_you_do:
            blocks.append(f"🚀 <b>کاربرد واقعی:</b> {post.what_can_you_do.strip()}")

        if post.analysis:
            blocks.append(f"🎯 <b>تحلیل کوتاه:</b> {post.analysis.strip()}")

        if post.try_now:
            action = post.try_now.strip()
            if not action.startswith("همین الان"):
                blocks.append(f"🧪 <b>همین الان تست کن:</b> {action}")
            else:
                blocks.append(f"🧪 <b>{action}</b>")

        return blocks

    def _render_tool(self, post: PostItemSchema) -> list[str]:
        """Format tool_ai / tool_tech post blocks."""
        blocks: list[str] = []
        blocks.append(self._format_hook(post.hook, "🛠"))

        if post.intro:
            blocks.append(post.intro.strip())

        if post.what_can_you_do:
            blocks.append(f"🚀 <b>باهاش چیکار میشه کرد؟</b>\n{post.what_can_you_do.strip()}")

        if post.steps:
            emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]
            step_lines = [
                f"{emojis[i] if i < len(emojis) else '▫️'} {s.strip()}"
                for i, s in enumerate(post.steps) if s.strip()
            ]
            if step_lines:
                blocks.append("\n".join(step_lines))

        if post.tip:
            blocks.append(f"💡 <b>قِلق کار:</b> {post.tip.strip()}")

        if post.prompt_to_copy:
            blocks.append(f"📋 <b>پرامپت آماده کپی:</b>\n<code>{post.prompt_to_copy.strip()}</code>")

        if post.try_now:
            action = post.try_now.strip()
            blocks.append(f"🧪 <b>همین الان تست کن:</b> {action}")

        return blocks

    def _render_lesson(self, post: PostItemSchema, lesson_number: int = 1, total_lessons: int = 30) -> list[str]:
        """Format lesson post blocks."""
        blocks: list[str] = []
        badge_header = f"🎓 <b>درس {persian_digit(lesson_number)} از {persian_digit(total_lessons)} | {post.hook.replace('💡', '').strip()}</b>"
        blocks.append(badge_header)

        if post.intro:
            blocks.append(f"💡 {post.intro.strip()}")

        if post.steps:
            step_lines = [f"▫️ {s.strip()}" for s in post.steps if s.strip()]
            if step_lines:
                blocks.append("\n".join(step_lines))

        if post.tip:
            blocks.append(f"📌 <b>نکته کلیدی:</b> {post.tip.strip()}")

        if post.try_now:
            blocks.append(f"🧪 <b>تمرین ۲ دقیقه‌ای:</b>\n{post.try_now.strip()}")

        if post.prompt_to_copy:
            blocks.append(f"📋 <code>{post.prompt_to_copy.strip()}</code>")

        return blocks

    def _find_glossary_term(self, post: PostItemSchema) -> Optional[tuple[str, str]]:
        """Look for terms in post that match glossary definitions."""
        candidate_terms = list(post.terms or [])
        text = f"{post.hook} {post.intro} {' '.join(post.points or [])} {post.what_can_you_do}".lower()

        for term in candidate_terms:
            t_clean = term.strip().lower()
            if t_clean in self.glossary:
                return (term.strip(), self.glossary[t_clean])

        for key, definition in self.glossary.items():
            if f" {key} " in f" {text} ":
                return (key.upper(), definition)

        return None

    def render_telegram_html(
        self,
        post: PostItemSchema,
        lesson_number: int = 1,
        total_lessons: int = 30,
    ) -> str:
        """
        Assemble the complete Telegram HTML post with exact <= 1024 char limit.
        """
        basket = post.basket or PostBasket.NEWS_AI
        if basket in (PostBasket.TOOL_AI, PostBasket.TOOL_TECH):
            blocks = self._render_tool(post)
        elif basket == PostBasket.LESSON:
            blocks = self._render_lesson(post, lesson_number, total_lessons)
        else:
            blocks = self._render_news(post)

        # Glossary box check
        glossary_match = self._find_glossary_term(post)
        glossary_box = ""
        if glossary_match:
            term, short_def = glossary_match
            glossary_box = f"📖 <b>واژه‌نامه:</b> <code>{term}</code> یعنی {short_def}"

        # Hashtags block
        tag_list = post.topic_tags or []
        tags = " ".join(f"#{t.replace(' ', '_')}" for t in tag_list if t)
        if not tags:
            tags = " ".join(self.hashtags[:2])

        # Bottom section that must be preserved
        footer_block = f"{tags}\n\n{BRAND_FOOTER}"

        # Assemble full content
        def assemble(main_blocks: list[str], include_glossary: bool) -> str:
            parts = list(main_blocks)
            if include_glossary and glossary_box:
                parts.append(glossary_box)
            parts.append(footer_block)
            return "\n\n".join(p for p in parts if p)

        # 1. Try full assembly with glossary
        full_text = assemble(blocks, include_glossary=True)
        if len(full_text) <= MAX_CAPTION_CHARS:
            return full_text

        # 2. Try without glossary
        full_text = assemble(blocks, include_glossary=False)
        if len(full_text) <= MAX_CAPTION_CHARS:
            return full_text

        # 3. Trim body blocks until <= 1024 characters
        # Blocks: [0=hook, 1=intro, 2=points/steps, ...]
        trimmed_blocks = list(blocks)
        for i in range(len(trimmed_blocks) - 1, 0, -1):
            # Do not delete the hook (index 0) or try_now (last element)
            if i == len(trimmed_blocks) - 1 and "🧪" in trimmed_blocks[i]:
                continue
            trimmed_blocks.pop(i)
            candidate = assemble(trimmed_blocks, include_glossary=False)
            if len(candidate) <= MAX_CAPTION_CHARS:
                return candidate

        # 4. Final safety slice if still over
        overflow = len(full_text) - MAX_CAPTION_CHARS + 5
        if len(trimmed_blocks) > 1:
            trimmed_blocks[1] = trimmed_blocks[1][:-overflow] + "…"
        return assemble(trimmed_blocks, include_glossary=False)

    def render_eitaa_plain_text(
        self,
        post: PostItemSchema,
        lesson_number: int = 1,
        total_lessons: int = 30,
    ) -> str:
        """
        Assemble plain-text post for Eitaa.
        """
        html_caption = self.render_telegram_html(post, lesson_number, total_lessons)
        return html_to_plain_text(html_caption)
