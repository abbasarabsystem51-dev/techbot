"""
src/techbot/review.py — 0-100 local quality and editorial reviewer.

Criteria:
  1. Punchy Hook (15 pts): <= 80 chars, starts with an attractive emoji.
  2. Zero Forbidden Clichés (25 pts): Free of forbidden phrases from editorial.md.
  3. Actionable Try-Now / CTA (20 pts): Practical 2-minute test present.
  4. Length Compliance (20 pts): Fully assembled post <= 1024 characters.
  5. HTML Validity (10 pts): Only valid, balanced Telegram HTML tags.
  6. Clean Bullet Structure (10 pts): Well-formatted points or steps.

Passing threshold: >= 70 / 100.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional

from src.models import FORBIDDEN_PHRASES, PostItemSchema

EXTENDED_FORBIDDEN = FORBIDDEN_PHRASES + (
    "شاهد هستیم",
    "حاکی از آن است",
    "دستخوش تغییر",
)

ALLOWED_HTML_TAGS = frozenset({"b", "strong", "i", "em", "code", "pre", "a", "s", "u"})


class HTMLTagValidator(HTMLParser):
    """Checks whether HTML contains only allowed tags and all tags are properly balanced."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.invalid_tags: list[str] = []
        self.unbalanced: bool = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        lower_tag = tag.lower()
        if lower_tag not in ALLOWED_HTML_TAGS:
            self.invalid_tags.append(lower_tag)
        else:
            self.stack.append(lower_tag)

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag in ALLOWED_HTML_TAGS:
            if self.stack and self.stack[-1] == lower_tag:
                self.stack.pop()
            else:
                self.unbalanced = True


@dataclass
class ReviewResult:
    score: int
    passed: bool
    feedback: list[str] = field(default_factory=list)


class QualityReviewer:
    """
    Evaluates posts against technical and editorial quality rules.
    """

    def __init__(self, passing_threshold: int = 70) -> None:
        self.passing_threshold = passing_threshold

    def validate_html(self, text: str) -> tuple[bool, list[str]]:
        """Verify HTML tags are valid and properly closed."""
        parser = HTMLTagValidator()
        issues: list[str] = []
        try:
            parser.feed(text)
            if parser.invalid_tags:
                issues.append(f"تگ‌های غیرمجاز HTML: {', '.join(parser.invalid_tags)}")
            if parser.stack or parser.unbalanced:
                issues.append("تگ‌های HTML باز بدون بسته شدن مناسب هستند.")
        except Exception as exc:
            issues.append(f"خطا در پارس HTML: {exc}")

        return (len(issues) == 0, issues)

    def review(
        self,
        post: PostItemSchema,
        rendered_text: Optional[str] = None,
    ) -> ReviewResult:
        """
        Score a post from 0 to 100 and report detailed feedback.
        """
        score = 0
        feedback: list[str] = []

        # 1. Hook Quality (15 pts)
        hook = post.hook.strip()
        if hook:
            if len(hook) <= 80:
                score += 10
            else:
                feedback.append(f"قلاب طولانی است ({len(hook)} کاراکتر، سقف ۸۰)")

            # Check emoji at beginning
            if re.match(r"^[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff]", hook):
                score += 5
            else:
                score += 2
        else:
            feedback.append("قلاب (hook) خالی است.")

        # 2. Cliché Compliance (25 pts)
        full_raw_text = f"{post.hook} {post.intro} {' '.join(post.points or [])} {post.what_can_you_do} {' '.join(post.steps or [])} {post.tip} {post.try_now}"
        check_cliche_text = f"{rendered_text} {full_raw_text}" if rendered_text else full_raw_text
        found_cliches = [c for c in EXTENDED_FORBIDDEN if c in check_cliche_text]
        if not found_cliches:
            score += 25
        else:
            feedback.append(f"عبارات ممنوعه کلیشه‌ای یافت شد: {', '.join(found_cliches)}")

        # 3. Actionable Try-Now / Practical Step (20 pts)
        if post.try_now and len(post.try_now.strip()) > 10:
            score += 20
        elif post.prompt_to_copy:
            score += 15
        else:
            feedback.append("بخش اقدام عملی (try_now) ناقص یا خالی است.")

        # 4. Length Compliance (20 pts)
        check_text = rendered_text or full_raw_text
        if len(check_text) <= 1024:
            score += 20
        else:
            feedback.append(f"طول متن بیش از ۱۰۲۴ کاراکتر است ({len(check_text)} کاراکتر).")

        # 5. HTML Validity (10 pts)
        is_html_valid, html_issues = self.validate_html(check_text)
        if is_html_valid:
            score += 10
        else:
            feedback.extend(html_issues)

        # 6. Bullet / Content Structure (10 pts)
        if post.points and len(post.points) >= 2:
            score += 10
        elif post.steps and len(post.steps) >= 2:
            score += 10
        elif post.quiz:
            score += 10
        else:
            score += 5

        passed = score >= self.passing_threshold
        return ReviewResult(score=score, passed=passed, feedback=feedback)
