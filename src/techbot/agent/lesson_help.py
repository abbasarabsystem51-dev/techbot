"""
src/techbot/agent/lesson_help.py — Persian help text for custom lesson JSON format.

The admin can send this text to Telegram via /lesson_help to see the required JSON format.
"""

LESSON_HELP_TEXT = """
📚 <b>راهنمای اضافه کردن درس سفارشی</b>

برای اضافه کردن یا جایگزین کردن یک درس، مراحل زیر را دنبال کنید:

1️⃣ دستور <code>/lesson_add</code> را ارسال کنید.
2️⃣ سپس JSON درس خود را ارسال کنید (می‌توانید چند پیام ارسال کنید).
3️⃣ در پایان دستور <code>/done</code> را بزنید تا درس ذخیره شود.

─────────────────────────
📋 <b>فرمت JSON مورد نیاز:</b>
─────────────────────────

<pre>{
  "lesson_id": 1,
  "title": "عنوان درس به فارسی",
  "level": "مبتدی",
  "topic": "AI Fundamentals",
  "key_terms": ["AI", "Machine Learning", "Algorithm"],
  "prerequisites": [],
  "prev_title": null,
  "next_title": null,
  "concept": {
    "summary": "خلاصه مفهوم اصلی درس در اینجا...",
    "everyday_example": "مثال روزمره که مفهوم را ساده می‌کند...",
    "allowed_facts": [
      "واقعیت تایید‌شده اول",
      "واقعیت تایید‌شده دوم",
      "واقعیت تایید‌شده سوم"
    ]
  },
  "example": {
    "scenario": "سناریوی مثال برای این درس...",
    "bad_case": "روش قدیمی یا اشتباه در استفاده از این مفهوم...",
    "good_case": "روش صحیح مبتنی بر AI یا روش جدید...",
    "takeaway": "نکته کلیدی که کاربر باید یاد بگیرد"
  },
  "practice": {
    "platform": "ChatGPT یا Gemini",
    "try_now": "دستورالعمل کوتاه برای امتحان کردن...",
    "prompt_to_copy": "پرامپت آماده برای کپی کردن در ابزار AI",
    "quiz": {
      "question": "متن سوال کوئیز چندگزینه‌ای؟",
      "options": [
        "گزینه اول",
        "گزینه دوم",
        "گزینه سوم"
      ],
      "correct_index": 0,
      "explanation": "توضیح کوتاه برای پاسخ صحیح"
    }
  }
}</pre>

─────────────────────────
⚠️ <b>نکات مهم:</b>
─────────────────────────
• <code>lesson_id</code> باید عدد صحیح مثبت باشد.
• اگر <code>lesson_id</code> از قبل وجود داشته باشد، درس قدیمی <b>جایگزین</b> می‌شود.
• کوئیز باید حداقل ۳ گزینه داشته باشد.
• <code>correct_index</code> از ۰ شروع می‌شود (گزینه اول = ۰).
• همه فیلدهای اجباری باید پر باشند.
"""
