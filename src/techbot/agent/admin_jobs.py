import json
import os
import httpx
import logging
from datetime import datetime
from pathlib import Path
from pydantic import ValidationError
from src.techbot.state.atomic import StateManager
from src.techbot.agent.lesson_help import LESSON_HELP_TEXT
from src.techbot.agent.lessons import save_lesson

logger = logging.getLogger(__name__)
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID", "")
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

# ---------------------------------------------------------------------------
# User Inbox helpers
# ---------------------------------------------------------------------------
_ROOT_DIR = Path(__file__).resolve().parents[4]
USER_INBOX_PATH = _ROOT_DIR / "data" / "user_inbox.json"


def _load_user_inbox() -> list[dict]:
    """Load the user inbox JSON file, returning an empty list if it doesn't exist or is corrupt."""
    if not USER_INBOX_PATH.exists():
        return []
    try:
        with open(USER_INBOX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as exc:
        logger.error("Failed to load user_inbox.json: %s", exc)
    return []


def _append_user_message(user_id: int | str, username: str | None, text: str) -> None:
    """Safely append a user message to data/user_inbox.json."""
    USER_INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    inbox = _load_user_inbox()
    record = {
        "user_id": user_id,
        "username": username or "",
        "text": text,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    inbox.append(record)
    try:
        with open(USER_INBOX_PATH, "w", encoding="utf-8") as f:
            json.dump(inbox, f, ensure_ascii=False, indent=2)
        logger.info("User message saved to inbox [user_id=%s]: %s", user_id, text[:60])
    except Exception as exc:
        logger.error("Failed to write user_inbox.json: %s", exc)


def _send_to_user(chat_id: int | str, text: str) -> None:
    """Send a plain-text or HTML message to any Telegram user."""
    if not TG_TOKEN:
        return
    try:
        httpx.post(
            f"{TG_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        logger.error("Failed to send message to user %s: %s", chat_id, exc)


def load_user_inbox() -> list[dict]:
    """Public helper: load all messages from data/user_inbox.json."""
    return _load_user_inbox()


def send_to_admin(text: str):
    if not ADMIN_CHAT_ID or not TG_TOKEN:
        return
    try:
        httpx.post(f"{TG_API}/sendMessage", json={"chat_id": ADMIN_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send admin message: {e}")

def drain_updates(state: StateManager) -> list[dict]:
    if not ADMIN_CHAT_ID or not TG_TOKEN:
        return []
    
    offset = state.get_state().get("inbox_offset", 0)
    try:
        resp = httpx.get(f"{TG_API}/getUpdates", params={"offset": offset, "timeout": 0}, timeout=15)
        data = resp.json()
    except Exception as e:
        logger.error(f"getUpdates failed: {e}")
        return []

    if not data.get("ok"):
        return []

    new_jobs = []
    new_offset = offset

    for upd in data.get("result", []):
        new_offset = max(new_offset, upd.get("update_id", 0) + 1)
        msg = upd.get("message")
        if not msg:
            continue

        chat_id = msg.get("chat", {}).get("id")
        text = msg.get("text", "").strip()

        # ----------------------------------------------------------------
        # Normal-user persona: route non-admin messages to user inbox
        # ----------------------------------------------------------------
        if str(chat_id) != str(ADMIN_CHAT_ID):
            if text == "/start":
                _send_to_user(
                    chat_id,
                    "سلام! 👋 به ربات <b>تکنیوز</b> خوش آمدید.\n\n"
                    "اینجا می‌توانید نظرات، انتقادات و پیشنهادات خود را برای ما بنویسید. "
                    "ادمین شخصاً پیام‌های شما را می‌بیند و در اسرع وقت بررسی می‌کند. 🚀",
                )
            elif text and not text.startswith("/"):
                from_ = msg.get("from", {})
                user_id = from_.get("id", chat_id)
                username = from_.get("username") or from_.get("first_name") or ""
                _append_user_message(user_id, username, text)
                _send_to_user(
                    chat_id,
                    "✅ پیام شما با موفقیت دریافت شد. از اینکه با ما در ارتباط هستید ممنونیم!",
                )
            continue

        # ----------------------------------------------------------------
        # Weekly analysis prompt interception (must be first admin check)
        # ----------------------------------------------------------------
        current_state_snap = state.get_state()
        if current_state_snap.get("awaiting_weekly_analysis_prompt") and text:
            admin_prompt = text
            pending_messages = current_state_snap.get("pending_weekly_messages", [])
            messages_text = "\n".join(
                f"[{m.get('date', '')}] "
                f"{m.get('username', 'ناشناس')} "
                f"(id:{m.get('user_id', '')}): "
                f"{m.get('text', '')}"
                for m in pending_messages
            )
            analysis_prompt = (
                f"اینجا لیست پیامهای کاربران است:\n{messages_text}\n\n"
                f"دستور ادمین برای تحلیل: {admin_prompt}\n\n"
                "لطفا یک تحلیل جامع و دقیق به زبان فارسی بر اساس دستور ادمین ارائه بده."
            )
            try:
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
                with httpx.Client(timeout=60.0) as _client:
                    _resp = _client.post(_url, json=_payload)
                _data = _resp.json()
                _candidates = _data.get("candidates", [])
                if _candidates:
                    _parts = _candidates[0].get("content", {}).get("parts", [])
                    analysis_result = _parts[0].get("text", "") if _parts else ""
                else:
                    analysis_result = ""
                if analysis_result:
                    send_to_admin(f"📊 نتیجه تحلیل هفتگی:\n\n{analysis_result}")
                else:
                    send_to_admin("❌ هوش مصنوعی پاسخی برای تحلیل هفتگی نداد.")
            except Exception as _weekly_exc:
                send_to_admin(f"❌ خطا در تحلیل هفتگی با هوش مصنوعی: {_weekly_exc}")
                logger.error("Weekly analysis LLM call failed: %s", _weekly_exc)
            # Clear state and inbox
            with state.transaction() as _st:
                _st["awaiting_weekly_analysis_prompt"] = False
                _st["pending_weekly_messages"] = []
            # Clear data/user_inbox.json
            try:
                USER_INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(USER_INBOX_PATH, "w", encoding="utf-8") as _f:
                    json.dump([], _f, ensure_ascii=False)
                logger.info("user_inbox.json cleared after weekly analysis.")
            except Exception as _clear_exc:
                logger.error("Failed to clear user_inbox.json: %s", _clear_exc)
            # Do NOT fall through to normal admin command processing
            continue

        # ----------------------------------------------------------------
        # /menu — send inline keyboard admin menu
        # ----------------------------------------------------------------
        if text == "/menu":
            if TG_TOKEN and ADMIN_CHAT_ID:
                try:
                    menu_keyboard = {
                        "inline_keyboard": [
                            [{"text": "📊 وضعیت ربات", "callback_data": "/status"}],
                            [{"text": "📥 تحلیل پیام‌ها", "callback_data": "/analyze"}],
                            [{"text": "📝 افزودن درس", "callback_data": "/lesson_add"}],
                            [{"text": "📖 راهنمای درس", "callback_data": "/lesson_help"}],
                            [{"text": "📢 پست متنی", "callback_data": "/raw"}],
                            [{"text": "🤖 پست با هوش مصنوعی", "callback_data": "/new"}],
                        ]
                    }
                    httpx.post(
                        f"{TG_API}/sendMessage",
                        json={
                            "chat_id": ADMIN_CHAT_ID,
                            "text": "🎛️ <b>منوی مدیریت TechNews Bot</b>\n\nیک گزینه را انتخاب کنید:",
                            "parse_mode": "HTML",
                            "reply_markup": menu_keyboard,
                        },
                        timeout=10,
                    )
                except Exception as exc:
                    logger.error("Failed to send menu: %s", exc)

        # ----------------------------------------------------------------
        # /lesson_help — send the JSON format guide to admin
        # ----------------------------------------------------------------
        elif text == "/lesson_help":
            send_to_admin(LESSON_HELP_TEXT)

        # ----------------------------------------------------------------
        # /lesson_add — start collecting JSON for a new/override lesson
        # ----------------------------------------------------------------
        elif text == "/lesson_add":
            with state.transaction() as st:
                st["lesson_add_mode"] = True
                st["lesson_json_buffer"] = []
            send_to_admin(
                "✅ حالت افزودن درس فعال شد.\n"
                "JSON درس خود را ارسال کنید (می‌توانید چند پیام ارسال کنید).\n"
                "وقتی تمام شد، <code>/done</code> را بزنید."
            )

        # ----------------------------------------------------------------
        # /done — finalise JSON buffer, validate, and save lesson
        # ----------------------------------------------------------------
        elif text == "/done":
            current_state = state.get_state()
            if not current_state.get("lesson_add_mode"):
                send_to_admin("⚠️ هیچ درسی در حال ویرایش نیست. ابتدا /lesson_add را بزنید.")
            else:
                buffer: list[str] = current_state.get("lesson_json_buffer", [])
                raw_json = "\n".join(buffer).strip()

                # Clear mode regardless of outcome
                with state.transaction() as st:
                    st["lesson_add_mode"] = False
                    st["lesson_json_buffer"] = []

                if not raw_json:
                    send_to_admin("❌ هیچ JSON‌ای دریافت نشد. لطفاً دوباره /lesson_add را بزنید.")
                else:
                    try:
                        lesson_data = json.loads(raw_json)
                    except json.JSONDecodeError as exc:
                        send_to_admin(
                            f"❌ JSON معتبر نیست:\n<code>{exc}</code>\n"
                            "لطفاً JSON خود را بررسی و دوباره ارسال کنید."
                        )
                        logger.warning("Admin sent invalid JSON for lesson_add: %s", exc)
                    else:
                        try:
                            lesson = save_lesson(lesson_data)
                            send_to_admin(
                                f"✅ درس شماره <b>{lesson.lesson_id}</b> با عنوان «{lesson.title}» "
                                f"با موفقیت ذخیره شد و در اجرای بعدی اعمال می‌شود."
                            )
                            logger.info(
                                "Admin saved custom lesson id=%d title=%s",
                                lesson.lesson_id, lesson.title,
                            )
                        except ValidationError as exc:
                            errors = "; ".join(
                                f"{e['loc']}: {e['msg']}" for e in exc.errors()
                            )
                            send_to_admin(
                                f"❌ اعتبارسنجی درس ناموفق بود:\n<code>{errors}</code>\n"
                                "لطفاً فیلدهای مشخص‌شده را اصلاح کنید."
                            )
                            logger.warning("Lesson validation failed: %s", exc)
                        except Exception as exc:
                            send_to_admin(f"❌ خطا در ذخیره‌سازی درس: {exc}")
                            logger.error("Failed to save custom lesson: %s", exc)

        # ----------------------------------------------------------------
        # Collect JSON buffer lines when lesson_add_mode is active
        # ----------------------------------------------------------------
        elif state.get_state().get("lesson_add_mode"):
            with state.transaction() as st:
                buf: list[str] = st.get("lesson_json_buffer", [])
                buf.append(text)
                st["lesson_json_buffer"] = buf
            send_to_admin("📥 متن دریافت شد. ادامه دهید یا /done بزنید.")

        # ----------------------------------------------------------------
        # Existing commands
        # ----------------------------------------------------------------
        elif text.startswith("/raw "):
            new_jobs.append({"id": f"job_{new_offset}", "type": "raw", "content": text[5:]})
            send_to_admin("✅ دستور /raw دریافت شد. در ران بعدی منتشر میشود.")
        elif text.startswith("/new "):
            new_jobs.append({"id": f"job_{new_offset}", "type": "new", "content": text[5:]})
            send_to_admin("✅ دستور /new دریافت شد. هوش مصنوعی در حال ساخت پست است...")
        elif text.startswith("/analyze "):
            admin_prompt = text[len("/analyze "):].strip()
            new_jobs.append({"id": f"job_{new_offset}", "type": "analyze", "content": admin_prompt})
            send_to_admin("🔍 دستور /analyze دریافت شد. در حال بارگذاری صندوق پیام‌ها و تحلیل...")
        elif text == "/status":
            send_to_admin("ربات آنلاین است و در حال کار.")

    with state.transaction() as st:
        st["inbox_offset"] = new_offset
        if new_jobs:
            jobs = st.get("admin_jobs", [])
            jobs.extend(new_jobs)
            st["admin_jobs"] = jobs

    return new_jobs
