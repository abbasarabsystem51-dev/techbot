"""
src/techbot/llm.py — LLM Client for Gemini API batch generation.

Features:
  1. Supports official `google-genai` SDK (`from google import genai`) with fallback to `google.generativeai`.
  2. Reads API key from `GEMINI_KEY` or `GEMINI_API_KEY` environment variables.
  3. Configurable model via `GEMINI_MODEL` (default: gemini-2.5-flash).
  4. Exactly 30 seconds timeout per attempt (LLM_TIMEOUT_SECONDS = 30).
  5. Retry max 3 attempts with exponential backoff (2s, 4s + jitter).
  6. Retry ONLY on timeout, 429, 500, 502, 503, 504, and connection errors; never retry 400/401/403.
  7. Clear Persian error logging with stage [۳/۵].
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from src.models import GeminiBatchOutput, PostItemSchema

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_print(text: str) -> None:
    """Print text safely with utf-8 encoding fallback."""
    try:
        print(text, flush=True)
    except Exception:
        try:
            sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()
        except Exception:
            pass

LLM_TIMEOUT_SECONDS = 30
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_MODELS = (DEFAULT_GEMINI_MODEL, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro")

# Try importing the new google-genai SDK first, then legacy google-generativeai
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    import google.generativeai as legacy_genai
except ImportError:
    legacy_genai = None


def is_retryable_llm_error(exc: Exception) -> bool:
    """
    Determine if an LLM error is transient and should be retried.
    Retry ONLY on timeout, 429, 500, 502, 503, 504, and connection errors.
    Never retry on 400/401/403.
    """
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in (400, 401, 403):
        return False

    err_str = str(exc).lower()

    # Never retry on 400, 401, 403 or permission / auth / invalid argument
    if any(k in err_str for k in ("400", "401", "403", "invalid argument", "invalid_argument", "permission_denied", "unauthenticated")):
        return False

    # Retry on timeout, 429, 500, 502, 503, 504, connection errors
    if any(k in err_str for k in ("504", "deadline exceeded", "timeout", "timed out", "429", "resource_exhausted", "500", "502", "503", "unavailable", "internal")):
        return True

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    try:
        import httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            return True
    except ImportError:
        pass

    return False


def clean_json_markdown(text: str) -> str:
    """Strip markdown code fence blocks if returned by the LLM."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def load_system_prompt() -> str:
    """Load system prompt from prompts/system.md or use default developer-cafe instructions."""
    root = Path(__file__).resolve().parents[2]
    prompt_path = root / "prompts" / "system.md"
    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    return (
        "شما یک دولوپر رفیق، باهوش و خفن هستید که برای کانال «تِک‌نیوز» (@tech_news_110) "
        "با لحن کافه‌ای، مدرن و پرانرژی محتوا می‌نویسید. هیچ عبارت کلیشه‌ای مانند «در دنیای امروز»، "
        "«بی‌شک»، «انقلابی» یا «بازی را عوض کرد» استفاده نکنید. "
        "قانون سخت‌گیرانه عدم توهم: تمام نکات کلیدی و تحلیل‌ها باید ۱۰۰٪ منطبق بر متن واقعی خبر ورودی باشد."
    )


class GeminiClient:
    """
    Robust Gemini API batch generation client with 30s timeout and exponential backoff retry.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        models: Optional[Sequence[str]] = None,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        timeout: float = LLM_TIMEOUT_SECONDS,
        max_retries: int = 2,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )
        primary_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        if models:
            self.models = list(models)
        else:
            self.models = [primary_model] + [m for m in DEFAULT_MODELS if m != primary_model]

        self.temperature = temperature
        self.system_prompt = system_prompt or load_system_prompt()
        self.timeout = LLM_TIMEOUT_SECONDS
        self.max_attempts = 3  # exactly 3 attempts max

        self._client: Any = None
        self._init_sdk()

    def _init_sdk(self) -> None:
        """Initialize the available Google GenAI client."""
        if not self.api_key:
            logger.warning("No Gemini API key found in environment (GEMINI_KEY / GEMINI_API_KEY).")
            return

        if genai is not None:
            try:
                http_opts = (
                    genai_types.HttpOptions(timeout=LLM_TIMEOUT_SECONDS * 1000)
                    if genai_types is not None
                    else None
                )
                self._client = genai.Client(api_key=self.api_key, http_options=http_opts)
                return
            except Exception as exc:
                logger.debug("Failed to init google-genai Client: %s", exc)

        if legacy_genai is not None:
            try:
                legacy_genai.configure(api_key=self.api_key)
                self._client = legacy_genai
            except Exception as exc:
                logger.debug("Failed to init legacy google-generativeai: %s", exc)

    def generate_batch(self, prompt: str) -> GeminiBatchOutput:
        """
        Execute a batch request to Gemini returning a validated GeminiBatchOutput.
        Retries max 3 attempts with exponential backoff (2s, 4s + jitter).
        Retries ONLY on timeout, 429, 500, 502, 503, 504, and connection errors.
        Never retries 400/401/403.
        On final failure, prints a clear Persian error and raises RuntimeError.
        If JSON parsing fails, returns empty GeminiBatchOutput(posts=[]) to allow fallback.
        """
        if not self.api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_KEY or GEMINI_API_KEY.")

        model_name = self.models[0]
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info("Calling Gemini (%s, attempt %d/%d)...", model_name, attempt, self.max_attempts)
                raw_text = self._call_model(model_name, prompt)
                cleaned_json = clean_json_markdown(raw_text)

                try:
                    data = json.loads(cleaned_json)
                    return GeminiBatchOutput.model_validate(data)
                except Exception as parse_err:
                    logger.warning("Failed to parse Gemini output as JSON (attempt %d): %s", attempt, parse_err)
                    # Unparsable JSON should not retry network; return empty to trigger candidate #1 fallback
                    return GeminiBatchOutput(posts=[])

            except Exception as exc:
                last_error = exc
                if not is_retryable_llm_error(exc):
                    logger.error("Non-retryable Gemini error (%s): %s", model_name, exc)
                    safe_print(
                        f"❌ [۳/۵] خطای غیرقابل تکرار در برقراری ارتباط با مدل هوش مصنوعی Gemini: {exc}"
                    )
                    raise RuntimeError(f"Non-retryable Gemini error: {exc}") from exc

                logger.warning("Gemini model '%s' failed on attempt %d/%d: %s", model_name, attempt, self.max_attempts, exc)
                if attempt < self.max_attempts:
                    backoff = (2.0 ** attempt) + random.uniform(0.05, 0.25)
                    time.sleep(backoff)

        # Final failure after 3 attempts
        safe_print(
            f"❌ [۳/۵] خطا: تلاش‌های اتصال به مدل هوش مصنوعی Gemini پس از ۳ بار تکرار ناموفق بود. آخرین خطا: {last_error}"
        )
        raise RuntimeError(f"All {self.max_attempts} Gemini attempts failed. Last error: {last_error}")

    def _call_model(self, model_name: str, prompt: str) -> str:
        """Direct call to model using either new or legacy SDK with 30-second timeout."""
        # 1. New google-genai SDK
        if genai is not None and hasattr(self._client, "models"):
            config: dict[str, Any] = {
                "temperature": self.temperature,
                "response_mime_type": "application/json",
            }
            if genai_types is not None:
                config["http_options"] = genai_types.HttpOptions(timeout=LLM_TIMEOUT_SECONDS * 1000)
            if self.system_prompt:
                config["system_instruction"] = self.system_prompt

            response = self._client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""

        # 2. Legacy google-generativeai SDK
        if legacy_genai is not None:
            kwargs: dict[str, Any] = {
                "generation_config": {
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                }
            }
            if self.system_prompt:
                kwargs["system_instruction"] = self.system_prompt

            model = legacy_genai.GenerativeModel(model_name, **kwargs)
            response = model.generate_content(
                prompt,
                request_options={"timeout": LLM_TIMEOUT_SECONDS},
            )
            return response.text or ""

        raise RuntimeError("No Google GenAI SDK is available in the Python environment.")
