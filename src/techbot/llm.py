"""
src/techbot/llm.py — LLM Client for Gemini API batch generation.
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

import httpx

from src.models import GeminiBatchOutput, PostItemSchema

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 25.0
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_FLASH_MODELS = ("gemini-3.8-flash", "gemini-3.0-flash")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except Exception:
        try:
            sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()
        except Exception:
            pass

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


def is_retryable_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in (400, 401, 403):
        return False

    err_str = str(exc).lower()
    if any(k in err_str for k in ("400", "401", "403", "invalid argument", "invalid_argument", "permission_denied", "unauthenticated")):
        return False

    if any(k in err_str for k in ("504", "deadline exceeded", "timeout", "timed out", "429", "resource_exhausted", "500", "502", "503", "unavailable", "internal")):
        return True

    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError)):
        return True

    return False


def clean_json_markdown(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def parse_llm_json(text: str) -> dict:
    import json
    import re

    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    no_trailing = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(no_trailing)
    except json.JSONDecodeError:
        pass

    fixed = no_trailing.replace("\u201c", '"').replace("\u201d", '"')
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    idx = fixed.find('"posts"')
    if idx < 0:
        raise ValueError("No 'posts' key found in LLM output")
    bracket = fixed.find("[", idx)
    if bracket < 0:
        raise ValueError("No array found after 'posts'")

    decoder = json.JSONDecoder(strict=False)
    posts = []
    pos = bracket + 1
    while pos < len(fixed):
        while pos < len(fixed) and fixed[pos] in " \n\r\t,":
            pos += 1
        if pos >= len(fixed) or fixed[pos] != "{":
            break
        try:
            obj, end = decoder.raw_decode(fixed, pos)
            posts.append(obj)
            pos = end
        except json.JSONDecodeError:
            break

    if not posts:
        raise ValueError("Could not recover any complete post objects")

    return {"posts": posts}


def load_system_prompt() -> str:
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
        self.primary_model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.custom_models = models is not None
        if models:
            self.models = list(models)
        else:
            self.models = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]

        self.temperature = temperature
        self.system_prompt = system_prompt or load_system_prompt()
        self.timeout = float(timeout)
        self.max_attempts = 3

        self._client: Any = None
        self._init_sdk()

    def _init_sdk(self) -> None:
        if not self.api_key:
            logger.warning("No Gemini API key found in environment.")
            return

        if genai is not None:
            try:
                http_opts = (
                    genai_types.HttpOptions(timeout=int(self.timeout * 1000))
                    if genai_types is not None
                    else None
                )
                self._client = genai.Client(api_key=self.api_key, http_options=http_opts)
            except Exception as exc:
                logger.debug("Failed to init google-genai Client: %s", exc)

    def discover_flash_models(self) -> list[str]:
        return self._fetch_and_prioritize_models()

    def discover_available_models(self) -> list[str]:
        return self._fetch_and_prioritize_models()

    def _fetch_and_prioritize_models(self) -> list[str]:
        discovered: list[str] = []
        if self._client is not None and hasattr(self._client, "models") and hasattr(self._client.models, "list"):
            try:
                for m in self._client.models.list():
                    name = getattr(m, "name", "") or ""
                    clean_name = name.replace("models/", "").strip()
                    methods = (
                        getattr(m, "supported_generation_methods", None)
                        or getattr(m, "supported_actions", None)
                        or []
                    )
                    if not methods or "generateContent" in methods:
                        if clean_name and "gemini" in clean_name.lower():
                            discovered.append(clean_name)
            except Exception as exc:
                logger.debug("SDK client.models.list() failed: %s", exc)

        if not discovered and self.api_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        models_data = resp.json().get("models", [])
                        for m in models_data:
                            name = m.get("name", "").replace("models/", "").strip()
                            methods = m.get("supportedGenerationMethods", [])
                            if "generateContent" in methods and "gemini" in name.lower():
                                discovered.append(name)
            except Exception as exc:
                logger.debug("REST model discovery failed: %s", exc)

        if not discovered:
            discovered = [self.primary_model] + [m for m in FALLBACK_FLASH_MODELS if m != self.primary_model]

        EXCLUDED_KEYWORDS = (
            "tts", "image", "audio", "transcribe", "embed", "embedding",
            "robotics", "computer-use", "live"
        )
        filtered_models = [m for m in discovered if not any(kw in m.lower() for kw in EXCLUDED_KEYWORDS)]
        if not filtered_models:
            filtered_models = list(FALLBACK_FLASH_MODELS)

        def model_version_key(name: str) -> tuple[tuple[float, ...], int]:
            matches = re.findall(r"(\d+(?:\.\d+)?)", name)
            ver = tuple(float(x) for x in matches) if matches else (0.0,)
            is_preview = 1 if ("preview" in name.lower() or "exp" in name.lower()) else 0
            return (ver, -is_preview)

        flash_lite: list[str] = []
        flash: list[str] = []
        other: list[str] = []

        for m in filtered_models:
            m_lower = m.lower()
            if "flash" in m_lower and ("lite" in m_lower or "8b" in m_lower):
                flash_lite.append(m)
            elif "flash" in m_lower:
                flash.append(m)
            elif "gemini" in m_lower:
                other.append(m)

        flash_lite.sort(key=model_version_key, reverse=True)
        flash.sort(key=model_version_key, reverse=True)
        other.sort(key=model_version_key, reverse=True)

        ordered: list[str] = []
        for m in flash + flash_lite + other:
            if m not in ordered:
                ordered.append(m)

        safe_print(f"Discovered available models: {ordered}")
        logger.info("Discovered available models: %s", ordered)
        return ordered

    def generate_batch(self, prompt: str) -> GeminiBatchOutput:
        if not self.api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_KEY or GEMINI_API_KEY.")

        # FORCED TO USE OUR CUSTOM MODELS LIST
        candidate_models = list(self.models)

        model_idx = 0
        last_error: Optional[Exception] = None

        while model_idx < len(candidate_models):
            current_model = candidate_models[model_idx]
            model_unavailable = False

            for attempt in range(1, self.max_attempts + 1):
                try:
                    logger.info("Calling Gemini (%s, attempt %d/%d)...", current_model, attempt, self.max_attempts)
                    raw_text = self._call_model(current_model, prompt)

                    try:
                        data = parse_llm_json(raw_text)
                        if isinstance(data, list):
                            data = {"posts": data}
                        elif isinstance(data, dict) and "posts" not in data:
                            data = {"posts": data}
                        raw_posts = data.get("posts", [])
                        valid_posts = []
                        for p_dict in raw_posts:
                            try:
                                from src.models import PostItemSchema
                                valid_post = PostItemSchema.model_validate(p_dict)
                                valid_posts.append(valid_post)
                            except Exception as e:
                                logger.warning(f"Skipping invalid post due to validation error: {e}")
                        return GeminiBatchOutput(posts=valid_posts)
                    except Exception as parse_err:
                        logger.error(
                            "Failed to parse Gemini output (attempt %d): %s | raw head: %s",
                            attempt, parse_err, raw_text[:200],
                        )
                        raise RuntimeError(f"Gemini output not parseable: {parse_err}") from parse_err

                except Exception as exc:
                    last_error = exc
                    err_str = str(exc).lower()

                    is_404 = "404" in err_str or "not_found" in err_str or "not found" in err_str
                    is_504 = "504" in err_str or "deadline" in err_str
                    is_503 = "503" in err_str or "unavailable" in err_str
                    is_429 = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
                    is_model_incompatible = any(k in err_str for k in (
                        "developer instruction is not enabled",
                        "not enabled for this model",
                        "not supported for this model",
                        "model does not support",
                        "unsupported model",
                    ))

                    if is_429:
                        logger.warning("Model '%s' returned 429 (Quota Exhausted). Moving to next model immediately.", current_model)
                        model_unavailable = True
                        break

                    if is_404 or is_model_incompatible:
                        logger.warning("Model '%s' is unavailable or incompatible (%s). Triggering fallback chain...", current_model, exc)
                        model_unavailable = True
                        break

                    if (is_504 or is_503) and attempt == self.max_attempts:
                        logger.warning("Model '%s' failed (%s) across all %d attempts. Triggering fallback chain...", current_model, "504" if is_504 else "503", self.max_attempts)
                        model_unavailable = True
                        break

                    if not is_retryable_llm_error(exc):
                        logger.error("Non-retryable Gemini error (%s): %s", current_model, exc)
                        safe_print(
                            f"❌ [۳/۵] خطای غیرقابل تکرار در برقراری ارتباط با مدل هوش مصنوعی Gemini: {exc}"
                        )
                        raise RuntimeError(f"Non-retryable Gemini error: {exc}") from exc

                    logger.warning("Gemini model '%s' failed on attempt %d/%d: %s", current_model, attempt, self.max_attempts, exc)
                    if attempt < self.max_attempts:
                        backoff = (2.0 ** attempt) + random.uniform(0.05, 0.25)
                        time.sleep(backoff)

            if model_unavailable:
                model_idx += 1
                if model_idx < len(candidate_models):
                    logger.info("Falling back from '%s' to '%s'...", current_model, candidate_models[model_idx])
                    continue
            else:
                break

        safe_print(
            f"❌ [۳/۵] خطا: تلاش‌های اتصال به مدل‌های هوش مصنوعی Gemini ناموفق بود. آخرین خطا: {last_error}"
        )
        raise RuntimeError(f"All Gemini attempts failed. Last error: {last_error}")

    def _call_model(self, model_name: str, prompt: str) -> str:
        if genai is not None and self._client is not None and hasattr(self._client, "models"):
            try:
                from src.models import GeminiBatchOutput as _Schema
                config_kwargs: dict[str, Any] = {
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                    "response_schema": _Schema,
                }
                if genai_types is not None:
                    config_kwargs["http_options"] = genai_types.HttpOptions(timeout=int(self.timeout * 1000))
                    if hasattr(genai_types, "AutomaticFunctionCallingConfig"):
                        config_kwargs["automatic_function_calling"] = genai_types.AutomaticFunctionCallingConfig(disable=True)
                if self.system_prompt:
                    config_kwargs["system_instruction"] = self.system_prompt

                if genai_types is not None and hasattr(genai_types, "GenerateContentConfig"):
                    call_config = genai_types.GenerateContentConfig(**config_kwargs)
                else:
                    call_config = config_kwargs

                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=call_config,
                )
                return response.text or ""
            except Exception as exc:
                raise exc

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        try:
            from src.models import GeminiBatchOutput as _Schema
            schema_dict = _Schema.model_json_schema()
        except Exception:
            schema_dict = None

        gen_config: dict[str, Any] = {
            "temperature": self.temperature,
            "responseMimeType": "application/json",
        }
        if schema_dict:
            gen_config["responseSchema"] = schema_dict

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }
        if self.system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": self.system_prompt}]
            }

        with httpx.Client(timeout=self.timeout, trust_env=True) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            if resp.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code} ({model_name}): {resp.text}",
                    request=resp.request,
                    response=resp,
                )
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""