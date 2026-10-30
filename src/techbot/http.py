"""
src/techbot/http.py — Robust HTTP client session with retry, backoff, and safe User-Agent.

Features:
  1. Default timeout = 10.0 seconds (strict per SPEC.md).
  2. Automatic retry with exponential backoff and randomized jitter on network errors and 5xx / 429 status codes.
  3. Safe default User-Agent headers: 'TechNewsBot/1.0 (+https://t.me/tech_news_110)'.
  4. Both async and sync method interfaces for high flexibility across fetchers.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Mapping, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_USER_AGENT = "TechNewsBot/1.0 (+https://t.me/tech_news_110)"
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MAX_DELAY = 10.0
DEFAULT_JITTER = 0.3

# Status codes that warrant an automatic retry
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class HttpClient:
    """
    Robust HTTP client with built-in retry, exponential backoff, and jitter.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        headers: Optional[Mapping[str, str]] = None,
        verify_ssl: bool = True,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.verify_ssl = verify_ssl

        base_headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
            "Accept-Encoding": "gzip, deflate",
        }
        if headers:
            base_headers.update(headers)
        self.headers = base_headers

        self._async_client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff duration with randomized jitter."""
        base = self.initial_delay * (2 ** attempt)
        delay = min(self.max_delay, base)
        jitter = random.uniform(0.01, DEFAULT_JITTER)
        return delay + jitter

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Lazy-loaded httpx.AsyncClient instance."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                headers=self.headers,
                timeout=httpx.Timeout(self.timeout),
                verify=self.verify_ssl,
                follow_redirects=True,
            )
        return self._async_client

    @property
    def sync_client(self) -> httpx.Client:
        """Lazy-loaded synchronous httpx.Client instance."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                headers=self.headers,
                timeout=httpx.Timeout(self.timeout),
                verify=self.verify_ssl,
                follow_redirects=True,
            )
        return self._sync_client

    async def get(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        follow_redirects: bool = True,
    ) -> httpx.Response:
        """
        Asynchronously perform HTTP GET with automatic retries and backoff.
        """
        client = self.async_client
        req_timeout = httpx.Timeout(timeout if timeout is not None else self.timeout)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=req_timeout,
                    follow_redirects=follow_redirects,
                )

                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        "HTTP GET %s returned %d (attempt %d/%d). Retrying in %.2fs...",
                        url,
                        response.status_code,
                        attempt + 1,
                        self.max_retries,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                response.raise_for_status()
                return response

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        "HTTP GET %s failed with %s (attempt %d/%d). Retrying in %.2fs...",
                        url,
                        type(exc).__name__,
                        attempt + 1,
                        self.max_retries,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "HTTP GET %s failed after %d retries: %s",
                        url,
                        self.max_retries,
                        exc,
                    )
                    raise

        if last_error:
            raise last_error
        raise httpx.RequestError(f"HTTP GET {url} failed without explicit response.")

    async def get_text(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Fetch and return response body as decoded text."""
        resp = await self.get(url, params=params, headers=headers, timeout=timeout)
        return resp.text

    async def get_json(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Fetch and parse JSON response."""
        resp = await self.get(url, params=params, headers=headers, timeout=timeout)
        return resp.json()

    async def get_bytes(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Fetch and return response body as raw bytes."""
        resp = await self.get(url, params=params, headers=headers, timeout=timeout)
        return resp.content

    def get_sync(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        follow_redirects: bool = True,
    ) -> httpx.Response:
        """
        Synchronously perform HTTP GET with automatic retries and backoff.
        """
        client = self.sync_client
        req_timeout = httpx.Timeout(timeout if timeout is not None else self.timeout)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=req_timeout,
                    follow_redirects=follow_redirects,
                )

                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    time.sleep(backoff)
                    continue

                response.raise_for_status()
                return response

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    time.sleep(backoff)
                else:
                    raise

        if last_error:
            raise last_error
        raise httpx.RequestError(f"HTTP GET {url} failed without explicit response.")

    def get_text_sync(self, url: str, **kwargs: Any) -> str:
        """Synchronously fetch and return text."""
        return self.get_sync(url, **kwargs).text

    def get_bytes_sync(self, url: str, **kwargs: Any) -> bytes:
        """Synchronously fetch and return bytes."""
        return self.get_sync(url, **kwargs).content

    async def aclose(self) -> None:
        """Close underlying async client."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()

    def close(self) -> None:
        """Close underlying clients."""
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()

    async def __aenter__(self) -> "HttpClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
        self.close()


def create_http_client(**kwargs: Any) -> HttpClient:
    """Factory function for creating an HttpClient instance."""
    return HttpClient(**kwargs)
