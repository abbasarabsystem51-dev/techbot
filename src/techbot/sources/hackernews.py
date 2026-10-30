"""
src/techbot/sources/hackernews.py — Official Hacker News Firebase API fetcher.

Fetches top/best stories from the official Hacker News Firebase REST API,
capturing hotness signals (score/upvotes and comment counts) to identify trending tech & AI topics.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.models import Item, PostBasket, PostCategory
from src.techbot.http import HttpClient, create_http_client
from src.techbot.sources.base import Source

logger = logging.getLogger(__name__)

HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"
AI_KEYWORDS: tuple[str, ...] = (
    "ai",
    "llm",
    "gpt",
    "gemini",
    "claude",
    "openai",
    "deepmind",
    "anthropic",
    "transformer",
    "diffusion",
    "machine learning",
    "deep learning",
    "neural",
    "rag",
    "agent",
    "robotics",
    "hugging face",
    "pytorch",
)


class HackerNewsSource(Source):
    """
    Hacker News Firebase API source collector.
    """

    def __init__(
        self,
        source_id: str = "hacker_news_api",
        name: str = "Hacker News",
        category: PostCategory = PostCategory.AI,
        basket: Optional[PostBasket] = None,
        weight: float = 1.3,
        enabled: bool = True,
        timeout: float = 10.0,
        story_limit: int = 30,
        min_score: int = 30,
    ) -> None:
        super().__init__(
            source_id=source_id,
            name=name,
            category=category,
            basket=basket,
            weight=weight,
            enabled=enabled,
            timeout=timeout,
        )
        self.story_limit = story_limit
        self.min_score = min_score

    async def _fetch_story(self, client: HttpClient, story_id: int) -> Optional[dict[str, Any]]:
        """Fetch details of an individual HN story."""
        url = f"{HN_BASE_URL}/item/{story_id}.json"
        try:
            data = await client.get_json(url)
            if isinstance(data, dict) and data.get("type") == "story" and not data.get("deleted"):
                return data
        except Exception as exc:
            logger.debug("Failed to fetch HN story %d: %s", story_id, exc)
        return None

    def _is_ai_related(self, title: str, text: str = "") -> bool:
        """Check if story title or text contains AI-related keywords."""
        content = f"{title} {text}".lower()
        # Word boundary or simple token match
        tokens = set(content.replace("-", " ").replace("_", " ").split())
        for kw in AI_KEYWORDS:
            if " " in kw:
                if kw in content:
                    return True
            elif kw in tokens:
                return True
        return False

    async def fetch(self, client: Optional[HttpClient] = None) -> list[Item]:
        """Fetch top story IDs, fetch their details concurrently, and filter."""
        http_client = client or create_http_client(timeout=self.timeout)

        # 1. Fetch top story IDs
        top_ids_url = f"{HN_BASE_URL}/topstories.json"
        story_ids = await http_client.get_json(top_ids_url)
        if not isinstance(story_ids, list):
            logger.warning("HN API topstories did not return a list.")
            return []

        selected_ids = story_ids[: self.story_limit]

        # 2. Fetch details concurrently
        tasks = [self._fetch_story(http_client, sid) for sid in selected_ids]
        stories = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[Item] = []
        for story in stories:
            if not isinstance(story, dict):
                continue

            title = story.get("title", "").strip()
            hn_id = story.get("id")
            score = story.get("score", 0)
            comments = story.get("descendants", 0)
            url = story.get("url") or f"https://news.ycombinator.com/item?id={hn_id}"

            if not title or score < self.min_score:
                continue

            # Publication time
            time_epoch = story.get("time")
            published_at = (
                datetime.fromtimestamp(time_epoch, tz=timezone.utc)
                if time_epoch
                else None
            )

            # Determine category & basket
            is_ai = self._is_ai_related(title, story.get("text", ""))
            category = PostCategory.AI if is_ai else PostCategory.TECH
            basket = PostBasket.NEWS_AI if is_ai else PostBasket.NEWS_TECH

            summary = story.get("text", "") or f"Hacker News discussion: {score} points, {comments} comments."

            item = self.make_item(
                title=title,
                url=url,
                summary=summary,
                published_at=published_at,
                category=category,
                basket=basket,
                raw_data={
                    "hn_id": hn_id,
                    "score": score,
                    "comments": comments,
                    "by": story.get("by", ""),
                },
            )
            items.append(item)

        return items
