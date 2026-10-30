"""
src/techbot/sources/github_search.py — GitHub Search API fetcher for trending AI & tech repos.

Searches GitHub API for trending or recently published open-source AI and developer repositories,
capturing stars, language, and topics as hotness and quality signals.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.models import Item, PostBasket, PostCategory
from src.techbot.http import HttpClient, create_http_client
from src.techbot.sources.base import Source

logger = logging.getLogger(__name__)

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
DEFAULT_QUERY = "topic:artificial-intelligence topic:llm stars:>100"


class GitHubSearchSource(Source):
    """
    GitHub repository search source.
    """

    def __init__(
        self,
        source_id: str = "github_trending_ai",
        name: str = "GitHub Trending AI",
        query: str = DEFAULT_QUERY,
        category: PostCategory = PostCategory.AI,
        basket: Optional[PostBasket] = PostBasket.TOOL_AI,
        weight: float = 1.2,
        enabled: bool = True,
        timeout: float = 10.0,
        limit: int = 20,
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
        self.query = query
        self.limit = limit

    async def fetch(self, client: Optional[HttpClient] = None) -> list[Item]:
        """Query GitHub Search API and convert repositories into candidate items."""
        http_client = client or create_http_client(timeout=self.timeout)
        headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        params = {
            "q": self.query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(self.limit, 50),
        }

        data = await http_client.get_json(GITHUB_SEARCH_URL, params=params, headers=headers)
        if not isinstance(data, dict):
            logger.warning("GitHub Search API returned unexpected non-dict structure.")
            return []

        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            return []

        items: list[Item] = []
        for repo in raw_items[: self.limit]:
            if not isinstance(repo, dict):
                continue

            full_name = repo.get("full_name") or repo.get("name", "")
            html_url = repo.get("html_url", "")
            description = repo.get("description") or ""
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            language = repo.get("language") or "Code"
            topics = repo.get("topics", [])

            if not full_name or not html_url:
                continue

            # Publication time
            created_str = repo.get("created_at") or repo.get("updated_at")
            published_at = None
            if created_str:
                try:
                    published_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                except Exception:
                    pass

            title = f"{full_name}: {description[:90]}" if description else full_name
            summary = (
                f"{description}\n\n"
                f"⭐ Stars: {stars:,} | 🍴 Forks: {forks:,} | 💻 Language: {language}"
            )

            # Assign category & basket
            is_ai = any("ai" in t.lower() or "llm" in t.lower() for t in topics) or "ai" in description.lower()
            category = PostCategory.AI if is_ai else PostCategory.TECH
            basket = PostBasket.TOOL_AI if is_ai else PostBasket.TOOL_TECH

            # Repo owner avatar as fallback image
            owner = repo.get("owner", {})
            avatar_url = owner.get("avatar_url") if isinstance(owner, dict) else None

            item = self.make_item(
                title=title,
                url=html_url,
                summary=summary,
                published_at=published_at,
                category=category,
                basket=basket,
                image_url=avatar_url,
                raw_data={
                    "stars": stars,
                    "forks": forks,
                    "language": language,
                    "topics": topics,
                    "full_name": full_name,
                },
            )
            items.append(item)

        return items
