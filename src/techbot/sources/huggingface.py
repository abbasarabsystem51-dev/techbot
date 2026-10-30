"""
src/techbot/sources/huggingface.py — Hugging Face trending papers and models fetcher.

Fetches trending research papers and models from the Hugging Face API,
capturing community upvotes, abstracts, and direct paper links.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.models import Item, PostBasket, PostCategory
from src.techbot.http import HttpClient, create_http_client
from src.techbot.sources.base import Source

logger = logging.getLogger(__name__)

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
HF_MODELS_URL = "https://huggingface.co/api/models"


class HuggingFaceSource(Source):
    """
    Hugging Face research papers and trending models source.
    """

    def __init__(
        self,
        source_id: str = "huggingface_papers",
        name: str = "Hugging Face Daily Papers",
        endpoint: str = HF_DAILY_PAPERS_URL,
        category: PostCategory = PostCategory.AI,
        basket: Optional[PostBasket] = PostBasket.NEWS_AI,
        weight: float = 1.4,
        enabled: bool = True,
        timeout: float = 10.0,
        limit: int = 15,
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
        self.endpoint = endpoint
        self.limit = limit

    async def fetch(self, client: Optional[HttpClient] = None) -> list[Item]:
        """Fetch papers from Hugging Face API and convert to Items."""
        http_client = client or create_http_client(timeout=self.timeout)
        data = await http_client.get_json(self.endpoint)

        if not isinstance(data, list):
            logger.warning("Hugging Face API returned non-list data.")
            return []

        items: list[Item] = []
        for entry in data[: self.limit]:
            if not isinstance(entry, dict):
                continue

            # Hugging Face daily papers structure: entry has "paper" object
            paper = entry.get("paper") if "paper" in entry else entry
            if not isinstance(paper, dict):
                continue

            title = paper.get("title", "").strip()
            paper_id = paper.get("id") or paper.get("_id", "")
            summary = paper.get("summary") or paper.get("description", "")
            upvotes = entry.get("upvotes") or paper.get("upvotes", 0)

            if not title:
                continue

            # Link to Hugging Face papers page
            url = f"https://huggingface.co/papers/{paper_id}" if paper_id else "https://huggingface.co/papers"
            published_str = paper.get("publishedAt")
            published_at = None
            if published_str:
                try:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except Exception:
                    pass

            thumbnail = paper.get("thumbnail") or entry.get("thumbnail")

            item = self.make_item(
                title=f"📄 {title}",
                url=url,
                summary=f"{summary}\n\n👍 Upvotes: {upvotes}",
                published_at=published_at,
                image_url=thumbnail,
                raw_data={
                    "paper_id": paper_id,
                    "upvotes": upvotes,
                    "authors": [a.get("name") for a in paper.get("authors", []) if isinstance(a, dict)],
                },
            )
            items.append(item)

        return items
