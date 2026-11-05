"""
src/techbot/sources package — Modular news and tool sources.
"""
from src.techbot.sources.base import Source
from src.techbot.sources.fulltext import (
    enrich_top_candidates,
    extract_fulltext,
    extract_fulltext_from_html,
)
from src.techbot.sources.github_search import GitHubSearchSource
from src.techbot.sources.hackernews import HackerNewsSource
from src.techbot.sources.huggingface import HuggingFaceSource
from src.techbot.sources.producthunt import ProductHuntSource
from src.techbot.sources.rss import RSSSource

__all__ = [
    "Source",
    "RSSSource",
    "HackerNewsSource",
    "ProductHuntSource",
    "GitHubSearchSource",
    "HuggingFaceSource",
    "extract_fulltext",
    "extract_fulltext_from_html",
    "enrich_top_candidates",
]
