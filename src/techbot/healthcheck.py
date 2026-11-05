"""
src/techbot/healthcheck.py — CLI diagnostic utility to verify status and latency of all configured feeds.

Usage:
  python -m techbot.healthcheck
  python -m src.techbot.healthcheck
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from src.models import PostCategory
from src.techbot.http import create_http_client
from src.techbot.sources.base import Source
from src.techbot.sources.github_search import GitHubSearchSource
from src.techbot.sources.hackernews import HackerNewsSource
from src.techbot.sources.huggingface import HuggingFaceSource
from src.techbot.sources.producthunt import ProductHuntSource
from src.techbot.sources.rss import RSSSource

logger = logging.getLogger("techbot.healthcheck")


def load_configured_sources(config_path: Optional[Path] = None) -> list[Source]:
    """Load sources defined in config.yaml and instantiate appropriate Source classes."""
    root = Path(__file__).resolve().parents[2]
    cfg_file = config_path or (root / "config.yaml")

    sources: list[Source] = []
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        for src_cfg in cfg.get("sources", []):
            if not src_cfg.get("enabled", True):
                continue
            cat = PostCategory.AI if src_cfg.get("category") == "ai" else PostCategory.TECH
            sources.append(
                RSSSource(
                    source_id=src_cfg.get("id", "rss_feed"),
                    name=src_cfg.get("name", "RSS Feed"),
                    url=src_cfg.get("url", ""),
                    category=cat,
                    weight=float(src_cfg.get("weight", 1.0)),
                    timeout=float(cfg.get("fetching", {}).get("timeout_seconds", 10.0)),
                )
            )

    # Add modular sources if not already present
    existing_ids = {s.source_id for s in sources}
    if "hacker_news_api" not in existing_ids:
        sources.append(HackerNewsSource(source_id="hacker_news_api", name="Hacker News API"))
    if "product_hunt" not in existing_ids:
        sources.append(ProductHuntSource(source_id="product_hunt", name="Product Hunt"))
    if "github_trending_ai" not in existing_ids:
        sources.append(GitHubSearchSource(source_id="github_trending_ai", name="GitHub Search AI"))
    if "huggingface_papers" not in existing_ids:
        sources.append(HuggingFaceSource(source_id="huggingface_papers", name="Hugging Face Daily"))

    return sources


async def check_source_health(source: Source) -> dict[str, Any]:
    """Test a single source and record status, item count, and latency."""
    client = create_http_client(timeout=source.timeout)
    start_time = time.perf_counter()
    status = "OK"
    items_count = 0
    error_msg = ""

    try:
        items = await source.fetch(client=client)
        items_count = len(items)
        if items_count == 0:
            status = "EMPTY"
    except Exception as exc:
        status = "FAIL"
        error_msg = str(exc)[:80]
    finally:
        await client.aclose()

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "source_id": source.source_id,
        "name": source.name,
        "type": type(source).__name__,
        "status": status,
        "items": items_count,
        "latency_ms": round(latency_ms, 1),
        "error": error_msg,
    }


async def run_healthcheck(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Run healthcheck on all sources and print formatted summary."""
    sources = load_configured_sources(config_path)
    print(f"\n🔍 Testing {len(sources)} news and tool sources...\n")

    tasks = [check_source_health(src) for src in sources]
    results = await asyncio.gather(*tasks)

    # Print ASCII table
    header = f"{'Source ID':<22} | {'Name':<22} | {'Status':<6} | {'Items':<5} | {'Latency':<8} | {'Note'}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    healthy_count = 0
    for r in results:
        status_str = r["status"]
        if status_str in ("OK", "EMPTY"):
            healthy_count += 1
        note = r["error"] if r["error"] else ("No items parsed" if status_str == "EMPTY" else "Healthy")
        latency_str = f"{r['latency_ms']}ms"
        print(
            f"{r['source_id']:<22} | {r['name'][:20]:<22} | {status_str:<6} | {r['items']:<5} | {latency_str:<8} | {note}"
        )

    print(separator)
    print(f"📊 Summary: {healthy_count}/{len(results)} sources operational.\n")

    return {
        "total": len(results),
        "healthy": healthy_count,
        "failed": len(results) - healthy_count,
        "results": results,
    }


def main() -> None:
    """CLI entry point for python -m techbot.healthcheck."""
    summary = asyncio.run(run_healthcheck())
    # Exit with error only if all sources failed
    sys.exit(0 if summary["healthy"] > 0 else 1)


if __name__ == "__main__":
    main()
