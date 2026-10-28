"""
src/main.py — نقطه ورود CLI ربات تک‌نیوز

دستورات پشتیبانی‌شده:
  python -m src.main run-once [--dry-run]   واکشی + تولید + انتشار یک‌باره
  python -m src.main serve   [--dry-run]   اجرای مداوم با زمان‌بندی
  python -m src.main status                 نمایش وضعیت فعلی

فلگ --dry-run: اجرا کامل اما بدون ارسال واقعی به تلگرام/ایتا.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import get_config
from src.database import Database
from src.logging_config import get_logger, setup_logging
from src.publisher import Publisher
from src.scheduler import TechBotScheduler
from src.state import get_state_manager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap():
    """
    Shared startup sequence for all CLI commands.

    Returns (cfg, state) ready to use.
    """
    cfg = get_config()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)

    db = Database(cfg.db_path, max_requests_per_day=cfg.gemini_max_requests_per_day)
    state = get_state_manager(db=db)

    # Repair any stale Gemini quota locks from crashed previous runs
    state.repair_stale_locks()

    # Initialize curriculum from JSON file (idempotent)
    curriculum_path: Path = cfg.curriculum_path
    if curriculum_path.exists():
        try:
            raw_data = json.loads(curriculum_path.read_text(encoding="utf-8"))
            lessons = raw_data if isinstance(raw_data, list) else raw_data.get("lessons", [])
            if isinstance(lessons, list) and lessons:
                state.initialize_curriculum(lessons)
                logger.debug("Curriculum initialized from %s (%d lessons).", curriculum_path, len(lessons))
            else:
                logger.warning("curriculum.json does not contain valid lessons — skipping curriculum init.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load curriculum.json: %s", exc)
    else:
        logger.debug("curriculum.json not found at %s — skipping.", curriculum_path)

    return cfg, state


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_run_once(args: argparse.Namespace) -> int:
    """Fetch, generate, and publish posts once, then exit."""
    try:
        cfg, state = _bootstrap()
        publisher = Publisher(cfg, state)
        scheduler = TechBotScheduler(cfg, state, publisher)
        logger.info("Running one-shot pipeline (dry_run=%s)...", args.dry_run)
        scheduler.run_once(dry_run=args.dry_run)
        state.checkpoint()
        logger.info("run-once complete.")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("run-once failed: %s", exc)
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the blocking scheduler and run indefinitely."""
    try:
        cfg, state = _bootstrap()
        publisher = Publisher(cfg, state)
        scheduler = TechBotScheduler(cfg, state, publisher)
        logger.info("Starting serve mode (dry_run=%s)...", args.dry_run)
        scheduler.serve(dry_run=args.dry_run)
        return 0
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("serve failed: %s", exc)
        return 1


def cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Print a human-readable status report and exit."""
    try:
        cfg, state = _bootstrap()
        report = state.format_status_report()
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        print(report)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("status command failed: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="techbot",
        description="TechNews (@tech_news_110) - Automated AI news bot",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Full run without actually sending to Telegram/Eitaa",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="COMMAND",
    )
    subparsers.required = True

    # ── run-once ─────────────────────────────────────────────────────────────
    run_once_parser = subparsers.add_parser(
        "run-once",
        help="Fetch + generate + publish once (suitable for cron job)",
    )
    run_once_parser.set_defaults(func=cmd_run_once)

    # ── serve ─────────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run continuously with daily schedule (blocking)",
    )
    serve_parser.set_defaults(func=cmd_serve)

    # ── status ────────────────────────────────────────────────────────────────
    status_parser = subparsers.add_parser(
        "status",
        help="Print current DB and Gemini quota status",
    )
    status_parser.set_defaults(func=cmd_status)

    for p in (run_once_parser, serve_parser):
        p.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Full run without actually sending to Telegram/Eitaa",
        )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # --dry-run must be forwarded to the subcommand handler
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
