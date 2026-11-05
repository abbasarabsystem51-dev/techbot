"""
main.py — Root CLI entry point for TechNews Bot.
"""
import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
for p in (root_dir, src_dir):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from src.techbot.main import cli_entry

if __name__ == "__main__":
    cli_entry()
