"""
src/techbot package root.
"""
import sys
from pathlib import Path

# Ensure project root (containing 'src') is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_ROOT_STR = str(_ROOT_DIR)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

try:
    from src.techbot import models, state
except ImportError:
    from techbot import models, state

__all__ = ["models", "state"]
