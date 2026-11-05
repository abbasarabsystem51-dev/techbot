"""
src/techbot/facts_check.py — Numeric and factual verification against source texts.

Rules:
  1. Normalize Persian/Arabic digits (۰۱۲۳۴۵۶۷۸۹) and symbols (٪, ٫) to standard ASCII.
  2. Extract numbers, prices, percentages, multipliers, and versions from generated posts.
  3. Verify that every non-structural numeric claim exists in the source article or allowed_facts.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from src.models import PostItemSchema

PERSIAN_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# Structural numbers inherently allowed in post templates (e.g., 2-minute try_now, step counts 1..3)
STRUCTURAL_ALLOWED_NUMBERS = frozenset({
    "1", "2", "3", "30", "10",  # steps 1..3, 2-minute test, 30 lessons, 10 posts
})


def normalize_digits_and_symbols(text: str) -> str:
    """Normalize Persian/Arabic digits to ASCII and standardize symbols."""
    if not text:
        return ""
    # Transliterate digits
    converted = text.translate(PERSIAN_TO_ASCII)
    # Standardize percent sign
    converted = converted.replace("٪", "%")
    # Standardize Persian decimal separator (٫) to dot (.)
    converted = re.sub(r"(?<=\d)٫(?=\d)", ".", converted)
    return converted


def extract_numeric_tokens(text: str) -> list[str]:
    """
    Extract numbers, versions, percentages, prices, and multipliers.
    Examples: 50%, $10, 1M, 500K, 3.7, 4o, 1000
    """
    normalized = normalize_digits_and_symbols(text)

    # Patterns for numbers with prefixes/suffixes ($10, 50%, 1.5M, v2.5, 4.0)
    patterns = [
        r"\$\s*\d+(?:\.\d+)?\s*[kmb]?",          # $10, $500k, $1.5M
        r"\d+(?:\.\d+)?\s*%",                    # 50%, 99.9%
        r"\d+(?:\.\d+)?\s*[kmb]\b",              # 1M, 500k, 10B
        r"\bv?\d+\.\d+(?:\.\d+)?\b",             # 3.7, v2.5, 1.5.0
        r"\b\d{2,}\b",                            # raw integers >= 10
    ]

    combined = re.compile("|".join(patterns), re.IGNORECASE)
    matches = combined.findall(normalized)

    # Clean extracted tokens
    tokens: list[str] = []
    for m in matches:
        token = re.sub(r"\s+", "", m.strip().lower())
        if token and token not in STRUCTURAL_ALLOWED_NUMBERS:
            tokens.append(token)

    return sorted(set(tokens))


def verify_facts(
    post_text: str,
    source_text: str,
    allowed_facts: Optional[Sequence[str]] = None,
) -> tuple[bool, list[str]]:
    """
    Check if numeric claims in post_text exist in source_text or allowed_facts.
    Returns: (is_valid, list_of_unverified_tokens)
    """
    tokens = extract_numeric_tokens(post_text)
    if not tokens:
        return True, []

    norm_source = normalize_digits_and_symbols(source_text).lower()
    allowed_normalized = [normalize_digits_and_symbols(f).lower() for f in (allowed_facts or [])]

    unverified: list[str] = []
    for tok in tokens:
        # 1. Direct match in source
        if tok in norm_source:
            continue

        # 2. Match without currency / percent sign
        clean_tok = tok.replace("$", "").replace("%", "").strip()
        if clean_tok in norm_source:
            continue

        # 3. Match in allowed_facts
        if any(tok in af or clean_tok in af for af in allowed_normalized):
            continue

        unverified.append(tok)

    return (len(unverified) == 0, unverified)


def verify_post_item(
    post: PostItemSchema,
    source_text: str,
    allowed_facts: Optional[Sequence[str]] = None,
) -> tuple[bool, list[str]]:
    """
    Verify all fields of a PostItemSchema against source text and allowed facts.
    """
    components = [
        post.hook,
        post.intro,
        " ".join(post.points or []),
        post.what_can_you_do,
        " ".join(post.steps or []),
        post.tip,
        post.try_now,
    ]
    full_post_text = " ".join(c for c in components if c)
    return verify_facts(full_post_text, source_text, allowed_facts=allowed_facts)
