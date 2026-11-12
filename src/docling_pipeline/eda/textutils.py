"""Parse accounting numbers and normalise heading text."""

from __future__ import annotations

import re

NUMERIC_TOKEN = re.compile(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\d+\.\d+\)?")
AMOUNT = re.compile(r"^\(?(-)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?%?$")
NIL_TOKENS = {"-", "–", "—", "--", "nil", "‒"}
UNIT_NOISE = re.compile(r"(rs\.?|lkr|usd|mn|bn|'000|’000|000s|\*)", re.I)
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def parse_amount(token: str) -> float | None:
    """Parse one accounting-format cell such as ``(1,234)``, ``-`` or ``12.5``.

    Args:
        token: Raw cell or word text.

    Returns:
        The value, ``0.0`` for nil markers, or ``None`` when the text is not a number.
    """
    text = token.strip().replace(" ", "")
    if text.lower() in NIL_TOKENS:
        return 0.0
    match = AMOUNT.match(text)
    if not match:
        return None
    value = float(match.group(2).replace(",", "") + (match.group(3) or ""))
    negative = text.startswith("(") and text.endswith(")") or bool(match.group(1))
    return -value if negative else value


def parse_cell(text: str) -> float | None:
    """Parse a table cell after removing currency and unit labels."""
    cleaned = UNIT_NOISE.sub("", text or "").strip()
    if not cleaned:
        return None
    return parse_amount(cleaned)


def count_numbers(text: str) -> int:
    """Count formatted figures (thousand separators or decimals) in text."""
    return len(NUMERIC_TOKEN.findall(text))


def collapse_letter_spacing(text: str) -> str:
    """Join letter-spaced headings such as ``S T A T E M E N T`` back into words."""
    tokens = text.split(" ")
    singles = sum(1 for t in tokens if len(t) == 1)
    if len(tokens) < 6 or singles / len(tokens) < 0.6:
        return text
    return re.sub(r"(?<=\S) (?=\S)", "", re.sub(r" {2,}", "\u0000", text)).replace("\u0000", " ")


def normalize_heading(text: str) -> str:
    """Lowercase a heading and strip quotes, numbering and punctuation for matching."""
    text = collapse_letter_spacing(text.translate(QUOTES)).lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"^\s*(page\s*)?[\divx.]+\s+", "", text)
    text = re.sub(r"[^a-z&]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
