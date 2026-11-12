"""Text similarity, repeated passages and sector vocabulary for steps B5 and B6."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

WORD = re.compile(r"[a-z][a-z'&-]*[a-z]|[a-z]")
PASSAGE_MIN_WORDS = 25
SHINGLE_SIZE = 5
TERM_MIN_LENGTH = 3
REPORT_STOP_WORDS = {
    "annual", "report", "page", "pages", "year", "years", "ended", "rs", "lkr", "mn", "bn", "million", "billion",
    "000", "plc", "ltd", "limited", "pvt", "sri", "lanka", "lankan", "colombo", "march", "december", "april",
    "january", "june", "september", "also", "per", "cent", "note", "notes", "group", "company", "companies",
}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | REPORT_STOP_WORDS


def tokens(text: str) -> list[str]:
    """Lowercase text and split it into words, ignoring digits and punctuation."""
    return WORD.findall(text.lower().replace("’", "'"))


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[int]:
    """Return hashed word n-grams of a text.

    The hashes come from Python's ``hash``, so they are only comparable within one process.
    """
    words = tokens(text)
    return {hash(gram) for gram in zip(*(words[i:] for i in range(size)))}


def jaccard(a: set, b: set) -> float | None:
    """Return the Jaccard similarity of two sets, or ``None`` if both are empty."""
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def passage_key(text: str) -> str | None:
    """Normalise a passage for exact-repeat detection, or return ``None`` if it is too short.

    Digits become ``#`` so passages that differ only in years or amounts still match.
    """
    words = re.sub(r"\d", "#", text.lower()).split()
    if len(words) < PASSAGE_MIN_WORDS:
        return None
    return re.sub(r"[^a-z# ]+", "", " ".join(words)).strip()


def term_counts(text: str, min_count: int = 1) -> Counter:
    """Count content words and two-word phrases, skipping stop words.

    Args:
        text: Text to count.
        min_count: Drop terms used fewer times than this, to keep corpus totals small.
    """
    words = tokens(text)
    counts: Counter = Counter()
    keep = [w if len(w) >= TERM_MIN_LENGTH and w not in STOP_WORDS else None for w in words]
    counts.update(w for w in keep if w)
    counts.update(f"{a} {b}" for a, b in zip(keep, keep[1:]) if a and b)
    if min_count > 1:
        counts = Counter({term: n for term, n in counts.items() if n >= min_count})
    return counts


def distinctive_terms(
    counts_by_class: Mapping[str, Counter],
    top_n: int = 15,
    min_count: int = 20,
    allowed: Mapping[str, set[str]] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Rank the terms that mark each class out from the others.

    A term scores ``p * log(p / q)``, where ``p`` is its share of the class's terms and ``q`` its
    share of all terms. This is a TF-IDF-like weight: it grows with use in the class and is zero
    for terms used at the same rate everywhere, so generic words such as "financial" drop out.

    Args:
        counts_by_class: Term counts per class, for example per sector.
        top_n: Number of terms to return per class.
        min_count: Ignore terms used fewer times than this in the class.
        allowed: Optional terms each class may return, for example terms used by several of
            its companies, so one company's name does not top its sector. Shares are still
            computed over all terms.

    Returns:
        Class mapped to ``(term, score)`` pairs, best first.
    """
    totals = {name: sum(counts.values()) for name, counts in counts_by_class.items()}
    overall: Counter = Counter()
    for counts in counts_by_class.values():
        overall.update(counts)
    grand_total = sum(totals.values())
    ranked = {}
    for name, counts in counts_by_class.items():
        scores = []
        for term, count in counts.items():
            if count < min_count or (allowed is not None and term not in allowed.get(name, set())):
                continue
            share = count / totals[name]
            ratio = share / (overall[term] / grand_total)
            if ratio > 1:
                scores.append((term, share * math.log(ratio)))
        ranked[name] = sorted(scores, key=lambda item: -item[1])[:top_n]
    return ranked


class PassageCounter:
    """Count how many companies use each passage word for word, keeping one example per repeated passage."""

    def __init__(self, keep_example_at: int = 3) -> None:
        """Start an empty counter.

        Args:
            keep_example_at: Store a passage's text once this many companies use it.
        """
        self.keep_example_at = keep_example_at
        self.companies: dict[int, set[str]] = {}
        self.examples: dict[int, str] = {}

    def add(self, company: str, passages: Iterable[str]) -> None:
        """Record the passages of one report."""
        for text in passages:
            normalised = passage_key(text)
            if normalised is None:
                continue
            key = hash(normalised)
            seen = self.companies.setdefault(key, set())
            seen.add(company)
            if len(seen) >= self.keep_example_at and key not in self.examples:
                self.examples[key] = text

    def repeated(self, min_companies: int) -> list[dict]:
        """Return ``{companies, words, text}`` for passages used by at least ``min_companies``, most widespread first."""
        rows = [
            {"companies": len(self.companies[key]), "words": len(text.split()), "text": " ".join(text.split())}
            for key, text in self.examples.items()
            if len(self.companies[key]) >= min_companies
        ]
        return sorted(rows, key=lambda r: (-r["companies"], -r["words"]))
