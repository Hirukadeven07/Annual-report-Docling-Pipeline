"""Measure report and section size, and sort pages into target and low-value content."""

from __future__ import annotations

import re
from collections.abc import Mapping

from docling_pipeline.eda.structure import NARRATIVE_TARGETS, STATEMENT_TARGETS

PHOTO_MIN_COVER = 0.5
PHOTO_MAX_WORDS = 80
LOW_VALUE_TITLE = re.compile(
    r"notice\s+of\s+(the\s+)?(\w+\s+)?(annual\s+)?general\s+meeting|notice\s+of\s+meeting|form\s+of\s+proxy|proxy\s+form"
    r"|corporate\s+information|corporate\s+profile|company\s+profile|about\s+(us|the\s+company)|our\s+(history|story|journey)"
    r"|milestones|group\s+structure|^(our\s+)?board\s+of\s+directors$|directors'?\s+profiles?|profiles?\s+of\s+(the\s+)?(board|directors)"
    r"|(senior\s+)?management\s+team|glossary|investor\s+feedback",
    re.I,
)
PAGE_CATEGORIES = ["financial statements", "narrative target", "photo page", "low-value", "other"]


def section_of_pages(sections: Mapping[str, Mapping], targets: list[str] | set[str]) -> set[int]:
    """Return every page that lies in one of the given target sections."""
    pages: set[int] = set()
    for key in targets:
        if key in sections:
            pages.update(range(sections[key]["start"], sections[key]["end"] + 1))
    return pages


def page_categories(features: Mapping, sections: Mapping[str, Mapping], segments: list[Mapping]) -> list[str]:
    """Label every page as statements, narrative target, photo page, low-value or other.

    Photo pages win over every other label, since a full-page photo carries no text to chunk.
    Low-value pages are those in a segment titled like an AGM notice, a proxy form or a profile.
    """
    statement_pages = section_of_pages(sections, STATEMENT_TARGETS)
    narrative_pages = section_of_pages(sections, NARRATIVE_TARGETS)
    low_value_pages: set[int] = set()
    for seg in segments:
        if LOW_VALUE_TITLE.search(seg["title"].strip()):
            low_value_pages.update(range(seg["start"], seg["end"] + 1))
    labels = []
    for page in features["pages"]:
        number = page["page"]
        if page["images"] and page["image_cover"] >= PHOTO_MIN_COVER and page["words"] < PHOTO_MAX_WORDS:
            labels.append("photo page")
        elif number in statement_pages:
            labels.append("financial statements")
        elif number in narrative_pages:
            labels.append("narrative target")
        elif number in low_value_pages or any(LOW_VALUE_TITLE.search(h["text"]) for h in page["headings"][:1]):
            labels.append("low-value")
        else:
            labels.append("other")
    return labels


def section_sizes(features: Mapping, sections: Mapping[str, Mapping]) -> list[dict]:
    """Return pages, words and formatted figures for each located target section."""
    by_page = {p["page"]: p for p in features["pages"]}
    rows = []
    for key, section in sections.items():
        pages = [by_page[n] for n in range(section["start"], section["end"] + 1) if n in by_page]
        words = sum(p["words"] for p in pages)
        numbers = sum(p["numbers"] for p in pages)
        rows.append({
            "section": key,
            "pages": len(pages),
            "words": words,
            "numbers": numbers,
            "numeric_density": numbers / words if words else 0.0,
        })
    return rows
