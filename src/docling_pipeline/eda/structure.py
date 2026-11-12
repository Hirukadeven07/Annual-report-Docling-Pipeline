"""Find each report's best structure source and locate the target sections in it."""

from __future__ import annotations

from collections import Counter

from rapidfuzz import fuzz

from docling_pipeline.eda.textutils import normalize_heading

MATCH_THRESHOLD = 85
STATEMENT_FLOOR_SHARE = 0.15

TARGET_SECTIONS: dict[str, list[str]] = {
    "chairman_message": [
        "chairman message", "chairman review", "chairman statement", "chairman letter", "chairperson message",
        "chairperson statement", "chairperson review", "message from the chairman", "letter from the chairman",
        "chairman address", "letter to shareholders",
    ],
    "management_discussion": [
        "management discussion and analysis", "management discussion & analysis", "md&a", "ceo review",
        "chief executive officer review", "managing director review", "group managing director review",
        "ceo message", "managing director message", "operational review", "operating review", "business review",
        "financial review", "chief financial officer review", "cfo review",
    ],
    "income_statement": [
        "statement of profit or loss", "statement of profit or loss and other comprehensive income", "income statement",
        "statement of comprehensive income", "statement of profit and loss", "profit and loss account", "statement of income",
    ],
    "balance_sheet": ["statement of financial position", "balance sheet"],
    "cash_flow": ["statement of cash flows", "cash flow statement", "statement of cash flow"],
    "changes_in_equity": ["statement of changes in equity"],
    "notes": ["notes to the financial statements", "notes to financial statements", "notes to the accounts"],
    "auditor_report": ["independent auditor report", "auditor report", "report of the auditors", "independent auditors report"],
    "risk": ["risk management", "enterprise risk management", "risk management report", "risk", "risks", "managing risk", "risks and opportunities", "risk review"],
    "governance": ["corporate governance", "corporate governance report", "governance", "governance report", "governance framework"],
    "sustainability": [
        "sustainability report", "sustainability", "corporate social responsibility", "csr", "esg report", "esg",
        "environmental social and governance", "social and relationship capital", "natural capital",
        "sustainable development", "sustainability review", "our approach to sustainability",
    ],
}

SECTION_LABELS = {
    "chairman_message": "Chairman's message",
    "management_discussion": "Management discussion",
    "income_statement": "Income statement",
    "balance_sheet": "Balance sheet",
    "cash_flow": "Cash flow statement",
    "changes_in_equity": "Changes in equity",
    "notes": "Notes",
    "auditor_report": "Auditor's report",
    "risk": "Risk",
    "governance": "Governance",
    "sustainability": "Sustainability",
}
STATEMENT_TARGETS = {"income_statement", "balance_sheet", "cash_flow", "changes_in_equity", "notes", "auditor_report"}
NARRATIVE_TARGETS = ["chairman_message", "management_discussion", "risk", "governance", "sustainability"]

_VARIANTS = [(target, normalize_heading(v)) for target, variants in TARGET_SECTIONS.items() for v in variants]


def match_heading(heading: str, threshold: int = MATCH_THRESHOLD) -> tuple[str | None, float]:
    """Match a heading to a target section with fuzzy matching.

    Multi-word variants also match headings that contain all of their words (for example
    ``Statement of Profit or Loss and Other Comprehensive Income``). Ties go to the variant
    with more words, which is the more specific one.

    Returns:
        The target key and score, or ``(None, best score)`` when below the threshold.
    """
    text = normalize_heading(heading)
    tokens = text.split()
    if not tokens or len(tokens) > 14:
        return None, 0.0
    best: tuple[float, int, str | None] = (0.0, 0, None)
    for target, variant in _VARIANTS:
        score = fuzz.ratio(text, variant)
        variant_tokens = variant.split()
        if len(variant_tokens) >= 2 and len(set(tokens) - set(variant_tokens)) <= 5:
            score = max(score, fuzz.token_set_ratio(text, variant))
        candidate = (score, len(variant_tokens), target)
        if candidate[:2] > best[:2]:
            best = candidate
    return (best[2], best[0]) if best[0] >= threshold else (None, best[0])


def folio_offset(pages: list[dict]) -> tuple[int | None, float]:
    """Return the most common gap between PDF page and printed page number, and its share of pages."""
    gaps = Counter(p["page"] - p["folio"] for p in pages if p.get("folio") is not None)
    if not gaps:
        return None, 0.0
    offset, count = gaps.most_common(1)[0]
    return offset, count / len(pages)


def _page_headings(pages: list[dict]) -> dict[int, list[str]]:
    """Map each page to its large-font headings."""
    return {p["page"]: [h["text"] for h in p["headings"]] for p in pages}


def contents_alignment(entries: list[list], pages: list[dict]) -> tuple[int | None, float]:
    """Find the offset that maps printed contents pages to PDF pages, and the share of entries it explains.

    An entry counts as aligned when a large-font heading on the mapped page, or the page after,
    resembles the entry title. When headings are too sparse, a consistent printed folio is used.
    """
    if not entries:
        return None, 0.0
    headings = _page_headings(pages)
    gaps: Counter = Counter()
    for title, printed in entries:
        for page, texts in headings.items():
            if abs(page - printed) <= 12 and any(fuzz.token_set_ratio(normalize_heading(title), normalize_heading(t)) >= 85 for t in texts):
                gaps[page - printed] += 1
    folio, folio_share = folio_offset(pages)
    if gaps:
        offset = gaps.most_common(1)[0][0]
        aligned = 0
        for title, printed in entries:
            near = headings.get(printed + offset, []) + headings.get(printed + offset + 1, [])
            if any(fuzz.token_set_ratio(normalize_heading(title), normalize_heading(t)) >= 85 for t in near):
                aligned += 1
        share = aligned / len(entries)
        if folio is not None and folio == offset:
            share = max(share, folio_share)
        return offset, share
    return folio, folio_share


def structure_profile(features: dict, docling_headers: list[tuple[int, str]] | None = None) -> dict:
    """Summarise the structure sources of one report and choose the best one (step A2)."""
    bookmarks = features["bookmarks"]
    entries = features["contents_entries"]
    offset, aligned = contents_alignment(entries, features["pages"])
    bookmark_pages = {b[2] for b in bookmarks if b[2] > 0}
    has_bookmarks = len(bookmarks) >= 5 and len(bookmark_pages) >= 5
    has_contents = bool(features["contents_pages"]) and len(entries) >= 6
    docling_count = None if docling_headers is None else len(docling_headers)
    if has_bookmarks:
        source = "bookmarks"
    elif has_contents and aligned >= 0.5:
        source = "contents_page"
    elif docling_count is not None and docling_count >= 10:
        source = "docling_only"
    else:
        source = "none"
    return {
        "has_bookmarks": has_bookmarks,
        "bookmark_count": len(bookmarks),
        "bookmark_levels": max((b[0] for b in bookmarks), default=0),
        "has_contents_page": has_contents,
        "contents_entry_count": len(entries),
        "page_offset": offset,
        "contents_aligned_share": round(aligned, 3),
        "docling_header_count": docling_count,
        "structure_source": source,
    }


def _runs(page_titles: dict[int, str], page_count: int) -> list[dict]:
    """Merge consecutive pages that share a heading into segments; pages without a heading continue the last one."""
    segments: list[dict] = []
    for page in range(1, page_count + 1):
        title = page_titles.get(page)
        if title and (not segments or normalize_heading(title) != normalize_heading(segments[-1]["title"])):
            segments.append({"title": title, "start": page, "end": page, "level": 1})
        elif segments:
            segments[-1]["end"] = page
    return segments


def section_segments(features: dict, source: str, docling_headers: list[tuple[int, str]] | None = None) -> list[dict]:
    """Return the report's sections as ``{title, start, end, level}`` from the chosen structure source.

    ``none`` falls back to the largest heading on each page of the PDF text layer.
    """
    page_count = features["page_count"]
    if source == "bookmarks":
        marks = [b for b in features["bookmarks"] if b[2] > 0]
        segments = []
        for i, (level, title, page) in enumerate(marks):
            later = [m[2] for m in marks[i + 1:] if m[0] <= level and m[2] >= page]
            end = (later[0] - 1) if later else page_count
            segments.append({"title": title, "start": page, "end": max(page, end), "level": level})
        return segments
    if source == "contents_page":
        offset, _ = contents_alignment(features["contents_entries"], features["pages"])
        mapped = sorted(
            ((printed + (offset or 0), title) for title, printed in features["contents_entries"]),
            key=lambda item: item[0],
        )
        mapped = [(p, t) for p, t in mapped if 1 <= p <= page_count]
        segments = []
        for i, (page, title) in enumerate(mapped):
            later = [p for p, _ in mapped[i + 1:] if p > page]
            segments.append({"title": title, "start": page, "end": (later[0] - 1) if later else page_count, "level": 1})
        return segments
    if source == "docling_only" and docling_headers:
        first_per_page: dict[int, str] = {}
        for page, text in docling_headers:
            first_per_page.setdefault(page, text)
        return _runs(first_per_page, page_count)
    largest = {p["page"]: p["headings"][0]["text"] for p in features["pages"] if p["headings"]}
    return _runs(largest, page_count)


def locate_sections(segments: list[dict], page_count: int, source: str, threshold: int = MATCH_THRESHOLD) -> tuple[dict[str, dict], list[str]]:
    """Match segments to target sections and return their page ranges (step A3).

    For heading-based sources, statement targets are only accepted after the front matter, so
    highlight pages such as ``Balance sheet highlights`` are not mistaken for the statements.

    Returns:
        Target key mapped to ``{start, end, heading, score}`` for the first match, and the
        normalised titles that matched no target.
    """
    found: dict[str, dict] = {}
    unmatched: list[str] = []
    floor = page_count * STATEMENT_FLOOR_SHARE if source not in ("bookmarks", "contents_page") else 0
    for seg in sorted(segments, key=lambda s: s["start"]):
        target, score = match_heading(seg["title"], threshold)
        if target is None:
            unmatched.append(normalize_heading(seg["title"]))
            continue
        if target in STATEMENT_TARGETS and seg["start"] < floor:
            continue
        if target not in found:
            found[target] = {"start": seg["start"], "end": seg["end"], "heading": seg["title"], "score": round(score, 1)}
        elif seg["start"] <= found[target]["end"] + 1 and seg["end"] > found[target]["end"]:
            found[target]["end"] = seg["end"]
    return found, unmatched
