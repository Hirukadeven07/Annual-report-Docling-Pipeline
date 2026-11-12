"""Combine the screening checks (A1 to A5) per report and assign each report a tier."""

from __future__ import annotations

import math
from collections.abc import Mapping

from docling_pipeline.eda.pdf_features import IMAGE_ONLY_MAX_CHARS, IMAGE_ONLY_MIN_COVER

TEXT_PASS_RATIO = 0.95
TEXT_FAIL_RATIO = 0.5
RELIABLE_SOURCES = {"bookmarks", "contents_page"}
CORE_STATEMENTS = ("income_statement", "balance_sheet", "cash_flow")
STATEMENT_KINDS = ("income_statement", "balance_sheet", "cash_flow", "changes_in_equity")
GOOD_PARSE_RATE = 0.9
GOOD_CHECK_SHARE = 0.5
STATEMENT_SECTION_PAGES = 3


def page_ranges(pages: list[int]) -> str:
    """Format page numbers compactly, for example ``3-5, 9``."""
    ranges: list[list[int]] = []
    for page in sorted(set(pages)):
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1][1] = page
        else:
            ranges.append([page, page])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def is_image_only(page: Mapping) -> bool:
    """Return True for a page with almost no text that is mostly covered by images."""
    return page["chars"] < IMAGE_ONLY_MAX_CHARS and page["image_cover"] >= IMAGE_ONLY_MIN_COVER


def text_layer_check(features: Mapping) -> dict:
    """Classify a report's text layer as ``pass``, ``partial`` or ``fail`` (step A1)."""
    image_only = [p["page"] for p in features["pages"] if is_image_only(p)]
    page_count = max(1, features["page_count"])
    ratio = 1 - len(image_only) / page_count
    if ratio >= TEXT_PASS_RATIO:
        status = "pass"
    elif ratio >= TEXT_FAIL_RATIO:
        status = "partial"
    else:
        status = "fail"
    return {
        "text_page_ratio": round(ratio, 3),
        "image_only_count": len(image_only),
        "image_only_pages": page_ranges(image_only),
        "chars_per_page": round(sum(p["chars"] for p in features["pages"]) / page_count),
        "text_status": status,
    }


def table_profile(tables: list[Mapping]) -> dict:
    """Summarise the Docling tables of one report (step A4).

    The mean numeric parse rate weights each table by its number of digit cells, so a
    two-cell table does not count as much as a full statement.
    """
    digit_cells = sum(t["digit_cells"] for t in tables if t["numeric_parse_rate"] is not None)
    parsed = sum(t["numeric_parse_rate"] * t["digit_cells"] for t in tables if t["numeric_parse_rate"] is not None)
    consistency = [t["row_consistency"] for t in tables if t["row_consistency"] is not None]
    return {
        "table_count": len(tables),
        "mean_numeric_parse_rate": round(parsed / digit_cells, 4) if digit_cells else None,
        "empty_cell_ratio": round(sum(t["empty_cell_ratio"] for t in tables) / len(tables), 4) if tables else None,
        "row_consistency": round(sum(consistency) / len(consistency), 4) if consistency else None,
        "multi_page_tables": sum(1 for t in tables if len(t["pages"]) > 1),
    }


def statement_pages(features: Mapping, sections: Mapping[str, Mapping], kind: str) -> list[int]:
    """Return the pages where a statement was found, by the PDF text layer or by its section heading."""
    pages = set(features["statement_primary"].get(kind, []))
    section = sections.get(kind)
    if section:
        last = min(section["end"], section["start"] + STATEMENT_SECTION_PAGES - 1)
        pages.update(range(section["start"], last + 1))
    return sorted(pages)


def image_statements(features: Mapping, sections: Mapping[str, Mapping], table_pages: set[int] | None = None) -> list[str]:
    """List the financial statements that look like images instead of text tables.

    With Docling output, a statement is flagged when none of its pages holds a Docling
    table. Without it, a statement is flagged when its heading was found but the PDF text
    layer has no page with statement figures for it.
    """
    flagged = []
    for kind in STATEMENT_KINDS:
        primary = features["statement_primary"].get(kind, [])
        located = statement_pages(features, sections, kind)
        if table_pages is not None:
            if located and not set(located) & table_pages:
                flagged.append(kind)
        elif not primary and (features["statement_image"].get(kind) or kind in sections):
            flagged.append(kind)
    return flagged


def _missing(value: object) -> bool:
    """Return True for ``None`` and NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def tables_ok(record: Mapping) -> bool | None:
    """Judge table extraction from the parse rate and the accounting checks.

    Returns:
        True or False, or ``None`` when there is no evidence either way.
    """
    rate = record.get("mean_numeric_parse_rate")
    run = int(record.get("accounting_checks_run") or 0)
    passed = int(record.get("accounting_checks_passed") or 0)
    checks_ok = run > 0 and passed / run >= GOOD_CHECK_SHARE
    if not _missing(rate):
        return rate >= GOOD_PARSE_RATE and (run == 0 or checks_ok)
    return checks_ok if run else None


def assign_tier(record: Mapping) -> tuple[str, str]:
    """Assign tier A, B or C to one screened report.

    Args:
        record: One row of the screening table, with ``text_status``, ``structure_source``,
            ``core_statements_found``, ``image_statements`` and the table evidence.

    Returns:
        The tier and the reasons it is not tier A, joined by ``; ``.
    """
    image = [k for k in record.get("image_statements") or [] if k in CORE_STATEMENTS]
    if record["text_status"] == "fail":
        return "C", "mostly image-only pages (scanned)"
    if image:
        return "C", "statement is an image: " + ", ".join(image)
    reasons = []
    if record["structure_source"] not in RELIABLE_SOURCES:
        reasons.append(f"weak structure ({record['structure_source']})")
    if int(record.get("core_statements_found") or 0) < len(CORE_STATEMENTS):
        reasons.append("core statements not all located")
    quality = tables_ok(record)
    if quality is None:
        reasons.append("table quality unverified")
    elif not quality:
        reasons.append("table checks failed")
    return ("B" if reasons else "A"), "; ".join(reasons)
