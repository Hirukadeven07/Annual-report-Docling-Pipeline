"""Compare each company's reports across years for step B7."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from docling_pipeline.eda.textutils import normalize_heading

ORDER_CHANGE_LIMIT = 0.7
HEADING_CHANGE_LIMIT = 0.3
PAGE_RATIO_RANGE = (2 / 3, 1.5)


def section_order(sections: Mapping[str, Mapping]) -> list[str]:
    """Return the located target sections in page order."""
    return [key for key, _ in sorted(sections.items(), key=lambda item: item[1]["start"])]


def order_agreement(a: list[str], b: list[str]) -> float | None:
    """Return the share of shared section pairs that keep their relative order, or ``None``."""
    shared = [key for key in a if key in b]
    if len(shared) < 2:
        return None
    position = {key: i for i, key in enumerate(b)}
    pairs = [(x, y) for i, x in enumerate(shared) for y in shared[i + 1:]]
    return sum(position[x] < position[y] for x, y in pairs) / len(pairs)


def heading_overlap(a: list[str], b: list[str]) -> float | None:
    """Return the Jaccard overlap of two lists of headings after normalisation, or ``None``."""
    set_a = {normalize_heading(h) for h in a} - {""}
    set_b = {normalize_heading(h) for h in b} - {""}
    if not set_a and not set_b:
        return None
    return len(set_a & set_b) / len(set_a | set_b)


def layout_changes(reports: pd.DataFrame) -> pd.DataFrame:
    """Compare consecutive reports of each company and flag large layout changes.

    Args:
        reports: One row per report with ``company, year, report_id, structure_source,
            has_bookmarks, page_count, section_order, headings``.

    Returns:
        One row per consecutive pair of years with the comparison measures and a
        ``layout_changed`` flag.
    """
    rows = []
    ordered = reports.dropna(subset=["year"]).sort_values(["company", "year"])
    for company, group in ordered.groupby("company"):
        records = group.to_dict("records")
        for before, after in zip(records, records[1:]):
            order = order_agreement(before["section_order"], after["section_order"])
            headings = heading_overlap(before["headings"], after["headings"])
            page_ratio = after["page_count"] / before["page_count"] if before["page_count"] else None
            reasons = []
            if before["structure_source"] != after["structure_source"]:
                reasons.append(f"structure {before['structure_source']} to {after['structure_source']}")
            if order is not None and order < ORDER_CHANGE_LIMIT:
                reasons.append("section order changed")
            if headings is not None and headings < HEADING_CHANGE_LIMIT:
                reasons.append("headings renamed")
            if page_ratio is not None and not PAGE_RATIO_RANGE[0] <= page_ratio <= PAGE_RATIO_RANGE[1]:
                reasons.append(f"page count x{page_ratio:.2f}")
            rows.append({
                "company": company,
                "from_year": before["year"],
                "to_year": after["year"],
                "from_report": before["report_id"],
                "to_report": after["report_id"],
                "bookmarks_changed": before["has_bookmarks"] != after["has_bookmarks"],
                "order_agreement": order,
                "heading_overlap": headings,
                "page_ratio": page_ratio,
                "layout_changed": bool(reasons),
                "change_reasons": "; ".join(reasons),
            })
    return pd.DataFrame(rows)
