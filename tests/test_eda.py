"""Tests for the EDA helpers, using small synthetic reports instead of real PDFs."""

from __future__ import annotations

import pandas as pd
import pytest

from docling_pipeline.eda.chunking import chunking_damage, elements_from_blocks, split_table_example
from docling_pipeline.eda.content import page_categories, section_sizes
from docling_pipeline.eda.docling_features import continuation_flags
from docling_pipeline.eda.ground_truth import FIGURES, classify_figure, compare_ground_truth, ground_truth_template, stratified_sample
from docling_pipeline.eda.metadata import parse_year
from docling_pipeline.eda.screening import assign_tier, image_statements, page_ranges, table_profile, tables_ok, text_layer_check
from docling_pipeline.eda.statements import accounting_checks
from docling_pipeline.eda.textstats import PassageCounter, distinctive_terms, jaccard, passage_key, shingles, term_counts
from docling_pipeline.eda.textutils import parse_amount
from docling_pipeline.eda.timeline import layout_changes, order_agreement


def make_page(number: int, chars: int = 2000, words: int = 300, numbers: int = 0, images: int = 0, cover: float = 0.0, heading: str | None = None) -> dict:
    """Build one cached page record."""
    return {
        "page": number, "chars": chars, "words": words, "numbers": numbers, "images": images, "image_cover": cover,
        "headings": [{"text": heading, "size": 20.0, "y": 0.1}] if heading else [], "folio": None, "continued": False,
    }


def make_features(pages: list[dict], primary: dict | None = None, image: dict | None = None) -> dict:
    """Build a cached feature record around the given pages."""
    kinds = ("income_statement", "balance_sheet", "cash_flow", "changes_in_equity")
    return {
        "report_id": "demo_2024",
        "page_count": len(pages),
        "pages": pages,
        "statement_primary": {k: (primary or {}).get(k, []) for k in kinds},
        "statement_image": {k: (image or {}).get(k, []) for k in kinds},
    }


def screening_record(**overrides) -> dict:
    """Build a screening row that reaches tier A unless overridden."""
    record = {
        "text_status": "pass", "structure_source": "bookmarks", "core_statements_found": 3, "image_statements": [],
        "mean_numeric_parse_rate": None, "accounting_checks_run": 4, "accounting_checks_passed": 4,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(("token", "expected"), [("(1,234)", -1234.0), ("-", 0.0), ("12.5", 12.5), ("2,000,000", 2e6), ("abc", None)])
def test_parse_amount_handles_accounting_formats(token: str, expected: float | None) -> None:
    """Parse negatives in brackets, nil dashes and thousand separators."""
    assert parse_amount(token) == expected


@pytest.mark.parametrize(("stem", "year"), [("johnkeells_2024", 2024), ("Report-2021-22", 2022), ("Abans-FY24-25", 2025), ("no_year", None)])
def test_parse_year_uses_the_year_the_period_ends(stem: str, year: int | None) -> None:
    """Read the financial year from the different file name styles."""
    assert parse_year(stem) == year


def test_page_ranges_collapses_runs() -> None:
    """Format consecutive pages as ranges."""
    assert page_ranges([9, 3, 4, 5, 12]) == "3-5, 9, 12"


def test_text_layer_check_classifies_scans() -> None:
    """A report of image-only pages fails, and a few photo pages make it partial."""
    scan = make_features([make_page(i, chars=0, words=0, images=1, cover=1.0) for i in range(1, 11)])
    assert text_layer_check(scan)["text_status"] == "fail"
    pages = [make_page(i) for i in range(1, 19)] + [make_page(19, chars=0, images=1, cover=0.9), make_page(20, chars=0, images=1, cover=0.9)]
    result = text_layer_check(make_features(pages))
    assert result["text_status"] == "partial"
    assert result["image_only_pages"] == "19-20"


def test_image_statements_without_docling_needs_a_heading_and_no_figures() -> None:
    """Flag a statement whose heading was found but whose figures are missing from the text layer."""
    features = make_features([make_page(i) for i in range(1, 11)], primary={"income_statement": [6]})
    sections = {"balance_sheet": {"start": 7, "end": 7}, "income_statement": {"start": 6, "end": 6}}
    assert image_statements(features, sections) == ["balance_sheet"]


def test_image_statements_with_docling_checks_table_pages() -> None:
    """With Docling output, flag a located statement that has no table on its pages."""
    features = make_features([make_page(i) for i in range(1, 11)], primary={"income_statement": [6], "balance_sheet": [7]})
    assert image_statements(features, {}, table_pages={6}) == ["balance_sheet"]


def test_table_profile_weights_parse_rate_by_digit_cells() -> None:
    """A large clean table outweighs a small messy one."""
    tables = [
        {"digit_cells": 90, "numeric_parse_rate": 1.0, "empty_cell_ratio": 0.1, "row_consistency": 1.0, "pages": [5]},
        {"digit_cells": 10, "numeric_parse_rate": 0.0, "empty_cell_ratio": 0.3, "row_consistency": 0.5, "pages": [6, 7]},
    ]
    profile = table_profile(tables)
    assert profile["mean_numeric_parse_rate"] == pytest.approx(0.9)
    assert profile["multi_page_tables"] == 1


def test_tables_ok_prefers_docling_parse_rate() -> None:
    """Use the parse rate when present and the accounting checks otherwise."""
    assert tables_ok(screening_record(mean_numeric_parse_rate=0.95)) is True
    assert tables_ok(screening_record(mean_numeric_parse_rate=0.5)) is False
    assert tables_ok(screening_record(accounting_checks_run=4, accounting_checks_passed=1)) is False
    assert tables_ok(screening_record(accounting_checks_run=0, accounting_checks_passed=0)) is None


@pytest.mark.parametrize(
    ("overrides", "tier", "reason"),
    [
        ({}, "A", ""),
        ({"text_status": "fail"}, "C", "scanned"),
        ({"image_statements": ["balance_sheet"]}, "C", "balance_sheet"),
        ({"image_statements": ["changes_in_equity"]}, "A", ""),
        ({"structure_source": "none"}, "B", "weak structure"),
        ({"core_statements_found": 2}, "B", "not all located"),
        ({"accounting_checks_run": 0, "accounting_checks_passed": 0}, "B", "unverified"),
    ],
)
def test_assign_tier(overrides: dict, tier: str, reason: str) -> None:
    """Assign tiers from the screening rules and explain anything below A."""
    assigned, reasons = assign_tier(screening_record(**overrides))
    assert assigned == tier
    assert reason in reasons


def test_accounting_checks_pass_on_a_consistent_balance_sheet() -> None:
    """A balance sheet whose totals add up passes the identity and subtotal checks."""
    rows = {
        "balance_sheet": [
            ["Property, plant and equipment", [600.0]], ["Inventories", [400.0]], ["Total assets", [1000.0]],
            ["Stated capital", [300.0]], ["Retained earnings", [300.0]], ["Total equity", [600.0]],
            ["Borrowings", [250.0]], ["Trade payables", [150.0]], ["Total liabilities", [400.0]],
        ],
    }
    result = accounting_checks(rows, [])
    assert result["bs_identity"] == "pass"
    assert result["subtotals_reconciled"] == result["subtotals_checked"] == 3


def test_page_categories_prefers_photo_then_statements() -> None:
    """Label photo pages first, then statement and narrative sections, then low-value segments."""
    pages = [make_page(1, words=10, images=1, cover=0.9), make_page(2), make_page(3), make_page(4), make_page(5)]
    sections = {"balance_sheet": {"start": 2, "end": 2}, "chairman_message": {"start": 3, "end": 3}}
    segments = [{"title": "Notice of Annual General Meeting", "start": 4, "end": 4, "level": 1}]
    labels = page_categories(make_features(pages), sections, segments)
    assert labels == ["photo page", "financial statements", "narrative target", "low-value", "other"]


def test_section_sizes_sum_words_and_figures() -> None:
    """Add up words and figures over a section's pages."""
    pages = [make_page(1, words=100, numbers=10), make_page(2, words=300, numbers=30)]
    [row] = section_sizes(make_features(pages), {"notes": {"start": 1, "end": 2}})
    assert (row["pages"], row["words"], row["numbers"]) == (2, 400, 40)
    assert row["numeric_density"] == pytest.approx(0.1)


def table_element(rows: int, header_rows: int = 1) -> dict:
    """Build a table element whose rows have two words each."""
    return {"kind": "table", "page": 1, "rows": [f"row {i}" for i in range(rows)], "header_rows": header_rows}


def test_chunking_damage_counts_split_tables_and_orphaned_rows() -> None:
    """A 10-row table starting at word 6 of 10-word chunks is split and loses its header for later rows."""
    elements = [{"kind": "text", "text": "one two three four five six"}, table_element(10)]
    stats = chunking_damage(elements, chunk_size=10)
    assert stats["paragraphs"] == 1 and stats["paragraphs_split"] == 0
    assert stats["tables"] == 1 and stats["tables_split"] == 1
    assert stats["table_rows"] == 9
    assert stats["rows_orphaned"] == 8


def test_chunking_damage_leaves_small_tables_whole_in_large_chunks() -> None:
    """Nothing is cut when every element fits in one chunk."""
    stats = chunking_damage([table_element(5), {"kind": "text", "text": "a b c d e f"}], chunk_size=100)
    assert stats["tables_split"] == stats["paragraphs_split"] == stats["rows_orphaned"] == 0


def test_split_table_example_shows_rows_on_both_sides() -> None:
    """Return the header and the rows either side of the first boundary after it."""
    example = split_table_example([table_element(20)], chunk_size=12, context_rows=2)
    assert example["header"] == ["row 0"]
    assert example["before"] == ["row 4", "row 5"]
    assert example["after"] == ["row 6", "row 7"]


def test_elements_from_blocks_merges_numeric_blocks_on_a_page() -> None:
    """Consecutive numeric blocks on one page become one table."""
    blocks = [[1, "text", "Intro"], [1, "numeric", "Revenue\n1,000"], [1, "numeric", "Profit\n200"], [2, "numeric", "Assets\n5,000"]]
    elements = elements_from_blocks(blocks)
    assert [e["kind"] for e in elements] == ["text", "table", "table"]
    assert elements[1]["rows"] == ["Revenue", "1,000", "Profit", "200"]


def test_continuation_flags_follow_repeated_headers() -> None:
    """A table on the next page with the same first row continues the previous table."""
    tables = [
        {"page": 10, "pages": [10], "cols": 4, "first_row": "Note | 2024 | 2023"},
        {"page": 11, "pages": [11], "cols": 4, "first_row": "Note | 2024 | 2023"},
        {"page": 13, "pages": [13], "cols": 4, "first_row": "Note | 2024 | 2023"},
        {"page": 14, "pages": [14], "cols": 4, "first_row": "Other"},
    ]
    assert continuation_flags(tables, continued_pages={14}) == [False, True, False, True]


def test_shingle_similarity_and_passage_keys() -> None:
    """Identical text has similarity 1, and passages differing only in figures share a key."""
    text = "the board reviewed the risk management framework during the year in detail"
    assert jaccard(shingles(text), shingles(text)) == 1.0
    assert jaccard(set(), set()) is None
    long_a = "We audited the financial statements of the company for 2024 " * 3
    long_b = "We audited the financial statements of the company for 2023 " * 3
    assert passage_key(long_a) == passage_key(long_b)
    assert passage_key("too short") is None


def test_passage_counter_finds_text_shared_by_companies() -> None:
    """Report passages used by at least the given number of companies."""
    shared = "In our opinion the accompanying financial statements give a true and fair view of the financial position " * 2
    counter = PassageCounter(keep_example_at=2)
    counter.add("alpha", [shared, "a unique passage " * 10])
    counter.add("beta", [shared])
    counter.add("beta", [shared])
    [row] = counter.repeated(min_companies=2)
    assert row["companies"] == 2


def test_term_counts_and_distinctive_terms_surface_sector_terms() -> None:
    """Sector-specific phrases outrank words that every sector uses."""
    banks = term_counts("capital adequacy ratio deposits loans capital adequacy " * 30 + "customers " * 30)
    makers = term_counts("cost of sales inventory exports factory " * 30 + "customers " * 30)
    assert banks["capital adequacy"] == 60 and "the" not in banks
    ranked = distinctive_terms({"Banks": banks, "Materials": makers}, top_n=3, min_count=5)
    assert "customers" not in [term for term, _ in ranked["Banks"]]
    assert ranked["Banks"][0][0] in {"capital adequacy", "capital", "adequacy"}
    only_deposits = distinctive_terms({"Banks": banks, "Materials": makers}, min_count=5, allowed={"Banks": {"deposits"}})
    assert [term for term, _ in only_deposits["Banks"]] == ["deposits"]
    assert only_deposits["Materials"] == []


def test_order_agreement_and_layout_changes() -> None:
    """Flag a company whose structure source changes between years."""
    assert order_agreement(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert order_agreement(["a", "b", "c"], ["c", "b", "a"]) == 0.0
    reports = pd.DataFrame([
        {"report_id": "x_2023", "company": "X", "year": 2023, "structure_source": "bookmarks", "has_bookmarks": True, "page_count": 200, "section_order": ["a", "b"], "headings": ["Chairman", "Risk"]},
        {"report_id": "x_2024", "company": "X", "year": 2024, "structure_source": "none", "has_bookmarks": False, "page_count": 210, "section_order": ["a", "b"], "headings": ["Chairman", "Risk"]},
    ])
    [change] = layout_changes(reports).to_dict("records")
    assert change["layout_changed"] and "structure bookmarks to none" in change["change_reasons"]


@pytest.mark.parametrize(
    ("true_value", "values", "result"),
    [
        (1000.0, [1000.0, 900.0], "match"),
        (-500.0, [500.0], "sign_lost"),
        (1_000_000.0, [1000.0], "unit_mismatch"),
        (900.0, [1000.0, 900.0], "column_shift"),
        (123.0, [456.0], "wrong_value"),
        (1.0, None, "missing"),
    ],
)
def test_classify_figure_names_error_types(true_value: float, values: list[float] | None, result: str) -> None:
    """Name each kind of mismatch between a hand-checked figure and the extraction."""
    assert classify_figure(true_value, values) == result


def test_ground_truth_sample_template_and_comparison() -> None:
    """Sample distinct companies per sector, and compare only the filled-in values."""
    meta = pd.DataFrame([
        {"report_id": f"{c}_{y}", "company": c, "gics_group": g, "year": y}
        for g, companies in {"Banks": ["b1", "b2", "b3"], "Materials": ["m1"]}.items()
        for c in companies for y in (2023, 2024)
    ])
    sample = stratified_sample(meta, per_group=2)
    assert sample.groupby("gics_group")["company"].nunique().to_dict() == {"Banks": 2, "Materials": 1}
    template = ground_truth_template(sample)
    assert len(template) == len(sample) * len(FIGURES)
    report = sample["report_id"].iloc[0]
    template.loc[(template["report_id"] == report) & (template["figure"] == "total_assets"), "true_value"] = "1000"
    template.loc[(template["report_id"] == report) & (template["figure"] == "section_start:notes"), "true_value"] = "50"
    rows = {report: {"balance_sheet": [["Total assets", [1000.0, 950.0]]]}}
    found = {report: {"notes": {"start": 51, "end": 90}}}
    comparison = compare_ground_truth(template, rows, found)
    assert dict(zip(comparison["figure"], comparison["result"])) == {"total_assets": "match", "section_start:notes": "match"}
