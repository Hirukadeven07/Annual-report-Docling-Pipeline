"""Run the per-report screening checks and the corpus-wide text pass used by the EDA notebook."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

from docling_pipeline.eda.chunking import chunking_damage, elements_from_blocks, split_table_example
from docling_pipeline.eda.docling_features import statement_rows_from_tables
from docling_pipeline.eda.pdf_features import load_blocks
from docling_pipeline.eda.screening import CORE_STATEMENTS, assign_tier, image_statements, table_profile, text_layer_check
from docling_pipeline.eda.statements import accounting_checks
from docling_pipeline.eda.structure import TARGET_SECTIONS, locate_sections, section_segments, structure_profile
from docling_pipeline.eda.textstats import PassageCounter, jaccard, shingles, term_counts

CHUNK_SIZES = (200, 400, 800)
EXAMPLE_CHUNK_SIZE = 400
MAX_EXAMPLES = 3
TERM_MIN_REPORT_COUNT = 2
CHECK_FIELDS = ("bs_identity", "subtotal_rate", "profit_cross_check", "highlights_check", "accounting_checks_run", "accounting_checks_passed")


def pdf_statement_rows(features: Mapping) -> dict[str, list[list]]:
    """Return the statement rows rebuilt from the PDF text layer, one list per statement."""
    return {kind: [row for page in pages for row in page["rows"]] for kind, pages in features["statement_table_rows"].items()}


def profile_report(features: Mapping, docling: Mapping | None = None) -> dict:
    """Run steps A1 to A5 on one report.

    Args:
        features: Cached PDF-layer features of the report.
        docling: Cached Docling features, or ``None`` when the report has no Docling output.

    Returns:
        ``record`` (one flat screening row), ``sections``, ``segments``, ``unmatched`` headings
        and the ``statement_rows`` used for the accounting checks.
    """
    headers = [(page, text) for page, text in docling["headers"]] if docling else None
    structure = structure_profile(features, headers)
    source = structure["structure_source"]
    segments = section_segments(features, source, headers)
    sections, unmatched = locate_sections(segments, features["page_count"], source)
    table_pages = {p for t in docling["tables"] for p in t["pages"]} if docling else None
    images = image_statements(features, sections, table_pages)
    pdf_rows = pdf_statement_rows(features)
    pdf_checks = accounting_checks(pdf_rows, features["highlight_numbers"])
    docling_rows = statement_rows_from_tables(docling, features["statement_primary"]) if docling else None
    docling_checks = accounting_checks(docling_rows, features["highlight_numbers"]) if docling else None
    chosen = docling_checks or pdf_checks
    record = {
        "report_id": features["report_id"],
        "page_count": features["page_count"],
        "has_docling": docling is not None,
        **text_layer_check(features),
        **structure,
        "section_coverage": len(sections),
        **{f"found_{key}": key in sections for key in TARGET_SECTIONS},
        **{f"start_{key}": sections[key]["start"] if key in sections else None for key in TARGET_SECTIONS},
        "core_statements_found": sum(1 for k in CORE_STATEMENTS if k in sections or features["statement_primary"].get(k)),
        **(table_profile(docling["tables"]) if docling else {"table_count": None, "mean_numeric_parse_rate": None}),
        "image_statements": images,
        "image_statement_flag": bool(images),
        **{f"pdf_{k}": pdf_checks[k] for k in CHECK_FIELDS},
        **{f"docling_{k}": (docling_checks or {}).get(k) for k in CHECK_FIELDS},
        "check_evidence": "docling" if docling_checks else "pdf_layer",
        "accounting_checks_run": chosen["accounting_checks_run"],
        "accounting_checks_passed": chosen["accounting_checks_passed"],
    }
    record["tier"], record["tier_reasons"] = assign_tier(record)
    return {
        "record": record,
        "sections": sections,
        "segments": segments,
        "unmatched": unmatched,
        "statement_rows": docling_rows or pdf_rows,
    }


def report_elements(blocks: list[list], docling: Mapping | None) -> tuple[list[dict], str]:
    """Return a report's reading-order elements and their source, preferring Docling."""
    if docling:
        return docling["elements"], "docling"
    return elements_from_blocks(blocks), "pdf_blocks"


def section_text(blocks: list[list], section: Mapping) -> str:
    """Join the text blocks that lie inside a section's page range."""
    return "\n".join(text for page, kind, text in blocks if kind == "text" and section["start"] <= page <= section["end"])


def text_pass(
    meta: pd.DataFrame,
    sections: Mapping[str, Mapping[str, Mapping]],
    cache_dir: Path,
    docling: Callable[[str], Mapping | None],
    chunk_sizes: tuple[int, ...] = CHUNK_SIZES,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Read every report's text once and gather the measures for steps B4, B5 and B6.

    Reports are read company by company, so consecutive years can be compared without
    keeping every report's text in memory.

    Args:
        meta: Master table with ``report_id, company, year, gics_group``.
        sections: Report id mapped to its located target sections.
        cache_dir: PDF-layer cache folder.
        docling: Returns a report's cached Docling features, or ``None``.
        chunk_sizes: Fixed chunk sizes in words to simulate.
        progress: Optional callable taking (done, total).

    Returns:
        ``chunking`` (one row per report and chunk size), ``split_examples``, ``similarity``
        (one row per company, section and consecutive pair of years), ``passages`` (a
        :class:`PassageCounter`), ``terms`` (term counts per GICS group) and ``term_companies``
        (per GICS group, how many companies use each term).
    """
    chunk_rows, examples, similarity_rows = [], [], []
    passages = PassageCounter()
    terms: dict[str, Counter] = defaultdict(Counter)
    term_companies: dict[str, Counter] = defaultdict(Counter)
    ordered = meta.sort_values(["company", "year"])
    done = 0
    for company, group in ordered.groupby("company", sort=False):
        previous: dict | None = None
        company_terms: set[str] = set()
        for record in group.to_dict("records"):
            report_id = record["report_id"]
            blocks = load_blocks(cache_dir, report_id)
            report_docling = docling(report_id)
            elements, source = report_elements(blocks, report_docling)
            for size in chunk_sizes:
                chunk_rows.append({"report_id": report_id, "chunk_size": size, "element_source": source, **chunking_damage(elements, size)})
            if sum(e["element_source"] == source for e in examples) < MAX_EXAMPLES:
                example = split_table_example(elements, EXAMPLE_CHUNK_SIZE)
                if example:
                    examples.append({"report_id": report_id, "element_source": source, **example})
            report_sections = sections.get(report_id, {})
            current = {"year": record["year"], "shingles": {key: shingles(section_text(blocks, sec)) for key, sec in report_sections.items()}}
            if previous and pd.notna(record["year"]) and pd.notna(previous["year"]) and record["year"] - previous["year"] == 1:
                for key, grams in current["shingles"].items():
                    if key in previous["shingles"]:
                        score = jaccard(previous["shingles"][key], grams)
                        if score is not None:
                            similarity_rows.append({"company": company, "section": key, "from_year": previous["year"], "to_year": record["year"], "similarity": score})
            previous = current
            texts = [text for _, kind, text in blocks if kind == "text"]
            passages.add(company, texts)
            report_terms = term_counts("\n".join(texts), min_count=TERM_MIN_REPORT_COUNT)
            terms[record["gics_group"]].update(report_terms)
            company_terms.update(report_terms)
            done += 1
            if progress:
                progress(done, len(ordered))
        term_companies[group["gics_group"].iloc[0]].update(company_terms)
    return {
        "chunking": pd.DataFrame(chunk_rows),
        "split_examples": examples,
        "similarity": pd.DataFrame(similarity_rows),
        "passages": passages,
        "terms": dict(terms),
        "term_companies": dict(term_companies),
    }
