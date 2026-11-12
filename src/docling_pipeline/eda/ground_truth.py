"""Sample reports for hand checking and compare hand-recorded figures with extracted ones (step A5)."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

from docling_pipeline.eda.statements import KEY_FIGURES, close, normalize_label, value_rows
from docling_pipeline.eda.structure import TARGET_SECTIONS

SECTION_PREFIX = "section_start:"
FIGURES = list(KEY_FIGURES) + [SECTION_PREFIX + key for key in TARGET_SECTIONS]
TEMPLATE_COLUMNS = ["report_id", "company", "gics_group", "year", "figure", "true_value", "pdf_page", "notes"]
UNIT_SCALES = (1e3, 1e6, 1e-3, 1e-6)
SECTION_PAGE_TOLERANCE = 1


def stratified_sample(meta: pd.DataFrame, per_group: int = 2, seed: int = 7) -> pd.DataFrame:
    """Pick up to ``per_group`` reports per GICS group from different companies and years."""
    picks = []
    for _, group in meta.sort_values("report_id").groupby("gics_group"):
        shuffled = group.sample(frac=1, random_state=seed)
        companies, years, chosen = set(), set(), []
        for record in shuffled.to_dict("records"):
            if record["company"] in companies or record["year"] in years:
                continue
            chosen.append(record)
            companies.add(record["company"])
            years.add(record["year"])
            if len(chosen) == per_group:
                break
        picks.extend(chosen)
    return pd.DataFrame(picks)


def ground_truth_template(sample: pd.DataFrame) -> pd.DataFrame:
    """Build a blank sheet with one row per sampled report and figure, for filling in by hand."""
    rows = [
        {**{k: record[k] for k in ("report_id", "company", "gics_group", "year")}, "figure": figure, "true_value": "", "pdf_page": "", "notes": ""}
        for record in sample.to_dict("records")
        for figure in FIGURES
    ]
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)


def figure_values(statement_rows: Mapping[str, list[list]], figure: str) -> list[float] | None:
    """Return every value on the first row that matches a key figure, or ``None`` if no row matches."""
    kind, pattern = KEY_FIGURES[figure]
    rx = re.compile(pattern)
    for label, values in value_rows(statement_rows.get(kind, [])):
        if rx.search(normalize_label(label)):
            return values
    return None


def classify_figure(true_value: float, values: list[float] | None) -> str:
    """Name the kind of error between a hand-recorded figure and the extracted row.

    Returns:
        ``match``, ``missing``, ``sign_lost``, ``unit_mismatch``, ``column_shift`` or ``wrong_value``.
    """
    if not values:
        return "missing"
    extracted = values[0]
    if close(true_value, extracted):
        return "match"
    if close(abs(true_value), abs(extracted)):
        return "sign_lost"
    if any(close(true_value, extracted * scale) for scale in UNIT_SCALES):
        return "unit_mismatch"
    if any(close(true_value, v) for v in values[1:]):
        return "column_shift"
    return "wrong_value"


def compare_ground_truth(
    truth: pd.DataFrame,
    statement_rows: Mapping[str, Mapping[str, list[list]]],
    sections: Mapping[str, Mapping[str, Mapping]],
) -> pd.DataFrame:
    """Compare the filled-in rows of the ground-truth sheet with the extraction.

    Args:
        truth: The ground-truth sheet; rows without ``true_value`` are skipped.
        statement_rows: Report id mapped to its statement rows.
        sections: Report id mapped to its located target sections.

    Returns:
        The filled rows with ``extracted`` and ``result`` columns added.
    """
    filled = truth[pd.to_numeric(truth["true_value"], errors="coerce").notna()].copy()
    filled["true_value"] = pd.to_numeric(filled["true_value"])
    extracted, results = [], []
    for record in filled.to_dict("records"):
        report, figure, value = record["report_id"], record["figure"], record["true_value"]
        if figure.startswith(SECTION_PREFIX):
            found = sections.get(report, {}).get(figure.removeprefix(SECTION_PREFIX))
            start = found["start"] if found else None
            extracted.append(start)
            if start is None:
                results.append("missing")
            else:
                results.append("match" if abs(start - value) <= SECTION_PAGE_TOLERANCE else "wrong_page")
        else:
            values = figure_values(statement_rows.get(report, {}), figure)
            extracted.append(values[0] if values else None)
            results.append(classify_figure(value, values))
    filled["extracted"] = extracted
    filled["result"] = results
    filled["kind"] = ["section" if f.startswith(SECTION_PREFIX) else "figure" for f in filled["figure"]]
    return filled
