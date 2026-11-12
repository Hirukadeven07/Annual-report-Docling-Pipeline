"""Automatic accounting checks on statement rows given as ``[label, [values...]]``."""

from __future__ import annotations

import re

Row = list
RELATIVE_TOLERANCE = 0.001
ABSOLUTE_TOLERANCE = 2.0

KEY_FIGURES: dict[str, tuple[str, str]] = {
    "revenue": ("income_statement", r"^(total |gross |net )?(revenue|turnover|income)( from contracts? with customers)?$|^interest income$|^gross written premiums?$|^total operating income$"),
    "profit_before_tax": ("income_statement", r"^(net )?profit( loss)? before (income )?tax(ation)?"),
    "profit_for_year": ("income_statement", r"^(net )?profit( loss)? (for|of) the (year|period)$"),
    "eps_basic": ("income_statement", r"(basic )?earnings( loss)? per (ordinary )?share"),
    "total_assets": ("balance_sheet", r"^total assets$"),
    "total_equity": ("balance_sheet", r"^total equity$"),
    "total_liabilities": ("balance_sheet", r"^total liabilities$"),
    "net_cash_operating": ("cash_flow", r"^net cash.*operating activities"),
    "cash_at_end": ("cash_flow", r"cash( and cash equivalents)? at (the )?end"),
}


def normalize_label(label: str) -> str:
    """Lowercase a row label and keep letters only."""
    label = re.sub(r"[^a-z ]+", " ", label.lower().replace("&", " and "))
    return re.sub(r"\s+", " ", label).strip()


def is_year_row(values: list[float]) -> bool:
    """Return True for header rows that only hold years, such as ``2022 2021``."""
    return bool(values) and all(v.is_integer() and 1990 <= v <= 2035 for v in values)


def value_rows(rows: list[Row]) -> list[Row]:
    """Drop year header rows and rows without figures."""
    return [r for r in rows if r[1] and not is_year_row(r[1])]


def close(a: float, b: float) -> bool:
    """Compare two amounts with a rounding tolerance."""
    return abs(a - b) <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * max(abs(a), abs(b)))


def first_value(rows: list[Row], pattern: str) -> float | None:
    """Return the first figure of the first row whose label matches the pattern."""
    rx = re.compile(pattern)
    for label, values in value_rows(rows):
        if rx.search(normalize_label(label)):
            return values[0]
    return None


def balance_sheet_identity(rows: list[Row]) -> str:
    """Check total assets against total equity plus total liabilities.

    Returns:
        ``pass``, ``fail`` or ``not_found``.
    """
    assets = first_value(rows, r"^total assets$")
    if assets is None:
        return "not_found"
    combined = first_value(rows, r"^total (equity and liabilities|liabilities and (shareholders )?equity)")
    equity = first_value(rows, r"^total (shareholders )?equity$")
    liabilities = first_value(rows, r"^total liabilities$")
    if combined is not None:
        return "pass" if close(assets, combined) else "fail"
    if equity is not None and liabilities is not None:
        return "pass" if close(assets, equity + liabilities) else "fail"
    return "not_found"


def subtotal_checks(rows: list[Row]) -> tuple[int, int]:
    """Count subtotal rows, and how many equal the sum of the line items above them.

    A subtotal is a row labelled ``Total ...`` or an unlabelled row of figures. It reconciles
    when a run of rows directly above it, a run of earlier subtotals, or a pair of earlier
    subtotals adds up to it.

    Returns:
        ``(checked, reconciled)``.
    """
    rows = [r for r in rows if not is_year_row(r[1])]
    checked = reconciled = 0
    totals_seen: list[float] = []
    for i, (label, values) in enumerate(rows):
        is_total = bool(values) and (normalize_label(label).startswith("total") or not label.strip())
        if not is_total:
            continue
        target = values[0]
        items: list[float] = []
        for prev_label, prev_values in reversed(rows[:i]):
            if not prev_values:
                break
            items.append(prev_values[0])
        run_sums = [sum(items[:k]) for k in range(2, len(items) + 1)]
        total_sums = [sum(totals_seen[-k:]) for k in range(2, len(totals_seen) + 1)]
        total_sums += [a + b for x, a in enumerate(totals_seen) for b in totals_seen[x + 1:]]
        if len(items) >= 2 or len(totals_seen) >= 2:
            checked += 1
            if any(close(target, s) for s in run_sums + total_sums):
                reconciled += 1
        totals_seen.append(target)
    return checked, reconciled


def profit_cross_check(income_rows: list[Row], equity_rows: list[Row]) -> str:
    """Check that profit for the year in the income statement appears in the changes-in-equity statement."""
    profit = first_value(income_rows, KEY_FIGURES["profit_for_year"][1])
    if profit is None:
        return "not_found"
    rx = re.compile(r"^(net )?profit( loss)? (for|of) the (year|period)")
    candidates = [v for label, values in value_rows(equity_rows) if rx.search(normalize_label(label)) for v in values]
    if not candidates:
        return "not_found"
    return "pass" if any(close(profit, v) for v in candidates) else "fail"


def highlights_check(income_rows: list[Row], highlight_numbers: list[float]) -> str:
    """Check that revenue or profit from the income statement appears on the highlights pages.

    Highlights often use a larger unit, so figures are compared at scales of a thousand and a
    million, within half a percent.
    """
    figures = [first_value(income_rows, KEY_FIGURES[k][1]) for k in ("revenue", "profit_for_year")]
    figures = [abs(f) for f in figures if f]
    if not figures or not highlight_numbers:
        return "not_found"
    scales = (1, 1e-3, 1e-6, 1e3, 1e6, 1e-9)
    for figure in figures:
        for h in highlight_numbers:
            if h >= 10 and any(abs(h - figure * s) <= 0.005 * figure * s for s in scales):
                return "pass"
    return "fail"


def key_figures(statement_rows: dict[str, list[Row]]) -> dict[str, float | None]:
    """Extract the headline figures used for ground-truth comparison."""
    return {name: first_value(statement_rows.get(kind, []), pattern) for name, (kind, pattern) in KEY_FIGURES.items()}


def accounting_checks(statement_rows: dict[str, list[Row]], highlight_numbers: list[float]) -> dict:
    """Run every automatic check for one report and summarise them."""
    checked = reconciled = 0
    for kind in ("income_statement", "balance_sheet", "cash_flow"):
        c, r = subtotal_checks(statement_rows.get(kind, []))
        checked += c
        reconciled += r
    result = {
        "bs_identity": balance_sheet_identity(statement_rows.get("balance_sheet", [])),
        "subtotals_checked": checked,
        "subtotals_reconciled": reconciled,
        "subtotal_rate": reconciled / checked if checked else None,
        "profit_cross_check": profit_cross_check(statement_rows.get("income_statement", []), statement_rows.get("changes_in_equity", [])),
        "highlights_check": highlights_check(statement_rows.get("income_statement", []), highlight_numbers),
    }
    statuses = [result["bs_identity"], result["profit_cross_check"], result["highlights_check"]]
    if result["subtotal_rate"] is not None:
        statuses.append("pass" if result["subtotal_rate"] >= 0.8 else "fail")
    run = [s for s in statuses if s != "not_found"]
    result["accounting_checks_run"] = len(run)
    result["accounting_checks_passed"] = sum(s == "pass" for s in run)
    return result
