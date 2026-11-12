"""Build the master metadata table: one row per annual report."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COMPANY_METADATA_COLUMNS = ["company", "ticker", "gics_group", "sector_source"]

SECTOR_RULES: list[tuple[str, str]] = [
    (r"HOSPITAL|NAWALOKA", "Health Care Equipment & Services"),
    (r"INSURANCE|\bLIFE\b", "Insurance"),
    (r"MERCHANT BANK", "Diversified Financials"),
    (r"\bBANK", "Banks"),
    (r"ENERGY|POWER|HYDRO|LAXAPANA", "Utilities"),
    (r"FINANCE|LEASING|CREDIT|TREASUR|\bFUND\b|INVESTMENT TRUST|CAPITAL|GUARDIAN|BIMPUTH|VENTURES|CEYLON INVESTMENT|MERCANTILE INVESTMENTS", "Diversified Financials"),
    (r"HOTEL|RESORT|LEISURE|BEACH|SIGIRIYA|JETWING|DOLPHIN|PALM GARDEN|CITRUS|SYMPHONY|HUNAS", "Consumer Services"),
    (r"PLANTATION|\bTEA\b|ESTATES|FOODS?\b|FARMS|BREWERY|BEVERAGE|DISTILLERIES|TOBACCO|COCONUT|GRAIN|MILK|MILLS|DILMAH|COLD STORES|MELSTACORP|KOTMALE|MAHARAJA|HVA", "Food, Beverage & Tobacco"),
    (r"CARGILLS|MULLER", "Food & Staples Retailing"),
    (r"LAND|REALTY|PROPERT|DEVELOPMENT|HOUSING|CITY HOLDINGS|CARGO BOAT|ONALLY", "Real Estate"),
    (r"\bIOC\b|GAS\b|LUBRICANTS", "Energy"),
    (r"DIALOG AXIATA", "Telecommunication Services"),
    (r"TILE|CERAMIC|PORCELAIN|ALUMINIUM|ALUMEX|GRAPHITE|ASPHALT|HAYCARB|CHEMANEX|LANKEM|PACKAGING|AGSTAR|JAT HOLDINGS|B P P L|DIPPED", "Materials"),
    (r"APPAREL|FABRIC|FIBRE|DIAMONDS|JEWELLERY", "Consumer Durables & Apparel"),
    (r"ABANS|ODEL|KAPRUKA|MACKIE|SINGER", "Retailing"),
    (r"SHIPPING|MOBILITY", "Transportation"),
    (r"HSENID|KERNER|SOFTWARE|SOLUTIONS LANKA", "Software & Services"),
    (r"PRINTERS|PUBLISHERS|GESTETNER|OFFICE EQUIPMENT", "Commercial & Professional Services"),
    (r"TYRES|DIESEL & MOTOR|ASHOK LEYLAND", "Automobiles & Components"),
    (r"HOLDINGS|ENGINEERING|CABLES?|DOCKYARD|AITKEN SPENCE|HAYLEYS|JOHN KEELLS|CARSON|BROWN & COMPANY|HUNTER|LUMINEX|LEE HEDGES|INDUSTRIES|ALPHA FIRE|BUKIT DARAH|CEYLINCO", "Capital Goods"),
]

FISCAL_PAIR = re.compile(r"(20\d{2})\D{1,2}(\d{2})(?!\d)")
FY_SHORT = re.compile(r"FY\s?(\d{2})\D(\d{2})", re.I)
PLAIN_YEAR = re.compile(r"(20\d{2})")


def parse_year(stem: str) -> int | None:
    """Return the financial year a report covers, as the year its period ends.

    ``johnkeells_2024`` gives 2024, ``Report-2021-22`` gives 2022 and ``FY24-25`` gives 2025.
    """
    short = FY_SHORT.search(stem)
    if short:
        return 2000 + int(short.group(2))
    pair = FISCAL_PAIR.search(stem)
    if pair and int(pair.group(2)) == (int(pair.group(1)) + 1) % 100:
        return int(pair.group(1)[:2] + pair.group(2))
    plain = PLAIN_YEAR.search(stem)
    return int(plain.group(1)) if plain else None


def guess_sector(company: str) -> str:
    """Guess a GICS industry group from the company name, or return ``Unclassified``."""
    name = company.upper()
    for pattern, group in SECTOR_RULES:
        if re.search(pattern, name):
            return group
    return "Unclassified"


def load_company_metadata(path: Path, companies: list[str]) -> pd.DataFrame:
    """Read the hand-maintained company table, creating it with guessed sectors if missing.

    Companies missing from an existing file are appended with guessed sectors, and rows
    already in the file are never changed, so manual corrections are kept.

    Args:
        path: CSV with ``company, ticker, gics_group, sector_source``.
        companies: Company folder names found in the corpus.

    Returns:
        One row per company.
    """
    existing = pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame(columns=COMPANY_METADATA_COLUMNS)
    known = set(existing["company"])
    new_rows = [
        {"company": c, "ticker": "", "gics_group": guess_sector(c), "sector_source": "name_rule"}
        for c in sorted(set(companies) - known)
    ]
    table = pd.concat([existing, pd.DataFrame(new_rows, columns=COMPANY_METADATA_COLUMNS)], ignore_index=True)
    if new_rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.sort_values("company").to_csv(path, index=False)
    return table


def company_of(path: Path, reports_dir: Path) -> str:
    """Return the company of a report: its parent folder name, or the file stem at the top level."""
    parts = path.relative_to(reports_dir).parts
    return parts[-2] if len(parts) >= 2 else path.stem


def build_metadata(reports: dict[str, Path], reports_dir: Path, company_table: pd.DataFrame) -> pd.DataFrame:
    """Build the master table with one row per report.

    Args:
        reports: Report id mapped to PDF path, as returned by ``discover_reports``.
        reports_dir: Root of the report folders.
        company_table: Output of :func:`load_company_metadata`.

    Returns:
        Columns ``report_id, company, ticker, gics_group, year, file_path, file_size_mb``.
    """
    rows = []
    for report_id, path in reports.items():
        rows.append({
            "report_id": report_id,
            "company": company_of(path, reports_dir),
            "year": parse_year(path.stem),
            "file_path": str(path),
            "file_size_mb": round(path.stat().st_size / 1e6, 2),
        })
    frame = pd.DataFrame(rows).merge(company_table[["company", "ticker", "gics_group"]], on="company", how="left")
    frame["year"] = frame["year"].astype("Int64")
    columns = ["report_id", "company", "ticker", "gics_group", "year", "file_path", "file_size_mb"]
    return frame[columns].sort_values(["company", "year"]).reset_index(drop=True)
