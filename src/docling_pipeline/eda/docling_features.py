"""Read Docling JSON output and measure its headings, tables and reading order."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from docling_pipeline.eda.pdf_features import read_json_gz, write_json_gz
from docling_pipeline.eda.textutils import parse_cell

DOCLING_CACHE_VERSION = 1
HEADER_LABELS = {"section_header", "title"}
TEXT_LABELS = {"text", "list_item", "caption", "footnote", "formula", "code"}
NOTE_HEADER = re.compile(r"^notes?$", re.I)


def index_docling_outputs(extracted_dir: Path) -> dict[str, Path]:
    """Map report ids to their Docling JSON files under the extracted folder."""
    if not extracted_dir.exists():
        return {}
    return {
        p.stem.lower(): p
        for p in extracted_dir.rglob("*.json")
        if not p.stem.endswith("_pictures")
    }


def _label(item: Any) -> str:
    """Return a Docling item's label as plain text."""
    return getattr(item.label, "value", str(item.label))


def _page(item: Any) -> int:
    """Return the first page an item appears on, or 0."""
    return item.prov[0].page_no if getattr(item, "prov", None) else 0


def table_metrics(grid: list[list[str]], header_flags: list[list[bool]]) -> dict:
    """Measure one table: size, empty cells, numeric parse rate and row consistency.

    The numeric parse rate is the share of body cells containing a digit that parse as an
    accounting figure, so garbled or merged numbers lower it.
    """
    n_rows = len(grid)
    n_cols = max((len(r) for r in grid), default=0)
    cells = [c for row in grid for c in row]
    empty = sum(1 for c in cells if not c.strip())
    header_rows = 0
    for flags, row in zip(header_flags, grid):
        if row and all(f or not c.strip() for f, c in zip(flags, row)):
            header_rows += 1
        else:
            break
    body = grid[header_rows:]
    digit_cells = [c for row in body for c in row[1:] if re.search(r"\d", c)]
    parsed = sum(1 for c in digit_cells if parse_cell(c) is not None)
    filled = [sum(1 for c in row if c.strip()) for row in body]
    modal = Counter(filled).most_common(1)[0][0] if filled else 0
    return {
        "rows": n_rows,
        "cols": n_cols,
        "header_rows": header_rows,
        "empty_cell_ratio": empty / len(cells) if cells else 0.0,
        "digit_cells": len(digit_cells),
        "numeric_parse_rate": parsed / len(digit_cells) if digit_cells else None,
        "row_consistency": sum(1 for f in filled if f == modal) / len(filled) if filled else None,
    }


def table_rows(grid: list[list[str]]) -> list[list]:
    """Turn a table grid into statement rows ``[label, [values...]]``, skipping a ``Note`` column."""
    note_cols = {j for row in grid[:3] for j, c in enumerate(row) if NOTE_HEADER.match(c.strip())}
    rows = []
    for row in grid:
        label = next((c for c in row if c.strip() and parse_cell(c) is None), "")
        values = [v for j, c in enumerate(row[1:], start=1) if j not in note_cols and (v := parse_cell(c)) is not None]
        rows.append([label, values])
    return rows


def extract_docling_features(json_path: Path) -> dict:
    """Read one Docling JSON file and return its headers, tables and reading-order elements."""
    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument.load_from_json(json_path)
    headers, tables, elements = [], [], []
    for item, _level in doc.iterate_items():
        label = _label(item)
        page = _page(item)
        if label in HEADER_LABELS:
            headers.append([page, item.text])
            elements.append({"kind": "heading", "page": page, "text": item.text})
        elif label in TEXT_LABELS and getattr(item, "text", "").strip():
            elements.append({"kind": "text", "page": page, "text": item.text})
        elif label == "table":
            grid = [[cell.text or "" for cell in row] for row in item.data.grid]
            flags = [[bool(cell.column_header) for cell in row] for row in item.data.grid]
            pages = sorted({p.page_no for p in item.prov})
            metrics = table_metrics(grid, flags)
            tables.append({"page": page, "pages": pages, **metrics, "first_row": " | ".join(grid[0]) if grid else ""})
            elements.append({"kind": "table", "page": page, "rows": [" | ".join(r) for r in grid], "header_rows": max(1, metrics["header_rows"]), "grid": grid})
    return {"version": DOCLING_CACHE_VERSION, "page_count": doc.num_pages(), "headers": headers, "tables": tables, "elements": elements}


def load_docling_features(json_path: Path, cache_dir: Path) -> dict:
    """Return cached Docling features, re-reading the JSON when it is newer than the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{json_path.stem.lower()}.docling.json.gz"
    if cache_file.exists() and cache_file.stat().st_mtime >= json_path.stat().st_mtime:
        cached = read_json_gz(cache_file)
        if cached.get("version") == DOCLING_CACHE_VERSION:
            return cached
    features = extract_docling_features(json_path)
    write_json_gz(cache_file, features)
    return features


def _cache_one(report_id: str, json_path: str, cache_dir: str) -> dict:
    """Cache one report's Docling features and return a status record."""
    try:
        load_docling_features(Path(json_path), Path(cache_dir))
        return {"report_id": report_id, "status": "ok", "error": ""}
    except Exception as exc:
        return {"report_id": report_id, "status": "error", "error": repr(exc)[:300]}


def build_docling_cache(index: dict[str, Path], cache_dir: Path, workers: int = 4, progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """Cache the Docling features of every report in ``index`` that is missing or stale, in parallel.

    Args:
        index: Report id mapped to its Docling JSON file.
        cache_dir: Folder for the cache files.
        workers: Number of worker processes.
        progress: Optional callable taking (done, total).

    Returns:
        Status records of the reports processed in this call.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = {}
    for report_id, json_path in index.items():
        cache_file = cache_dir / f"{json_path.stem.lower()}.docling.json.gz"
        if not cache_file.exists() or cache_file.stat().st_mtime < json_path.stat().st_mtime:
            todo[report_id] = json_path
    results: list[dict] = []
    if not todo:
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_cache_one, rid, str(path), str(cache_dir)) for rid, path in todo.items()]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if progress:
                progress(done, len(futures))
    return results


def continuation_flags(tables: list[dict], continued_pages: set[int]) -> list[bool]:
    """Flag tables, given in reading order, that continue the table on the previous page.

    A table continues the previous one when it starts on the next page and either repeats
    its first row, or has the same number of columns on a page marked "continued".
    """
    flags = []
    previous = None
    for table in tables:
        follows = previous is not None and table["page"] == previous["pages"][-1] + 1
        same_header = bool(table["first_row"].strip(" |")) and table["first_row"] == (previous or {}).get("first_row")
        same_shape = previous is not None and table["cols"] == previous["cols"] and table["page"] in continued_pages
        flags.append(bool(follows and (same_header or same_shape)))
        previous = table
    return flags


def statement_rows_from_tables(features: dict, pages: dict[str, list[int]]) -> dict[str, list[list]]:
    """Collect statement rows from the Docling tables on each statement's pages."""
    result: dict[str, list[list]] = {}
    for kind, kind_pages in pages.items():
        wanted = set(kind_pages)
        rows: list[list] = []
        for element in features["elements"]:
            if element["kind"] == "table" and element["page"] in wanted:
                rows.extend(table_rows(element["grid"]))
        result[kind] = rows
    return result
