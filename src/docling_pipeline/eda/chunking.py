"""Simulate naive fixed-size chunking and count the tables and paragraphs it cuts (step B4)."""

from __future__ import annotations

from collections.abc import Mapping

MIN_PARAGRAPH_WORDS = 5
MIN_TABLE_ROWS = 2


def elements_from_blocks(blocks: list[list]) -> list[dict]:
    """Turn cached PDF text blocks into reading-order elements.

    Consecutive numeric blocks on the same page are merged into one table whose rows are
    the block lines. This is a rough stand-in for real tables when Docling output is missing.
    """
    elements: list[dict] = []
    previous_page = None
    for page, kind, text in blocks:
        if kind == "numeric":
            lines = [line for line in text.split("\n") if line.strip()]
            if elements and elements[-1]["kind"] == "table" and previous_page == page:
                elements[-1]["rows"].extend(lines)
            else:
                elements.append({"kind": "table", "page": page, "rows": lines, "header_rows": 1})
        else:
            elements.append({"kind": "text", "page": page, "text": text})
        previous_page = page
    return elements


def _row_lengths(element: Mapping) -> list[int]:
    """Return the word count of each table row, at least one word per row."""
    return [max(1, len(row.split())) for row in element["rows"]]


def chunking_damage(elements: list[Mapping], chunk_size: int) -> dict:
    """Count paragraphs and tables cut by fixed-size chunks of ``chunk_size`` words.

    A table row is orphaned when it ends up in a later chunk than the table's header rows,
    so the chunk holding it has no column headings.

    Returns:
        Counts of paragraphs, split paragraphs, tables, split tables, body rows and orphaned rows.
    """
    stats = {"paragraphs": 0, "paragraphs_split": 0, "tables": 0, "tables_split": 0, "table_rows": 0, "rows_orphaned": 0}
    position = 0
    for element in elements:
        if element["kind"] == "table":
            lengths = _row_lengths(element)
            start = position
            position += sum(lengths)
            if len(lengths) < MIN_TABLE_ROWS:
                continue
            stats["tables"] += 1
            header_rows = min(element.get("header_rows", 1), len(lengths))
            header_chunk = (start + sum(lengths[:header_rows]) - 1) // chunk_size
            if start // chunk_size != (position - 1) // chunk_size:
                stats["tables_split"] += 1
            row_start = start + sum(lengths[:header_rows])
            for length in lengths[header_rows:]:
                stats["table_rows"] += 1
                if row_start // chunk_size > header_chunk:
                    stats["rows_orphaned"] += 1
                row_start += length
        else:
            words = len(element.get("text", "").split())
            start = position
            position += words
            if words < MIN_PARAGRAPH_WORDS:
                continue
            stats["paragraphs"] += 1
            if start // chunk_size != (position - 1) // chunk_size:
                stats["paragraphs_split"] += 1
    return stats


def split_table_example(elements: list[Mapping], chunk_size: int, context_rows: int = 4) -> dict | None:
    """Return the first table that a chunk boundary cuts after its header, with rows on both sides.

    Returns:
        ``{page, header, before, after}`` with the header rows, the last rows of the first
        chunk and the first rows of the next chunk, or ``None`` if no table is cut that way.
    """
    position = 0
    for element in elements:
        if element["kind"] != "table":
            position += len(element.get("text", "").split())
            continue
        lengths = _row_lengths(element)
        header_rows = min(element.get("header_rows", 1), len(lengths))
        offsets = [position + sum(lengths[:i]) for i in range(len(lengths))]
        position += sum(lengths)
        chunks = [offset // chunk_size for offset in offsets]
        for i in range(header_rows + context_rows, len(chunks) - context_rows + 1):
            if chunks[i] != chunks[i - 1] and chunks[header_rows - 1] == chunks[i - 1]:
                return {
                    "page": element.get("page"),
                    "header": element["rows"][:header_rows],
                    "before": element["rows"][i - context_rows:i],
                    "after": element["rows"][i:i + context_rows],
                }
    return None
