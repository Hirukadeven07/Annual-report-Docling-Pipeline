"""Extract page-level features from the PDF text layer with PyMuPDF and cache them per report."""

from __future__ import annotations

import gzip
import json
import re
import statistics
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymupdf

from docling_pipeline.eda.textutils import collapse_letter_spacing, count_numbers, parse_amount

CACHE_VERSION = 3
IMAGE_ONLY_MAX_CHARS = 50
IMAGE_ONLY_MIN_COVER = 0.5
STATEMENT_MIN_NUMBERS = 20
FRONT_MATTER_PAGES = 15
ROW_Y_TOLERANCE = 3.0

LEAD = r"^\s*((consolidated|group|company|separate|the|bank)\s+)*"
STATEMENT_HEADINGS = {
    "income_statement": re.compile(LEAD + r"(income\s+statements?|statements?\s+of\s+(profit\s+(or|and|&)\s+loss|comprehensive\s+income|income)|profit\s+(and|&)\s+loss)", re.I),
    "balance_sheet": re.compile(LEAD + r"(statements?\s+of\s+financial\s+position|balance\s+sheets?)", re.I),
    "cash_flow": re.compile(LEAD + r"(statements?\s+of\s+cash\s*flows?|cash\s*flows?\s+statements?)", re.I),
    "changes_in_equity": re.compile(LEAD + r"statements?\s+of\s+changes\s+in\s+(equity|shareholders)", re.I),
}
STATEMENT_ROWS = {
    "income_statement": (r"profit\s*(/\s*\(loss\)\s*)?\s*(for|of)\s+the\s+(year|period)|earnings\s+per\s+share", r"revenue|turnover|interest\s+income|premium", r"operating\s+activities|balance\s+as\s+at"),
    "balance_sheet": (r"total\s+assets", r"total\s+(equity|liabilities)", None),
    "cash_flow": (r"operating\s+activities", r"investing\s+activities", None),
    "changes_in_equity": (r"balance\s+as\s+at", r"dividend|profit\s+for\s+the\s+year", None),
}
CONTENTS_WORDS = re.compile(r"\b(table\s+of\s+)?contents?\b|what'?s\s+inside|inside\s+this\s+report|\bindex\b|navigat", re.I)
CONTENTS_TOPICS = [r"chair", r"governance", r"risk", r"financial\s+statements", r"auditor", r"directors", r"notes\s+to", r"sustainab"]
TITLE_THEN_NUMBER = re.compile(r"^(?P<title>.*?[A-Za-z].*?)[\s.…_\-]*?(?P<num>\d{1,3})$")
NUMBER_THEN_TITLE = re.compile(r"^(page\s*)?(?P<num>\d{1,3})\s+(?P<title>.*[A-Za-z].*)$", re.I)
PURE_NUMBER = re.compile(r"^\d{1,3}$")
CONTINUED = re.compile(r"\b(cont'?d|continued)\b", re.I)
RUNNING_HEADER_PATTERN = re.compile(r"annualreport(20\d{2}|\d{4}/)|^\d{1,3}$|\|\d{1,3}$", re.I)
HIGHLIGHTS = re.compile(r"highlights|at a glance|key\s+(figures|indicators)", re.I)


def is_running_header(text: str) -> bool:
    """Return True for page furniture such as ``Annual Report 2021/22 | 5`` or a bare page number."""
    return bool(RUNNING_HEADER_PATTERN.search(re.sub(r"\s", "", text)))


def cache_paths(cache_dir: Path, report_id: str) -> tuple[Path, Path]:
    """Return the feature file and the block file for one report."""
    return cache_dir / f"{report_id}.features.json.gz", cache_dir / f"{report_id}.blocks.json.gz"


def write_json_gz(path: Path, payload: object) -> None:
    """Write JSON to a gzip file atomically."""
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)


def read_json_gz(path: Path) -> object:
    """Read a gzip JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def cached_version(path: Path) -> int | None:
    """Return the cache version stored in a feature file, or ``None`` if unreadable."""
    try:
        return read_json_gz(path).get("version")
    except (OSError, ValueError, EOFError):
        return None


def _line_text(line: dict) -> str:
    """Join the spans of one PyMuPDF line."""
    return "".join(span["text"] for span in line["spans"]).strip()


def _line_size(line: dict) -> float:
    """Return the largest font size in a line."""
    return max((span["size"] for span in line["spans"]), default=0.0)


def _body_size(blocks: list[dict]) -> float:
    """Estimate the body font size as the character-weighted median span size."""
    sizes = [round(span["size"], 1) for b in blocks for line in b["lines"] for span in line["spans"] for _ in range(min(len(span["text"]), 200))]
    return statistics.median(sizes) if sizes else 0.0


def _image_cover(page: pymupdf.Page) -> float:
    """Return the share of the page area covered by images, capped at 1."""
    area = abs(page.rect)
    if not area:
        return 0.0
    covered = sum(abs(pymupdf.Rect(info["bbox"]) & page.rect) for info in page.get_image_info())
    return min(1.0, covered / area)


def _folio(lines: list[tuple[str, float]], height: float) -> int | None:
    """Return the printed page number found in the header or footer band."""
    for text, y in lines:
        if (y < height * 0.08 or y > height * 0.92) and re.fullmatch(r"\d{1,3}", text.replace(" ", "")):
            return int(text.replace(" ", ""))
    for text, y in lines:
        if y < height * 0.08 or y > height * 0.92:
            edge = re.match(r"^(\d{1,3})\b|\b(\d{1,3})$", text.strip())
            if edge:
                return int(edge.group(1) or edge.group(2))
    return None


def _statement_kinds(short_lines: list[str], text: str, numbers: int, page_share: float) -> tuple[list[str], list[str]]:
    """Return statements named by a heading, and statements recognised from their row labels."""
    titled, by_rows = [], []
    if numbers < STATEMENT_MIN_NUMBERS:
        return titled, by_rows
    for kind, rx in STATEMENT_HEADINGS.items():
        if any(rx.search(line) for line in short_lines):
            titled.append(kind)
        need_a, need_b, veto = STATEMENT_ROWS[kind]
        if page_share >= 0.15 and re.search(need_a, text, re.I) and re.search(need_b, text, re.I) and not (veto and re.search(veto, text, re.I)):
            by_rows.append(kind)
    return titled, by_rows


def _split_row(tokens: list[tuple[str, float, float]], value_x_min: float | None) -> tuple[str, list[float]]:
    """Split one visual row into its label and the run of figures at its right end."""
    values: list[float] = []
    cut = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        text = tokens[i][0]
        value = parse_amount(text)
        if value is None:
            break
        is_year = re.fullmatch(r"(19|20)\d{2}", text) is not None
        if is_year and i > 0 and parse_amount(tokens[i - 1][0]) is None:
            break
        values.insert(0, value)
        cut = i
    if values and value_x_min is not None:
        first_text, _, first_x1 = tokens[cut]
        if re.fullmatch(r"\d{1,2}", first_text) and first_x1 < value_x_min - 5 and len(values) >= 2:
            values = values[1:]
    label = " ".join(t[0] for t in tokens[:cut]).strip()
    return label, values


def page_rows(page: pymupdf.Page) -> list[list]:
    """Rebuild table rows from word positions as ``[label, [values...]]``.

    Words are grouped into visual rows by vertical position. A leading small integer
    that sits left of every comma-formatted figure on the page is treated as a note
    reference and dropped.
    """
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    comma_x = [w[0] for w in words if "," in w[4] and parse_amount(w[4]) is not None]
    value_x_min = min(comma_x) if comma_x else None
    rows: list[list[tuple]] = []
    anchor = None
    for word in words:
        centre = (word[1] + word[3]) / 2
        if rows and anchor is not None and abs(centre - anchor) <= ROW_Y_TOLERANCE:
            rows[-1].append(word)
        else:
            rows.append([word])
            anchor = centre
    result = []
    for row in rows:
        tokens = [(w[4], w[0], w[2]) for w in sorted(row, key=lambda w: w[0])]
        label, values = _split_row(tokens, value_x_min)
        if label or values:
            result.append([label, values])
    return result


def _contents_entries(lines: list[str]) -> list[list]:
    """Pair section titles with printed page numbers on a contents page."""
    entries = []
    pending_title = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if PURE_NUMBER.match(line):
            if pending_title:
                entries.append([pending_title, int(line)])
                pending_title = None
            continue
        match = NUMBER_THEN_TITLE.match(line) or TITLE_THEN_NUMBER.match(line)
        if match and len(match.group("title")) >= 3:
            entries.append([match.group("title").strip(" .…\t"), int(match.group("num"))])
            pending_title = None
        elif re.search(r"[A-Za-z]{3,}", line) and len(line) <= 90:
            pending_title = line
    return entries


def extract_report_features(pdf_path: str, report_id: str, cache_dir: str) -> dict:
    """Extract and cache the PDF-layer features of one report.

    Args:
        pdf_path: Path to the PDF.
        report_id: Report id used to name the cache files.
        cache_dir: Folder for the cache files.

    Returns:
        A short status record: ``report_id, status, pages, error``.
    """
    features_path, blocks_path = cache_paths(Path(cache_dir), report_id)
    try:
        doc = pymupdf.open(pdf_path)
        pages, blocks_out = [], []
        statement_titled: dict[str, list[int]] = {k: [] for k in STATEMENT_HEADINGS}
        statement_rows: dict[str, list[int]] = {k: [] for k in STATEMENT_HEADINGS}
        statement_image: dict[str, list[int]] = {k: [] for k in STATEMENT_HEADINGS}
        contents_pages, contents_entries = [], []
        highlight_numbers: list[float] = []
        for page in doc:
            info, page_blocks = _page_features(page, doc.page_count)
            pages.append(info)
            blocks_out.extend(page_blocks)
            for kind in info.pop("_titled"):
                statement_titled[kind].append(info["page"])
            for kind in info.pop("_by_rows"):
                statement_rows[kind].append(info["page"])
            for kind in info.pop("_image_statement"):
                statement_image[kind].append(info["page"])
            lines = info.pop("_lines")
            if info["page"] <= FRONT_MATTER_PAGES:
                entries = [e for e in _contents_entries(lines) if e[1] <= doc.page_count + 20 and not is_running_header(e[0])]
                text = "\n".join(lines)
                topics = sum(bool(re.search(t, text, re.I)) for t in CONTENTS_TOPICS)
                dense = len(entries) >= 6 and len(entries) / max(1, len(lines)) >= 0.2
                continues = bool(contents_pages) and contents_pages[-1] == info["page"] - 1
                if dense and (CONTENTS_WORDS.search(text) or topics >= 3) and (not contents_pages or continues):
                    contents_pages.append(info["page"])
                    contents_entries.extend(entries)
            in_front = info["page"] <= max(FRONT_MATTER_PAGES, doc.page_count * 0.2)
            if in_front and any(HIGHLIGHTS.search(h["text"]) for h in info["headings"]) and len(highlight_numbers) < 3000:
                highlight_numbers.extend(abs(v) for _, vals in page_rows(page) for v in vals if v)
        _drop_running_headers(pages)
        primary = _primary_statement_pages(statement_titled, statement_rows, doc.page_count)
        rows = {kind: [{"page": p, "rows": page_rows(doc[p - 1])} for p in pages_] for kind, pages_ in primary.items()}
        features = {
            "version": CACHE_VERSION,
            "report_id": report_id,
            "page_count": doc.page_count,
            "bookmarks": [[lvl, title, page] for lvl, title, page in doc.get_toc(simple=True)],
            "pages": pages,
            "contents_pages": contents_pages,
            "contents_entries": contents_entries,
            "statement_titled": statement_titled,
            "statement_rows": statement_rows,
            "statement_image": statement_image,
            "statement_primary": primary,
            "statement_table_rows": rows,
            "highlight_numbers": highlight_numbers,
        }
        write_json_gz(blocks_path, blocks_out)
        write_json_gz(features_path, features)
        return {"report_id": report_id, "status": "ok", "pages": doc.page_count, "error": ""}
    except Exception as exc:
        return {"report_id": report_id, "status": "error", "pages": 0, "error": repr(exc)[:300]}


def _page_features(page: pymupdf.Page, page_count: int) -> tuple[dict, list[list]]:
    """Measure one page and split it into text blocks."""
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    blocks = [b for b in data["blocks"] if b.get("type") == 0]
    height = page.rect.height or 1.0
    body = _body_size(blocks)
    lines, positioned, headings, short_lines, block_records = [], [], [], [], []
    for block in blocks:
        block_lines = []
        for line in block["lines"]:
            text = _line_text(line)
            if not text:
                continue
            text = collapse_letter_spacing(text)
            block_lines.append(text)
            lines.append(text)
            positioned.append((text, line["bbox"][1]))
            if len(text) < 160:
                short_lines.append(text)
            size = _line_size(line)
            if body and size >= body * 1.25 and 3 <= len(text) <= 120 and re.search(r"[A-Za-z]{3,}", text):
                headings.append({"text": text, "size": round(size, 1), "y": round(line["bbox"][1] / height, 3)})
        if block_lines:
            block_text = "\n".join(block_lines)
            tokens = block_text.split()
            numeric_share = count_numbers(block_text) / max(1, len(tokens))
            kind = "numeric" if numeric_share >= 0.3 and len(tokens) >= 3 else "text"
            block_records.append([page.number + 1, kind, block_text])
    text = "\n".join(lines)
    chars = len(re.sub(r"\s", "", text))
    numbers = count_numbers(text)
    image_count = len(page.get_images(full=False))
    cover = _image_cover(page) if image_count and chars < 800 else 0.0
    page_share = page.number / max(1, page_count)
    titled, by_rows = _statement_kinds(short_lines, text, numbers, page_share)
    image_statement = []
    if numbers < STATEMENT_MIN_NUMBERS and image_count and chars < 800:
        image_statement = [k for k, rx in STATEMENT_HEADINGS.items() if any(rx.search(line) for line in short_lines)]
    merged_headings = _merge_heading_lines(headings)
    info = {
        "page": page.number + 1,
        "chars": chars,
        "words": len(text.split()),
        "numbers": numbers,
        "images": image_count,
        "image_cover": round(cover, 3),
        "headings": [{**h, "norm": re.sub(r"[^a-z]", "", h["text"].lower())} for h in merged_headings[:6]],
        "folio": _folio(positioned, height),
        "continued": bool(CONTINUED.search(text[:600])),
        "_titled": titled,
        "_by_rows": by_rows,
        "_image_statement": image_statement,
        "_lines": lines,
    }
    return info, block_records


def _drop_running_headers(pages: list[dict]) -> None:
    """Remove headings that repeat on many pages, such as running headers, in place."""
    counts = Counter(h["norm"] for p in pages for h in p["headings"])
    limit = max(4, len(pages) * 0.05)
    for page in pages:
        page["headings"] = [
            {k: v for k, v in h.items() if k != "norm"}
            for h in page["headings"]
            if counts[h["norm"]] <= limit and not is_running_header(h["text"])
        ]


def _merge_heading_lines(headings: list[dict]) -> list[dict]:
    """Merge consecutive heading lines of the same size into one heading, largest first."""
    merged: list[dict] = []
    for h in headings:
        if merged and abs(merged[-1]["size"] - h["size"]) < 0.6 and 0 <= h["y"] - merged[-1]["y_last"] < 0.06:
            merged[-1]["text"] += " " + h["text"]
            merged[-1]["y_last"] = h["y"]
        else:
            merged.append({**h, "y_last": h["y"]})
    for h in merged:
        h.pop("y_last")
    return sorted(merged, key=lambda h: (-h["size"], h["y"]))


def _primary_statement_pages(titled: dict[str, list[int]], by_rows: dict[str, list[int]], page_count: int) -> dict[str, list[int]]:
    """Pick the main pages of each statement.

    Candidates are pages after the front matter, preferring pages with both a statement
    heading and matching row labels. The primary statements sit close together, so each
    statement takes the run of candidate pages nearest to the page where most statements cluster.
    """
    floor = max(1, int(page_count * 0.15))
    candidates = {}
    for kind in STATEMENT_HEADINGS:
        row_pages = set(by_rows[kind])
        both = [p for p in titled[kind] if p > floor and p in row_pages]
        candidates[kind] = both or [p for p in titled[kind] if p > floor] or [p for p in by_rows[kind] if p > floor]
    all_pages = sorted({p for pages in candidates.values() for p in pages})
    if not all_pages:
        return {kind: [] for kind in STATEMENT_HEADINGS}
    anchor = max(all_pages, key=lambda p: (sum(any(abs(q - p) <= 8 for q in pages) for pages in candidates.values()), -p))
    primary = {}
    for kind, pages in candidates.items():
        runs: list[list[int]] = []
        for p in pages:
            if runs and p - runs[-1][-1] <= 1:
                runs[-1].append(p)
            else:
                runs.append([p])
        primary[kind] = min(runs, key=lambda run: abs(run[0] - anchor))[:3] if runs else []
    return primary


def build_cache(reports: dict[str, Path], cache_dir: Path, workers: int = 8, force: bool = False, progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """Extract features for every report not yet cached, in parallel processes.

    Args:
        reports: Report id mapped to PDF path.
        cache_dir: Folder for the cache files.
        workers: Number of worker processes.
        force: Re-extract reports that are already cached.
        progress: Optional callable taking (done, total) for progress display.

    Returns:
        Status records of the reports processed in this call.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = {
        rid: path for rid, path in reports.items()
        if force or cached_version(cache_paths(cache_dir, rid)[0]) != CACHE_VERSION or not cache_paths(cache_dir, rid)[1].exists()
    }
    results = []
    if not todo:
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract_report_features, str(path), rid, str(cache_dir)) for rid, path in todo.items()]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if progress:
                progress(done, len(futures))
    return results


def load_features(cache_dir: Path, report_id: str) -> dict:
    """Load the cached feature record of one report."""
    return read_json_gz(cache_paths(cache_dir, report_id)[0])


def load_blocks(cache_dir: Path, report_id: str) -> list[list]:
    """Load the cached text blocks of one report as ``[page, kind, text]``."""
    return read_json_gz(cache_paths(cache_dir, report_id)[1])
