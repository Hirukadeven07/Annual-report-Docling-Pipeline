"""Parse report PDFs to markdown with Docling."""

from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, fields
from pathlib import Path

from docling_pipeline.config import Config, ParseConfig
from docling_pipeline.schemas import ExtractionRecord

LOW_TEXT_CHARS_PER_PAGE = 200
LOG_FILENAME = "extraction_log.csv"

Converter = Callable[[Path], tuple[str, int]]

logger = logging.getLogger(__name__)


class DuplicateReportError(ValueError):
    """Raised when two PDFs map to the same report name."""


def discover_reports(reports_dir: str | Path, exclude_dirs: Iterable[str] = ()) -> dict[str, Path]:
    """Find report PDFs recursively and key them by lowercased filename stem.

    Args:
        reports_dir: Root folder to search.
        exclude_dirs: Folder names to skip anywhere below the root, case-insensitive.

    Returns:
        Report name mapped to PDF path, sorted by name.

    Raises:
        DuplicateReportError: If two PDFs share the same report name.
    """
    root = Path(reports_dir)
    excluded = {name.lower() for name in exclude_dirs}
    found: dict[str, Path] = {}
    for pdf in sorted(root.rglob("*")):
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            continue
        parent_parts = pdf.relative_to(root).parts[:-1]
        if any(part.lower() in excluded for part in parent_parts):
            continue
        name = pdf.stem.lower()
        if name in found:
            raise DuplicateReportError(
                f"report name '{name}' is used by both {found[name]} and {pdf}; rename one"
            )
        found[name] = pdf
    return dict(sorted(found.items()))


def build_converter(parse_cfg: ParseConfig) -> Converter:
    """Create a Docling-backed converter returning markdown and page count."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = parse_cfg.ocr
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode(parse_cfg.table_mode)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    def convert(pdf: Path) -> tuple[str, int]:
        """Convert one PDF and return its markdown and page count."""
        document = converter.convert(pdf).document
        return document.export_to_markdown(), document.num_pages()

    return convert


def run_parse(
    config: Config,
    run_dir: str | Path,
    force: bool = False,
    report_filter: Iterable[str] | None = None,
    converter: Converter | None = None,
) -> list[ExtractionRecord]:
    """Convert every discovered PDF to markdown and write the extraction log.

    Reports whose markdown already exists are skipped unless forced. A failure
    on one report is logged and does not stop the others. The log is rewritten
    after each report so an interrupted run keeps the rows it finished.

    Args:
        config: The run configuration.
        run_dir: The run folder that receives extraction_log.csv.
        force: Re-convert reports that already have markdown.
        report_filter: Report names to process; all reports when None.
        converter: Conversion function; a Docling converter is built when None.

    Returns:
        One record per processed report.
    """
    reports = discover_reports(config.data.reports_dir, config.parse.exclude_dirs)
    selected = select_reports(reports, report_filter)
    parsed_dir = Path(config.data.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(run_dir) / LOG_FILENAME
    previous = read_log(log_path)

    records = []
    for name, pdf in selected.items():
        out_path = parsed_dir / f"{name}.md"
        if out_path.exists() and not force:
            records.append(_skipped_record(name, pdf, out_path, previous.get(name)))
            logger.info("skip %s (markdown exists)", name)
        else:
            if converter is None:
                converter = build_converter(config.parse)
            records.append(_convert_one(name, pdf, out_path, converter))
        write_log(log_path, previous, records)
    return records


def select_reports(reports: dict[str, Path], report_filter: Iterable[str] | None) -> dict[str, Path]:
    """Restrict discovered reports to the requested names, warning on unknown ones."""
    if report_filter is None:
        return reports
    wanted = [name.lower() for name in report_filter]
    for name in wanted:
        if name not in reports:
            logger.warning("report '%s' not found in reports folder", name)
    return {name: path for name, path in reports.items() if name in wanted}


def _convert_one(name: str, pdf: Path, out_path: Path, converter: Converter) -> ExtractionRecord:
    """Convert one PDF, write its markdown atomically and describe the outcome."""
    logger.info("parse %s", name)
    start = time.perf_counter()
    try:
        markdown, pages = converter(pdf)
        write_atomic(out_path, markdown)
    except Exception as exc:
        seconds = time.perf_counter() - start
        logger.error("failed %s: %s", name, exc)
        return ExtractionRecord(
            report=name,
            source=str(pdf),
            pages=None,
            chars=None,
            seconds=round(seconds, 2),
            status=f"failed: {type(exc).__name__}: {exc}",
            low_text=False,
        )
    seconds = time.perf_counter() - start
    record = ExtractionRecord(
        report=name,
        source=str(pdf),
        pages=pages,
        chars=len(markdown),
        seconds=round(seconds, 2),
        status="ok",
        low_text=is_low_text(len(markdown), pages),
    )
    if record.low_text:
        logger.warning("%s has little text per page, possibly a scan", name)
    return record


def _skipped_record(
    name: str, pdf: Path, out_path: Path, previous: ExtractionRecord | None
) -> ExtractionRecord:
    """Describe a report skipped because its markdown already exists.

    Pages and conversion time are carried over from the earlier log row, when
    there is one, so the log keeps the original measurements.
    """
    chars = len(out_path.read_text(encoding="utf-8"))
    pages = previous.pages if previous else None
    return ExtractionRecord(
        report=name,
        source=str(pdf),
        pages=pages,
        chars=chars,
        seconds=previous.seconds if previous else 0.0,
        status="skipped",
        low_text=is_low_text(chars, pages),
    )


def is_low_text(chars: int, pages: int | None) -> bool:
    """Return whether a document has too little text per page to be text-based."""
    if not pages:
        return False
    return chars / pages < LOW_TEXT_CHARS_PER_PAGE


def write_atomic(path: Path, text: str) -> None:
    """Write text through a temporary file so a partial file never looks finished."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def read_log(path: Path) -> dict[str, ExtractionRecord]:
    """Load an existing extraction log keyed by report name."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["report"]: _record_from_row(row) for row in csv.DictReader(handle)}


def _record_from_row(row: dict[str, str]) -> ExtractionRecord:
    """Convert a CSV row back into an extraction record."""
    return ExtractionRecord(
        report=row["report"],
        source=row["source"],
        pages=int(row["pages"]) if row["pages"] else None,
        chars=int(row["chars"]) if row["chars"] else None,
        seconds=float(row["seconds"] or 0),
        status=row["status"],
        low_text=row["low_text"] == "True",
    )


def write_log(
    path: Path, previous: dict[str, ExtractionRecord], records: list[ExtractionRecord]
) -> None:
    """Merge new records into the extraction log and write it sorted by report."""
    merged = dict(previous)
    merged.update({record.report: record for record in records})
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f.name for f in fields(ExtractionRecord)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name in sorted(merged):
            row = asdict(merged[name])
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
