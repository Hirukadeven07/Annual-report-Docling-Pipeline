"""Import the Colab notebook's Docling output into the project."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from docling_pipeline.config import Config
from docling_pipeline.schemas import ExtractionRecord
from docling_pipeline.stages.parse import (
    LOG_FILENAME,
    DuplicateReportError,
    discover_reports,
    is_low_text,
    read_log,
    select_reports,
    write_log,
)

SUCCESS_STATUS = "success"
PICTURES_SUFFIX = "_pictures"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColabLogRow:
    """One report's row from the notebook's extraction log."""

    status: str
    pages: int | None
    seconds: float


def run_colab_import(
    config: Config,
    run_dir: str | Path,
    force: bool = False,
    report_filter: Iterable[str] | None = None,
) -> list[ExtractionRecord]:
    """Copy finished Colab output into the project and write the extraction log.

    Each report's Docling files go to the extracted folder, in the same
    subfolder its PDF has under the reports folder, and its markdown goes to
    the parsed folder. Reports with markdown already in place are skipped
    unless forced, so older local conversions are replaced only on request.
    A report with no finished output is logged and does not stop the others.

    Args:
        config: The run configuration.
        run_dir: The run folder that receives extraction_log.csv.
        force: Replace reports that already have markdown.
        report_filter: Report names to process; all reports when None.

    Returns:
        One record per processed report.

    Raises:
        ValueError: If the Colab output folder does not exist.
        DuplicateReportError: If two Colab outputs map to the same report name.
    """
    import_dir = Path(config.parse.import_dir)
    if not import_dir.is_dir():
        raise ValueError(
            f"Colab output folder not found: {import_dir}; check that Google Drive "
            "for Desktop is running and parse.import_dir is correct"
        )
    reports_dir = Path(config.data.reports_dir)
    reports = discover_reports(reports_dir, config.parse.exclude_dirs)
    selected = select_reports(reports, report_filter)
    outputs = _find_outputs(import_dir)
    colab_log = _read_colab_log(import_dir)
    parsed_dir = Path(config.data.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    extracted_root = Path(config.data.extracted_dir)
    log_path = Path(run_dir) / LOG_FILENAME
    previous = read_log(log_path)

    records = []
    for name, pdf in selected.items():
        markdown_path = parsed_dir / f"{name}.md"
        target_dir = extracted_root / pdf.parent.relative_to(reports_dir)
        source = outputs.get(name)
        if markdown_path.exists() and not force:
            records.append(_skipped_record(name, pdf, markdown_path, previous.get(name)))
            _warn_if_superseded(name, source, target_dir)
        else:
            records.append(
                _import_one(name, pdf, source, target_dir, markdown_path, colab_log.get(name))
            )
        write_log(log_path, previous, records)
    return records


def _find_outputs(import_dir: Path) -> dict[str, Path]:
    """Map report names to their Docling JSON files in the Colab output folder."""
    found: dict[str, Path] = {}
    for path in sorted(import_dir.rglob("*.json")):
        if path.stem.endswith(PICTURES_SUFFIX):
            continue
        name = path.stem.lower()
        if name in found:
            raise DuplicateReportError(
                f"report name '{name}' has two Colab outputs: {found[name]} and {path}; remove one"
            )
        found[name] = path
    return found


def _read_colab_log(import_dir: Path) -> dict[str, ColabLogRow]:
    """Read the notebook's log, keeping the newest row for each report."""
    path = import_dir / LOG_FILENAME
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            PurePosixPath(row["file"].replace("\\", "/")).stem.lower(): ColabLogRow(
                status=row["status"],
                pages=int(row["pages"]) if row["pages"] else None,
                seconds=float(row["seconds"] or 0),
            )
            for row in csv.DictReader(handle)
        }


def _import_one(
    name: str,
    pdf: Path,
    source: Path | None,
    target_dir: Path,
    markdown_path: Path,
    log_row: ColabLogRow | None,
) -> ExtractionRecord:
    """Copy one report's output and describe the outcome."""
    if source is None or not _is_complete(source):
        return _unavailable_record(name, pdf, log_row)
    logger.info("import %s", name)
    try:
        _copy_output(source, target_dir)
        _copy_atomic(source.with_suffix(".md"), markdown_path)
        chars = len(markdown_path.read_text(encoding="utf-8"))
        pages = (log_row.pages if log_row else None) or _count_pages(source)
    except (OSError, ValueError) as exc:
        logger.error("failed %s: %s", name, exc)
        return ExtractionRecord(
            report=name,
            source=str(pdf),
            pages=None,
            chars=None,
            seconds=0.0,
            status=f"failed: {type(exc).__name__}: {exc}",
            low_text=False,
        )
    record = ExtractionRecord(
        report=name,
        source=str(pdf),
        pages=pages,
        chars=chars,
        seconds=log_row.seconds if log_row else 0.0,
        status="ok" if log_row is None or log_row.status == SUCCESS_STATUS else log_row.status,
        low_text=is_low_text(chars, pages),
    )
    if record.low_text:
        logger.warning("%s has little text per page, possibly a scan", name)
    return record


def _is_complete(source: Path) -> bool:
    """Return whether the notebook has finished every file a report needs."""
    return source.with_suffix(".md").is_file() and _pictures_json(source).is_file()


def _pictures_json(source: Path) -> Path:
    """Return the pictures file that belongs to a Docling JSON file."""
    return source.with_name(f"{source.stem}{PICTURES_SUFFIX}.json")


def _copy_output(source: Path, target_dir: Path) -> None:
    """Copy a report's Docling files, replacing any images from an earlier import."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in (source, source.with_suffix(".md"), _pictures_json(source)):
        _copy_atomic(path, target_dir / path.name)
    images = source.with_name(f"{source.stem}{PICTURES_SUFFIX}")
    images_target = target_dir / images.name
    if images_target.exists():
        shutil.rmtree(images_target)
    if images.is_dir():
        shutil.copytree(images, images_target)


def _copy_atomic(source: Path, target: Path) -> None:
    """Copy a file through a temporary name so a partial copy never looks finished."""
    tmp_path = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp_path)
    os.replace(tmp_path, target)


def _count_pages(source: Path) -> int | None:
    """Count the pages recorded in a Docling JSON file."""
    return len(json.loads(source.read_text(encoding="utf-8")).get("pages", {})) or None


def _unavailable_record(name: str, pdf: Path, log_row: ColabLogRow | None) -> ExtractionRecord:
    """Describe a report with no finished Colab output."""
    failed = log_row is not None and log_row.status != SUCCESS_STATUS
    status = f"failed in colab: {log_row.status}" if failed else "missing"
    logger.warning("%s: %s", name, status)
    return ExtractionRecord(
        report=name,
        source=str(pdf),
        pages=None,
        chars=None,
        seconds=log_row.seconds if log_row else 0.0,
        status=status,
        low_text=False,
    )


def _skipped_record(
    name: str, pdf: Path, markdown_path: Path, previous: ExtractionRecord | None
) -> ExtractionRecord:
    """Describe a report skipped because its markdown already exists."""
    chars = len(markdown_path.read_text(encoding="utf-8"))
    pages = previous.pages if previous else None
    logger.info("skip %s (markdown exists)", name)
    return ExtractionRecord(
        report=name,
        source=str(pdf),
        pages=pages,
        chars=chars,
        seconds=previous.seconds if previous else 0.0,
        status="skipped",
        low_text=is_low_text(chars, pages),
    )


def _warn_if_superseded(name: str, source: Path | None, target_dir: Path) -> None:
    """Warn when Colab output exists for a report still using older markdown."""
    if source is None or (target_dir / source.name).exists():
        return
    logger.warning(
        "%s keeps its older markdown; use --force to replace it with the Colab output", name
    )
