"""Tests for report discovery and the parse stage, using a stub converter."""

from __future__ import annotations

import csv
import dataclasses
import time
from pathlib import Path

import pytest

from slm_bench.config import Config, load_config
from slm_bench.stages.parse import DuplicateReportError, discover_reports, run_parse

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
LONG_PAGE = "Revenue grew strongly. " * 20


def touch_pdf(path: Path) -> Path:
    """Create an empty placeholder PDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n")
    return path


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    """Build a nested reports folder with annual and interim reports."""
    root = tmp_path / "reports"
    touch_pdf(root / "Commercial Bank" / "commercialbank_2024.pdf")
    touch_pdf(root / "John Keells" / "Annual" / "johnkeells_2024.pdf")
    touch_pdf(root / "John Keells" / "Interim" / "johnkeells_2024_june.pdf")
    return root


@pytest.fixture
def config(tmp_path: Path, reports_dir: Path) -> Config:
    """Return the baseline config pointed at temporary folders."""
    base = load_config(BASELINE)
    data = dataclasses.replace(
        base.data, reports_dir=str(reports_dir), parsed_dir=str(tmp_path / "parsed")
    )
    return dataclasses.replace(base, data=data)


class StubConverter:
    """Converter that returns fixed markdown and records the PDFs it saw."""

    def __init__(
        self, pages: int = 2, fail_on: str | None = None, text: str = LONG_PAGE, delay: float = 0.0
    ) -> None:
        """Configure the page count, a failing report, the page text and a per-call delay."""
        self.pages = pages
        self.fail_on = fail_on
        self.text = text
        self.delay = delay
        self.calls: list[str] = []

    def __call__(self, pdf: Path) -> tuple[str, int]:
        """Return markdown for a PDF, or raise for the configured failing report."""
        self.calls.append(pdf.stem)
        time.sleep(self.delay)
        if pdf.stem == self.fail_on:
            raise RuntimeError("broken pdf")
        return f"# {pdf.stem}\n\n| a | b |\n|---|---|\n\n" + self.text * self.pages, self.pages


def read_log(run_dir: Path) -> list[dict[str, str]]:
    """Read the extraction log rows."""
    with (run_dir / "extraction_log.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_discover_skips_excluded_and_finds_nested(reports_dir: Path) -> None:
    """Find nested annual PDFs and skip the Interim folder."""
    found = discover_reports(reports_dir, ["interim"])
    assert list(found) == ["commercialbank_2024", "johnkeells_2024"]
    assert found["johnkeells_2024"].parent.name == "Annual"


def test_discover_without_exclusions_includes_interim(reports_dir: Path) -> None:
    """Include every PDF when nothing is excluded."""
    assert "johnkeells_2024_june" in discover_reports(reports_dir)


def test_discover_rejects_duplicate_names(reports_dir: Path) -> None:
    """Refuse two PDFs that map to the same report name."""
    touch_pdf(reports_dir / "Other" / "CommercialBank_2024.PDF")
    with pytest.raises(DuplicateReportError, match="commercialbank_2024"):
        discover_reports(reports_dir)


def test_parse_writes_markdown_and_log(config: Config, tmp_path: Path) -> None:
    """Convert each annual report and log pages, chars and status."""
    run_dir = tmp_path / "runs" / "baseline"
    stub = StubConverter()
    records = run_parse(config, run_dir, converter=stub)
    assert sorted(stub.calls) == ["commercialbank_2024", "johnkeells_2024"]
    assert [r.status for r in records] == ["ok", "ok"]
    parsed = tmp_path / "parsed"
    assert sorted(p.name for p in parsed.iterdir()) == ["commercialbank_2024.md", "johnkeells_2024.md"]
    rows = read_log(run_dir)
    assert list(rows[0]) == ["report", "source", "pages", "chars", "seconds", "status", "low_text"]
    assert rows[0]["pages"] == "2"
    assert int(rows[0]["chars"]) == len((parsed / "commercialbank_2024.md").read_text(encoding="utf-8"))
    assert rows[0]["low_text"] == "False"


def test_parse_skips_existing_unless_forced(config: Config, tmp_path: Path) -> None:
    """Skip reports with markdown on rerun, and redo them with force."""
    run_dir = tmp_path / "runs" / "baseline"
    initial = run_parse(config, run_dir, converter=StubConverter(delay=0.05))
    assert all(r.seconds > 0 for r in initial)
    second = StubConverter()
    records = run_parse(config, run_dir, converter=second)
    assert second.calls == []
    assert {r.status for r in records} == {"skipped"}
    assert {row["pages"] for row in read_log(run_dir)} == {"2"}
    assert all(r.seconds == first.seconds for r, first in zip(records, initial))
    forced = StubConverter()
    run_parse(config, run_dir, force=True, converter=forced)
    assert len(forced.calls) == 2


def test_parse_report_filter(config: Config, tmp_path: Path) -> None:
    """Convert only the requested report and keep other log rows."""
    run_dir = tmp_path / "runs" / "baseline"
    run_parse(config, run_dir, converter=StubConverter())
    stub = StubConverter()
    run_parse(config, run_dir, force=True, report_filter=["JohnKeells_2024"], converter=stub)
    assert stub.calls == ["johnkeells_2024"]
    assert [row["report"] for row in read_log(run_dir)] == ["commercialbank_2024", "johnkeells_2024"]


def test_parse_continues_after_failure(config: Config, tmp_path: Path) -> None:
    """Log a failed report without writing markdown and keep going."""
    run_dir = tmp_path / "runs" / "baseline"
    records = run_parse(config, run_dir, converter=StubConverter(fail_on="commercialbank_2024"))
    status = {r.report: r.status for r in records}
    assert status["commercialbank_2024"].startswith("failed: RuntimeError")
    assert status["johnkeells_2024"] == "ok"
    parsed = tmp_path / "parsed"
    assert not (parsed / "commercialbank_2024.md").exists()
    assert not list(parsed.glob("*.tmp"))


def test_log_written_after_each_report(config: Config, tmp_path: Path) -> None:
    """Keep finished rows in the log when a later report crashes the run."""
    run_dir = tmp_path / "runs" / "baseline"

    def crash_on_second(pdf: Path) -> tuple[str, int]:
        """Convert the first report, then abort as if the process was killed."""
        if pdf.stem == "johnkeells_2024":
            raise KeyboardInterrupt
        return "# ok\n\n" + LONG_PAGE * 2, 2

    with pytest.raises(KeyboardInterrupt):
        run_parse(config, run_dir, converter=crash_on_second)
    assert [row["report"] for row in read_log(run_dir)] == ["commercialbank_2024"]


def test_failed_report_is_retried(config: Config, tmp_path: Path) -> None:
    """Retry a report that failed earlier, since it has no markdown."""
    run_dir = tmp_path / "runs" / "baseline"
    run_parse(config, run_dir, converter=StubConverter(fail_on="commercialbank_2024"))
    retry = StubConverter()
    run_parse(config, run_dir, converter=retry)
    assert retry.calls == ["commercialbank_2024"]


def test_low_text_flagged(config: Config, tmp_path: Path) -> None:
    """Flag reports with under about 200 characters per page."""
    records = run_parse(config, tmp_path / "run", converter=StubConverter(pages=10, text="x"))
    assert all(r.low_text for r in records)
