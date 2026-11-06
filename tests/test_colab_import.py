"""Tests for importing Colab extraction output into the project."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import pytest

from slm_bench.config import Config, load_config
from slm_bench.pipeline import run_pipeline
from slm_bench.stages.colab_import import run_colab_import
from slm_bench.stages.parse import DuplicateReportError

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
LOG_HEADER = "file,status,pages,seconds,pages_per_second\n"
MARKDOWN = "# Report\n\n| a | b |\n|---|---|\n\n" + "Revenue grew strongly. " * 50


def touch_pdf(path: Path) -> Path:
    """Create an empty placeholder PDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n")
    return path


def write_colab_output(root: Path, relative: str, markdown: str = MARKDOWN, pages: int = 2) -> None:
    """Write the files the notebook produces for one report."""
    json_path = root / f"{relative}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    document = {"pages": {str(n): {} for n in range(1, pages + 1)}}
    json_path.write_text(json.dumps(document), encoding="utf-8")
    json_path.with_suffix(".md").write_bytes(markdown.encode("utf-8"))
    stem = json_path.stem
    pictures = json_path.with_name(f"{stem}_pictures.json")
    pictures.write_text(json.dumps([{"image": f"{stem}_pictures/p001_001.png"}]), encoding="utf-8")
    image_dir = json_path.with_name(f"{stem}_pictures")
    image_dir.mkdir()
    (image_dir / "p001_001.png").write_bytes(b"png")


def write_colab_log(root: Path, *rows: str) -> None:
    """Write the notebook's extraction log."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "extraction_log.csv").write_text(LOG_HEADER + "".join(rows), encoding="utf-8")


def read_log(run_dir: Path) -> list[dict[str, str]]:
    """Read the extraction log rows."""
    with (run_dir / "extraction_log.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Return a colab-engine config pointed at temporary folders with two reports."""
    reports = tmp_path / "reports"
    touch_pdf(reports / "Commercial Bank" / "commercialbank_2024.pdf")
    touch_pdf(reports / "John Keells" / "Annual" / "johnkeells_2024.pdf")
    touch_pdf(reports / "John Keells" / "Interim" / "johnkeells_2024_june.pdf")
    base = load_config(BASELINE)
    data = dataclasses.replace(
        base.data,
        reports_dir=str(reports),
        parsed_dir=str(tmp_path / "parsed"),
        extracted_dir=str(tmp_path / "extracted"),
    )
    parse = dataclasses.replace(base.parse, import_dir=str(tmp_path / "drive"))
    return dataclasses.replace(base, data=data, parse=parse)


@pytest.fixture
def drive(tmp_path: Path) -> Path:
    """Return the Colab output folder holding both annual reports."""
    root = tmp_path / "drive"
    write_colab_output(root, "Commercial Bank/commercialbank_2024")
    write_colab_output(root, "John Keells/Annual/johnkeells_2024", pages=3)
    write_colab_log(
        root,
        "Commercial Bank/commercialbank_2024.pdf,success,2,120.5,0.02\n",
        "John Keells/Annual/johnkeells_2024.pdf,partial_success,3,300.0,0.01\n",
    )
    return root


def test_import_sorts_output_like_the_reports_folder(
    config: Config, drive: Path, tmp_path: Path
) -> None:
    """Copy every artefact into the report's own subfolder."""
    run_colab_import(config, tmp_path / "run")
    folder = tmp_path / "extracted" / "John Keells" / "Annual"
    assert (folder / "johnkeells_2024.json").is_file()
    assert (folder / "johnkeells_2024.md").is_file()
    assert (folder / "johnkeells_2024_pictures.json").is_file()
    assert (folder / "johnkeells_2024_pictures" / "p001_001.png").read_bytes() == b"png"
    assert (tmp_path / "extracted" / "Commercial Bank" / "commercialbank_2024.json").is_file()


def test_import_copies_markdown_for_later_stages(
    config: Config, drive: Path, tmp_path: Path
) -> None:
    """Place each markdown file flat in the parsed folder, byte for byte."""
    run_colab_import(config, tmp_path / "run")
    parsed = tmp_path / "parsed"
    assert sorted(p.name for p in parsed.iterdir()) == [
        "commercialbank_2024.md",
        "johnkeells_2024.md",
    ]
    source = drive / "Commercial Bank" / "commercialbank_2024.md"
    assert (parsed / "commercialbank_2024.md").read_bytes() == source.read_bytes()


def test_import_logs_colab_measurements(config: Config, drive: Path, tmp_path: Path) -> None:
    """Carry pages and seconds from the notebook log and report partial conversions."""
    run_colab_import(config, tmp_path / "run")
    rows = {row["report"]: row for row in read_log(tmp_path / "run")}
    assert rows["commercialbank_2024"]["pages"] == "2"
    assert rows["commercialbank_2024"]["seconds"] == "120.5"
    assert rows["commercialbank_2024"]["status"] == "ok"
    assert int(rows["commercialbank_2024"]["chars"]) == len(MARKDOWN)
    assert rows["johnkeells_2024"]["status"] == "partial_success"


def test_last_notebook_log_row_wins(config: Config, drive: Path, tmp_path: Path) -> None:
    """Use the newest row after a report was retried in Colab."""
    write_colab_log(
        drive,
        "Commercial Bank/commercialbank_2024.pdf,failure,0,5.0,0.0\n",
        "Commercial Bank/commercialbank_2024.pdf,success,2,120.5,0.02\n",
    )
    records = run_colab_import(config, tmp_path / "run", report_filter=["commercialbank_2024"])
    assert records[0].status == "ok"


def test_pages_fall_back_to_the_docling_json(config: Config, drive: Path, tmp_path: Path) -> None:
    """Count pages from the JSON when the notebook log has no row."""
    write_colab_log(drive)
    records = run_colab_import(config, tmp_path / "run")
    assert [r.pages for r in records] == [2, 3]
    assert {r.seconds for r in records} == {0.0}


def test_import_ignores_interim_output(config: Config, drive: Path, tmp_path: Path) -> None:
    """Leave out reports excluded from the benchmark even when Colab converted them."""
    write_colab_output(drive, "John Keells/Interim/johnkeells_2024_june")
    records = run_colab_import(config, tmp_path / "run")
    assert [r.report for r in records] == ["commercialbank_2024", "johnkeells_2024"]
    assert not (tmp_path / "extracted" / "John Keells" / "Interim").exists()


def test_import_matches_names_case_insensitively(config: Config, tmp_path: Path) -> None:
    """Find Colab output whose folder and filename casing differ from the local PDF."""
    write_colab_output(tmp_path / "drive", "COMMERCIAL/CommercialBank_2024")
    records = run_colab_import(config, tmp_path / "run", report_filter=["commercialbank_2024"])
    assert records[0].status == "ok"
    folder = tmp_path / "extracted" / "Commercial Bank"
    assert (folder / "CommercialBank_2024_pictures" / "p001_001.png").is_file()


def test_missing_report_does_not_stop_the_run(config: Config, tmp_path: Path) -> None:
    """Log a report with no Colab output and still import the others."""
    write_colab_output(tmp_path / "drive", "Commercial Bank/commercialbank_2024")
    records = run_colab_import(config, tmp_path / "run")
    status = {r.report: r.status for r in records}
    assert status == {"commercialbank_2024": "ok", "johnkeells_2024": "missing"}
    assert not (tmp_path / "parsed" / "johnkeells_2024.md").exists()


def test_incomplete_output_is_not_imported(config: Config, tmp_path: Path) -> None:
    """Treat a report without its pictures file as not finished yet."""
    drive = tmp_path / "drive"
    write_colab_output(drive, "Commercial Bank/commercialbank_2024")
    (drive / "Commercial Bank" / "commercialbank_2024_pictures.json").unlink()
    records = run_colab_import(config, tmp_path / "run", report_filter=["commercialbank_2024"])
    assert records[0].status == "missing"
    assert not (tmp_path / "parsed" / "commercialbank_2024.md").exists()


def test_failed_conversion_is_reported(config: Config, tmp_path: Path) -> None:
    """Show the notebook's failure status for a report with no output."""
    write_colab_log(tmp_path / "drive", "Commercial Bank/commercialbank_2024.pdf,failure,0,5.0,0.0\n")
    records = run_colab_import(config, tmp_path / "run", report_filter=["commercialbank_2024"])
    assert records[0].status == "failed in colab: failure"


def test_rerun_skips_existing_unless_forced(config: Config, drive: Path, tmp_path: Path) -> None:
    """Skip finished reports and keep their measurements, then replace them with force."""
    run_dir = tmp_path / "run"
    run_colab_import(config, run_dir)
    (drive / "Commercial Bank" / "commercialbank_2024.md").write_text("# new\n", encoding="utf-8")
    records = run_colab_import(config, run_dir)
    assert {r.status for r in records} == {"skipped"}
    assert {row["seconds"] for row in read_log(run_dir)} == {"120.5", "300.0"}
    parsed = tmp_path / "parsed" / "commercialbank_2024.md"
    assert parsed.read_bytes() == MARKDOWN.encode("utf-8")
    run_colab_import(config, run_dir, force=True)
    assert parsed.read_text(encoding="utf-8") == "# new\n"


def test_older_markdown_needs_force(
    config: Config, drive: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep older local markdown by default and warn that force replaces it."""
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "commercialbank_2024.md").write_text("# local docling\n", encoding="utf-8")
    records = run_colab_import(config, tmp_path / "run", report_filter=["commercialbank_2024"])
    assert records[0].status == "skipped"
    assert (parsed / "commercialbank_2024.md").read_text(encoding="utf-8") == "# local docling\n"
    assert "--force" in caplog.text
    run_colab_import(config, tmp_path / "run", force=True, report_filter=["commercialbank_2024"])
    assert (parsed / "commercialbank_2024.md").read_bytes() == MARKDOWN.encode("utf-8")


def test_force_keeps_existing_output_when_colab_has_none(config: Config, tmp_path: Path) -> None:
    """Leave a report's existing markdown alone when forced with nothing to import."""
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "johnkeells_2024.md").write_text("# local docling\n", encoding="utf-8")
    (tmp_path / "drive").mkdir()
    records = run_colab_import(config, tmp_path / "run", force=True, report_filter=["johnkeells_2024"])
    assert records[0].status == "missing"
    assert (parsed / "johnkeells_2024.md").read_text(encoding="utf-8") == "# local docling\n"


def test_force_replaces_stale_pictures(config: Config, drive: Path, tmp_path: Path) -> None:
    """Remove images from an earlier import that the new output no longer has."""
    run_colab_import(config, tmp_path / "run")
    folder = tmp_path / "extracted" / "Commercial Bank" / "commercialbank_2024_pictures"
    stale = folder / "p999_999.png"
    stale.write_bytes(b"old")
    run_colab_import(config, tmp_path / "run", force=True)
    assert not stale.exists()
    assert (folder / "p001_001.png").is_file()


def test_missing_import_folder_is_a_clear_error(config: Config, tmp_path: Path) -> None:
    """Stop with a useful message when Drive is not available."""
    with pytest.raises(ValueError, match="Colab output folder not found"):
        run_colab_import(config, tmp_path / "run")


def test_duplicate_colab_output_is_refused(config: Config, drive: Path, tmp_path: Path) -> None:
    """Refuse to guess between two outputs for one report."""
    write_colab_output(drive, "Other/commercialbank_2024")
    with pytest.raises(DuplicateReportError, match="commercialbank_2024"):
        run_colab_import(config, tmp_path / "run")


def test_low_text_flagged(config: Config, tmp_path: Path) -> None:
    """Flag a likely scan using the page count from Colab."""
    write_colab_output(tmp_path / "drive", "Commercial Bank/commercialbank_2024", "x", pages=10)
    records = run_colab_import(config, tmp_path / "run", report_filter=["commercialbank_2024"])
    assert records[0].low_text


def test_pipeline_uses_the_import_for_the_colab_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route the parse stage to the import without touching Docling."""
    root = tmp_path / "project"
    touch_pdf(root / "data" / "reports" / "Commercial Bank" / "commercialbank_2024.pdf")
    write_colab_output(root / "drive", "Commercial Bank/commercialbank_2024")
    lines = [
        "  import_dir: drive" if line.strip().startswith("import_dir:") else line
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
    ]
    (root / "config.yaml").write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.chdir(root)
    run_dir = run_pipeline("config.yaml", stage="parse", runs_dir="runs")
    assert (root / "data" / "parsed" / "commercialbank_2024.md").is_file()
    assert read_log(run_dir)[0]["status"] == "ok"
