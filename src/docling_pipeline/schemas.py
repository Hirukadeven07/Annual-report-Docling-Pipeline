"""Dataclasses for the records the pipeline writes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractionRecord:
    """One row of extraction_log.csv: the outcome of extracting one PDF."""

    report: str
    source: str
    pages: int | None
    chars: int | None
    seconds: float
    status: str
    low_text: bool
