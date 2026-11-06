"""Dataclasses for the records passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Unit:
    """One piece of model input produced by a strategy from a report."""

    report: str
    unit_id: str
    text: str
    section: str | None = None
    pages: tuple[int, ...] | None = None
    tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawOutput:
    """One row of raw_outputs.csv: a single model call and its measurements."""

    model: str
    report: str
    unit_id: str
    prompt: str
    run: int
    ctx: int
    report_tokens: int
    tokens_fed: int
    truncated: bool
    chars_fed: int
    prompt_eval_count: int | None
    eval_count: int
    latency_s: float
    ttft_s: float | None
    tok_s: float | None
    peak_mem_mb: float
    error: str
    output: str


@dataclass
class GoldFile:
    """Hand-verified reference data for one report."""

    company: str
    year: int
    facts: dict[str, str | None]
    table_keys: list[str] = field(default_factory=list)
    risks: list[str | list[str]] = field(default_factory=list)
    reference_summary: list[str] = field(default_factory=list)


@dataclass
class ExtractionRecord:
    """One row of extraction_log.csv: the outcome of parsing one PDF."""

    report: str
    source: str
    pages: int | None
    chars: int | None
    seconds: float
    status: str
    low_text: bool
