"""Command-line entry point for the extraction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from docling_pipeline.config import ConfigError
from docling_pipeline.pipeline import run_pipeline
from docling_pipeline.stages.parse import DuplicateReportError

app = typer.Typer(add_completion=False, help="Extract annual reports from PDF with Docling.")


@app.callback()
def main() -> None:
    """Extract annual reports from PDF with Docling."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="Path to the YAML config.")],
    force: Annotated[bool, typer.Option(help="Redo work that would otherwise be skipped.")] = False,
    reports: Annotated[
        Optional[list[str]], typer.Option(help="Report names to include (repeat or comma-separate).")
    ] = None,
) -> None:
    """Run extraction for one config."""
    try:
        run_dir = run_pipeline(config, force=force, reports=_split(reports))
    except (ConfigError, DuplicateReportError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"done: {run_dir}")


def _split(values: list[str] | None) -> list[str] | None:
    """Flatten repeated and comma-separated option values."""
    if not values:
        return None
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    app()
