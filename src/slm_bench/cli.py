"""Command-line entry point for the benchmarking pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from slm_bench.config import ConfigError
from slm_bench.pipeline import STAGES, StageNotImplementedError, run_pipeline
from slm_bench.stages.parse import DuplicateReportError

app = typer.Typer(add_completion=False, help="Benchmark small language models on CSE annual reports.")


@app.callback()
def main() -> None:
    """Benchmark small language models on CSE annual reports."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="Path to the YAML config.")],
    stage: Annotated[
        Optional[str], typer.Option(help=f"Run a single stage: {', '.join(STAGES)}.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Redo work that would otherwise be skipped.")] = False,
    reports: Annotated[
        Optional[list[str]], typer.Option(help="Report names to include (repeat or comma-separate).")
    ] = None,
    models: Annotated[
        Optional[list[str]], typer.Option(help="Model names to include (repeat or comma-separate).")
    ] = None,
    runs: Annotated[Optional[int], typer.Option(help="Override the number of runs per key.")] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Use the fake backend and a separate run folder.")
    ] = False,
) -> None:
    """Run the pipeline, or a single stage of it, for one config."""
    try:
        run_dir = run_pipeline(
            config,
            stage=stage,
            force=force,
            reports=_split(reports),
            models=_split(models),
            runs=runs,
            dry_run=dry_run,
        )
    except (ConfigError, DuplicateReportError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except StageNotImplementedError as exc:
        typer.echo(f"stopped: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"done: {run_dir}")


def _split(values: list[str] | None) -> list[str] | None:
    """Flatten repeated and comma-separated option values."""
    if not values:
        return None
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    app()
