"""Orchestrate the pipeline stages for one run."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path

from slm_bench.config import Config, freeze_config, load_config
from slm_bench.stages.colab_import import run_colab_import
from slm_bench.stages.parse import run_parse

STAGES = ("parse", "prepare", "infer", "score", "report")
DRY_RUN_SUFFIX = "_dryrun"

logger = logging.getLogger(__name__)


class StageNotImplementedError(RuntimeError):
    """Raised when a requested stage has not been built yet."""


def run_pipeline(
    config_path: str | Path,
    stage: str | None = None,
    force: bool = False,
    reports: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    runs: int | None = None,
    dry_run: bool = False,
    runs_dir: str | Path = "runs",
) -> Path:
    """Load and freeze the config, then run one stage or all stages in order.

    Args:
        config_path: Path to the YAML config.
        stage: Single stage to run; all stages when None.
        force: Redo work that would otherwise be skipped.
        reports: Restrict to these report names.
        models: Restrict to these model names.
        runs: Override the number of runs per key.
        dry_run: Use the fake backend and a separate run folder.
        runs_dir: Folder that holds all run folders.

    Returns:
        The run folder.

    Raises:
        ValueError: If the stage name is unknown.
        StageNotImplementedError: If a requested stage has not been built yet.
    """
    if stage is not None and stage not in STAGES:
        raise ValueError(f"unknown stage '{stage}'; choose from {', '.join(STAGES)}")
    config = load_config(config_path)
    if dry_run:
        config = dataclasses.replace(config, run_id=config.run_id + DRY_RUN_SUFFIX)
    freeze_config(config, config_path, runs_dir)
    run_dir = Path(runs_dir) / config.run_id

    for name in [stage] if stage else STAGES:
        _run_stage(name, config, run_dir, force, reports)
    return run_dir


def _run_stage(
    name: str, config: Config, run_dir: Path, force: bool, reports: Sequence[str] | None
) -> None:
    """Dispatch one stage by name."""
    logger.info("stage %s (run %s)", name, config.run_id)
    if name == "parse":
        parse_stage = run_colab_import if config.parse.engine == "colab" else run_parse
        parse_stage(config, run_dir, force=force, report_filter=reports)
        return
    raise StageNotImplementedError(f"stage '{name}' is not implemented yet")
