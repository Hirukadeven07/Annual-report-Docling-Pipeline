"""Run the extraction pipeline for one config."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from docling_pipeline.config import freeze_config, load_config
from docling_pipeline.stages.colab_import import run_colab_import
from docling_pipeline.stages.parse import run_parse

logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str | Path,
    force: bool = False,
    reports: Sequence[str] | None = None,
    runs_dir: str | Path = "runs",
) -> Path:
    """Load and freeze the config, then extract with the configured engine.

    Args:
        config_path: Path to the YAML config.
        force: Redo work that would otherwise be skipped.
        reports: Restrict to these report names.
        runs_dir: Folder that holds all run folders.

    Returns:
        The run folder.
    """
    config = load_config(config_path)
    freeze_config(config, config_path, runs_dir)
    run_dir = Path(runs_dir) / config.run_id
    logger.info("extract with %s (run %s)", config.parse.engine, config.run_id)
    extract = run_colab_import if config.parse.engine == "colab" else run_parse
    extract(config, run_dir, force=force, report_filter=reports)
    return run_dir
