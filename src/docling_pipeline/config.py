"""Load, validate and freeze the YAML run configuration."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

TABLE_MODES = ("accurate", "fast")
PARSE_ENGINES = ("docling", "colab")


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class DataConfig:
    """Locations of the input and output data folders."""

    reports_dir: str
    parsed_dir: str
    extracted_dir: str = "data/extracted"


@dataclass(frozen=True)
class ParseConfig:
    """Options for converting PDFs locally or importing the Colab notebook's output."""

    engine: str
    ocr: bool
    table_mode: str
    exclude_dirs: list[str] = field(default_factory=list)
    import_dir: str = ""


@dataclass(frozen=True)
class Config:
    """Complete, validated run configuration."""

    run_id: str
    data: DataConfig
    parse: ParseConfig


def load_config(path: str | Path) -> Config:
    """Read a YAML config file and validate it.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing, malformed or fails validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    return config_from_dict(raw)


def config_from_dict(raw: Any) -> Config:
    """Build and validate a configuration from parsed YAML data.

    Raises:
        ConfigError: If keys are unknown or missing, or values are invalid.
    """
    config = _build(Config, raw, "")
    _check_values(config)
    return config


def freeze_config(config: Config, source_path: str | Path, runs_dir: str | Path) -> Path:
    """Write a frozen copy of the config into the run folder.

    The source file is copied verbatim. An existing frozen copy is accepted
    only when it describes the same configuration.

    Args:
        config: The validated configuration.
        source_path: The YAML file the configuration was loaded from.
        runs_dir: Folder that holds all run folders.

    Returns:
        Path of the frozen config file.

    Raises:
        ConfigError: If the run folder already holds a different config.
    """
    run_dir = Path(runs_dir) / config.run_id
    frozen_path = run_dir / "config.yaml"
    if frozen_path.exists():
        existing = load_config(frozen_path)
        if asdict(existing) != asdict(config):
            raise ConfigError(
                f"{frozen_path} holds a different config; "
                "use a new run_id instead of changing a finished run"
            )
        return frozen_path
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(Path(source_path).read_text(encoding="utf-8"), encoding="utf-8")
    return frozen_path


def _build(target: Any, value: Any, path: str) -> Any:
    """Convert a parsed YAML value into the annotated type, checking it."""
    if dataclasses.is_dataclass(target):
        return _build_dataclass(target, value, path)
    if typing.get_origin(target) is list:
        if not isinstance(value, list):
            raise ConfigError(f"{_label(path)} must be a list")
        (item_type,) = typing.get_args(target)
        return [_build(item_type, item, f"{path}[{i}]") for i, item in enumerate(value)]
    return _check_scalar(target, value, path)


def _build_dataclass(cls: type, value: Any, path: str) -> Any:
    """Build one dataclass section, rejecting unknown and missing keys."""
    if not isinstance(value, dict):
        raise ConfigError(f"{_label(path)} must be a mapping")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ConfigError(f"unknown key: {_join(path, unknown[0])}")
    kwargs = {}
    for name, spec in fields.items():
        key_path = _join(path, name)
        if name not in value:
            has_default = (
                spec.default is not dataclasses.MISSING
                or spec.default_factory is not dataclasses.MISSING
            )
            if not has_default:
                raise ConfigError(f"missing key: {key_path}")
            continue
        kwargs[name] = _build(hints[name], value[name], key_path)
    return cls(**kwargs)


def _check_scalar(target: Any, value: Any, path: str) -> Any:
    """Check a scalar value against its annotated type."""
    if target is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{_label(path)} must be true or false")
        return value
    if target is str:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{_label(path)} must be a non-empty string")
        return value
    raise ConfigError(f"unsupported type at {_label(path)}")


def _check_values(config: Config) -> None:
    """Validate allowed choices across sections."""
    if config.parse.engine not in PARSE_ENGINES:
        raise ConfigError(f"parse.engine must be one of {list(PARSE_ENGINES)}")
    if config.parse.engine == "colab" and not config.parse.import_dir:
        raise ConfigError("parse.import_dir is required when parse.engine is colab")
    if config.parse.table_mode not in TABLE_MODES:
        raise ConfigError(f"parse.table_mode must be one of {list(TABLE_MODES)}")


def _join(path: str, key: str) -> str:
    """Append a key to a dotted config path."""
    return f"{path}.{key}" if path else key


def _label(path: str) -> str:
    """Return a readable name for a config path."""
    return path or "config"
