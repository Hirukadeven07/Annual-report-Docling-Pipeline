"""Tests for config loading, validation and freezing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from docling_pipeline.config import ConfigError, config_from_dict, freeze_config, load_config

EXTRACTION_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "extraction.yaml"


def config_dict() -> dict:
    """Return the shipped config as a fresh mutable dict."""
    return yaml.safe_load(EXTRACTION_CONFIG.read_text(encoding="utf-8"))


def write_config(path: Path, data: dict) -> Path:
    """Write a config dict to a YAML file and return its path."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_extraction_config_loads() -> None:
    """Load the shipped config with the expected values."""
    config = load_config(EXTRACTION_CONFIG)
    assert config.run_id == "colab_extraction"
    assert config.data.extracted_dir == "data/extracted"
    assert config.parse.engine == "colab"
    assert config.parse.import_dir.endswith("chunkwise-cse/extracted_egret_large")
    assert config.parse.exclude_dirs == ["Interim"]


def test_unknown_key_is_named() -> None:
    """Reject an unknown nested key and report its dotted path."""
    data = config_dict()
    data["parse"]["ocr_mode"] = "x"
    with pytest.raises(ConfigError, match=r"unknown key: parse\.ocr_mode"):
        config_from_dict(data)


def test_unknown_top_level_key() -> None:
    """Reject an unknown top-level key."""
    data = config_dict()
    data["models"] = []
    with pytest.raises(ConfigError, match="unknown key: models"):
        config_from_dict(data)


def test_missing_key_is_named() -> None:
    """Reject a missing required key and report its path."""
    data = config_dict()
    del data["parse"]["ocr"]
    with pytest.raises(ConfigError, match=r"missing key: parse\.ocr"):
        config_from_dict(data)


def test_exclude_dirs_is_optional() -> None:
    """Default exclude_dirs to an empty list when omitted."""
    data = config_dict()
    del data["parse"]["exclude_dirs"]
    assert config_from_dict(data).parse.exclude_dirs == []


def test_extracted_dir_has_a_default() -> None:
    """Default extracted_dir when a config omits it."""
    data = config_dict()
    del data["data"]["extracted_dir"]
    assert config_from_dict(data).data.extracted_dir == "data/extracted"


def test_colab_engine_requires_import_dir() -> None:
    """Reject the colab engine when it has no folder to import from."""
    data = config_dict()
    del data["parse"]["import_dir"]
    with pytest.raises(ConfigError, match=r"parse\.import_dir"):
        config_from_dict(data)


def test_docling_engine_needs_no_import_dir() -> None:
    """Keep local Docling usable without a Colab output folder."""
    data = config_dict()
    data["parse"]["engine"] = "docling"
    del data["parse"]["import_dir"]
    assert config_from_dict(data).parse.engine == "docling"


def test_wrong_type_rejected() -> None:
    """Reject a string where a boolean is required."""
    data = config_dict()
    data["parse"]["ocr"] = "no"
    with pytest.raises(ConfigError, match=r"parse\.ocr"):
        config_from_dict(data)


def test_exclude_dirs_must_be_a_list() -> None:
    """Reject a single folder name where a list is required."""
    data = config_dict()
    data["parse"]["exclude_dirs"] = "Interim"
    with pytest.raises(ConfigError, match=r"parse\.exclude_dirs"):
        config_from_dict(data)


@pytest.mark.parametrize(("key", "value"), [("table_mode", "slow"), ("engine", "pymupdf")])
def test_invalid_values_rejected(key: str, value: str) -> None:
    """Reject values outside the allowed choices."""
    data = config_dict()
    data["parse"][key] = value
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_freeze_writes_copy_and_accepts_same_config(tmp_path: Path) -> None:
    """Freeze a config, then accept freezing the identical config again."""
    source = write_config(tmp_path / "cfg.yaml", config_dict())
    config = load_config(source)
    frozen = freeze_config(config, source, tmp_path / "runs")
    assert frozen == tmp_path / "runs" / "colab_extraction" / "config.yaml"
    assert frozen.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert freeze_config(config, source, tmp_path / "runs") == frozen


def test_freeze_rejects_changed_config(tmp_path: Path) -> None:
    """Refuse to freeze a different config into an existing run folder."""
    source = write_config(tmp_path / "cfg.yaml", config_dict())
    freeze_config(load_config(source), source, tmp_path / "runs")
    changed = config_dict()
    changed["parse"]["ocr"] = True
    changed_source = write_config(tmp_path / "changed.yaml", changed)
    with pytest.raises(ConfigError, match="new run_id"):
        freeze_config(load_config(changed_source), changed_source, tmp_path / "runs")
