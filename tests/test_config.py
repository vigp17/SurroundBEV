"""Tests for configuration loading."""

from pathlib import Path

import pytest

from surround_bev import REPO_ROOT
from surround_bev.config import Config, load_config, parse_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_parses_fields(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path,
        "dataset:\n  root: /data/nuscenes\n  version: v1.0-mini\n\nsample_index: 3\n\noutput:\n  file: out/mosaic.jpg\n",
    )
    cfg = load_config(cfg_path, base=tmp_path)

    assert isinstance(cfg, Config)
    assert cfg.dataset_root == Path("/data/nuscenes")
    assert cfg.version == "v1.0-mini"
    assert cfg.sample_index == 3
    # Relative output paths resolve against the provided base.
    assert cfg.output_file == tmp_path / "out" / "mosaic.jpg"
    assert cfg.output_file.is_absolute()


def test_load_config_expands_home_and_defaults_sample_index(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "dataset:\n  root: ~/nuscenes\n  version: v1.0-mini\noutput:\n  file: /tmp/x.jpg\n")
    cfg = load_config(cfg_path, base=tmp_path)

    assert cfg.dataset_root == Path("~/nuscenes").expanduser()
    assert cfg.sample_index == 0


def test_repo_mini_config_is_valid() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "mini.yaml")

    assert cfg.version == "v1.0-mini"
    assert cfg.sample_index == 0
    assert cfg.dataset_root.is_absolute()
    assert cfg.output_file == REPO_ROOT / "outputs" / "sample_mosaic.jpg"


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


@pytest.mark.parametrize(
    "raw, exc",
    [
        ([], TypeError),  # not a mapping
        ({"output": {"file": "x.jpg"}}, KeyError),  # missing dataset
        ({"dataset": {"root": "/d", "version": "v"}}, KeyError),  # missing output
        ({"dataset": {"root": "/d"}, "output": {"file": "x.jpg"}}, KeyError),  # missing version
        ({"dataset": "nope", "output": {"file": "x.jpg"}}, TypeError),  # section not mapping
        ({"dataset": {"root": "/d", "version": "v"}, "output": {"file": "x"}, "sample_index": "0"}, TypeError),
        ({"dataset": {"root": "/d", "version": "v"}, "output": {"file": "x"}, "sample_index": True}, TypeError),
    ],
)
def test_parse_config_rejects_invalid(raw: object, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        parse_config(raw)  # type: ignore[arg-type]
