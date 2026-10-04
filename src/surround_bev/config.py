"""Configuration loading for SurroundBEV.

All filesystem locations (dataset root, output file) come from a YAML config
file; nothing in the Python code hardcodes a dataset path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from surround_bev import REPO_ROOT


@dataclass(frozen=True)
class Config:
    """Typed view of ``configs/*.yaml``.

    Attributes:
        dataset_root: Directory containing the nuScenes ``samples/`` folder and
            the ``<version>/`` table folder.
        version: nuScenes version string, e.g. ``"v1.0-mini"``.
        sample_index: Index into the nuScenes ``sample`` table to visualize.
        output_file: Where the mosaic JPEG is written.
    """

    dataset_root: Path
    version: str
    sample_index: int
    output_file: Path


def resolve_path(path_str: str, base: Path = REPO_ROOT) -> Path:
    """Expand ``~`` and make relative paths relative to ``base``.

    Args:
        path_str: Path as written in the YAML file.
        base: Directory that relative paths are resolved against.

    Returns:
        An absolute :class:`~pathlib.Path`.
    """
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else base / path


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Return ``mapping[key]`` or raise a ``KeyError`` naming the missing field."""
    if key not in mapping:
        raise KeyError(f"Missing required config field '{context}.{key}'" if context else f"Missing required config field '{key}'")
    return mapping[key]


def parse_config(raw: dict[str, Any], base: Path = REPO_ROOT) -> Config:
    """Build a :class:`Config` from an already-parsed YAML mapping.

    Args:
        raw: Mapping as returned by :func:`yaml.safe_load`.
        base: Directory used to resolve relative paths.

    Returns:
        A validated :class:`Config`.

    Raises:
        TypeError: If ``raw`` or one of its sections is not a mapping, or
            ``sample_index`` is not an integer.
        KeyError: If a required field is missing.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Config must be a mapping, got {type(raw).__name__}")

    dataset = _require(raw, "dataset", "")
    output = _require(raw, "output", "")
    if not isinstance(dataset, dict):
        raise TypeError("'dataset' section must be a mapping")
    if not isinstance(output, dict):
        raise TypeError("'output' section must be a mapping")

    sample_index = raw.get("sample_index", 0)
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError(f"'sample_index' must be an integer, got {sample_index!r}")

    return Config(
        dataset_root=resolve_path(str(_require(dataset, "root", "dataset")), base),
        version=str(_require(dataset, "version", "dataset")),
        sample_index=sample_index,
        output_file=resolve_path(str(_require(output, "file", "output")), base),
    )


def load_config(path: Path, base: Path = REPO_ROOT) -> Config:
    """Read a YAML file and return a validated :class:`Config`.

    Args:
        path: Path to the YAML config file.
        base: Directory used to resolve relative paths in the config.

    Returns:
        A validated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError, KeyError: See :func:`parse_config`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse_config(raw, base)
