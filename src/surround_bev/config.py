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
class ProjectionConfig:
    """Settings for projecting 3D boxes into camera images.

    Attributes:
        minimum_depth: Near-plane distance in meters. A point in the camera
            frame (z forward) is in front of the plane only when ``z`` is
            strictly greater than this value.
        line_width: Stroke width, in pixels, of each box edge on the original
            camera image, before the mosaic resizes it.
        show_labels: When true, draw each visible box's category name.
    """

    minimum_depth: float
    line_width: int
    show_labels: bool


@dataclass(frozen=True)
class Config:
    """Typed view of ``configs/*.yaml``.

    Attributes:
        dataset_root: Directory containing the nuScenes ``samples/`` folder and
            the ``<version>/`` table folder.
        version: nuScenes version string, e.g. ``"v1.0-mini"``.
        sample_index: Index into the nuScenes ``sample`` table to visualize.
        output_file: Where the unlabeled six-camera mosaic is written.
        boxes_mosaic: Where the box-projection mosaic is written.
        projection: Near plane, stroke width, and label toggle.
    """

    dataset_root: Path
    version: str
    sample_index: int
    output_file: Path
    boxes_mosaic: Path
    projection: ProjectionConfig


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


def _as_bool(value: Any, field: str) -> bool:
    """Return ``value`` when it is a real boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"'{field}' must be a boolean, got {value!r}")
    return value


def _as_positive_float(value: Any, field: str) -> float:
    """Return ``value`` as a float that is strictly greater than zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"'{field}' must be a number, got {value!r}")
    number = float(value)
    if not number > 0.0:
        raise ValueError(f"'{field}' must be > 0, got {number}")
    return number


def _as_positive_int(value: Any, field: str) -> int:
    """Return ``value`` when it is an integer strictly greater than zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"'{field}' must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"'{field}' must be > 0, got {value}")
    return value


def _parse_projection(raw: dict[str, Any]) -> ProjectionConfig:
    """Validate the ``projection`` section."""
    projection = _require(raw, "projection", "")
    if not isinstance(projection, dict):
        raise TypeError("'projection' section must be a mapping")
    return ProjectionConfig(
        minimum_depth=_as_positive_float(
            _require(projection, "minimum_depth", "projection"),
            "projection.minimum_depth",
        ),
        line_width=_as_positive_int(
            _require(projection, "line_width", "projection"),
            "projection.line_width",
        ),
        show_labels=_as_bool(
            _require(projection, "show_labels", "projection"),
            "projection.show_labels",
        ),
    )


def parse_config(raw: dict[str, Any], base: Path = REPO_ROOT) -> Config:
    """Build a :class:`Config` from an already-parsed YAML mapping.

    Args:
        raw: Mapping as returned by :func:`yaml.safe_load`.
        base: Directory used to resolve relative paths.

    Returns:
        A validated :class:`Config`.

    Raises:
        TypeError: If ``raw`` or one of its sections is not a mapping, or a
            numeric or boolean field has the wrong type.
        KeyError: If a required field is missing.
        ValueError: If ``projection.minimum_depth`` or ``projection.line_width``
            is not strictly positive.
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
        boxes_mosaic=resolve_path(str(_require(output, "boxes_mosaic", "output")), base),
        projection=_parse_projection(raw),
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
        TypeError, KeyError, ValueError: See :func:`parse_config`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse_config(raw, base)
