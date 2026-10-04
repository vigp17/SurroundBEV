"""Tests for configuration loading."""

from pathlib import Path

import pytest

from surround_bev import REPO_ROOT
from surround_bev.config import (
    BevConfig,
    Config,
    ProjectionConfig,
    load_config,
    parse_config,
)

_BEV_YAML = (
    "bev:\n"
    "  x_min: -30\n"
    "  x_max: 50\n"
    "  y_min: -30\n"
    "  y_max: 30\n"
    "  resolution: 0.25\n"
    "  layers:\n"
    "    - drivable_area\n"
    "  output_file: out/bev.png\n"
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_parses_fields(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path,
        "dataset:\n  root: /data/nuscenes\n  version: v1.0-mini\n\n"
        "sample_index: 3\n\n"
        "output:\n  file: out/mosaic.jpg\n  boxes_mosaic: out/boxes.jpg\n\n"
        "projection:\n  minimum_depth: 1.0\n  line_width: 3\n  show_labels: true\n"
        + _BEV_YAML,
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
    cfg_path = _write(
        tmp_path,
        "dataset:\n  root: ~/nuscenes\n  version: v1.0-mini\n"
        "output:\n  file: /tmp/x.jpg\n  boxes_mosaic: /tmp/boxes.jpg\n"
        "projection:\n  minimum_depth: 1\n  line_width: 1\n  show_labels: true\n" + _BEV_YAML,
    )
    cfg = load_config(cfg_path, base=tmp_path)

    assert cfg.dataset_root == Path("~/nuscenes").expanduser()
    assert cfg.sample_index == 0


def test_repo_mini_config_is_valid() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "mini.yaml")

    assert cfg.version == "v1.0-mini"
    assert cfg.sample_index == 0
    assert cfg.dataset_root.is_absolute()
    assert cfg.output_file == REPO_ROOT / "outputs" / "sample_mosaic.jpg"
    assert cfg.boxes_mosaic == REPO_ROOT / "outputs" / "sample_boxes_mosaic.jpg"
    assert cfg.projection == ProjectionConfig(minimum_depth=1.0, line_width=3, show_labels=False)
    assert cfg.bev == BevConfig(
        x_min=-30.0,
        x_max=50.0,
        y_min=-30.0,
        y_max=30.0,
        resolution=0.25,
        layers=("drivable_area",),
        output_file=REPO_ROOT / "outputs" / "sample_bev_target.png",
    )


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


def _valid_raw(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "dataset": {"root": "/data/nuscenes", "version": "v1.0-mini"},
        "sample_index": 0,
        "output": {"file": "mosaics/plain.jpg", "boxes_mosaic": "mosaics/boxes.jpg"},
        "projection": {"minimum_depth": 1.0, "line_width": 3, "show_labels": True},
        "bev": {
            "x_min": -30.0,
            "x_max": 50.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "resolution": 0.25,
            "layers": ["drivable_area"],
            "output_file": "out/bev.png",
        },
    }
    raw.update(overrides)
    return raw


def test_projection_config_loading(tmp_path: Path) -> None:
    text = (
        "dataset:\n  root: /data/nuscenes\n  version: v1.0-mini\n"
        "output:\n  file: a.jpg\n  boxes_mosaic: b.jpg\n"
        "projection:\n  minimum_depth: 2.5\n  line_width: 4\n  show_labels: false\n" + _BEV_YAML
    )
    cfg = load_config(_write(tmp_path, text), base=tmp_path)

    assert cfg.projection == ProjectionConfig(minimum_depth=2.5, line_width=4, show_labels=False)
    assert isinstance(cfg.projection.minimum_depth, float)
    assert isinstance(cfg.projection.line_width, int)
    assert isinstance(cfg.projection.show_labels, bool)


@pytest.mark.parametrize(
    "projection, exc",
    [
        ({"line_width": 3, "show_labels": True}, KeyError),
        ({"minimum_depth": 1.0, "show_labels": True}, KeyError),
        ({"minimum_depth": 1.0, "line_width": 3}, KeyError),
        ({"minimum_depth": True, "line_width": 3, "show_labels": True}, TypeError),
        ({"minimum_depth": "1.0", "line_width": 3, "show_labels": True}, TypeError),
        ({"minimum_depth": 0, "line_width": 3, "show_labels": True}, ValueError),
        ({"minimum_depth": -2.0, "line_width": 3, "show_labels": True}, ValueError),
        ({"minimum_depth": 1.0, "line_width": True, "show_labels": True}, TypeError),
        ({"minimum_depth": 1.0, "line_width": 1.5, "show_labels": True}, TypeError),
        ({"minimum_depth": 1.0, "line_width": 0, "show_labels": True}, ValueError),
        ({"minimum_depth": 1.0, "line_width": 3, "show_labels": "true"}, TypeError),
    ],
)
def test_projection_config_rejects_invalid(projection: dict[str, object], exc: type[Exception]) -> None:
    raw = _valid_raw(projection=projection)
    with pytest.raises(exc):
        parse_config(raw)


def test_output_paths_resolve_against_base(tmp_path: Path) -> None:
    cfg = parse_config(_valid_raw(), base=tmp_path)

    assert cfg.output_file == tmp_path / "mosaics" / "plain.jpg"
    assert cfg.boxes_mosaic == tmp_path / "mosaics" / "boxes.jpg"
    assert cfg.output_file.is_absolute()
    assert cfg.boxes_mosaic.is_absolute()


def test_output_paths_keep_absolute_and_expand_home(tmp_path: Path) -> None:
    raw = _valid_raw(
        output={
            "file": "/var/out/plain.jpg",
            "boxes_mosaic": "~/mosaics/boxes.jpg",
        }
    )
    cfg = parse_config(raw, base=tmp_path)

    assert cfg.output_file == Path("/var/out/plain.jpg")
    assert cfg.boxes_mosaic == Path.home() / "mosaics" / "boxes.jpg"


def test_missing_boxes_mosaic_raises() -> None:
    raw = _valid_raw(output={"file": "plain.jpg"})
    with pytest.raises(KeyError, match="boxes_mosaic"):
        parse_config(raw)


def test_bev_output_path_resolution(tmp_path: Path) -> None:
    relative = parse_config(_valid_raw(), base=tmp_path)
    assert relative.bev.output_file == tmp_path / "out" / "bev.png"

    absolute = parse_config(
        _valid_raw(bev={**_valid_raw()["bev"], "output_file": "/var/bev/target.png"}),  # type: ignore[index]
        base=tmp_path,
    )
    assert absolute.bev.output_file == Path("/var/bev/target.png")

    home = parse_config(
        _valid_raw(bev={**_valid_raw()["bev"], "output_file": "~/bev/target.png"}),  # type: ignore[index]
        base=tmp_path,
    )
    assert home.bev.output_file == Path.home() / "bev" / "target.png"


@pytest.mark.parametrize(
    "bev, exc, match",
    [
        ({"x_max": 50, "y_min": -30, "y_max": 30, "resolution": 0.25, "layers": ["drivable_area"], "output_file": "a.png"}, KeyError, "x_min"),
        ({"x_min": 10, "x_max": 0, "y_min": -30, "y_max": 30, "resolution": 0.25, "layers": ["drivable_area"], "output_file": "a.png"}, ValueError, "invalid extent"),
        ({"x_min": -30, "x_max": -30, "y_min": -30, "y_max": 30, "resolution": 0.25, "layers": ["drivable_area"], "output_file": "a.png"}, ValueError, "invalid extent"),
        ({"x_min": -30, "x_max": 50, "y_min": 5, "y_max": -5, "resolution": 0.25, "layers": ["drivable_area"], "output_file": "a.png"}, ValueError, "invalid extent"),
        ({"x_min": -30, "x_max": 50, "y_min": -30, "y_max": 30, "resolution": 0, "layers": ["drivable_area"], "output_file": "a.png"}, ValueError, "invalid resolution"),
        ({"x_min": -30, "x_max": 50, "y_min": -30, "y_max": 30, "resolution": -0.25, "layers": ["drivable_area"], "output_file": "a.png"}, ValueError, "invalid resolution"),
        ({"x_min": -30, "x_max": 50, "y_min": -30, "y_max": 30, "resolution": 0.3, "layers": ["drivable_area"], "output_file": "a.png"}, ValueError, "invalid resolution"),
        ({"x_min": -30, "x_max": 50, "y_min": -30, "y_max": 30, "resolution": 0.25, "layers": ["lane"], "output_file": "a.png"}, ValueError, "unsupported map layer"),
        ({"x_min": -30, "x_max": 50, "y_min": -30, "y_max": 30, "resolution": 0.25, "layers": [], "output_file": "a.png"}, ValueError, "bev.layers"),
    ],
)
def test_bev_config_rejects_invalid(bev: dict[str, object], exc: type[Exception], match: str) -> None:
    with pytest.raises(exc, match=match):
        parse_config(_valid_raw(bev=bev))
