"""Tests for nuScenes sample access (no dataset required)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from surround_bev.data import CAMERA_CHANNELS, get_sample, load_nuscenes


def _fake_nusc(n: int) -> SimpleNamespace:
    """Object exposing a ``sample`` list, like ``NuScenes`` does."""
    return SimpleNamespace(sample=[{"token": f"sample{i}"} for i in range(n)])


def test_camera_channels_are_the_six_surround_cameras() -> None:
    assert CAMERA_CHANNELS == (
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    )
    assert len(set(CAMERA_CHANNELS)) == 6


@pytest.mark.parametrize("index", [0, 1, 4])
def test_get_sample_valid_index(index: int) -> None:
    nusc = _fake_nusc(5)
    assert get_sample(nusc, index) is nusc.sample[index]


@pytest.mark.parametrize("index", [-1, 5, 100])
def test_get_sample_invalid_index_raises(index: int) -> None:
    nusc = _fake_nusc(5)
    with pytest.raises(IndexError, match="out of range"):
        get_sample(nusc, index)


def test_get_sample_empty_table_raises() -> None:
    with pytest.raises(IndexError):
        get_sample(_fake_nusc(0), 0)


def test_load_nuscenes_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_nuscenes(tmp_path / "missing", "v1.0-mini")


def test_load_nuscenes_missing_version_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tables not found"):
        load_nuscenes(tmp_path, "v1.0-mini")
