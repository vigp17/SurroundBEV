"""Tests for geometry-only inverse perspective mapping."""

import math

import numpy as np
import pytest
from PIL import Image

from surround_bev.bev import BevGrid, ego_to_grid
from surround_bev.data import CAMERA_CHANNELS
from surround_bev.ipm import (
    IPM_CAMERA_CHANNELS,
    camera_in_bev_frame,
    ground_to_pixels,
    pixels_to_ground,
    project_camera_to_bev,
    require_camera_channel,
    stitch_bev,
)

# Camera at (0, 0, 1.5) m looking along ego +x, with +x to the right and +y down.
# Pixel (50, 50) is 10 px below the principal point and lands 15 m forward.
_INTRINSIC = np.array(
    [
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 40.0],
        [0.0, 0.0, 1.0],
    ]
)
_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)
_TRANSLATION = np.array([0.0, 0.0, 1.5])


def test_pixel_to_ground_known_ray() -> None:
    points, valid = pixels_to_ground(
        np.array([[50.0, 60.0, 50.0], [50.0, 50.0, 40.0]]),
        _INTRINSIC,
        _ROTATION,
        _TRANSLATION,
    )

    assert valid.tolist() == [True, True, False]
    assert points[:, 0] == pytest.approx([15.0, 0.0, 0.0])
    assert points[:, 1] == pytest.approx([15.0, -1.5, 0.0])
    assert np.isnan(points[:, 2]).all()


def test_pixel_to_ground_round_trips() -> None:
    points, valid = pixels_to_ground(
        np.array([[50.0], [50.0]]),
        _INTRINSIC,
        _ROTATION,
        _TRANSLATION,
    )
    pixels, back = ground_to_pixels(points[:, valid], _INTRINSIC, _ROTATION, _TRANSLATION)

    assert back.tolist() == [True]
    assert pixels[:, 0] == pytest.approx([50.0, 50.0])


def test_ego_to_grid_places_ground_hit() -> None:
    points, valid = pixels_to_ground(
        np.array([[50.0], [50.0]]),
        _INTRINSIC,
        _ROTATION,
        _TRANSLATION,
    )
    assert valid[0]
    grid = BevGrid(-30.0, 50.0, -30.0, 30.0, 0.25)
    row, col = ego_to_grid(float(points[0, 0]), float(points[1, 0]), grid)

    # 15 m forward of an origin whose continuous index is (200, 120).
    assert row == pytest.approx(140.0)
    assert col == pytest.approx(120.0)
    assert grid.shape == (320, 240)


def test_output_dimensions() -> None:
    grid = BevGrid(-30.0, 50.0, -30.0, 30.0, 0.25)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    bev, distance = project_camera_to_bev(image, _INTRINSIC, _ROTATION, _TRANSLATION, grid)

    assert bev.shape == (320, 240, 3)
    assert bev.dtype == np.uint8
    assert distance.shape == (320, 240)
    assert Image.fromarray(bev, mode="RGB").size == (240, 320)


def test_project_samples_cell_at_ground_hit() -> None:
    grid = BevGrid(1.5, 2.5, -0.5, 0.5, 1.0)
    image = np.zeros((200, 120, 3), dtype=np.uint8)
    image[115, 50] = (255, 0, 0)
    bev, distance = project_camera_to_bev(image, _INTRINSIC, _ROTATION, _TRANSLATION, grid)

    assert bev.shape == (1, 1, 3)
    assert bev[0, 0].tolist() == [255, 0, 0]
    assert distance[0, 0] == pytest.approx(math.sqrt(2.0**2 + 1.5**2))


def test_invalid_calibration() -> None:
    pixels = np.array([[50.0], [50.0]])
    singular = np.zeros((3, 3))
    reflection = np.diag([1.0, 1.0, -1.0])
    cases = [
        (singular, _ROTATION, _TRANSLATION),
        (_INTRINSIC, reflection, _TRANSLATION),
        (_INTRINSIC, np.ones((3, 3)), _TRANSLATION),
        (np.ones((2, 2)), _ROTATION, _TRANSLATION),
        (_INTRINSIC, _ROTATION, np.array([0.0, 0.0])),
        (_INTRINSIC, _ROTATION, np.array([np.nan, 0.0, 1.5])),
    ]
    for intrinsic, rotation, translation in cases:
        with pytest.raises(ValueError, match="invalid calibration"):
            pixels_to_ground(pixels, intrinsic, rotation, translation)


def test_invalid_resolution() -> None:
    with pytest.raises(ValueError, match="invalid resolution"):
        BevGrid(-30.0, 50.0, -30.0, 30.0, 0.3)
    with pytest.raises(ValueError, match="invalid resolution"):
        BevGrid(-30.0, 50.0, -30.0, 30.0, 0.0)
    with pytest.raises(ValueError, match="invalid resolution"):
        project_camera_to_bev(
            np.zeros((4, 4, 3), dtype=np.uint8),
            _INTRINSIC,
            _ROTATION,
            _TRANSLATION,
            BevGrid(-30.0, 50.0, -30.0, 30.0, -0.25),
        )


def test_unsupported_camera_channel() -> None:
    with pytest.raises(ValueError, match="unsupported camera channel"):
        require_camera_channel("CAM_FRONT_EXTRA")
    with pytest.raises(ValueError, match="unsupported camera channel"):
        require_camera_channel("LIDAR_TOP")


def test_camera_ordering_preserved() -> None:
    assert IPM_CAMERA_CHANNELS == (
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    )
    assert IPM_CAMERA_CHANNELS == CAMERA_CHANNELS

    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[:] = (255, 0, 0)
    blue = np.zeros((2, 2, 3), dtype=np.uint8)
    blue[:] = (0, 0, 255)
    tied = np.ones((2, 2))
    assert (stitch_bev([(red, tied), (blue, tied)]) == red).all()

    nearer = np.full((2, 2), 0.5)
    assert (stitch_bev([(red, tied), (blue, nearer)]) == blue).all()


def test_camera_pose_matches_yaw_frame() -> None:
    identity = np.eye(3)
    rotation, translation = camera_in_bev_frame(
        identity,
        np.zeros(3),
        _ROTATION,
        _TRANSLATION,
        np.zeros(3),
        0.0,
    )
    assert rotation == pytest.approx(_ROTATION)
    assert translation == pytest.approx(_TRANSLATION)

    # Yaw of +90 deg: global +y is ego forward.
    rotation, translation = camera_in_bev_frame(
        identity,
        np.array([0.0, 1.0, 0.0]),
        identity,
        np.array([0.0, 0.0, 2.0]),
        np.zeros(3),
        math.pi / 2.0,
    )
    assert translation == pytest.approx([1.0, 0.0, 2.0])
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, -1.0, 0.0])
