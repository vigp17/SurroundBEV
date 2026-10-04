"""Tests for the ego-centric BEV grid, transforms, and binary mask."""

import numpy as np
import pytest
from shapely.geometry import box

from surround_bev.bev import (
    SUPPORTED_BEV_LAYERS,
    BevGrid,
    GlobalPolygon,
    ego_to_grid,
    ego_to_world,
    grid_shape,
    pixel_center,
    rasterize_drivable_mask,
    render_bev,
    world_to_ego,
)

_STANDARD = BevGrid(-30.0, 50.0, -30.0, 30.0, 0.25)


def _square(x0: float, y0: float, x1: float, y1: float) -> GlobalPolygon:
    """Axis-aligned ego-frame square stored as if yaw is zero and ego is at the origin.

    ``rasterize_drivable_mask`` applies ``world_to_ego``. With a zero yaw and an
    ego origin at ``(0, 0)``, global coordinates and ego coordinates match, so
    the square can be written directly in ego meters.
    """
    ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return GlobalPolygon(exterior=ring)


def test_valid_bev_dimensions() -> None:
    assert grid_shape(-30.0, 50.0, -30.0, 30.0, 0.25) == (320, 240)
    assert _STANDARD.shape == (320, 240)
    assert _STANDARD.rows == 320
    assert _STANDARD.cols == 240


def test_invalid_extent() -> None:
    with pytest.raises(ValueError, match="invalid extent"):
        grid_shape(10.0, 0.0, -30.0, 30.0, 0.25)
    with pytest.raises(ValueError, match="invalid extent"):
        grid_shape(-30.0, 50.0, 5.0, -5.0, 0.25)
    with pytest.raises(ValueError, match="invalid extent"):
        BevGrid(-30.0, -30.0, -30.0, 30.0, 0.25)


def test_invalid_resolution() -> None:
    with pytest.raises(ValueError, match="invalid resolution"):
        grid_shape(-30.0, 50.0, -30.0, 30.0, 0.0)
    with pytest.raises(ValueError, match="invalid resolution"):
        grid_shape(-30.0, 50.0, -30.0, 30.0, -0.25)
    with pytest.raises(ValueError, match="invalid resolution"):
        grid_shape(-30.0, 50.0, -30.0, 30.0, 0.3)


def test_world_to_ego_translation_and_yaw() -> None:
    ego = np.array([10.0, 5.0])
    points = np.array([[12.0], [8.0]])
    np.testing.assert_allclose(world_to_ego(points, ego, yaw_rad=0.0), [[2.0], [3.0]])

    # yaw = 90 deg: ego forward is global +y, ego left is global -x.
    yaw = float(np.pi / 2.0)
    origin = np.zeros(2)
    forward = world_to_ego(np.array([[0.0], [2.0]]), origin, yaw)
    left = world_to_ego(np.array([[-2.0], [0.0]]), origin, yaw)
    np.testing.assert_allclose(forward, [[2.0], [0.0]], atol=1e-8)
    np.testing.assert_allclose(left, [[0.0], [2.0]], atol=1e-8)

    sample = np.array([[3.0, -4.0], [1.0, 2.0]])
    round_trip = ego_to_world(world_to_ego(sample, ego, yaw_rad=0.4), ego, yaw_rad=0.4)
    np.testing.assert_allclose(round_trip, sample, atol=1e-8)
    np.testing.assert_allclose(world_to_ego(ego.reshape(2, 1), ego, yaw_rad=1.2), np.zeros((2, 1)), atol=1e-8)


def test_ego_to_grid_places_the_origin_on_the_asymmetric_boundary() -> None:
    row, col = ego_to_grid(0.0, 0.0, _STANDARD)
    assert (row, col) == (200.0, 120.0)
    assert row != _STANDARD.rows / 2


def test_forward_direction_maps_to_image_top() -> None:
    ahead_row, _ = ego_to_grid(10.0, 0.0, _STANDARD)
    behind_row, _ = ego_to_grid(-10.0, 0.0, _STANDARD)
    assert ahead_row < behind_row

    top_x, _ = pixel_center(0, _STANDARD.cols // 2, _STANDARD)
    bottom_x, _ = pixel_center(_STANDARD.rows - 1, _STANDARD.cols // 2, _STANDARD)
    assert top_x > bottom_x
    assert top_x > 0.0
    assert bottom_x < 0.0

    mask = rasterize_drivable_mask([_square(0.0, -30.0, 50.0, 30.0)], np.zeros(2), 0.0, _STANDARD)
    assert mask.shape == (320, 240)
    assert np.all(mask[:200] == 1)
    assert np.all(mask[200:] == 0)


def test_left_direction_maps_to_image_left() -> None:
    _, left_col = ego_to_grid(0.0, 10.0, _STANDARD)
    _, right_col = ego_to_grid(0.0, -10.0, _STANDARD)
    assert left_col < right_col

    _, left_y = pixel_center(_STANDARD.rows // 2, 0, _STANDARD)
    _, right_y = pixel_center(_STANDARD.rows // 2, _STANDARD.cols - 1, _STANDARD)
    assert left_y > right_y

    mask = rasterize_drivable_mask([_square(-30.0, 0.0, 50.0, 30.0)], np.zeros(2), 0.0, _STANDARD)
    assert np.all(mask[:, :120] == 1)
    assert np.all(mask[:, 120:] == 0)


def test_binary_mask_contains_only_zero_and_one() -> None:
    mask = rasterize_drivable_mask([_square(0.0, 0.0, 20.0, 10.0)], np.zeros(2), 0.0, _STANDARD)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert 0 in mask and 1 in mask


def test_render_does_not_modify_the_raw_mask() -> None:
    mask = rasterize_drivable_mask([_square(0.0, -30.0, 50.0, 30.0)], np.zeros(2), 0.0, _STANDARD)
    before = mask.copy()
    image = render_bev(mask, _STANDARD)

    assert np.array_equal(mask, before)
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert image.mode == "RGB"
    assert image.size == (_STANDARD.cols, _STANDARD.rows)
    # The ego disc is yellow and is not a color stored in the raw mask.
    pixels = np.asarray(image)
    assert (pixels == np.array([255, 196, 32])).all(axis=2).any()


def test_unsupported_map_layer_constant() -> None:
    assert "drivable_area" in SUPPORTED_BEV_LAYERS
    assert "lane" not in SUPPORTED_BEV_LAYERS


def test_polygon_hole_is_not_drivable() -> None:
    grid = BevGrid(-4.0, 4.0, -4.0, 4.0, 1.0)
    exterior = np.array([[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]], dtype=float)
    hole = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
    mask = rasterize_drivable_mask([GlobalPolygon(exterior, (hole,))], np.zeros(2), 0.0, grid)
    assert set(np.unique(mask).tolist()) <= {0, 1}
    # Cell centers near the origin fall inside the hole. A corner cell does not.
    assert mask[grid.rows // 2, grid.cols // 2] == 0
    assert mask[0, 0] == 1


def test_shapely_box_agrees_with_the_forward_window() -> None:
    """Guard the raster window against an accidental swap of x and y."""
    window = box(_STANDARD.x_min, _STANDARD.y_min, _STANDARD.x_max, _STANDARD.y_max)
    assert window.bounds == (-30.0, -30.0, 50.0, 30.0)
