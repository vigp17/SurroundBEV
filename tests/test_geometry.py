"""Tests for box edges, camera projection, and the near-plane rejection."""

import numpy as np
import pytest
from pyquaternion import Quaternion

from surround_bev.data import quaternion_to_matrix
from surround_bev.geometry import (
    BOX_EDGES,
    box_corners,
    clip_edge_to_depth,
    is_fully_behind,
    project_box_to_image,
    project_points,
    transform_points_to_camera,
)

_INTRINSIC = np.array(
    [
        [100.0, 0.0, 400.0],
        [0.0, 100.0, 300.0],
        [0.0, 0.0, 1.0],
    ]
)
_IMAGE_SIZE = (800, 600)


def _axis_aligned_corners(center: np.ndarray, size_wlh: np.ndarray) -> np.ndarray:
    return box_corners(center, size_wlh, np.eye(3))


def test_box_has_the_twelve_cube_edges() -> None:
    """The 12 edges are exactly the pairs of corners that differ on one axis."""
    corners = _axis_aligned_corners(np.zeros(3), np.array([2.0, 2.0, 2.0]))
    assert len(BOX_EDGES) == 12
    undirected = {frozenset(edge) for edge in BOX_EDGES}
    assert len(undirected) == 12

    for start, end in BOX_EDGES:
        delta = np.abs(corners[:, start] - corners[:, end])
        assert np.count_nonzero(delta > 1e-8) == 1

    # Every corner of a cube has degree 3.
    degree = [0] * 8
    for start, end in BOX_EDGES:
        degree[start] += 1
        degree[end] += 1
    assert degree == [3, 3, 3, 3, 3, 3, 3, 3]


def test_box_corner_convention_matches_nuscenes() -> None:
    """Corner 0 is front-left-top in the box frame: +length/2, +width/2, +height/2."""
    corners = _axis_aligned_corners(np.array([10.0, 20.0, 30.0]), np.array([2.0, 4.0, 6.0]))
    # width 2, length 4, height 6 -> local (x, y, z) = (2, 1, 3), then translated.
    np.testing.assert_allclose(corners[:, 0], [12.0, 21.0, 33.0])
    # Corner 2 is front-right-bottom: +length/2, -width/2, -height/2.
    np.testing.assert_allclose(corners[:, 2], [12.0, 19.0, 27.0])
    # Corner 4 is the rear partner of corner 0, so only x (length) flips.
    np.testing.assert_allclose(corners[:, 4], [8.0, 21.0, 33.0])


def test_project_points_with_known_intrinsic() -> None:
    points = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [10.0, 10.0, 0.5],
        ]
    )
    pixels, valid = project_points(points, _INTRINSIC, minimum_depth=1.0)

    assert valid.tolist() == [True, True, False]
    # (0, 0, 10) falls on the principal point.
    np.testing.assert_allclose(pixels[:, 0], [400.0, 300.0])
    # u = 100 * 1/10 + 400, v = 100 * 2/10 + 300.
    np.testing.assert_allclose(pixels[:, 1], [410.0, 320.0])
    assert np.isnan(pixels[:, 2]).all()


def test_points_on_or_behind_the_near_plane_are_rejected() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-2.0, 1.0, 0.5],
        ]
    )
    assert is_fully_behind(points, minimum_depth=1.0)
    pixels, valid = project_points(points, np.eye(3), minimum_depth=1.0)
    assert valid.tolist() == [False, False, False]
    assert np.isnan(pixels).all()

    mixed = np.array([[0.0, 0.0], [0.0, 0.0], [-1.0, 5.0]])
    assert not is_fully_behind(mixed, minimum_depth=1.0)


def test_box_fully_behind_camera_is_rejected() -> None:
    corners = _axis_aligned_corners(np.array([0.0, 0.0, -4.0]), np.array([2.0, 2.0, 2.0]))
    assert is_fully_behind(corners, minimum_depth=1.0)
    assert project_box_to_image(corners, _INTRINSIC, _IMAGE_SIZE, 1.0, "vehicle.car") is None


def test_edge_crossing_the_near_plane_is_clipped() -> None:
    start = np.array([-2.0, 0.0, 0.0])
    end = np.array([2.0, 0.0, 4.0])
    clipped = clip_edge_to_depth(start, end, minimum_depth=2.0)
    assert clipped is not None
    np.testing.assert_allclose(clipped[0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(clipped[1], end)

    behind = clip_edge_to_depth(np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.5]), minimum_depth=1.0)
    assert behind is None


def test_straddling_box_keeps_only_edges_beyond_the_near_plane() -> None:
    # Height 6 at z = 2 puts half the corners at z = -1 and half at z = 5.
    corners = _axis_aligned_corners(np.array([0.0, 0.0, 2.0]), np.array([2.0, 2.0, 6.0]))
    projected = project_box_to_image(corners, _INTRINSIC, _IMAGE_SIZE, 1.0, "vehicle.car")
    assert projected is not None
    assert 0 < len(projected.segments) < 12
    for (u0, v0), (u1, v1) in projected.segments:
        assert np.isfinite([u0, v0, u1, v1]).all()


def test_in_front_box_projects_all_twelve_edges() -> None:
    corners = _axis_aligned_corners(np.array([0.0, 0.0, 20.0]), np.array([2.0, 4.0, 2.0]))
    projected = project_box_to_image(corners, _INTRINSIC, _IMAGE_SIZE, 1.0, "vehicle.car")
    assert projected is not None
    assert len(projected.segments) == 12

    # Corner 0 is center + (length/2, width/2, height/2) = (2, 1, 21).
    target_u = 400.0 + 100.0 * 2.0 / 21.0
    target_v = 300.0 + 100.0 * 1.0 / 21.0
    flat = [np.array(point) for segment in projected.segments for point in segment]
    assert any(np.allclose(point, [target_u, target_v], atol=1e-6) for point in flat)
    assert projected.label_anchor is not None
    np.testing.assert_allclose(projected.label_anchor, [400.0, 300.0])


def test_box_in_front_but_outside_the_image_is_not_visible() -> None:
    corners = _axis_aligned_corners(np.array([100.0, 0.0, 10.0]), np.array([2.0, 2.0, 2.0]))
    assert not is_fully_behind(corners, minimum_depth=1.0)
    assert project_box_to_image(corners, _INTRINSIC, _IMAGE_SIZE, 1.0, "vehicle.car") is None


def test_transform_points_to_camera_with_known_poses() -> None:
    identity = np.eye(3)
    points = np.array([[12.0], [3.0], [4.0]])
    translated = transform_points_to_camera(
        points,
        ego_translation=np.array([10.0, 0.0, 0.0]),
        ego_rotation=identity,
        camera_translation=np.array([0.0, 1.0, 0.0]),
        camera_rotation=identity,
    )
    np.testing.assert_allclose(translated, [[2.0], [2.0], [4.0]])

    # camera_rotation maps camera vectors into ego. R @ [1, 0, 0] = [0, 1, 0].
    camera_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = transform_points_to_camera(
        np.array([[1.0], [0.0], [5.0]]),
        ego_translation=np.zeros(3),
        ego_rotation=identity,
        camera_translation=np.zeros(3),
        camera_rotation=camera_rotation,
    )
    np.testing.assert_allclose(rotated, [[0.0], [-1.0], [5.0]])


def test_transform_points_to_camera_matches_nuscenes_transform_matrix() -> None:
    from nuscenes.utils.geometry_utils import transform_matrix

    ego_translation = np.array([1.0, 2.0, 3.0])
    camera_translation = np.array([0.5, -0.2, 1.4])
    ego_quaternion = Quaternion(axis=[0.0, 0.0, 1.0], angle=0.3)
    camera_quaternion = Quaternion(axis=[0.0, 1.0, 0.0], angle=-0.4)
    points = np.array([[10.0, -4.0], [4.0, 1.5], [2.0, 8.0]])

    ego_from_global = transform_matrix(ego_translation, ego_quaternion, inverse=True)
    camera_from_ego = transform_matrix(camera_translation, camera_quaternion, inverse=True)
    camera_from_global = camera_from_ego @ ego_from_global
    homogeneous = np.vstack([points, np.ones((1, points.shape[1]))])
    expected = (camera_from_global @ homogeneous)[:3]

    got = transform_points_to_camera(
        points,
        ego_translation,
        ego_quaternion.rotation_matrix,
        camera_translation,
        camera_quaternion.rotation_matrix,
    )
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_nuscenes_quaternion_is_wxyz() -> None:
    """A +90 degree yaw is (w, x, y, z) = (cos 45, 0, 0, sin 45)."""
    half = float(np.sqrt(2.0) / 2.0)
    matrix = quaternion_to_matrix([half, 0.0, 0.0, half])
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-8)


def test_project_box_rejects_non_positive_near_plane() -> None:
    corners = _axis_aligned_corners(np.array([0.0, 0.0, 10.0]), np.ones(3))
    with pytest.raises(ValueError, match="minimum_depth"):
        project_box_to_image(corners, _INTRINSIC, _IMAGE_SIZE, 0.0, "vehicle.car")
