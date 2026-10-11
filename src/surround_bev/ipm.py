"""Geometry-only inverse perspective mapping from cameras onto the BEV grid.

Pipeline
--------
Pixel
    A pixel ``(u, v)`` is a column and a row. The pinhole ray in the camera
    frame is ``d_cam = K^{-1} [u, v, 1]``. The camera frame has +x right,
    +y down, and +z forward.

Camera ray to the ground plane
    ``K`` is the nuScenes intrinsic. ``R`` maps camera vectors into the BEV
    frame, and ``t`` is the camera center in that frame, in meters. The ray
    in the BEV frame is ``d = R d_cam``. It meets the ground plane ``z = 0``
    at ``t + s d`` with ``s = -t_z / d_z``.

    The hit is kept only when the ray points downward (``d_z < 0``) and the
    intersection is in front of the camera (``s > 0``). A horizontal ray,
    a ray into the sky, and a ray that meets the plane behind the camera are
    left empty.

Ego coordinate to the BEV grid
    The kept ``(x, y)`` is ego forward and ego left, in meters. It uses the
    Milestone 2 index map: row 0 is the forward edge, column 0 is the left
    edge. See :func:`surround_bev.bev.ego_to_grid`.

Assumptions
-----------
* The road is the horizontal plane through the LIDAR_TOP ego origin. nuScenes
  puts that origin at the rear axle, so the axle height stands in for the
  road. Hills, banks, and curbs are not modeled.
* Vehicle pitch and roll tilt the camera relative to that plane. They are not
  folded into the output axes. Those axes stay the yaw-only ego frame from
  Milestone 2, so an IPM pixel and a drivable-area cell share ``(x, y)``.
* There is no lens-distortion model. nuScenes intrinsics are used as a pinhole.
* Anything above the road (vehicles, buildings, the sky) is either dropped or
  painted onto the plane at the ray's ground hit.
* The saved image is the inverse of the same intersection: each cell center
  is projected into the cameras. That fills the grid without splat holes.
  Overlaps keep the nearest camera. An equal range keeps the earlier channel
  in :data:`IPM_CAMERA_CHANNELS`.
* Unobserved cells stay black. No network, detector, or segmenter is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from surround_bev.bev import BevGrid
from surround_bev.data import CAMERA_CHANNELS

IPM_CAMERA_CHANNELS: tuple[str, ...] = CAMERA_CHANNELS
"""Surround cameras in mosaic order. The stitcher walks this tuple."""

_RAY_EPSILON = 1e-8


def require_camera_channel(channel: str) -> None:
    """Raise ``ValueError`` when ``channel`` is not one of the six cameras.

    Args:
        channel: nuScenes channel name.

    Raises:
        ValueError: If ``channel`` is not in :data:`IPM_CAMERA_CHANNELS`.
    """
    if channel not in IPM_CAMERA_CHANNELS:
        raise ValueError(f"unsupported camera channel {channel!r}")


def pixels_to_ground(
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    rotation_cam_to_ego: np.ndarray,
    camera_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect camera rays through ``pixels`` with the ground plane ``z = 0``.

    Args:
        pixels: Shape ``(2, N)``. Row 0 is ``u`` (column), row 1 is ``v`` (row).
        intrinsic: Shape ``(3, 3)``, pixels. Maps camera-frame meters to pixels.
        rotation_cam_to_ego: Shape ``(3, 3)``. Maps camera vectors into the
            ego frame whose ``z = 0`` plane is the ground.
        camera_translation: Shape ``(3,)``, meters. Camera center in that frame.

    Returns:
        ``(points, valid)``. ``points`` has shape ``(3, N)`` and is in meters
        in the same frame as ``camera_translation``. Invalid columns are NaN.
        Valid points have ``z == 0``. ``valid`` has shape ``(N,)``.

    Raises:
        ValueError: If the calibration or ``pixels`` is not usable.
    """
    uv = _as_pixels(pixels)
    intrinsic, rotation, translation = _validate_calibration(
        intrinsic, rotation_cam_to_ego, camera_translation
    )
    count = uv.shape[1]
    rays_cam = np.linalg.inv(intrinsic) @ np.vstack((uv[0], uv[1], np.ones(count)))
    rays_ego = rotation @ rays_cam
    pointing_down = rays_ego[2] < -_RAY_EPSILON
    scale = np.zeros(count)
    scale[pointing_down] = -translation[2] / rays_ego[2, pointing_down]
    valid = pointing_down & (scale > _RAY_EPSILON)
    points = np.full((3, count), np.nan)
    if np.any(valid):
        points[:, valid] = translation.reshape(3, 1) + scale[valid] * rays_ego[:, valid]
        points[2, valid] = 0.0
    return points, valid


def ground_to_pixels(
    points_ego: np.ndarray,
    intrinsic: np.ndarray,
    rotation_cam_to_ego: np.ndarray,
    camera_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame points into the image. Inverse of :func:`pixels_to_ground`.

    Args:
        points_ego: Shape ``(3, N)``, meters, same frame as ``camera_translation``.
        intrinsic: Shape ``(3, 3)``.
        rotation_cam_to_ego: Shape ``(3, 3)``. Maps camera vectors into that frame.
        camera_translation: Shape ``(3,)``, meters.

    Returns:
        ``(pixels, valid)``. ``pixels`` has shape ``(2, N)`` with ``u`` then
        ``v``. Invalid columns are NaN. A point is valid when it is in front
        of the camera (camera-frame ``z > 0``).

    Raises:
        ValueError: If the calibration or ``points_ego`` is not usable.
    """
    points = _as_points3(points_ego, "points_ego")
    intrinsic, rotation, translation = _validate_calibration(
        intrinsic, rotation_cam_to_ego, camera_translation
    )
    rays_cam = rotation.T @ (points - translation.reshape(3, 1))
    valid = rays_cam[2] > _RAY_EPSILON
    pixels = np.full((2, points.shape[1]), np.nan)
    if np.any(valid):
        projected = intrinsic @ rays_cam[:, valid]
        pixels[:, valid] = projected[:2] / projected[2]
    return pixels, valid


def camera_in_bev_frame(
    ego_rotation: np.ndarray,
    ego_translation: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
    reference_translation: np.ndarray,
    reference_yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Express one camera in the Milestone 2 ego frame.

    The reference frame has its origin at ``reference_translation``, +x along
    the reference yaw, +y left, and +z world-up. The ground plane is ``z = 0``
    in that frame. The camera's own ego rotation still tilts the ray, so a
    pitched vehicle looks down the slope of its cameras, and the hit is then
    written in the yaw-only axes shared with the drivable-area mask.

    Args:
        ego_rotation: Shape ``(3, 3)``. Maps the camera-time ego frame into global.
        ego_translation: Shape ``(3,)``, meters, global. Camera-time ego origin.
        camera_rotation: Shape ``(3, 3)``. Maps camera vectors into the
            camera-time ego frame.
        camera_translation: Shape ``(3,)``, meters, camera center in the
            camera-time ego frame.
        reference_translation: Shape ``(3,)``, meters, global LIDAR_TOP origin.
        reference_yaw_rad: Planar heading of the reference ego x-axis, radians.

    Returns:
        ``(rotation_cam_to_bev, camera_translation_bev)``. The rotation maps
        camera vectors into the BEV frame. The translation is the camera
        center in that frame, meters.

    Raises:
        ValueError: If a matrix or vector is not a usable pose.
    """
    ego_rotation, ego_translation = _validate_pose(ego_rotation, ego_translation, "ego")
    camera_rotation, camera_translation = _validate_pose(camera_rotation, camera_translation, "camera")
    reference_translation = _as_vec3(reference_translation, "reference translation")
    if isinstance(reference_yaw_rad, bool) or not isinstance(reference_yaw_rad, (int, float)):
        raise TypeError(f"'reference_yaw_rad' must be a number, got {reference_yaw_rad!r}")
    if not math.isfinite(float(reference_yaw_rad)):
        raise ValueError(f"invalid calibration: reference yaw is not finite ({reference_yaw_rad!r})")
    bev_from_global = _yaw_rotation(float(reference_yaw_rad)).T
    camera_global = ego_rotation @ camera_translation + ego_translation
    translation_bev = bev_from_global @ (camera_global - reference_translation)
    rotation_bev = bev_from_global @ ego_rotation @ camera_rotation
    return rotation_bev, translation_bev


def project_camera_to_bev(
    image: np.ndarray,
    intrinsic: np.ndarray,
    rotation_cam_to_ego: np.ndarray,
    camera_translation: np.ndarray,
    grid: BevGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Paint one camera onto ``grid`` by projecting each cell center.

    Each cell center is a point on ``z = 0``. :func:`ground_to_pixels` finds
    the source pixel, which is the inverse of :func:`pixels_to_ground` for the
    same plane. Cells whose ray misses the image stay black.

    Args:
        image: ``uint8`` array of shape ``(height, width, 3)``.
        intrinsic: Shape ``(3, 3)``.
        rotation_cam_to_ego: Shape ``(3, 3)``. Maps camera vectors into the
            frame of ``grid``.
        camera_translation: Shape ``(3,)``, meters, camera center in that frame.
        grid: Metric window. Its shape is the output shape.

    Returns:
        ``(rgb, distance)``. ``rgb`` is ``uint8`` of shape
        ``(rows, cols, 3)``. ``distance`` is float of shape ``(rows, cols)``,
        the 3D range from the camera to the cell center, or ``inf`` where the
        cell was not painted.

    Raises:
        ValueError: If ``image`` or the calibration is not usable. An invalid
            grid resolution raises from :class:`~surround_bev.bev.BevGrid`.
    """
    photo = _as_rgb_image(image)
    rows, cols = grid.shape
    row_index, col_index = np.indices((rows, cols))
    x = grid.x_max - (row_index + 0.5) * grid.resolution
    y = grid.y_max - (col_index + 0.5) * grid.resolution
    points = np.stack((x.ravel(), y.ravel(), np.zeros(rows * cols)), axis=0)
    pixels, in_front = ground_to_pixels(points, intrinsic, rotation_cam_to_ego, camera_translation)
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    distance = np.full((rows, cols), np.inf)
    if not np.any(in_front):
        return rgb, distance

    height, width = photo.shape[:2]
    u = pixels[0, in_front]
    v = pixels[1, in_front]
    u_index = np.floor(u).astype(np.int32)
    v_index = np.floor(v).astype(np.int32)
    inside = (u_index >= 0) & (u_index < width) & (v_index >= 0) & (v_index < height)
    if not np.any(inside):
        return rgb, distance

    flat = np.flatnonzero(in_front)[inside]
    rgb.reshape(-1, 3)[flat] = photo[v_index[inside], u_index[inside]]
    translation = np.asarray(camera_translation, dtype=float).reshape(3, 1)
    distance.ravel()[flat] = np.linalg.norm(points[:, flat] - translation, axis=0)
    return rgb, distance


def stitch_bev(frames: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Composite camera BEV images. The nearest camera wins.

    ``frames`` is walked in order. A later camera replaces a cell only when
    its range is strictly smaller, so a tie keeps the earlier frame. Callers
    pass :data:`IPM_CAMERA_CHANNELS` order.

    Args:
        frames: ``(rgb, distance)`` pairs as returned by
            :func:`project_camera_to_bev`.

    Returns:
        ``uint8`` array of shape ``(rows, cols, 3)``. Cells no camera saw
        stay ``(0, 0, 0)``.

    Raises:
        ValueError: If ``frames`` is empty or the shapes disagree.
    """
    if len(frames) == 0:
        raise ValueError("no camera frames to stitch")
    rgb0, dist0 = frames[0]
    if rgb0.ndim != 3 or rgb0.shape[2] != 3 or dist0.shape != rgb0.shape[:2]:
        raise ValueError(f"frame 0 has rgb shape {rgb0.shape} and distance shape {dist0.shape}")
    stitched = np.zeros(rgb0.shape, dtype=np.uint8)
    best = np.full(rgb0.shape[:2], np.inf)
    for index, (rgb, distance) in enumerate(frames):
        if rgb.shape != rgb0.shape or distance.shape != dist0.shape:
            raise ValueError(f"frame {index} shape {rgb.shape} does not match {rgb0.shape}")
        nearer = np.asarray(distance, dtype=float) < best
        stitched[nearer] = rgb[nearer]
        best[nearer] = distance[nearer]
    return stitched


def _yaw_rotation(yaw_rad: float) -> np.ndarray:
    """Return the matrix that maps BEV vectors into the global frame."""
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _validate_calibration(
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite ``K``, ``R``, and ``t``, or raise ``ValueError``."""
    matrix_k = _as_matrix3(intrinsic, "intrinsic")
    matrix_r = _as_matrix3(rotation, "rotation")
    vector_t = _as_vec3(translation, "translation")
    if abs(np.linalg.det(matrix_k)) < 1e-8:
        raise ValueError("invalid calibration: intrinsic matrix is singular")
    _require_rotation(matrix_r, "camera rotation")
    return matrix_k, matrix_r, vector_t


def _validate_pose(rotation: np.ndarray, translation: np.ndarray, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite rotation and translation, or raise ``ValueError``."""
    matrix = _as_matrix3(rotation, f"{what} rotation")
    vector = _as_vec3(translation, f"{what} translation")
    _require_rotation(matrix, f"{what} rotation")
    return matrix, vector


def _require_rotation(rotation: np.ndarray, what: str) -> None:
    """Raise ``ValueError`` unless ``rotation`` is a proper rotation."""
    if abs(np.linalg.det(rotation) - 1.0) > 1e-3 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError(f"invalid calibration: {what} is not a rotation matrix")


def _as_matrix3(value: np.ndarray, what: str) -> np.ndarray:
    """Return ``value`` as a finite ``(3, 3)`` float array."""
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"invalid calibration: {what} must have shape (3, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"invalid calibration: {what} contains non-finite values")
    return array


def _as_vec3(value: np.ndarray, what: str) -> np.ndarray:
    """Return ``value`` as a finite ``(3,)`` float array."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"invalid calibration: {what} must have shape (3,), got {np.shape(value)}")
    if not np.isfinite(array).all():
        raise ValueError(f"invalid calibration: {what} contains non-finite values")
    return array


def _as_pixels(pixels: np.ndarray) -> np.ndarray:
    """Return ``pixels`` as a finite ``(2, N)`` float array."""
    array = np.asarray(pixels, dtype=float)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"pixels must have shape (2, N), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("invalid calibration: pixels contain non-finite values")
    return array


def _as_points3(points: np.ndarray, what: str) -> np.ndarray:
    """Return ``points`` as a finite ``(3, N)`` float array."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] != 3:
        raise ValueError(f"{what} must have shape (3, N), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{what} contains non-finite values")
    return array


def _as_rgb_image(image: np.ndarray) -> np.ndarray:
    """Return ``image`` when it is a ``uint8`` RGB array."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"image must have shape (height, width, 3), got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {array.dtype}")
    return array
