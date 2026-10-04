"""Camera-frame geometry for ground-truth 3D boxes.

Coordinate frames and units
---------------------------
Global frame
    nuScenes world frame. Translations are meters.

Ego frame
    Vehicle frame at the camera timestamp. ``ego_rotation`` (shape ``(3, 3)``)
    maps ego vectors into the global frame, and ``ego_translation`` (shape
    ``(3,)``, meters) is the ego origin in the global frame:

    ``p_ego = R_ego.T @ (p_global - t_ego)``

Camera frame
    Optical frame of one camera. +x is right, +y is down, +z is forward, and
    distances are meters. ``camera_rotation`` maps camera vectors into the ego
    frame, and ``camera_translation`` is the camera origin in the ego frame:

    ``p_cam = R_cam.T @ (p_ego - t_cam)``

    ``NuScenes.get_sample_data`` applies this same global → ego → camera
    sequence and returns boxes already in the camera frame. This module
    performs the transform itself so callers can reject only boxes that are
    fully behind the near plane. The devkit's default visibility filter also
    requires every corner to be in front of the camera.

Box frame
    nuScenes box-local frame: +x forward, +y left, +z up, meters. Size is
    ``(width, length, height)``: width along y, length along x, height along z.
    The box rotation maps these local vectors into the global frame.

Image plane
    Pixels, origin at the top-left, +u right, +v down. With the ``(3, 3)``
    intrinsic matrix ``K`` and a camera-frame point ``(x, y, z)``:

    ``[u, v, 1] ~ K @ [x, y, z]``, then divide by the third component.

    Division is defined only for points strictly in front of the camera
    (``z > 0``). Points with ``z <= minimum_depth`` are behind the configured
    near plane and are not projected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Corner order matches nuscenes.utils.data_classes.Box.corners:
# indices 0..3 are the +x (forward) face, 4..7 the -x (rear) face.
# Within each face the order is top-left, top-right, bottom-right, bottom-left
# in the box frame (y left, z up), which is not the image-plane order.
BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # forward face
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # rear face
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # edges joining the two faces
)
"""The 12 edges of a box, as pairs of indices into :func:`box_corners`."""


@dataclass(frozen=True)
class ProjectedBox:
    """One box projected into a single image.

    Attributes:
        name: Category name. Drawn only when labels are enabled.
        segments: Visible edges in pixel coordinates ``((u0, v0), (u1, v1))``.
            Each edge was in front of the near plane, clipped to that plane
            when it crossed it, and intersects the image rectangle.
        label_anchor: Pixel ``(u, v)`` of the box center when that center is
            beyond the near plane and inside the image. Otherwise ``None``.
    """

    name: str
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    label_anchor: tuple[float, float] | None


def _as_points(points: np.ndarray, what: str) -> np.ndarray:
    """Return ``points`` as a float array of shape ``(3, N)``."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] != 3:
        raise ValueError(f"{what} must have shape (3, N), got {array.shape}")
    return array


def _as_vec3(vector: np.ndarray, what: str) -> np.ndarray:
    """Return ``vector`` as a float array of shape ``(3,)``."""
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{what} must have shape (3,), got {array.shape}")
    return array


def _as_matrix3(matrix: np.ndarray, what: str) -> np.ndarray:
    """Return ``matrix`` as a float array of shape ``(3, 3)``."""
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"{what} must have shape (3, 3), got {array.shape}")
    return array


def box_corners(center: np.ndarray, size_wlh: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return the eight corners of a box.

    Args:
        center: Shape ``(3,)``, meters, in the parent frame (global, when used
            on a :class:`~surround_bev.data.GlobalBox`).
        size_wlh: Shape ``(3,)``, meters, ``(width, length, height)``.
        rotation: Shape ``(3, 3)``. Maps box-local vectors into the parent frame.

    Returns:
        Shape ``(3, 8)``, meters, in the same frame as ``center``. Column ``i``
        is corner ``i``; see :data:`BOX_EDGES` for which columns are connected.
        The local corner signs follow ``nuscenes.Box.corners``: x is
        ``±length/2`` (forward), y is ``±width/2`` (left), z is ``±height/2`` (up).
    """
    center = _as_vec3(center, "center")
    width, length, height = _as_vec3(size_wlh, "size_wlh")
    rotation = _as_matrix3(rotation, "rotation")

    x_corners = length / 2.0 * np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    y_corners = width / 2.0 * np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    z_corners = height / 2.0 * np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    local = np.vstack((x_corners, y_corners, z_corners))
    return rotation @ local + center.reshape(3, 1)


def transform_points_to_camera(
    points_global: np.ndarray,
    ego_translation: np.ndarray,
    ego_rotation: np.ndarray,
    camera_translation: np.ndarray,
    camera_rotation: np.ndarray,
) -> np.ndarray:
    """Transform points from the global frame into the camera frame.

    Args:
        points_global: Shape ``(3, N)``, meters, global frame.
        ego_translation: Shape ``(3,)``, meters, ego origin in the global frame.
        ego_rotation: Shape ``(3, 3)``. Maps ego vectors into the global frame.
        camera_translation: Shape ``(3,)``, meters, camera origin in the ego frame.
        camera_rotation: Shape ``(3, 3)``. Maps camera vectors into the ego frame.

    Returns:
        Shape ``(3, N)``, meters, camera frame (+x right, +y down, +z forward).
    """
    points_global = _as_points(points_global, "points_global")
    ego_translation = _as_vec3(ego_translation, "ego_translation")
    ego_rotation = _as_matrix3(ego_rotation, "ego_rotation")
    camera_translation = _as_vec3(camera_translation, "camera_translation")
    camera_rotation = _as_matrix3(camera_rotation, "camera_rotation")

    points_ego = ego_rotation.T @ (points_global - ego_translation.reshape(3, 1))
    return camera_rotation.T @ (points_ego - camera_translation.reshape(3, 1))


def is_fully_behind(points_cam: np.ndarray, minimum_depth: float) -> bool:
    """Return whether every point is on or behind the near plane.

    Args:
        points_cam: Shape ``(3, N)``, meters, camera frame.
        minimum_depth: Near-plane distance in meters. A point is beyond the
            plane when its forward coordinate ``z`` is strictly greater.

    Returns:
        True when ``N > 0`` and every ``z`` is less than or equal to
        ``minimum_depth``.
    """
    points_cam = _as_points(points_cam, "points_cam")
    if points_cam.shape[1] == 0:
        raise ValueError("points_cam must contain at least one point")
    return bool(np.all(points_cam[2] <= minimum_depth))


def project_points(
    points_cam: np.ndarray,
    intrinsic: np.ndarray,
    minimum_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to pixels, rejecting the near plane and behind.

    Args:
        points_cam: Shape ``(3, N)``, meters, camera frame.
        intrinsic: Shape ``(3, 3)``. Maps camera-frame meters to pixels.
        minimum_depth: Near-plane distance in meters.

    Returns:
        ``(pixels, valid)``. ``pixels`` has shape ``(2, N)`` and is ``u, v``
        in pixels. ``valid`` has shape ``(N,)`` and is true where ``z`` is
        strictly greater than ``minimum_depth``. Rejected columns of
        ``pixels`` are ``nan``.
    """
    points_cam = _as_points(points_cam, "points_cam")
    intrinsic = _as_matrix3(intrinsic, "intrinsic")
    count = points_cam.shape[1]
    valid = points_cam[2] > minimum_depth
    pixels = np.full((2, count), np.nan, dtype=float)
    if np.any(valid):
        pixels[:, valid] = _project_in_front(points_cam[:, valid], intrinsic)
    return pixels, valid


def _project_in_front(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project points that are strictly in front of the camera origin.

    Args:
        points_cam: Shape ``(3, N)``, meters, with every ``z > 0``.
        intrinsic: Shape ``(3, 3)``.

    Returns:
        Shape ``(2, N)``, pixels.

    Raises:
        ValueError: If any point has ``z <= 0`` or the homogeneous depth is 0.
    """
    if np.any(points_cam[2] <= 0.0):
        raise ValueError("cannot project points at or behind the camera origin (z <= 0)")
    homogeneous = intrinsic @ points_cam
    depth = homogeneous[2]
    if np.any(depth == 0.0):
        raise ValueError("camera projection has zero homogeneous depth")
    return homogeneous[:2] / depth


def clip_edge_to_depth(
    start: np.ndarray,
    end: np.ndarray,
    minimum_depth: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip one 3D edge to the half-space ``z > minimum_depth``.

    Args:
        start: Shape ``(3,)``, meters, camera frame.
        end: Shape ``(3,)``, meters, camera frame.
        minimum_depth: Near-plane distance in meters.

    Returns:
        The clipped endpoints, or ``None`` when the whole edge is on or behind
        the plane. An endpoint beyond the plane is unchanged. An endpoint on
        or behind the plane is moved to the intersection with ``z = minimum_depth``.
    """
    start = _as_vec3(start, "start")
    end = _as_vec3(end, "end")
    start_in_front = float(start[2]) > minimum_depth
    end_in_front = float(end[2]) > minimum_depth
    if start_in_front and end_in_front:
        return start, end
    if not start_in_front and not end_in_front:
        return None

    depth_delta = float(end[2] - start[2])
    if depth_delta == 0.0:
        raise ValueError("edge is parallel to the near plane but classified as straddling it")
    fraction = (minimum_depth - float(start[2])) / depth_delta
    intersection = start + fraction * (end - start)
    intersection = intersection.copy()
    intersection[2] = minimum_depth
    if start_in_front:
        return start, intersection
    return intersection, end


def _segment_intersects_image(
    start: tuple[float, float],
    end: tuple[float, float],
    image_size: tuple[int, int],
) -> bool:
    """Return whether a pixel segment overlaps the image rectangle.

    The rectangle is ``0 <= u <= width`` and ``0 <= v <= height``. Uses
    Cohen–Sutherland clipping.
    """
    width, height = image_size
    x0, y0 = start
    x1, y1 = end

    def code(x: float, y: float) -> int:
        bits = 0
        if x < 0.0:
            bits |= 1
        elif x > width:
            bits |= 2
        if y < 0.0:
            bits |= 4
        elif y > height:
            bits |= 8
        return bits

    c0, c1 = code(x0, y0), code(x1, y1)
    for _ in range(8):
        if c0 == 0 and c1 == 0:
            return True
        if c0 & c1:
            return False
        outside = c0 or c1
        if outside & 8:  # below the bottom edge
            if y1 == y0:
                raise ValueError("horizontal edge was classified as crossing y = height")
            x = x0 + (x1 - x0) * (height - y0) / (y1 - y0)
            y = float(height)
        elif outside & 4:  # above the top edge
            if y1 == y0:
                raise ValueError("horizontal edge was classified as crossing y = 0")
            x = x0 + (x1 - x0) * (0.0 - y0) / (y1 - y0)
            y = 0.0
        elif outside & 2:  # past the right edge
            if x1 == x0:
                raise ValueError("vertical edge was classified as crossing x = width")
            y = y0 + (y1 - y0) * (width - x0) / (x1 - x0)
            x = float(width)
        else:  # past the left edge
            if x1 == x0:
                raise ValueError("vertical edge was classified as crossing x = 0")
            y = y0 + (y1 - y0) * (0.0 - x0) / (x1 - x0)
            x = 0.0
        if outside == c0:
            x0, y0 = x, y
            c0 = code(x0, y0)
        else:
            x1, y1 = x, y
            c1 = code(x1, y1)
    raise RuntimeError("segment clipping did not converge")


def _inside_image(pixel: np.ndarray, image_size: tuple[int, int]) -> bool:
    """Return whether a pixel lies inside ``image_size`` (width, height)."""
    width, height = image_size
    u, v = float(pixel[0]), float(pixel[1])
    return 0.0 <= u <= width and 0.0 <= v <= height


def project_box_to_image(
    corners_cam: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    minimum_depth: float,
    name: str,
) -> ProjectedBox | None:
    """Project one camera-frame box into an image.

    Args:
        corners_cam: Shape ``(3, 8)``, meters, camera frame, in
            :func:`box_corners` order.
        intrinsic: Shape ``(3, 3)``, pixels.
        image_size: ``(width, height)`` in pixels.
        minimum_depth: Near-plane distance in meters.
        name: Category name copied onto the result.

    Returns:
        The visible edges, or ``None`` when every corner is on or behind the
        near plane or when no clipped edge intersects the image.
    """
    if not minimum_depth > 0.0:
        raise ValueError(f"minimum_depth must be > 0, got {minimum_depth}")
    corners_cam = _as_points(corners_cam, "corners_cam")
    if corners_cam.shape[1] != 8:
        raise ValueError(f"corners_cam must have shape (3, 8), got {corners_cam.shape}")
    if is_fully_behind(corners_cam, minimum_depth):
        return None

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for start_index, end_index in BOX_EDGES:
        clipped = clip_edge_to_depth(corners_cam[:, start_index], corners_cam[:, end_index], minimum_depth)
        if clipped is None:
            continue
        # Clipped endpoints sit on z = minimum_depth, which is in front of the
        # camera origin whenever minimum_depth > 0. Pass a zero near plane so
        # those endpoints are projected instead of rejected again.
        pixels, valid = project_points(np.column_stack(clipped), intrinsic, minimum_depth=0.0)
        if not bool(np.all(valid)):
            raise ValueError("clipped edge still has a point that cannot be projected")
        start_px = (float(pixels[0, 0]), float(pixels[1, 0]))
        end_px = (float(pixels[0, 1]), float(pixels[1, 1]))
        if _segment_intersects_image(start_px, end_px, image_size):
            segments.append((start_px, end_px))
    if not segments:
        return None

    label_anchor: tuple[float, float] | None = None
    center = corners_cam.mean(axis=1, keepdims=True)
    center_pixels, center_valid = project_points(center, intrinsic, minimum_depth)
    if bool(center_valid[0]) and _inside_image(center_pixels[:, 0], image_size):
        label_anchor = (float(center_pixels[0, 0]), float(center_pixels[1, 0]))
    return ProjectedBox(name=name, segments=tuple(segments), label_anchor=label_anchor)
