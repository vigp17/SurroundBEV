"""Ego-centric BEV grid math and drivable-area rasterization.

Coordinate frames and units
---------------------------
Global map frame
    nuScenes map plane. ``x`` and ``y`` are meters. This is the same plane as
    ``ego_pose.translation`` x and y.

Ego frame
    Origin at the ego position. Positive x is ego forward, positive y is ego
    left, both in meters. The transform uses the planar heading of the ego
    x-axis and ignores pitch, roll, and height. See :func:`world_to_ego`.

Grid and image
    The mask is indexed ``mask[row, col]`` with shape ``(rows, cols)``.

    * Row 0 is the forward edge (``x = x_max``), so image top is ego forward.
    * The last row is the backward edge (``x = x_min``).
    * Column 0 is the left edge (``y = y_max``), so image left is ego left.
    * The last column is the right edge (``y = y_min``).

    Continuous indices of an ego-frame point ``(x, y)`` are

    ``row = (x_max - x) / resolution``
    ``col = (y_max - y) / resolution``

    Cell ``(row, col)`` is drivable when its center is strictly inside a
    drivable polygon and outside that polygon's holes. The center is

    ``x = x_max - (row + 0.5) * resolution``
    ``y = y_max - (col + 0.5) * resolution``

    The ego origin is the continuous index, which lies on a cell boundary when
    ``x_max`` and ``y_max`` are integer multiples of the resolution. It is not
    written into the raw mask.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from shapely import contains_xy, make_valid
from shapely.geometry import Polygon, box

SUPPORTED_BEV_LAYERS: frozenset[str] = frozenset({"drivable_area"})
"""Map layers this milestone can rasterize. Any other name is rejected."""

_NONDRIVABLE_RGB: tuple[int, int, int] = (32, 32, 32)
_DRIVABLE_RGB: tuple[int, int, int] = (176, 176, 176)
_EGO_RGB: tuple[int, int, int] = (255, 196, 32)
_HEADING_RGB: tuple[int, int, int] = (220, 48, 48)
_ARROW_METERS = 8.0


@dataclass(frozen=True)
class BevGrid:
    """Metric window of an ego-centric BEV mask.

    Attributes:
        x_min: Rear edge, meters. Negative is behind the ego vehicle.
        x_max: Forward edge, meters.
        y_min: Right edge, meters. Negative ego-y is to the right.
        y_max: Left edge, meters.
        resolution: Cell size in meters. Both spans must be integer multiples
            of this value.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    def __post_init__(self) -> None:
        grid_shape(self.x_min, self.x_max, self.y_min, self.y_max, self.resolution)

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the mask."""
        return grid_shape(self.x_min, self.x_max, self.y_min, self.y_max, self.resolution)

    @property
    def rows(self) -> int:
        """Number of rows. This is the forward/backward dimension."""
        return self.shape[0]

    @property
    def cols(self) -> int:
        """Number of columns. This is the left/right dimension."""
        return self.shape[1]


@dataclass(frozen=True)
class GlobalPolygon:
    """One map polygon in the nuScenes global plane.

    Attributes:
        exterior: Shape ``(N, 2)``, meters. Column 0 is global x, column 1 is
            global y. ``N >= 3``.
        holes: Interior rings, each shape ``(M, 2)``, same units and columns.
    """

    exterior: np.ndarray
    holes: tuple[np.ndarray, ...] = ()


def grid_shape(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: float,
) -> tuple[int, int]:
    """Return ``(rows, cols)`` for a metric BEV window.

    Args:
        x_min: Rear edge, meters.
        x_max: Forward edge, meters. Must be strictly greater than ``x_min``.
        y_min: Right edge, meters.
        y_max: Left edge, meters. Must be strictly greater than ``y_min``.
        resolution: Cell size in meters. Must be strictly positive, and both
            spans must be integer multiples of it.

    Returns:
        ``(rows, cols)``. Rows span x. Columns span y.

    Raises:
        ValueError: If the extent is empty or reversed, or if the resolution
            does not divide both spans into a positive integer number of cells.
    """
    if not x_min < x_max or not y_min < y_max:
        raise ValueError(f"invalid extent: x [{x_min}, {x_max}], y [{y_min}, {y_max}]")
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)) or not resolution > 0:
        raise ValueError(f"invalid resolution: {resolution!r}")
    rows = _cell_count(x_max - x_min, float(resolution), "x")
    cols = _cell_count(y_max - y_min, float(resolution), "y")
    return rows, cols


def _cell_count(span: float, resolution: float, axis: str) -> int:
    """Return the integer cell count for one axis, or raise ``ValueError``."""
    count = span / resolution
    rounded = round(count)
    if rounded < 1 or abs(count - rounded) > 1e-6:
        raise ValueError(
            f"invalid resolution {resolution}: {axis} span {span} m is not an integer number of cells ({count})"
        )
    return rounded


def world_to_ego(points_world: np.ndarray, ego_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Transform global map points into the ego frame.

    Args:
        points_world: Shape ``(2, N)``, meters. Row 0 is global x, row 1 is
            global y.
        ego_xy: Shape ``(2,)``, meters, ego position in the global map plane.
        yaw_rad: Ego heading in radians. Zero means ego forward points along
            global +x. Positive yaw rotates that heading toward global +y.
            This is the planar heading of the ego x-axis, not a camera yaw.

    Returns:
        Shape ``(2, N)``, meters. Row 0 is ego forward, row 1 is ego left.

        ``x_ego =  cos(yaw) * dx + sin(yaw) * dy``
        ``y_ego = -sin(yaw) * dx + cos(yaw) * dy``
    """
    points = _as_points2(points_world, "points_world")
    origin = _as_vec2(ego_xy, "ego_xy").reshape(2, 1)
    delta = points - origin
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    forward = cosine * delta[0] + sine * delta[1]
    left = -sine * delta[0] + cosine * delta[1]
    return np.vstack((forward, left))


def ego_to_world(points_ego: np.ndarray, ego_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Inverse of :func:`world_to_ego`.

    Args:
        points_ego: Shape ``(2, N)``, meters. Row 0 is forward, row 1 is left.
        ego_xy: Shape ``(2,)``, meters, global map position.
        yaw_rad: Ego heading in radians. See :func:`world_to_ego`.

    Returns:
        Shape ``(2, N)``, meters, global map plane.
    """
    points = _as_points2(points_ego, "points_ego")
    origin = _as_vec2(ego_xy, "ego_xy").reshape(2, 1)
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    x = cosine * points[0] - sine * points[1]
    y = sine * points[0] + cosine * points[1]
    return np.vstack((x, y)) + origin


def ego_to_grid(x: float, y: float, grid: BevGrid) -> tuple[float, float]:
    """Map one ego-frame point to continuous ``(row, col)`` indices.

    Args:
        x: Ego forward coordinate, meters.
        y: Ego left coordinate, meters.
        grid: BEV window.

    Returns:
        ``(row, col)``. Row 0 is the forward edge. Column 0 is the left edge.
        Values may fall outside ``[0, rows]`` and ``[0, cols]`` when the point
        is outside the window. The ego origin is this continuous index, which
        sits on a cell boundary when the edges are multiples of the resolution.
    """
    row = (grid.x_max - x) / grid.resolution
    col = (grid.y_max - y) / grid.resolution
    return float(row), float(col)


def pixel_center(row: int, col: int, grid: BevGrid) -> tuple[float, float]:
    """Return the ego-frame ``(x, y)`` meters at the center of one cell.

    Args:
        row: Row index, ``0`` at the forward edge.
        col: Column index, ``0`` at the left edge.
        grid: BEV window.

    Returns:
        ``(x, y)`` in meters. x is forward, y is left.
    """
    x = grid.x_max - (row + 0.5) * grid.resolution
    y = grid.y_max - (col + 0.5) * grid.resolution
    return float(x), float(y)


def ego_window_global_bounds(ego_xy: np.ndarray, yaw_rad: float, grid: BevGrid) -> tuple[float, float, float, float]:
    """Return the global axis-aligned box that covers the ego window.

    Args:
        ego_xy: Shape ``(2,)``, meters.
        yaw_rad: Ego heading in radians.
        grid: Ego-frame window.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` in the global map plane, meters.
    """
    corners = np.array(
        [
            [grid.x_min, grid.x_min, grid.x_max, grid.x_max],
            [grid.y_min, grid.y_max, grid.y_min, grid.y_max],
        ],
        dtype=float,
    )
    world = ego_to_world(corners, ego_xy, yaw_rad)
    return (
        float(np.min(world[0])),
        float(np.min(world[1])),
        float(np.max(world[0])),
        float(np.max(world[1])),
    )


def rasterize_drivable_mask(
    polygons: Sequence[GlobalPolygon],
    ego_xy: np.ndarray,
    yaw_rad: float,
    grid: BevGrid,
) -> np.ndarray:
    """Rasterize global drivable polygons into a binary ego-centric mask.

    The returned array is the raw training target. It contains only ``0`` and
    ``1`` and has no ego marker, arrow, text, or color.

    Args:
        polygons: Drivable polygons in the global map plane.
        ego_xy: Shape ``(2,)``, meters, global position of the ego origin.
        yaw_rad: Ego heading in radians.
        grid: Output window.

    Returns:
        ``uint8`` array of shape ``(rows, cols)``. ``1`` is drivable.
    """
    rows, cols = grid.shape
    col_index, row_index = np.meshgrid(np.arange(cols), np.arange(rows))
    center_x = grid.x_max - (row_index + 0.5) * grid.resolution
    center_y = grid.y_max - (col_index + 0.5) * grid.resolution
    window = box(grid.x_min, grid.y_min, grid.x_max, grid.y_max)
    covered = np.zeros((rows, cols), dtype=bool)

    for polygon in polygons:
        exterior = _ring_to_ego(polygon.exterior, ego_xy, yaw_rad)
        holes = tuple(_ring_to_ego(hole, ego_xy, yaw_rad) for hole in polygon.holes)
        if exterior.shape[0] < 3:
            continue
        geometry = Polygon(exterior, [hole for hole in holes if hole.shape[0] >= 3])
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        clipped = geometry.intersection(window)
        _mark_covered(covered, clipped, center_x, center_y)

    return covered.astype(np.uint8)


def render_bev(mask: np.ndarray, grid: BevGrid) -> Image.Image:
    """Draw a visualization copy of a binary BEV mask.

    The ego marker and heading arrow are painted only on the returned image.
    ``mask`` is not modified.

    Args:
        mask: ``uint8`` array of shape ``grid.shape`` containing only ``0`` and
            ``1``.
        grid: Window that produced ``mask``.

    Returns:
        RGB image of size ``(cols, rows)``. Image top is ego forward. A yellow
        disc marks the ego origin and a red arrow points forward.
    """
    if mask.shape != grid.shape:
        raise ValueError(f"mask shape {mask.shape} does not match grid shape {grid.shape}")
    if mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must be a uint8 array containing only 0 and 1")

    canvas = np.empty((*mask.shape, 3), dtype=np.uint8)
    canvas[mask == 0] = _NONDRIVABLE_RGB
    canvas[mask == 1] = _DRIVABLE_RGB
    image = Image.fromarray(canvas, mode="RGB")
    _draw_ego_overlay(image, grid)
    return image


def _draw_ego_overlay(image: Image.Image, grid: BevGrid) -> None:
    """Paint the ego disc and forward arrow. Both are visualization-only."""
    origin_row, origin_col = ego_to_grid(0.0, 0.0, grid)
    if not _inside_image(origin_col, origin_row, image.size):
        return
    draw = ImageDraw.Draw(image)
    arrow_length = min(_ARROW_METERS, max(grid.x_max, 0.0))
    if arrow_length > 0.0:
        tip_row, tip_col = ego_to_grid(arrow_length, 0.0, grid)
        draw.line([(origin_col, origin_row), (tip_col, tip_row)], fill=_HEADING_RGB, width=3)
        _draw_arrow_head(draw, origin_col, origin_row, tip_col, tip_row)
    radius = max(4, round(1.0 / grid.resolution))
    draw.ellipse(
        (origin_col - radius, origin_row - radius, origin_col + radius, origin_row + radius),
        fill=_EGO_RGB,
        outline=(0, 0, 0),
    )


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    """Draw a small triangular head at ``(x1, y1)`` pointing away from ``(x0, y0)``."""
    direction = np.array([x1 - x0, y1 - y0], dtype=float)
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    tip = np.array([x1, y1])
    base = tip - unit * 10.0
    left = base + normal * 5.0
    right = base - normal * 5.0
    draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=_HEADING_RGB)


def _inside_image(x: float, y: float, image_size: tuple[int, int]) -> bool:
    width, height = image_size
    return 0.0 <= x < width and 0.0 <= y < height


def _ring_to_ego(ring: np.ndarray, ego_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Return a global ``(N, 2)`` ring as ego-frame ``(N, 2)``."""
    points = np.asarray(ring, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"polygon ring must have shape (N, 2), got {points.shape}")
    if points.shape[0] == 0:
        return points
    return world_to_ego(points.T, ego_xy, yaw_rad).T


def _mark_covered(covered: np.ndarray, geometry: Polygon, center_x: np.ndarray, center_y: np.ndarray) -> None:
    """Set ``covered`` where cell centers lie in ``geometry`` but not in a hole."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        parts: list[Polygon] = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        parts = list(geometry.geoms)
    elif geometry.geom_type == "GeometryCollection":
        for part in geometry.geoms:
            _mark_covered(covered, part, center_x, center_y)
        return
    else:
        return

    flat_x = center_x.ravel()
    flat_y = center_y.ravel()
    for part in parts:
        if part.is_empty or part.exterior is None:
            continue
        inside = contains_xy(part, flat_x, flat_y).reshape(covered.shape)
        for interior in part.interiors:
            hole = Polygon(interior)
            if hole.is_empty:
                continue
            inside &= ~contains_xy(hole, flat_x, flat_y).reshape(covered.shape)
        covered |= inside


def _as_points2(points: np.ndarray, what: str) -> np.ndarray:
    """Return ``points`` as a float array of shape ``(2, N)``."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"{what} must have shape (2, N), got {array.shape}")
    return array


def _as_vec2(vector: np.ndarray, what: str) -> np.ndarray:
    """Return ``vector`` as a float array of shape ``(2,)``."""
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape != (2,):
        raise ValueError(f"{what} must have shape (2,), got {array.shape}")
    return array
