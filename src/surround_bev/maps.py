"""nuScenes semantic-map access for BEV targets.

The map name is the log location of the configured sample. This module does
not rasterize and does not draw.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from surround_bev.bev import SUPPORTED_BEV_LAYERS, GlobalPolygon

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from nuscenes.map_expansion.map_api import NuScenesMap


def load_nuscenes_map(dataset_root: Path, location: str) -> NuScenesMap:
    """Open the semantic map for a discovered nuScenes location.

    Args:
        dataset_root: Dataset directory that contains ``maps/expansion``.
        location: Map name from the sample's log, for example the value of
            ``log["location"]``. This is not hardcoded by the caller.

    Returns:
        An initialized :class:`nuscenes.map_expansion.map_api.NuScenesMap`.

    Raises:
        ValueError: If ``location`` is not one of the maps shipped with the
            devkit.
        FileNotFoundError: If the map expansion file is missing.
    """
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.map_expansion.map_api import locations as known_locations

    if location not in known_locations:
        raise ValueError(f"unsupported map location {location!r}; known locations: {list(known_locations)}")
    map_file = Path(dataset_root) / "maps" / "expansion" / f"{location}.json"
    if not map_file.is_file():
        raise FileNotFoundError(f"nuScenes map expansion not found: {map_file}")
    return NuScenesMap(dataroot=str(dataset_root), map_name=location)


def load_layer_polygons(
    nusc_map: NuScenesMap,
    layer: str,
    global_bounds: tuple[float, float, float, float],
) -> list[GlobalPolygon]:
    """Load polygons of one map layer that meet a global axis-aligned window.

    Args:
        nusc_map: Open semantic map.
        layer: Layer name. Only :data:`~surround_bev.bev.SUPPORTED_BEV_LAYERS`
            are accepted.
        global_bounds: ``(x_min, y_min, x_max, y_max)`` in the global map
            plane, meters.

    Returns:
        Polygons in global meters. Rings are ``(N, 2)`` with columns ``(x, y)``.

    Raises:
        ValueError: If ``layer`` is not supported.
    """
    if layer not in SUPPORTED_BEV_LAYERS:
        supported = ", ".join(sorted(SUPPORTED_BEV_LAYERS))
        raise ValueError(f"unsupported map layer {layer!r}; supported: {supported}")

    records = nusc_map.get_records_in_patch(global_bounds, layer_names=[layer], mode="intersect")
    polygons: list[GlobalPolygon] = []
    for token in records[layer]:
        record = nusc_map.get(layer, token)
        for polygon_token in record["polygon_tokens"]:
            geometry = nusc_map.extract_polygon(polygon_token)
            exterior = _ring(geometry.exterior.coords)
            holes = tuple(_ring(interior.coords) for interior in geometry.interiors)
            polygons.append(GlobalPolygon(exterior=exterior, holes=holes))
    return polygons


def _ring(coords: Any) -> np.ndarray:
    """Return shapely coordinates as a float array of shape ``(N, 2)``."""
    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"polygon ring must have shape (N, 2), got {points.shape}")
    return points[:, :2]
