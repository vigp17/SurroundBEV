"""nuScenes data access.

This module opens the dataset and returns images, global-frame annotations, and
per-camera calibration. It does not transform coordinates or draw; see
:mod:`surround_bev.geometry` and :mod:`surround_bev.draw`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from nuscenes.nuscenes import NuScenes

CAMERA_CHANNELS: tuple[str, ...] = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
"""The six surround-view camera channels in nuScenes."""


class SampleTable(Protocol):
    """Minimal interface of ``NuScenes`` needed by :func:`get_sample`.

    Declared as a protocol so the index logic can be tested without the
    devkit or a dataset on disk.
    """

    sample: list[dict[str, Any]]


def load_nuscenes(dataset_root: Path, version: str, verbose: bool = False) -> NuScenes:
    """Open a nuScenes database.

    Args:
        dataset_root: Directory holding ``samples/`` and ``<version>/``.
        version: nuScenes version, e.g. ``"v1.0-mini"``.
        verbose: Forwarded to the devkit; prints table statistics when true.

    Returns:
        An initialized :class:`nuscenes.nuscenes.NuScenes` instance.

    Raises:
        FileNotFoundError: If ``dataset_root`` or its table folder is missing.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {dataset_root}")
    if not (dataset_root / version).is_dir():
        raise FileNotFoundError(f"nuScenes tables not found at {dataset_root / version}")

    # Imported lazily so the rest of the package (and tests) do not need the devkit.
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=version, dataroot=str(dataset_root), verbose=verbose)


def get_sample(nusc: SampleTable, index: int) -> dict[str, Any]:
    """Return the sample record at ``index``.

    Args:
        nusc: Any object exposing a ``sample`` list (normally ``NuScenes``).
        index: Zero-based position in the sample table. Negative indices are
            rejected rather than wrapping around.

    Returns:
        The sample record (a dict with ``token``, ``scene_token``, ``data``...).

    Raises:
        IndexError: If ``index`` is negative or beyond the table length.
    """
    n = len(nusc.sample)
    if not 0 <= index < n:
        raise IndexError(f"sample_index {index} out of range; dataset has {n} samples (valid: 0..{n - 1})")
    return nusc.sample[index]


def get_camera_image_paths(
    nusc: NuScenes,
    sample: dict[str, Any],
    channels: Sequence[str] = CAMERA_CHANNELS,
) -> dict[str, Path]:
    """Resolve the image file for each requested camera channel of a sample.

    Args:
        nusc: Open ``NuScenes`` instance.
        sample: A sample record from ``nusc.sample``.
        channels: Camera channel names to look up.

    Returns:
        Mapping from channel name to absolute image path.

    Raises:
        KeyError: If the sample has no key-frame data for a channel.
    """
    paths: dict[str, Path] = {}
    for channel in channels:
        if channel not in sample["data"]:
            raise KeyError(f"Sample {sample['token']} has no data for channel {channel}")
        paths[channel] = Path(nusc.get_sample_data_path(sample["data"][channel]))
    return paths


def load_camera_images(
    nusc: NuScenes,
    sample: dict[str, Any],
    channels: Sequence[str] = CAMERA_CHANNELS,
) -> dict[str, Image.Image]:
    """Load the camera images for a sample as RGB PIL images.

    Args:
        nusc: Open ``NuScenes`` instance.
        sample: A sample record from ``nusc.sample``.
        channels: Camera channel names to load.

    Returns:
        Mapping from channel name to an RGB :class:`PIL.Image.Image`.
    """
    paths = get_camera_image_paths(nusc, sample, channels)
    return {channel: Image.open(path).convert("RGB") for channel, path in paths.items()}


@dataclass(frozen=True)
class GlobalBox:
    """One ground-truth box in the nuScenes global frame.

    Attributes:
        center: Shape ``(3,)``, meters, global coordinates.
        size_wlh: Shape ``(3,)``, meters, ordered ``(width, length, height)``.
            Width is along the box's local y axis (left), length along local x
            (forward), and height along local z (up).
        rotation: Shape ``(3, 3)``. Maps box-local vectors into the global frame.
        name: Category name, for example ``"vehicle.car"``.
    """

    center: np.ndarray
    size_wlh: np.ndarray
    rotation: np.ndarray
    name: str


@dataclass(frozen=True)
class CameraCalibration:
    """Extrinsics and intrinsics for one camera image.

    The camera frame has +x right, +y down, and +z forward, in meters.
    ``ego_rotation`` maps ego-frame vectors into the global frame.
    ``camera_rotation`` maps camera-frame vectors into the ego frame.
    Both translations are in meters: ego in the global frame, camera in the
    ego frame. ``intrinsic`` has shape ``(3, 3)`` and maps camera-frame
    meters to pixels. ``image_size`` is ``(width, height)`` in pixels.

    Attributes:
        channel: nuScenes channel name, for example ``"CAM_FRONT"``.
        image_path: Absolute path of the image file.
        image_size: ``(width, height)`` in pixels.
        intrinsic: Shape ``(3, 3)``, pixels.
        ego_translation: Shape ``(3,)``, meters, global frame.
        ego_rotation: Shape ``(3, 3)``.
        camera_translation: Shape ``(3,)``, meters, ego frame.
        camera_rotation: Shape ``(3, 3)``.
    """

    channel: str
    image_path: Path
    image_size: tuple[int, int]
    intrinsic: np.ndarray
    ego_translation: np.ndarray
    ego_rotation: np.ndarray
    camera_translation: np.ndarray
    camera_rotation: np.ndarray


def quaternion_to_matrix(wxyz: Sequence[float]) -> np.ndarray:
    """Convert a nuScenes quaternion ``(w, x, y, z)`` to a rotation matrix.

    The returned matrix has shape ``(3, 3)`` and maps vectors from the child
    frame into the parent frame. This is the convention used by
    ``ego_pose.rotation`` and ``calibrated_sensor.rotation``.

    Args:
        wxyz: Quaternion components in ``(w, x, y, z)`` order.

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    matrix = np.asarray(Quaternion(wxyz).rotation_matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"quaternion produced shape {matrix.shape}, expected (3, 3)")
    return matrix


def _vec3(values: Sequence[float], what: str) -> np.ndarray:
    """Return ``values`` as a float array of shape ``(3,)``."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{what} must have shape (3,), got {vector.shape}")
    return vector


def load_global_boxes(nusc: NuScenes, sample: dict[str, Any]) -> list[GlobalBox]:
    """Load every ground-truth box annotated on ``sample``.

    Boxes stay in the global frame. Keyframe camera images referenced by
    ``sample["data"]`` share this annotation timestamp, so no interpolation
    is applied.

    Args:
        nusc: Open ``NuScenes`` instance.
        sample: A sample record from ``nusc.sample``.

    Returns:
        One :class:`GlobalBox` per annotation, in table order.
    """
    boxes: list[GlobalBox] = []
    for token in sample["anns"]:
        record = nusc.get("sample_annotation", token)
        boxes.append(
            GlobalBox(
                center=_vec3(record["translation"], f"annotation {token} translation"),
                size_wlh=_vec3(record["size"], f"annotation {token} size"),
                rotation=quaternion_to_matrix(record["rotation"]),
                name=str(record["category_name"]),
            )
        )
    return boxes


def load_camera_calibration(nusc: NuScenes, sample: dict[str, Any], channel: str) -> CameraCalibration:
    """Load intrinsics and ego/camera poses for one channel of ``sample``.

    Args:
        nusc: Open ``NuScenes`` instance.
        sample: A sample record from ``nusc.sample``.
        channel: Camera channel name.

    Returns:
        Calibration for that camera's keyframe.

    Raises:
        KeyError: If ``sample`` has no data for ``channel``.
        ValueError: If the intrinsic matrix is not ``(3, 3)``.
    """
    if channel not in sample["data"]:
        raise KeyError(f"Sample {sample['token']} has no data for channel {channel}")
    sample_data_token = sample["data"][channel]
    sample_data = nusc.get("sample_data", sample_data_token)
    calibrated = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    intrinsic = np.asarray(calibrated["camera_intrinsic"], dtype=float)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"{channel} camera_intrinsic has shape {intrinsic.shape}, expected (3, 3)")
    return CameraCalibration(
        channel=channel,
        image_path=Path(nusc.get_sample_data_path(sample_data_token)),
        image_size=(int(sample_data["width"]), int(sample_data["height"])),
        intrinsic=intrinsic,
        ego_translation=_vec3(ego_pose["translation"], f"{channel} ego translation"),
        ego_rotation=quaternion_to_matrix(ego_pose["rotation"]),
        camera_translation=_vec3(calibrated["translation"], f"{channel} camera translation"),
        camera_rotation=quaternion_to_matrix(calibrated["rotation"]),
    )
