"""nuScenes data access.

This module only knows how to open the dataset and fetch samples/images. It
contains no drawing or layout logic; see :mod:`surround_bev.viz` for that.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from PIL import Image

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
