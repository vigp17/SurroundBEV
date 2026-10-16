"""Milestone 3: project the six cameras onto the ego ground plane.

Usage (from the repo root):
    python scripts/generate_ipm_bev.py --config configs/mini.yaml

The dataset path, sample index, BEV window, and output paths come from the
YAML config. No learned model is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from surround_bev.bev import BevGrid, draw_ego_marker
from surround_bev.config import Config, load_config
from surround_bev.data import (
    get_sample,
    load_camera_calibration,
    load_nuscenes,
    reference_ego_pose,
)
from surround_bev.ipm import (
    IPM_CAMERA_CHANNELS,
    camera_in_bev_frame,
    project_camera_to_bev,
    require_camera_channel,
    stitch_bev,
)
from surround_bev.viz import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "mini.yaml",
        help="Path to the YAML config (default: configs/mini.yaml)",
    )
    return parser.parse_args(argv)


def run(config: Config) -> tuple[Path, Path]:
    """Project the configured sample's cameras and write both IPM images.

    ``front_output`` is the configured front camera alone. ``bev_output`` is
    all six cameras, nearest camera first, ties broken by
    :data:`IPM_CAMERA_CHANNELS` order. Both images use the configured ego
    window. A yellow disc and a red forward arrow mark the ego vehicle.

    Args:
        config: Parsed configuration.

    Returns:
        ``(front_output, bev_output)``.
    """
    window = config.ipm
    require_camera_channel(window.front_camera)
    grid = BevGrid(window.x_min, window.x_max, window.y_min, window.y_max, window.resolution)
    nusc = load_nuscenes(config.dataset_root, config.version)
    sample = get_sample(nusc, config.sample_index)
    scene = nusc.get("scene", sample["scene_token"])
    reference = reference_ego_pose(nusc, sample)

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    front: np.ndarray | None = None
    for channel in IPM_CAMERA_CHANNELS:
        require_camera_channel(channel)
        calibration = load_camera_calibration(nusc, sample, channel)
        image = np.array(Image.open(calibration.image_path).convert("RGB"))
        rotation, translation = camera_in_bev_frame(
            calibration.ego_rotation,
            calibration.ego_translation,
            calibration.camera_rotation,
            calibration.camera_translation,
            reference.translation,
            reference.yaw_rad,
        )
        painted, distance = project_camera_to_bev(
            image,
            calibration.intrinsic,
            rotation,
            translation,
            grid,
        )
        frames.append((painted, distance))
        if channel == window.front_camera:
            front = painted
    if front is None:
        raise ValueError(f"front camera {window.front_camera!r} was not projected")

    front_image = draw_ego_marker(Image.fromarray(front, mode="RGB"), grid)
    bev_image = draw_ego_marker(Image.fromarray(stitch_bev(frames), mode="RGB"), grid)
    front_path = save_image(front_image, window.front_output)
    bev_path = save_image(bev_image, window.bev_output)

    print(f"Sample index:   {config.sample_index}")
    print(f"Sample token:   {sample['token']}")
    print(f"Scene:          {scene['name']}")
    print(f"Front camera:   {window.front_camera}")
    print(f"Cameras:        {', '.join(IPM_CAMERA_CHANNELS)}")
    print(
        "BEV extent:     "
        f"x [{grid.x_min:.1f}, {grid.x_max:.1f}] m, "
        f"y [{grid.y_min:.1f}, {grid.y_max:.1f}] m"
    )
    print(f"BEV resolution: {grid.resolution:.2f} m/px")
    print(f"Image shape:    {grid.shape}")
    print(f"Front IPM:      {front_path}")
    print(f"Stitched IPM:   {bev_path}")
    return front_path, bev_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        run(config)
    except PermissionError as exc:
        print(
            f"Permission denied reading the dataset: {exc}\n"
            "On macOS this usually means the terminal/app lacks access to that folder. "
            "Grant access in System Settings > Privacy & Security > Files and Folders "
            "(or Full Disk Access), or run from a terminal that already has access.",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, IndexError, KeyError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
