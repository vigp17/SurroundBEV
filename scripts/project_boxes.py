"""Milestone 1: project ground-truth 3D boxes onto a six-camera mosaic.

Usage (from the repo root):
    python scripts/project_boxes.py --config configs/mini.yaml

The dataset location, sample index, projection settings, and output path are
read from the YAML config. The dataset path is not hardcoded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surround_bev.config import Config, load_config
from surround_bev.data import (
    CAMERA_CHANNELS,
    CameraCalibration,
    GlobalBox,
    get_sample,
    load_camera_calibration,
    load_camera_images,
    load_global_boxes,
    load_nuscenes,
)
from surround_bev.draw import draw_projected_boxes
from surround_bev.geometry import (
    box_corners,
    project_box_to_image,
    transform_points_to_camera,
)
from surround_bev.viz import CAMERA_GRID, build_mosaic, save_image


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


def _annotate_camera(
    image: Image.Image,
    boxes: list[GlobalBox],
    calibration: CameraCalibration,
    config: Config,
) -> tuple[Image.Image, int]:
    """Project every global box into one camera and draw the visible ones.

    A box is visible when it is not fully behind the near plane and at least
    one clipped edge intersects the image.
    """
    projected = []
    minimum_depth = config.projection.minimum_depth
    for box in boxes:
        corners_global = box_corners(box.center, box.size_wlh, box.rotation)
        corners_cam = transform_points_to_camera(
            corners_global,
            calibration.ego_translation,
            calibration.ego_rotation,
            calibration.camera_translation,
            calibration.camera_rotation,
        )
        visible = project_box_to_image(
            corners_cam,
            calibration.intrinsic,
            calibration.image_size,
            minimum_depth,
            box.name,
        )
        if visible is not None:
            projected.append(visible)
    annotated = draw_projected_boxes(
        image,
        projected,
        line_width=config.projection.line_width,
        show_labels=config.projection.show_labels,
    )
    return annotated, len(projected)


def run(config: Config) -> Path:
    """Project the configured sample's boxes and write the mosaic.

    Args:
        config: Parsed configuration.

    Returns:
        Path of the saved mosaic.
    """
    nusc = load_nuscenes(config.dataset_root, config.version)
    sample = get_sample(nusc, config.sample_index)
    scene = nusc.get("scene", sample["scene_token"])

    print(f"Sample index: {config.sample_index}")
    print(f"Sample token: {sample['token']}")
    print(f"Scene:        {scene['name']}")

    boxes = load_global_boxes(nusc, sample)
    images = load_camera_images(nusc, sample, CAMERA_CHANNELS)
    for channel in CAMERA_CHANNELS:
        calibration = load_camera_calibration(nusc, sample, channel)
        images[channel], count = _annotate_camera(images[channel], boxes, calibration, config)
        print(f"  {channel:<16} {count} visible boxes")

    # 800x450 tiles keep the 1600x900 camera aspect ratio (16:9).
    mosaic = build_mosaic(images, grid=CAMERA_GRID)
    out = save_image(mosaic, config.boxes_mosaic)
    print(f"Saved boxes mosaic ({mosaic.width}x{mosaic.height}) to {out}")
    return out


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
