"""Milestone 0: load one nuScenes sample and save a labeled 6-camera mosaic.

Usage (from the repo root):
    python scripts/inspect_sample.py [--config configs/mini.yaml]

The dataset location, version, sample index and output file are all read
from the YAML config; nothing here hardcodes a dataset path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surround_bev.config import Config, load_config
from surround_bev.data import (
    CAMERA_CHANNELS,
    get_sample,
    load_camera_images,
    load_nuscenes,
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


def run(config: Config) -> Path:
    """Load the configured sample, build the mosaic, and write it to disk.

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
    print(f"Timestamp:    {sample['timestamp']}")

    images = load_camera_images(nusc, sample, CAMERA_CHANNELS)
    for channel in CAMERA_CHANNELS:
        w, h = images[channel].size
        print(f"  {channel:<16} {w}x{h}")

    mosaic = build_mosaic(images, grid=CAMERA_GRID)
    out = save_image(mosaic, config.output_file)
    print(f"Saved mosaic ({mosaic.width}x{mosaic.height}) to {out}")
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
    except (FileNotFoundError, IndexError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
