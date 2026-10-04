"""Milestone 2: build an ego-centric drivable-area BEV target for one sample.

Usage (from the repo root):
    python scripts/generate_bev_target.py --config configs/mini.yaml

The dataset path, sample index, BEV window, map layers, and output path are
read from the YAML config. The map name comes from the sample's log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surround_bev.bev import (
    BevGrid,
    ego_window_global_bounds,
    rasterize_drivable_mask,
    render_bev,
)
from surround_bev.config import Config, load_config
from surround_bev.data import (
    BEV_REFERENCE_CHANNEL,
    get_sample,
    load_nuscenes,
    map_location,
    reference_ego_pose,
)
from surround_bev.maps import load_layer_polygons, load_nuscenes_map
from surround_bev.viz import save_image


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


def raw_mask_path(visualization_path: Path) -> Path:
    """Return the ``.npy`` path of the raw mask that pairs with a visualization."""
    return visualization_path.with_suffix(".npy")


def run(config: Config) -> Path:
    """Rasterize the configured sample's drivable area and write both outputs.

    The ``.npy`` file is the raw ``uint8`` mask with values ``0`` and ``1``.
    The PNG is a visualization copy with an ego marker and a forward arrow.

    Args:
        config: Parsed configuration.

    Returns:
        Path of the visualization PNG.
    """
    grid = BevGrid(
        config.bev.x_min,
        config.bev.x_max,
        config.bev.y_min,
        config.bev.y_max,
        config.bev.resolution,
    )
    nusc = load_nuscenes(config.dataset_root, config.version)
    sample = get_sample(nusc, config.sample_index)
    scene = nusc.get("scene", sample["scene_token"])
    location = map_location(nusc, sample)
    pose = reference_ego_pose(nusc, sample)
    ego_xy = pose.translation[:2]

    nusc_map = load_nuscenes_map(config.dataset_root, location)
    bounds = ego_window_global_bounds(ego_xy, pose.yaw_rad, grid)
    polygons = []
    for layer in config.bev.layers:
        polygons.extend(load_layer_polygons(nusc_map, layer, bounds))
    mask = rasterize_drivable_mask(polygons, ego_xy, pose.yaw_rad, grid)
    visualization = render_bev(mask, grid)

    raw_path = raw_mask_path(config.bev.output_file)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, mask)
    out = save_image(visualization, config.bev.output_file)

    yaw_deg = float(np.degrees(pose.yaw_rad))
    print(f"Sample index:   {config.sample_index}")
    print(f"Sample token:   {sample['token']}")
    print(f"Scene:          {scene['name']}")
    print(f"Map location:   {location}")
    print(f"Ego pose:       {BEV_REFERENCE_CHANNEL} ({pose.ego_pose_token})")
    print(f"Ego x:          {pose.translation[0]:.3f} m")
    print(f"Ego y:          {pose.translation[1]:.3f} m")
    print(f"Ego yaw:        {yaw_deg:.3f} deg")
    print(
        "BEV extent:     "
        f"x [{grid.x_min:.1f}, {grid.x_max:.1f}] m, "
        f"y [{grid.y_min:.1f}, {grid.y_max:.1f}] m"
    )
    print(f"BEV resolution: {grid.resolution:.2f} m/px")
    print(f"Mask shape:     {mask.shape}")
    print(f"Raw mask:       {raw_path}")
    print(f"Output path:    {out}")
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
