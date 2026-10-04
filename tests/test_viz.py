"""Tests for mosaic layout, camera ordering, and labeling."""

from pathlib import Path

import pytest
from PIL import Image

from surround_bev.data import CAMERA_CHANNELS
from surround_bev.viz import CAMERA_GRID, build_mosaic, label_image, save_image

# Distinct, saturated colors so each tile is identifiable from a single pixel.
COLORS: dict[str, tuple[int, int, int]] = {
    "CAM_FRONT_LEFT": (255, 0, 0),
    "CAM_FRONT": (0, 255, 0),
    "CAM_FRONT_RIGHT": (0, 0, 255),
    "CAM_BACK_LEFT": (255, 255, 0),
    "CAM_BACK": (0, 255, 255),
    "CAM_BACK_RIGHT": (255, 0, 255),
}


def _solid_images(size: tuple[int, int] = (160, 90)) -> dict[str, Image.Image]:
    return {ch: Image.new("RGB", size, color) for ch, color in COLORS.items()}


def test_camera_grid_ordering() -> None:
    assert CAMERA_GRID == (
        ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
        ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"),
    )
    # The grid is exactly the six channels, each used once, in row-major order.
    assert tuple(ch for row in CAMERA_GRID for ch in row) == CAMERA_CHANNELS


def test_mosaic_places_cameras_in_grid_order() -> None:
    tile_w, tile_h = 80, 45
    mosaic = build_mosaic(_solid_images(), grid=CAMERA_GRID, tile_size=(tile_w, tile_h), label=False)

    assert mosaic.size == (3 * tile_w, 2 * tile_h)
    for r, row in enumerate(CAMERA_GRID):
        for c, channel in enumerate(row):
            center = (c * tile_w + tile_w // 2, r * tile_h + tile_h // 2)
            assert mosaic.getpixel(center) == COLORS[channel], f"{channel} not at row {r}, col {c}"


def test_mosaic_labels_draw_on_tiles() -> None:
    tile_w, tile_h = 200, 100
    unlabeled = build_mosaic(_solid_images(), tile_size=(tile_w, tile_h), label=False)
    labeled = build_mosaic(_solid_images(), tile_size=(tile_w, tile_h), label=True)

    for r, row in enumerate(CAMERA_GRID):
        for c, channel in enumerate(row):
            corner = (c * tile_w + 2, r * tile_h + 2)
            assert unlabeled.getpixel(corner) == COLORS[channel]
            # Label backing box is black, so the corner no longer shows the tile color.
            assert labeled.getpixel(corner) == (0, 0, 0), f"no label box on {channel}"


def test_label_image_returns_same_image() -> None:
    img = Image.new("RGB", (100, 50), (255, 0, 0))
    out = label_image(img, "CAM_FRONT")
    assert out is img
    assert img.getpixel((1, 1)) == (0, 0, 0)


def test_mosaic_missing_channel_raises() -> None:
    images = _solid_images()
    del images["CAM_BACK"]
    with pytest.raises(KeyError, match="CAM_BACK"):
        build_mosaic(images)


@pytest.mark.parametrize("grid", [(), ((),), (("A", "B"), ("C",))])
def test_mosaic_rejects_bad_grid(grid: tuple) -> None:
    with pytest.raises(ValueError):
        build_mosaic(_solid_images(), grid=grid)


def test_save_image_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "mosaic.jpg"
    save_image(Image.new("RGB", (10, 10), (1, 2, 3)), out)
    assert out.is_file()
    assert Image.open(out).size == (10, 10)
