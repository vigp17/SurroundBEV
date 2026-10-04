"""Visualization helpers: labeled camera tiles and the 2x3 surround mosaic.

This module works purely on PIL images and has no dependency on nuScenes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CAMERA_GRID: tuple[tuple[str, ...], ...] = (
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"),
)
"""Mosaic layout: row-major, front cameras on top, back cameras on the bottom."""

DEFAULT_TILE_SIZE: tuple[int, int] = (800, 450)
"""Per-camera tile size (width, height); nuScenes frames are 1600x900."""

_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Return a TrueType font if one is available, else PIL's bitmap default.

    Args:
        size: Font size in pixels (ignored for the bitmap fallback).
    """
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def label_image(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None,
    padding: int = 8,
) -> Image.Image:
    """Draw ``text`` in the top-left corner of ``image`` on a black backing box.

    The image is modified in place and also returned for convenience.

    Args:
        image: RGB image to annotate.
        text: Label text (e.g. a camera channel name).
        font: Font to use; defaults to :func:`load_font` at 28 px.
        padding: Pixels of padding around the text inside the box.
    """
    font = font or load_font(28)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    draw.rectangle((0, 0, text_w + 2 * padding, text_h + 2 * padding), fill=(0, 0, 0))
    draw.text((padding - left, padding - top), text, fill=(255, 255, 255), font=font)
    return image


def build_mosaic(
    images: Mapping[str, Image.Image],
    grid: Sequence[Sequence[str]] = CAMERA_GRID,
    tile_size: tuple[int, int] = DEFAULT_TILE_SIZE,
    label: bool = True,
) -> Image.Image:
    """Arrange camera images into a grid, optionally labeling each tile.

    Args:
        images: Mapping from channel name to image. Must contain every channel
            named in ``grid``.
        grid: Row-major layout of channel names. All rows must have equal length.
        tile_size: ``(width, height)`` each image is resized to.
        label: Whether to draw the channel name on each tile.

    Returns:
        A new RGB image of size ``(cols * width, rows * height)``.

    Raises:
        KeyError: If a channel in ``grid`` is missing from ``images``.
        ValueError: If ``grid`` is empty or ragged.
    """
    if not grid or not grid[0]:
        raise ValueError("grid must have at least one row and one column")
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("all rows in grid must have the same length")

    missing = [ch for row in grid for ch in row if ch not in images]
    if missing:
        raise KeyError(f"Missing images for channels: {missing}")

    tile_w, tile_h = tile_size
    mosaic = Image.new("RGB", (cols * tile_w, len(grid) * tile_h), color=(0, 0, 0))
    font = load_font(28) if label else None

    for r, row in enumerate(grid):
        for c, channel in enumerate(row):
            tile = images[channel].convert("RGB").resize((tile_w, tile_h), Image.BILINEAR)
            if label:
                label_image(tile, channel, font)
            mosaic.paste(tile, (c * tile_w, r * tile_h))
    return mosaic


def save_image(image: Image.Image, path: Path, quality: int = 92) -> Path:
    """Write ``image`` to ``path``, creating parent directories as needed.

    Args:
        image: Image to save.
        path: Destination file; format is inferred from the suffix.
        quality: JPEG quality (ignored for other formats).

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality)
    return path
