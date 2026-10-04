"""Draw projected 3D boxes onto camera images.

This module only paints pixels. It does not load nuScenes records or project
coordinates; see :mod:`surround_bev.geometry` for that.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw

from surround_bev.geometry import ProjectedBox
from surround_bev.viz import load_font

# Distinct, readable colors. The index is a stable sum of the category name,
# not Python's randomized hash().
_CATEGORY_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 56, 56),
    (56, 220, 56),
    (56, 160, 255),
    (255, 200, 40),
    (220, 80, 255),
    (40, 220, 220),
    (255, 140, 40),
)

# Drawn on the original (typically 1600x900) image, then the mosaic scales
# the image by one half, so this lands near 16 px on a tile.
_LABEL_FONT_PX = 32


def category_color(name: str) -> tuple[int, int, int]:
    """Return a stable RGB color for a category name."""
    index = sum(name.encode("utf-8")) % len(_CATEGORY_COLORS)
    return _CATEGORY_COLORS[index]


def _draw_label(canvas: Image.Image, anchor: tuple[float, float], text: str) -> None:
    """Draw ``text`` on a black backing box with its top-left near ``anchor``."""
    draw = ImageDraw.Draw(canvas)
    font = load_font(_LABEL_FONT_PX)
    padding = 4
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    x = min(max(round(anchor[0]), 0), max(canvas.width - 1, 0))
    y = min(max(round(anchor[1]) - text_h - 2 * padding, 0), max(canvas.height - 1, 0))
    draw.rectangle((x, y, x + text_w + 2 * padding, y + text_h + 2 * padding), fill=(0, 0, 0))
    draw.text((x + padding - left, y + padding - top), text, fill=(255, 255, 255), font=font)


def draw_projected_boxes(
    image: Image.Image,
    boxes: Sequence[ProjectedBox],
    line_width: int,
    show_labels: bool,
) -> Image.Image:
    """Draw projected box edges, and optionally category labels, on a copy.

    Args:
        image: RGB camera image. Not modified.
        boxes: Boxes already projected into this image's pixel frame.
        line_width: Stroke width in pixels on ``image``.
        show_labels: When true, draw each box name at its label anchor.

    Returns:
        A new RGB image.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        color = category_color(box.name)
        for (u0, v0), (u1, v1) in box.segments:
            draw.line([(u0, v0), (u1, v1)], fill=color, width=line_width)
        if show_labels and box.label_anchor is not None and box.name:
            _draw_label(canvas, box.label_anchor, box.name)
    return canvas
