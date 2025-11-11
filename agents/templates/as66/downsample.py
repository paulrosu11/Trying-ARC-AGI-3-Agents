"""
AS66 downsampling utilities.

AS66 raw frames are 64×64 where each semantic tile is a uniform 4×4 block.
We reduce to 16×16 by averaging each 4×4 block (robust to small edge variations)
and rounding to the nearest integer code.

Text mode: never expose colors; only integers.
Visual mode: you may render PNGs, but do not leak numbers into text prompts.
"""

from __future__ import annotations
from typing import Iterable, List, Callable
import io
import base64
from PIL import Image, ImageDraw, ImageFont


Number = float | int


def _mean(vals: Iterable[Number]) -> float:
    vs = list(vals)
    return (sum(vs) / float(len(vs))) if vs else 0.0


def downsample_blocks(
    grid: List[List[int]],
    block_h: int = 4,
    block_w: int = 4,
    reducer: Callable[[Iterable[Number]], float] = _mean,
    *,
    round_to_int: bool = True,
) -> List[List[int | float]]:
    if not grid or not grid[0]:
        return []
    H, W = len(grid), len(grid[0])
    out_h = (H + block_h - 1) // block_h
    out_w = (W + block_w - 1) // block_w
    out: List[List[int | float]] = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for by in range(out_h):
        y0, y1 = by * block_h, min(H, (by + 1) * block_h)
        for bx in range(out_w):
            x0, x1 = bx * block_w, min(W, (bx + 1) * block_w)
            acc: list[Number] = []
            for y in range(y0, y1):
                acc.extend(grid[y][x0:x1])
            val = reducer(acc)
            out[by][bx] = int(round(val)) if round_to_int else val
    return out


def downsample_4x4(
    frame_3d: List[List[List[int]]] | None,
    *,
    take_last_grid: bool = True,
    round_to_int: bool = True,
) -> List[List[int]]:
    """
    Select one 2D grid from the 3D frame list, then 4×4-average → 16×16.
    """
    if not frame_3d:
        return []
    grid = frame_3d[-1] if take_last_grid else frame_3d[0]
    if not grid or not grid[0]:
        return []
    # type: ignore[return-value]
    return downsample_blocks(grid, 4, 4, _mean, round_to_int=round_to_int)


def matrix16_to_lines(mat: List[List[int]]) -> str:
    """
    Numeric-only textual form (no headers/legends).
    """
    if not mat:
        return "(empty)"
    return "\n".join(" ".join(str(v) for v in row) for row in mat)


#tiny 16×16 visualization for vision/inspection (never used in text prompts) 

KEY_COLORS = {
    0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999",
    3: "#666666", 4: "#000000", 5: "#202020",
    6: "#1E93FF", 7: "#F93C31", 8: "#FF851B",
    9: "#921231", 10: "#88D8F1", 11: "#FFDC00",
    12: "#FF7BCC", 13: "#4FCC30", 14: "#2ECC71",
    15: "#7F3FBF",
}

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return (136, 136, 136)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def render_grid_to_png_bytes(grid: List[List[int]], cell: int = 22) -> bytes:
    """
    Generates a color PNG from a grid (e.g., 16x16 or 64x64).
    
    Args:
        grid: The 2D integer grid.
        cell: The pixel size (width and height) for each grid cell.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    if h == 0 or w == 0:
        # Return a 1x1 black pixel as a fallback
        im = Image.new("RGB", (1, 1), (0, 0, 0))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return buf.getvalue()

    H, W = h * cell, w * cell
    im = Image.new("RGB", (W, H), (0, 0, 0))
    px = im.load()
    for y in range(h):
        row = grid[y]
        for x in range(w):
            code = row[x]
            rgb = _hex_to_rgb(KEY_COLORS.get(int(code) & 15, "#888888"))
            for dy in range(cell):
                for dx in range(cell):
                    px[x * cell + dx, y * cell + dy] = rgb
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def generate_numeric_grid_image_bytes(grid: List[List[int]]) -> bytes:
    """
    Generates a PNG 'screenshot' of the 16x16 grid with numbers and headers.
    """
    cell_size = 24
    header_size = 24
    grid_size = 16
    img_size_w = (grid_size * cell_size) + header_size
    img_size_h = (grid_size * cell_size) + header_size

    bg_color = "#FFFFFF"
    line_color = "#000000"
    text_color = "#000000"
    header_bg = "#EEEEEE"

    img = Image.new("RGB", (img_size_w, img_size_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        # Use a basic, widely available font. Fallback to default if needed.
        try:
            # Try a common truetype font
            font = ImageFont.truetype("Arial.ttf", 10)
        except IOError:
            # Fallback to default bitmap font
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Draw headers
    draw.rectangle([0, 0, img_size_w, header_size], fill=header_bg)
    draw.rectangle([0, 0, header_size, img_size_h], fill=header_bg)
    for i in range(grid_size):
        # Column headers
        x_center = header_size + (i * cell_size) + (cell_size // 2)
        y_center = header_size // 2
        draw.text((x_center, y_center), str(i), fill=text_color, font=font, anchor="mm")
        # Row headers
        x_center = header_size // 2
        y_center = header_size + (i * cell_size) + (cell_size // 2)
        draw.text((x_center, y_center), str(i), fill=text_color, font=font, anchor="mm")

    # Draw grid lines and numbers
    for y in range(grid_size):
        for x in range(grid_size):
            x0 = header_size + (x * cell_size)
            y0 = header_size + (y * cell_size)
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            # Draw cell border
            draw.rectangle([x0, y0, x1, y1], outline=line_color)
            
            # Draw number
            if y < len(grid) and x < len(grid[y]):
                num_str = str(grid[y][x])
                draw.text((x0 + cell_size // 2, y0 + cell_size // 2), num_str, fill=text_color, font=font, anchor="mm")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()