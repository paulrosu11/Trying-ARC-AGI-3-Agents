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
from PIL import Image


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


# ---- tiny 16×16 visualization for vision/inspection (never used in text prompts) ----

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

def ds16_png_bytes(ds16: List[List[int]], cell: int = 22) -> bytes:
    h = len(ds16) or 16
    w = len(ds16[0]) if ds16 and ds16[0] else 16
    H, W = h * cell, w * cell
    im = Image.new("RGB", (W, H), (0, 0, 0))
    px = im.load()
    for y in range(h):
        row = ds16[y] if y < len(ds16) else [0] * w
        for x in range(w):
            code = row[x] if x < len(row) else 0
            rgb = _hex_to_rgb(KEY_COLORS.get(int(code) & 15, "#888888"))
            for dy in range(cell):
                for dx in range(cell):
                    px[x * cell + dx, y * cell + dy] = rgb
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()
