"""
Text-only prompts for AS66 (no images). STRICTLY numeric codes; no color words.

Semantic guide (codes), text mode ONLY:
- 4 = walls (impassable).
- 0 = goal region (e.g., a 2×3 “U” with one non-zero “hole” to fill).
- 8 = frequently the movable character in early examples; do not assume—identify the distinct
      contiguous “odd one out” block(s) that actually move.
- 15 = background play area.
- 1 and 14 = fence/border surrounding the play area.
- 6 = remaining-moves indicator (more 6s → fewer moves).

Movement (codes-only):
- Choose Up/Down/Left/Right; the moving block(s) slide as far as possible until stopped by 4s,
  wrapping across edges if unobstructed. Multiple characters (if present) move together.

Goal:
- Place the moving block into the intended 0-region (e.g., fill the purple gap inside a white U).

Observation output (≤ 60 tokens):
- 1–3 sentences: (a) identify the moving cluster vs. background, (b) predict final landing per direction
  with wrap+wall logic, (c) state which direction best advances toward the correct 0-slot.
"""

from __future__ import annotations
from typing import List


def build_observation_system_text() -> str:
    return (
        "You will receive a 16×16 matrix of integer CODES in [0,15]. "
        "Reason ONLY in codes and grid structure—no colors. "
        "Walls=4, goals=0, background=15, borders=1/14, moves indicator=6. "
        "The movable cluster is the distinct contiguous block(s) (often 8 in simple levels, but infer). "
        "Sliding movement with wrap: stop adjacent to a 4. Multiple movers (if any) move together. "
        "Keep rationale ≤ 60 tokens."
    )


def build_observation_user_text(ds16: List[List[int]], score: int, step: int) -> str:
    rows = [" ".join(str(v) for v in r) for r in ds16]
    grid_txt = "\n".join(rows)
    return (
        f"Score: {score}\n"
        f"Step: {step}\n"
        "Matrix 16x16 (codes only):\n"
        f"{grid_txt}\n\n"
        "Respond with two lines:\n"
        "Rationale: <≤60 tokens, codes-only, sliding/wrap reasoning, which 0-slot to target>\n"
        "ProposedMove: <Up|Down|Left|Right>\n"
    )


def build_action_system_text() -> str:
    return (
        "Select EXACTLY ONE move via a function call and nothing else. "
        "Available: ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right). "
        "Do not output text; only a single tool/function call."
    )


def build_action_user_text(ds16: List[List[int]], last_observation_text: str) -> str:
    rows = [" ".join(str(v) for v in r) for r in ds16]
    grid_txt = "\n".join(rows)
    return (
        "Choose the single best move as a function call.\n"
        "Matrix 16x16 (codes only):\n"
        f"{grid_txt}\n\n"
        "Previous observation summary:\n"
        f"{last_observation_text}\n"
    )
