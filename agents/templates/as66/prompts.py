# agents/templates/as66/prompts.py
"""
Prompt builders for the AS66 game.

This game has 64x64 frames where each semantic tile is a uniform 4x4 color block.
We pre-process by averaging each 4x4 block (robust to small edge differences) to
obtain a 16x16 “logical” board.

AS66 rules (from spec you provided):
- Movable character is a unique-colored 4x4 block (e.g., red in level 1).
- A move goes as FAR AS POSSIBLE in the chosen direction (wrap around edges).
- Black squares are walls and stop movement.
- The goal is to place the character on a white region (a "goal slot").
- Board edges are gray; green lines indicate prior move direction.
- Purple is the playable background area.
- Enemies: orange/red blobs with a "red eye" pointing toward the player, and they move
  one cell per player move toward that direction.
- There may be multiple characters with color-matching white goals; some tiles recolor
  the character (e.g., passing through a red area turns you red).
- Only 4 directions: up, down, left, right.

We keep color indices consistent with your repo’s VisualGuidedLLM palette.
"""

from __future__ import annotations
from typing import List

# Keep consistent with agents/templates/llm_agents.py VisualGuidedLLM.KEY_COLORS
INT_TO_COLOR = {
    0: "white",
    1: "light-gray",
    2: "gray",
    3: "dim-gray",
    4: "dark-gray",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "orange-red",
    9: "blue",
    10: "light-blue",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}


def matrix_to_color_text(ds16: List[List[int | float]]) -> str:
    """
    Render a compact color-name version of the down-sampled 16x16 board.
    """
    if not ds16:
        return "(no matrix)"
    lines = []
    for row in ds16:
        names = []
        for v in row:
            i = int(round(v)) if isinstance(v, float) else int(v)
            names.append(INT_TO_COLOR.get(i, f"#{i}"))
        lines.append("  " + " | ".join(names))
    return "16x16 colors (approx):\n" + "\n".join(lines)


def build_text_agent_prompt(ds16_str: str, ds16_color_str: str) -> str:
    """
    Text agent system/user prompt (single-phase) that gives AS66 rules and the 16x16 board.
    This is intended to be used by a tool-calling LLM (e.g., GPT‑5) with ACTION1..4.
    """
    return f"""
# AS66 — Rules & Objective (Text)
You are playing AS66. The raw 64×64 screen was down-sampled to a logical 16×16
grid by averaging each 4×4 block (robust to small edge differences). Treat the
16×16 grid as the ground-truth board.

## Movement
- You control a single 4×4 colored block (the player). It moves as FAR AS POSSIBLE
  in the chosen direction until blocked by BLACK walls. Movement WRAPS across edges:
  if there is no wall before the border, the player reappears on the opposite side
  and continues until a wall stops it.
- Legal moves: Up (ACTION1), Down (ACTION2), Left (ACTION3), Right (ACTION4).

## Terrain / Colors (approximate)
- black (5): walls (impassable blocks).
- white (0): goal slots / goal region. End your movement on the correct white slot to finish the level.
- purple (15): general background (playable).
- gray/dark-gray (2/4): borders/edges; green lines may show the last move direction.
- orange/red blobs with a red “eye”: enemies; after you move, they step one square toward the “eye” direction.
- Some areas recolor your character—match your character color to the color of the matching white goal if needed.

## Goal
- Navigate to place your character onto the correct white target cell/region while avoiding walls and enemies.
- Some levels require recoloring before reaching the white region that expects your color.

## Observations Requested
1) Identify the player's current 16×16 coordinates.
2) For each of Up/Down/Left/Right: what *final* stop square would result (wrap is allowed unless blocked by black)?
3) Do enemies put any directions at risk on the next step?
4) Decide the single best move that progresses toward the white goal (account for recoloring when necessary).

## 16×16 (numeric)
{ds16_str}

## 16×16 (colors, approximate)
{ds16_color_str}

# TURN
Call exactly one action tool: ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right).
Never call RESET unless you see GAME_OVER or NOT_PLAYED.
""".strip()


def build_visual_context_header() -> str:
    """
    The vision agent’s shared SYSTEM text (observation+action phases in VisualGuidedLLM style).
    """
    return """
You are playing AS66. You will receive an image of the current state plus a 16×16
down-sampled grid derived by 4×4 averaging from the 64×64 source.

Key rules:
- The player is a unique-colored 4×4 block that moves as far as possible in the chosen direction.
- Wrap-around at edges; walls are black and stop movement.
- The objective is to place the player onto the correct white goal region.
- Background is purple; borders/edges are gray; green marks indicate previous move direction.
- Enemies (orange/red with a "red eye") move one cell per player move toward the indicated direction.
- Only four moves: Up, Down, Left, Right.
""".strip()


def build_visual_observation_user_text(ds16_str: str, ds16_color_str: str, state_name: str, score: int) -> str:
    """
    Observation-phase user text. We attach the image outside this string. We also include the down-sampled matrix.
    """
    return f"""
# OBSERVATION (image attached)
State: {state_name} | Score: {score}

Use *only* visual evidence + the 16×16 matrix to:
- Locate the player (16×16 coords).
- For each of Up/Down/Left/Right, predict the *final* stop cell (with wrap if unblocked).
- Note any enemies that threaten the next landing squares.
- State whether recoloring is needed before entering the white goal region.
- End with a single sentence of intent (which direction and why). Do NOT call a tool.

## 16×16 (numeric)
{ds16_str}

## 16×16 (colors, approximate)
{ds16_color_str}
""".strip()


def build_visual_action_user_text() -> str:
    return "Choose exactly one tool (ACTION1..ACTION4). Favor the move that cleanly progresses toward placing the player onto the correct white region while avoiding walls/enemy contact."
