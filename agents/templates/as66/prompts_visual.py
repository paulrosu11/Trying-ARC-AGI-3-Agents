"""
Visual-only prompts for AS66 (with images). STRICTLY color/appearance words; never mention numbers.

Visual semantics:
- Black squares are walls (impassable).
- The goal is the white target region (e.g., a white “U” with a purple gap).
- Floor/background is purple.
- Borders/fence are gray; green lines can indicate last move direction.
- The uniquely colored 4×4 moving block(s) slide; if multiple exist, they move together.
- Movement slides to the farthest reachable cell until a black wall stops it, with wrap-around across edges.

Observation: 1–3 short sentences; colors only; pick best direction to approach/fill the target white gap.
"""

from __future__ import annotations


def build_visual_context_header() -> str:
    return (
        "You will receive an AS66 board image. Use only color/appearance words (no numbers). "
        "The uniquely colored moving block(s) slide until black walls stop them; wrap across edges is allowed. "
        "The objective is to place the block into the correct white target region. "
        "Borders are gray; background is purple."
    )


def build_observation_system_visual() -> str:
    return (
        "Observation phase (colors only). ≤ 60 tokens. "
        "Briefly: where the uniquely colored moving block(s) are, final landings for Up/Down/Left/Right "
        "with wrap + black-wall stopping, and which direction best approaches the white target."
    )


def build_observation_user_visual(state_name: str, score: int) -> str:
    return (
        f"State: {state_name} | Score: {score}\n"
        "Analyze the attached image (colors only; no numbers). End with one sentence of intended direction."
    )


def build_action_system_visual() -> str:
    return (
        "Action phase. Call exactly one tool: ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right). "
        "No prose; only the tool call."
    )


def build_action_user_visual() -> str:
    return "Choose the single best action now (colors-only reasoning already completed in observation)."
