from __future__ import annotations
from typing import List
from .prompts_text import build_observation_system_text

PRIMER_HEADER = """\
You are the AS66 game-playing agent. You will conduct a thorough OBSERVATION, then choose
exactly one move by calling a function tool (ACTION1..4). The environment will return the
result as a tool message, after which you continue to the next turn. Stay strictly in the
integer-codes domain (no color words). Use the 16×16 matrix as ground truth (it is a 4×4
average of the raw 64×64 board).

Conversation protocol (multi-turn):
1) system (this message)
2) user (provides starting state, score, step, matrix)
3) assistant (your OBSERVATION, then a function tool call: ACTION1..4)
4) tool (environment result: post-state, score, matrix)
5) assistant (next OBSERVATION + tool call), tool (...), and so on, until WIN or GAME_OVER.

OBSERVATION rules (multi-paragraph, codes-only):
- Identify the movable cluster(s) by structure vs. background (15), borders (1/14), walls (4), targets (0).
- For each direction Up/Down/Left/Right, reason through full sliding with wrap; stopping occurs adjacent to 4 or into the 0 region.
- Explain landing locations, risks (enemies 8/9 if present), and why your chosen move is best.
- Penalize repeating no-op directions observed earlier in this episode (if a move didn’t change state, try others).
- Keep the rationale focused on the current step; do not mention future moves explicitly.

IMPORTANT (imputed observation for training quality):
- You may use knowledge derived from the full trajectory to craft a better explanation,
  but phrase it as if you are at the current step (no explicit future references).
- End by emitting exactly one function call (ACTION1..4). Do not include prose after the call.
"""

def build_primer_system_text() -> str:
    """
    The unified SYSTEM primer for multi-turn SFT conversations.
    We reuse and extend the existing observation system prompt to keep semantics identical.
    """
    base = build_observation_system_text()
    return base + "\n\n" + PRIMER_HEADER

def build_user_step_text(ds16: List[List[int]], score: int, step: int, note: str | None = None) -> str:
    rows = [" ".join(str(v) for v in r) for r in ds16]
    grid_txt = "\n".join(rows) if rows else "(empty)"
    extra = f"\nNote: {note}" if note else ""
    return (
        f"Step: {step}\n"
        f"Score: {score}\n"
        "Matrix 16x16 (integer codes):\n"
        f"{grid_txt}\n"
        f"{extra}\n"
        "Respond with an OBSERVATION (multi-paragraph, codes-only) and then call exactly one function tool."
    )
