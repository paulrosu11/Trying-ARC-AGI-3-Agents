"""
AS66 agent using the full 64×64 codes (no downsampling).

This is the ablation counterpart to the 16×16 text-only agent. The prompts are
intentionally parallel to the 16×16 version: same rules, goals, movement,
formatting, and required outputs — the only change is that we feed a 64×64
matrix (codes only) instead of a 16×16 downsample.

TEXT-ONLY CONTRACT (ablation fairness):
• Never mention or infer colors in text.
• Reason solely over integer codes and grid structure.
• Movement/goal semantics are identical to the 16×16 agent.
• Output schema identical: two lines, “Rationale:” then “ProposedMove:”.

Key semantic notes (codes domain; no colors):
- The raw board is 64×64. Each semantic tile occupies a 4×4 block.
- The movable character(s) appear as contiguous 4×4 (or unions of such) blocks
  with a distinct code compared to surrounding background (in some early levels
  this is often 8, but never assume — infer the “odd one out” structurally).
- Code 4 behaves as walls (impassable). Sliding stops adjacent to 4.
- Code 0 denotes the target region (e.g., a 2×3 arrangement containing a single
  non-zero “gap” you’re meant to occupy).
- Code 15 is background/play area.
- Codes 1 and 14 form fences/borders around the playable area.
- Code 6 functions as a moves/lives indicator: more 6s → fewer remaining moves.
- If multiple characters exist, all move together in the same direction.
- Movement is wrap-aware: in a chosen direction, the character slides as far as
  possible, wrapping across edges until blocked by walls (4), and must stop just
  before overlap with a wall.

The observation/user prompts below mirror the 16×16 agent, with “64×64” replacing
“16×16” and explicit mention that semantic tiles are 4×4 within the 64×64 grid.
"""

from __future__ import annotations
from typing import List

from ..llm_agents import GuidedLLM
from ...structs import FrameData


# ---------- prompt builders (64×64 codes-only; identical schema to 16×16) ----------

def _build_observation_system_text_64() -> str:
    return (
        "You will receive a 64×64 MATRIX of integer CODES in [0,15]. "
        "Reason ONLY in terms of codes and grid structure (no colors). "
        "Semantics: the board is 64×64 and each semantic tile is a 4×4 block. "
        "The movable character appears as a distinct contiguous block pattern (often 4×4 blocks), "
        "the 'odd one out' compared to background; do not assume a fixed code — infer it structurally. "
        "Code 4 = walls (impassable). Code 0 = goal region (e.g., a U-shaped cluster whose single non-zero gap should be filled). "
        "Code 15 = background. Codes 1 and 14 form borders/fences. Code 6 = moves/lives indicator (more 6s → fewer remaining moves). "
        "Movement rule: choose one of Up/Down/Left/Right; the character slides as far as possible in that direction, "
        "wrapping across edges, and must stop just before it would overlap code 4 (a wall). "
        "If multiple characters exist they all move together. "
        "Keep the rationale ≤ 60 tokens. Do not mention colors or anything outside of integer codes."
    )


def _build_observation_user_text_64(grid64: List[List[int]], score: int, step: int) -> str:
    rows = [" ".join(str(v) for v in r) for r in grid64]
    grid_txt = "\n".join(rows)
    return (
        f"Score: {score}\n"
        f"Step: {step}\n"
        "Matrix 64x64 (codes only):\n"
        f"{grid_txt}\n\n"
        "Respond with exactly two lines:\n"
        "Rationale: <≤60 tokens, codes-only; identify the odd contiguous character block; "
        "predict final landings for all directions under wrap+walls; state which 0-slot is being targeted>\n"
        "ProposedMove: <Up|Down|Left|Right>\n"
    )


def _build_action_system_text_64() -> str:
    return (
        "Select EXACTLY ONE move via a function call and nothing else. "
        "Available: ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right). "
        "Do not output prose; only the single tool/function call."
    )


def _build_action_user_text_64(grid64: List[List[int]], last_observation_text: str) -> str:
    rows = [" ".join(str(v) for v in r) for r in grid64]
    grid_txt = "\n".join(rows)
    return (
        "Choose the single best move as a function call.\n"
        "Matrix 64x64 (codes only):\n"
        f"{grid_txt}\n\n"
        "Previous observation summary (codes-only):\n"
        f"{last_observation_text}\n"
    )


# ---------- agent ----------

class AS66GuidedAgent64(GuidedLLM):
    """
    Text-only guided agent for AS66 that uses the full 64×64 matrix (codes only).

    This class mirrors the 16×16 downsampled agent’s behavior and output contract so that
    ablation is fair: identical rule framing, identical response format, identical tool set.
    The only change is the input matrix dimensionality (64×64 instead of 16×16).
    """

    MAX_ACTIONS = 80
    MODEL = "gpt-5"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 8
    REASONING_EFFORT = "low"

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        """
        Observation turn. We embed both the 'system' rules paragraph and the 'user' content
        into one string because the parent template sends this via the function/tool message.
        """
        grid = latest_frame.frame[-1] if latest_frame.frame else []
        return (
            "# OBSERVATION (codes only)\n"
            + _build_observation_system_text_64()
            + "\n\n"
            + _build_observation_user_text_64(grid, latest_frame.score, len(self.frames))
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """
        Action turn prompt. Same contract as the 16×16 agent, but on the 64×64 matrix.
        """
        grid = latest_frame.frame[-1] if latest_frame.frame else []
        return (
            "# ACTION (codes only)\n"
            + _build_action_system_text_64()
            + "\n\n"
            + _build_action_user_text_64(grid, "(previous observation above)")
        )
