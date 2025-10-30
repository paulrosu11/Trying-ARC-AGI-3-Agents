# agents/templates/as66/prompts_text.py
from __future__ import annotations
from typing import List, Optional, Dict
import os

# Optionally use the helper to render the 16x16 neatly. If you prefer the inline join below, you can remove this import.
from .downsample import matrix16_to_lines

"""
Multi-game prompt selector with a runtime switch:

- Two "modes":
    1) DETAILED (per-game): uses your current AS66 prompt (unchanged) and filler stubs for LS20, FT09, VC33, LP85, SP80.
    2) GENERAL: a self-learning prompt usable for any game; teaches the model to infer rules and also documents ACTION6 (click).

- How to choose the mode:
    • In code: flip USE_GENERAL_PROMPTS_DEFAULT = True / False (below).
    • Or at runtime: set env ARCGAME_GENERAL_PROMPTS=1 (overrides the default).

- Backward compatible function signatures:
    build_observation_system_text(game_id=None, use_general=None)
    build_observation_user_text(ds16, score, step, game_id=None, use_general=None)
    build_action_system_text(game_id=None, use_general=None)
    build_action_user_text(ds16, last_obs, game_id=None, use_general=None)
"""

# -------------------------
# Switch: set in code (default), or override via env var
# -------------------------
USE_GENERAL_PROMPTS_DEFAULT: bool = False # ← flip to True in code to force GENERAL prompts

def _use_general(use_general: Optional[bool]) -> bool:
    if use_general is not None:
        return bool(use_general)
    env = os.getenv("ARCGAME_GENERAL_PROMPTS", "").strip().lower()
    env_flag = env in ("1", "true", "yes", "on")
    return env_flag or USE_GENERAL_PROMPTS_DEFAULT


# -------------------------
# DETAILED (per-game) packs
# -------------------------

# AS66 — your existing, fully fleshed-out observation text (unchanged)
_AS66_OBS_SYSTEM = (
    "You are playing a game which is represented by a 16×16 matrix of integer codes. "
    "This matrix is the current state after a reset or after your most recent move. "
    "Your task is to observe the position and analyze the potential benefits and drawbacks of each possible move.\n\n"
    "Movement model:\n"
    "• There is one main movable integer. Its specific value may vary across levels, but it will be unique among the surrounding integers (it could even be a small square of 8/9s). "
    "  There may also be multiple movable integers; they remain distinct from the rest of the board and move together under the same command.\n"
    "• When you choose a direction (Up, Down, Left, Right), each movable integer slides as far as possible in that direction. "
    "  Sliding wraps across the board edges when unobstructed (after passing beyond one edge, it reappears from the opposite edge and continues). "
    "  Sliding stops as soon as an obstacle blocks further motion.\n"
    "If you move in a direction with no obstacles ever, you move back to where you started. For example, if one moves up and the entire wraparound has no obstacles, the game state does not change due to your action and you remain stationary.\n \n"
    "Obstacles and board semantics:\n"
    "• 4 are walls. Sliding stops adjacent to a 4 and cannot overlap 4s.\n"
    "• 15 are background cells that constitute the playable area (free to traverse and occupy).\n"
    "• The board is typically delimited by boundaries such as 1 or 14. You can generally localize the playable field by the region filled with 15s.\n"
    "• 0 are target cells. They form a U shape that can be viewed as a 2×3 rectangle with the center cell removed. "
    "  Your objective is to navigate the movable integer(s) into this space to complete the U by filling its cavity. "
    "  The 0 region also interrupts sliding (you stop upon reaching it when the motion would otherwise continue).\n"
    "• You may observe an increasing number of 6 near the top; this indicates the consumption of available moves in this downsampled 16×16 view.\n\n"
    "Hostile entity (avoid at all costs but note frequently there is not one. They are large 3 by 3 and easy to spot):\n"
    "• Larger composite blocks consisting of 8 and 9 indicate an enemy. If any movable integer collides with this enemy, it is game over.\n"
    "• The position of 9 within that 8/9 block indicates the direction in which this enemy will step per your move (one tile, row, or column, potentially diagonally depending on level behavior). "
    "  If the 9 is centered and the entity is stationary, it remains stationary. Use history to infer whether it moves.\n\n"
    "Multiple movers:\n"
    "• If multiple movable integers exist, they all move together in the same chosen direction. Avoid any collision with the hostile entity while advancing the objective of completing the U.\n\n"
    "Target matching with multiple movers:\n"
    "• On later levels, there may be multiple target U regions that expect specific movers. The 0 U may include a single cell of the intended mover’s code to indicate which mover should fill that particular U.\n\n"
    "What to produce during observation (rationale only):\n"
    "• Identify the locations of the movable integer(s) and all relevant structures (0 region, 4 walls, 15 background, 1/14 boundaries, any 8/9 enemy mass). "
    "• For each direction (Up, Down, Left, Right), reason carefully about full wrap-around sliding: what blocking elements will be met, what will be the final resting locations, and how these outcomes change proximity/alignment to the U cavity. "
    "• Consider the enemy’s response (8/9), including whether a move would cause immediate collision or a forced collision on the subsequent step. "
    "• Conclude which direction best progresses toward completing the 2×3 cavity in the 0 region while avoiding risk. "
    "This is a text-only analysis turn; do not name or call an action tool here."
    "**THE MOST IMPORTANT THING TO KEEP IN MIND IS THE RESULTS OF YOUR PAST ACTIONS AND PREVIOUSLY WHAT STATE CHANGE CAME FROM THEM, DO NOT REPREAT ACTIONS THAT CHANGED NOTHING!"
)

_AS66_OBS_USER_TMPL = (
    "Score: {score}\n"
    "Step: {step}\n"
    "{matrix_block}" # This will be either the matrix string or empty
    "Rationale:\n"
    "  • Identify the movable integer(s) and relevant structures (0/4/15/1/14 and any 8/9 enemy mass).\n"
    "  • For Up, Down, Left, Right: fully simulate wrap-around sliding, state blockers, and final landing positions.\n"
    "  • Explain how each landing affects progress toward completing the 2×3 U cavity (0 region) and whether the enemy’s response threatens collision.\n"
    "  • Conclude which direction is best and why. Do not output an action here.\n"
    "{format_clarification}" # NEW: Add clarification here
)

# Keep the original ACTION set for AS66 "detailed" to match your current wording
_AS66_ACT_SYSTEM = (
    "Select exactly one move by calling a single tool. Do not include prose.\n"
    "Available tools:\n"
    "• ACTION1 = Up\n"
    "• ACTION2 = Down\n"
    "• ACTION3 = Left\n"
    "• ACTION4 = Right"
)

_AS66_ACT_USER_TMPL = (
    "Choose the best single move as a function call.\n"
    "{matrix_block}" # This will be either the matrix string or empty
    "Previous observation summary:\n"
    "{last_obs}\n"
    "{format_clarification}" # NEW: Add clarification here
)

# Filler stubs for the other five games (you can paste real rules later)
# Each includes a brief name header and mirrors the AS66 structure.

_LS20_OBS_SYSTEM = (
    "LS20 — Placeholder observation rules.\n"
    "You are playing a 16×16 integer-code game state. Provide a rationale-only observation.\n"
    "• Identify controllable elements vs. walls/targets/hazards (LS20 semantics).\n"
    "• Simulate outcomes for Up/Down/Left/Right (LS20 motion rules; wrap/stop/merge if applicable).\n"
    "• Avoid tool calls in observation."
)
_FT09_OBS_SYSTEM = (
    "FT09 — Placeholder observation rules.\n"
    "Treat the 16×16 integers as ground truth for reasoning.\n"
    "• State FT09 objective/win condition (when known), and key structures.\n"
    "• Simulate Up/Down/Left/Right effects and blockers.\n"
    "• No tool calls now."
)
_VC33_OBS_SYSTEM = (
    "VC33 — Placeholder observation rules.\n"
    "• Hypothesize VC33 movement (slide/step/gravity? wrap?).\n"
    "• Mark targets, hazards, resources, and likely constraints.\n"
    "• Compare directions qualitatively; observation only."
)
_LP85_OBS_SYSTEM = (
    "LP85 — Placeholder observation rules.\n"
    "• Explain special tiles (LP85) and interactions.\n"
    "• Simulate legal outcomes for Up/Down/Left/Right.\n"
    "• Focus on next-step progress toward the goal; no tool calls."
)
_SP80_OBS_SYSTEM = (
    "SP80 — Placeholder observation rules.\n"
    "• Clarify objective/termination when visible.\n"
    "• Simulate direction outcomes and note immediate risks/opportunities.\n"
    "• Observation only; no tool calls."
)

GENERIC_STUB_OBS_USER_TMPL = (
    "Score: {score}\n"
    "Step: {step}\n"
    "{matrix_block}\n"
    "Rationale (tailored to this game):\n"
    "  • Identify controllable piece(s) and key structures.\n"
    "  • For Up/Down/Left/Right: describe blockers and final landings.\n"
    "  • Choose the best direction (no tool call here).\n"
    "{format_clarification}"
)

# For stubs, we expose the full action set (incl. click/space) since your code supports it.
STUB_ACT_SYSTEM = (
    "Select exactly one action (function call only; no prose).\n"
    "Available tools:\n"
    "• RESET\n"
    "• ACTION1 = Up\n"
    "• ACTION2 = Down\n"
    "• ACTION3 = Left\n"
    "• ACTION4 = Right\n"
    "• ACTION5 = Space/Enter/No-op\n"
    "• ACTION6 = Click(x,y)  [x,y may be 16×16 cell indices (0..15) or absolute source coords]"
)

STUB_ACT_USER_TMPL = (
    "Choose exactly one action.\n"
    "{matrix_block}\n"
    "Previous observation summary:\n"
    "{last_obs}\n"
    "{format_clarification}"
)

DETAILED_PACKS: Dict[str, Dict[str, str]] = {
    "AS66": {
        "obs_system": _AS66_OBS_SYSTEM,
        "obs_user_tmpl": _AS66_OBS_USER_TMPL,
        "act_system": _AS66_ACT_SYSTEM,
        "act_user_tmpl": _AS66_ACT_USER_TMPL,
    },
    "LS20": {
        "obs_system": _LS20_OBS_SYSTEM,
        "obs_user_tmpl": GENERIC_STUB_OBS_USER_TMPL,
        "act_system": STUB_ACT_SYSTEM,
        "act_user_tmpl": STUB_ACT_USER_TMPL,
    },
    "FT09": {
        "obs_system": _FT09_OBS_SYSTEM,
        "obs_user_tmpl": GENERIC_STUB_OBS_USER_TMPL,
        "act_system": STUB_ACT_SYSTEM,
        "act_user_tmpl": STUB_ACT_USER_TMPL,
    },
    "VC33": {
        "obs_system": _VC33_OBS_SYSTEM,
        "obs_user_tmpl": GENERIC_STUB_OBS_USER_TMPL,
        "act_system": STUB_ACT_SYSTEM,
        "act_user_tmpl": STUB_ACT_USER_TMPL,
    },
    "LP85": {
        "obs_system": _LP85_OBS_SYSTEM,
        "obs_user_tmpl": GENERIC_STUB_OBS_USER_TMPL,
        "act_system": STUB_ACT_SYSTEM,
        "act_user_tmpl": STUB_ACT_USER_TMPL,
    },
    "SP80": {
        "obs_system": _SP80_OBS_SYSTEM,
        "obs_user_tmpl": GENERIC_STUB_OBS_USER_TMPL,
        "act_system": STUB_ACT_SYSTEM,
        "act_user_tmpl": STUB_ACT_USER_TMPL,
    },
}

# Map the game_id prefix (before '-') to the pack key above
GAME_PREFIX_TO_PACK = {
    "as66": "AS66",
    "ls20": "LS20",
    "ft09": "FT09",
    "vc33": "VC33",
    "lp85": "LP85",
    "sp80": "SP80",
}


# -------------------------
# GENERAL (self-learning) pack
# -------------------------

_GENERAL_OBS_SYSTEM = (
    "GENERAL 16×16 GAME — Learn by observation.\n\n"
    "You are playing a tile-based game whose rules are unknown a priori. The board is a 16×16 matrix of integers.\n"
    "Your job is to (1) read the current state, (2) form careful hypotheses about the rules, and (3) propose an action\n"
    "plan that helps test those hypotheses. Do NOT assume a specific genre: the same inputs can mean different things\n"
    "across levels/games.\n\n"
    "Inputs the environment MAY support (semantics vary by game and level): Up, Down, Left, Right; clicking a cell; Space.\n"
    "Do not over-specify what these do—treat them as probes to learn the rules.\n\n"
    "Core approach:\n"
    "• Identify structures and roles: which integers might be movable entities, terrain/borders, goals/targets, counters,\n"
    "  hazards, keys/locks, or UI/score indicators. Note any center ‘arena’ vs. surrounding ‘context’ band.\n"
    "• Seek invariants and conserved quantities (e.g., counts of special integers, sums, connected components, symmetry).\n"
    "• Predict qualitative outcomes of candidate actions (including clicks/space): what is likely to change vs. remain\n"
    "  fixed; which cells may block or enable change; what could be illegal.\n"
    "• Compare with prior states, if any, to refine hypotheses: explicitly state what moved, what stayed the same, and why.\n"
    "• If an action yields no state change, record that it was illegal/no-op under current conditions and update beliefs.\n"
    "• Levels evolve: success increases score and may introduce new mechanics. Re-evaluate assumptions each level.\n\n"
    "Deliverables (no tool call in OBSERVATION):\n"
    "1) Key observations (salient features, suspected roles of notable integers, potential arena vs. perimeter).\n"
    "2) Hypotheses (concise, testable; include invariants and expected effects of inputs without asserting certainty).\n"
    "3) Change-tracking plan (how you will detect/measure differences after an action: moved cells, count deltas, etc.).\n"
    "4) One-sentence recommended next action (prose only) chosen to maximally reduce uncertainty about the rules."
    "    **THE MOST IMPORTANT THING TO KEEP IN MIND IS THE RESULTS OF YOUR PAST ACTIONS AND PREVIOUSLY WHAT STATE CHANGE CAME FROM THEM, DO NOT REPREAT ACTIONS THAT CHANGED NOTHING!"
    "    I repeat, do not reselect an action from the past if the state is the same. **try something new like clicking, or action 5, or moving in a different direction from before**"
    "    Please start with stating what changed from last time, explicitly noting if the state is identical, and recalling what moves caused changes and in what way in the past"
)

_GENERAL_OBS_USER_TMPL = (
    "Score: {score}\n"
    "Step: {step}\n"
    "{matrix_block}\n"
    "In your OBSERVATION (no tool call):\n"
    "  • Note controllable pieces and important structures.\n"
    "  • Hypothesize rules (slide/step, wrap/stop, merges, keys/locks, hazards, counters).\n"
    "  • For Up/Down/Left/Right (and possibly click/space), describe expected outcomes and blockers.\n"
    "  • Conclude the best action type now (one sentence; prose only).\n"
    "{format_clarification}"
)

_GENERAL_ACT_SYSTEM = (
    "ACTION PHASE — Call exactly ONE tool; no prose.\n"
    "Available tools:\n"
    "• RESET                  (restart when NOT_PLAYED or after GAME_OVER)\n"
    "• ACTION1 = Up           (directional move)\n"
    "• ACTION2 = Down         (directional move)\n"
    "• ACTION3 = Left         (directional move)\n"
    "• ACTION4 = Right        (directional move)\n"
    "• ACTION5 = Space/Enter/No-op (some games bind a key; otherwise may do nothing)\n"
    "• ACTION6 = Click(x,y)     (click at a coordinate)\n\n"
    "CLICK COORDS (ACTION6):\n"
    "  - Provide x,y as 16×16 cell indices (0..15) to click that int. "
)

_GENERAL_ACT_USER_TMPL = (
    "Choose and call exactly ONE tool.\n"
    "{matrix_block}\n"
    "Previous observation summary:\n"
    "{last_obs}\n"
    "If clicking (ACTION6), supply integers x,y (cell indices 0..15). No prose.\n"
    "{format_clarification}"
)

GENERAL_PACK = {
    "obs_system": _GENERAL_OBS_SYSTEM,
    "obs_user_tmpl": _GENERAL_OBS_USER_TMPL,
    "act_system": _GENERAL_ACT_SYSTEM,
    "act_user_tmpl": _GENERAL_ACT_USER_TMPL,
}


# -------------------------
# Pack selection helpers
# -------------------------

def _prefix_of(game_id: Optional[str]) -> str:
    if not game_id:
        return ""
    return (game_id.split("-", 1)[0] or "").lower().strip()

def _select_pack(game_id: Optional[str], use_general: Optional[bool]) -> Dict[str, str]:
    if _use_general(use_general):
        return GENERAL_PACK
    key = GAME_PREFIX_TO_PACK.get(_prefix_of(game_id), "AS66")
    return DETAILED_PACKS.get(key, DETAILED_PACKS["AS66"])


# -------------------------
# Public builders (backward-compatible)
# -------------------------

def build_observation_system_text(
    game_id: Optional[str] = None,
    use_general: Optional[bool] = None,
) -> str:
    return _select_pack(game_id, use_general)["obs_system"].strip()

def build_observation_user_text(
    ds16: List[List[int]],
    score: int,
    step: int,
    game_id: Optional[str] = None,
    use_general: Optional[bool] = None,
    # NEW KWARGS
    format_clarification: str = "",
    include_text_matrix: bool = True,
) -> str:
    if include_text_matrix:
        matrix = matrix16_to_lines(ds16)
        matrix_block = (
            "Matrix 16x16 (integer codes):\n"
            f"{matrix}\n"
        )
    else:
        matrix_block = "" # Omit the matrix block entirely
    
    tmpl = _select_pack(game_id, use_general)["obs_user_tmpl"]
    return tmpl.format(
        matrix_block=matrix_block,
        score=score,
        step=step,
        format_clarification=format_clarification.strip()
    ).strip()

def build_action_system_text(
    game_id: Optional[str] = None,
    use_general: Optional[bool] = None,
) -> str:
    return _select_pack(game_id, use_general)["act_system"].strip()

def build_action_user_text(
    ds16: List[List[int]],
    last_observation_text: str,
    game_id: Optional[str] = None,
    use_general: Optional[bool] = None,
    # NEW KWARGS
    format_clarification: str = "",
    include_text_matrix: bool = True,
) -> str:
    if include_text_matrix:
        matrix = matrix16_to_lines(ds16)
        matrix_block = (
            "Matrix 16x16 (integer codes):\n"
            f"{matrix}\n"
        )
    else:
        matrix_block = "" # Omit the matrix block entirely
    
    tmpl = _select_pack(game_id, use_general)["act_user_tmpl"]
    return tmpl.format(
        matrix_block=matrix_block,
        last_obs=last_observation_text,
        format_clarification=format_clarification.strip()
    ).strip()