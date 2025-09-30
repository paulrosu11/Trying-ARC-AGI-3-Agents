from __future__ import annotations
from typing import List
import json
import logging

from openai import OpenAI

from ..llm_agents import GuidedLLM
from ...structs import FrameData, GameAction, GameState

log = logging.getLogger(__name__)


# ----------------------- 64×64 TEXT-ONLY PROMPTS (your spec) -----------------------

def _build_observation_system_text_64() -> str:
    return (
        "You are playing a game which is represented by a 64×64 matrix of integer codes. "
        "This matrix is the current state after a reset or after your most recent move. "
        "Your task is to observe the position and analyze the potential benefits and drawbacks of each possible move.\n\n"
        "Movement model:\n"
        "• There is one main movable square of integers. Its specific value may vary across levels, but it will be unique among the surrounding integers (it could even be a square of 2/6/8/9 among other ints). "
        "  There may also be multiple movable integers; they remain distinct from the rest of the board and move together under the same command.\n"
        "• When you choose a direction (Up, Down, Left, Right), each movable integer slides as far as possible in that direction. "
        "  Sliding wraps across the board edges when unobstructed (after passing beyond one edge, it reappears from the opposite edge and continues). "
        "  Sliding stops as soon as an obstacle blocks further motion.\n\n"
        "If you move in a direction with no obstacles ever, you move back to where you started. For example, if one moves up and the entire wraparound has no obstacles, the game state does not change due to your action and you remain stationary. \n\n"
        "Obstacles and board semantics:\n"
        "• 4 are walls. Sliding stops adjacent to a 4 and cannot overlap 4s.\n"
        "• 15 are background cells that constitute the playable area (free to traverse and occupy).\n"
        "• The board is typically delimited by boundaries such as 1 or 14. You can generally localize the playable field by the region filled with 15s.\n"
        "• 0 are target cells. They form a U shape that can be viewed as a larger 8×12 rectangle with the a center block removed. "
        "  Your objective is to navigate the movable integer(s) into this space to complete the U by filling its cavity. "
        "  The 0 region also interrupts sliding (you stop upon reaching it when the motion would otherwise continue).\n"
        "• You may observe a perimeter/track behavior using 8 and 9; this indicates the consumption of available moves, the more 9s you see, the less moves you have\n\n"
        "Hostile entity (avoid at all costs but note frequently there is not one. They are large 12 by 12 and easy to spot):\n"
        "• Larger composite blocks consisting of 8 and 9 indicate an enemy. If any movable integer collides with this enemy, it is game over.\n"
        "• The position of 9 within that 8/9 block indicates the direction in which this enemy will step per your move (one tile, row, or column, potentially diagonally depending on level behavior). "
        "  If the 9 is centered and the entity is stationary, it remains stationary. Use history to infer whether it moves.\n\n"
        "Multiple movers:\n"
        "• If multiple movable integers exist, they all move together in the same chosen direction. Avoid any collision with the hostile entity while advancing the objective of completing the U.\n\n"
        "Target matching with multiple movers:\n"
        "• On later levels, there may be multiple target U regions that expect specific movers. The 0 U may include a single block of the intended mover’s code to indicate which mover should fill that particular U.\n\n"
        "What to produce during observation (rationale only):\n"
        "• Identify the locations of the movable integer(s) and all relevant structures (0 region, 4 walls, 15 background, 1/14 boundaries, any 8/9 enemy mass). "
        "• For each direction (Up, Down, Left, Right), reason carefully about full wrap-around sliding: what blocking elements will be met, what will be the final resting locations, and how these outcomes change proximity/alignment to the U cavity. "
        "• Consider the enemy’s response (8/9), including whether a move would cause immediate collision or a forced collision on the subsequent step. "
        "• Conclude which direction best progresses toward completing the 8×12 cavity in the 0 region while avoiding risk. "
        "This is a text-only analysis turn; do not name or call an action tool here."
        "**THE MOST IMPORTANT THING TO KEEP IN MIND IS THE RESULTS OF YOUR PAST ACTIONS AND PREVIOUSLY WHAT STATE CHANGE CAME FROM THEM, DO NOT REPREAT ACTIONS THAT CHANGED NOTHING!"
    )


def _build_observation_user_text_64(grid64: List[List[int]], score: int, step: int) -> str:
    rows = [" ".join(str(v) for v in r) for r in grid64]
    grid_txt = "\n".join(rows)
    return (
        f"Score: {score}\n"
        f"Step: {step}\n"
        "Matrix 64x64 (integer codes):\n"
        f"{grid_txt}\n\n"
        "Rationale:\n"
        "  • Identify the movable integer(s) and relevant structures (0/4/15/1/14 and any 8/9 enemy mass).\n"
        "  • For Up, Down, Left, Right: fully simulate wrap-around sliding, state blockers, and final landing positions.\n"
        "  • Explain how each landing affects progress toward completing the 8×12 U cavity (0 region) and whether the enemy’s response threatens collision.\n"
        "  • Conclude which direction is best and why. Do not output an action here."
    )


def _build_action_system_text_64() -> str:
    return (
        "Select exactly one move by calling a single tool. Do not include prose.\n"
        "Available tools:\n"
        "• ACTION1 = Up\n"
        "• ACTION2 = Down\n"
        "• ACTION3 = Left\n"
        "• ACTION4 = Right"
    )


def _build_action_user_text_64(grid64: List[List[int]], last_observation_text: str) -> str:
    rows = [" ".join(str(v) for v in r) for r in grid64]
    grid_txt = "\n".join(rows)
    return (
        "Choose the best single move as a function call.\n"
        "Matrix 64x64 (integer codes):\n"
        f"{grid_txt}\n\n"
        "Previous observation summary:\n"
        f"{last_observation_text}\n"
    )


# ----------------------------- Agent with CLI prints & clean phases -----------------------------

class AS66GuidedAgent64(GuidedLLM):
    """
    64×64 ablation agent (text-only, codes-only) with:
      • Fresh messages per phase (no history bleed)
      • Observation: no tools (cannot act)
      • Action: tool_choice='required'
      • CLI prints for observation text, action, tokens, score, step, running totals
    """

    MAX_ACTIONS = 80
    MODEL = "gpt-5"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 14
    REASONING_EFFORT = "low"

    _transcript: str = ""
    _token_total: int = 0

    def _append(self, s: str) -> None:
        self._transcript = (self._transcript + s.rstrip() + "\n")[-8000:]

    def _observation(self, latest_frame: FrameData) -> tuple[str, int]:
        client = OpenAI()
        grid = latest_frame.frame[-1] if latest_frame.frame else []
        sys_msg = _build_observation_system_text_64()
        user_msg = _build_observation_user_text_64(grid, latest_frame.score, len(self.frames))

        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            reasoning_effort=self.REASONING_EFFORT,
        )
        text = (resp.choices[0].message.content or "").strip()
        if resp.choices[0].message.tool_calls:
            text = "(observation only; tool call suppressed)"
        for bad in ("{\"id\":\"ACTION", "ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if bad in text and "Rationale" not in text:
                text = "(observation only; action-like content suppressed)"
        used = getattr(resp.usage, "total_tokens", 0) or 0
        return text, used

    def _action(self, latest_frame: FrameData, last_obs: str) -> tuple[GameAction, int]:
        client = OpenAI()
        grid = latest_frame.frame[-1] if latest_frame.frame else []
        sys_msg = _build_action_system_text_64()
        user_msg = _build_action_user_text_64(grid, last_obs)

        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            tools=self.build_tools(),
            tool_choice="required",
            reasoning_effort=self.REASONING_EFFORT,
        )
        m = resp.choices[0].message
        used = getattr(resp.usage, "total_tokens", 0) or 0

        if not m.tool_calls:
            act = GameAction.ACTION5
            act.reasoning = {"error": "model did not call a tool"}
            return act, used

        tc = m.tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}
        act = GameAction.from_name(name)
        act.set_data(args)
        return act, used

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # RESET as needed
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            log.info("▶️  RESET required (state=%s)", latest_frame.state.name)
            return GameAction.RESET

        step = self.action_counter + 1
        log.info("──── Turn %d | Score=%d ────", step, latest_frame.score)

        # Observation (no tools)
        obs_text, obs_tokens = self._observation(latest_frame)
        self._token_total += obs_tokens
        self._append("Observation agent message:\n" + obs_text + "\n")
        log.info("OBS (%d tok, total %d):\n%s", obs_tokens, self._token_total, obs_text)

        # Action (required tool)
        act, act_tokens = self._action(latest_frame, obs_text)
        self._token_total += act_tokens
        self._append(f"Action agent choice: {act.name}\n")
        log.info("ACT  (%d tok, total %d): %s", act_tokens, self._token_total, act.name)

        # attach transcript
        act.reasoning = {
            "agent": "as66guidedagent64",
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "tokens_this_turn": {"observation": obs_tokens, "action": act_tokens},
            "tokens_total": self._token_total,
            "transcript_tail": self._transcript[-2000:],
        }
        return act
