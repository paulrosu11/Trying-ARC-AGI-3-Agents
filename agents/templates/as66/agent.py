# agents/templates/as66/agent.py
from __future__ import annotations
from typing import Any, Optional, Tuple
import json
import logging

from openai import OpenAI

from ..llm_agents import GuidedLLM, VisualGuidedLLM
from ...structs import FrameData, GameAction, GameState
from .downsample import downsample_4x4
from .prompts_text import (
    build_observation_system_text,
    build_observation_user_text,
    build_action_system_text,
    build_action_user_text,
)
from .prompts_visual import (
    build_visual_context_header,
    build_observation_system_visual,
    build_observation_user_visual,
    build_action_system_visual,
    build_action_user_visual,
)

log = logging.getLogger(__name__)


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, bool):
            return default
        return max(0, int(float(v)))
    except Exception:
        return default


def _grid_hw_from_frame(latest_frame: FrameData) -> Tuple[int, int]:
    """
    Return (H, W) of the topmost 2D grid in the latest frame.
    Falls back to (64, 64) if unknown.
    """
    try:
        grid = latest_frame.frame[-1]
        H = len(grid)
        W = len(grid[0]) if H else 0
        if H > 0 and W > 0:
            return (H, W)
    except Exception:
        pass
    return (64, 64)


def _map_click_to_source_xy(
    latest_frame: FrameData,
    x_in: Optional[int],
    y_in: Optional[int],
) -> Tuple[int, int, str]:
    """
    Map model-provided click coordinates to source-grid (H×W) coordinates.

    Rules:
      - If both x,y are in [0..15], interpret as 16×16 cell indices and
        click at the *center* of the corresponding block in the source grid.
      - Otherwise, treat as absolute source coordinates and clamp to [0..W-1], [0..H-1].

    Returns: (x64, y64, note)
    """
    H, W = _grid_hw_from_frame(latest_frame)
    x = 0 if x_in is None else _coerce_int(x_in, 0)
    y = 0 if y_in is None else _coerce_int(y_in, 0)

    # 16x16 cell → source center mapping (generalized to H×W)
    if 0 <= x <= 15 and 0 <= y <= 15:
        cell_w = max(1, W // 16)
        cell_h = max(1, H // 16)
        x64 = min(W - 1, x * cell_w + (cell_w // 2))
        y64 = min(H - 1, y * cell_h + (cell_h // 2))
        return (x64, y64, f"mapped 16×16 cell ({x},{y}) → source ({x64},{y64}) in {W}×{H}")

    # Absolute → clamp
    x64 = max(0, min(W - 1, x))
    y64 = max(0, min(H - 1, y))
    return (x64, y64, f"used absolute ({x},{y}) → clamped ({x64},{y64}) in {W}×{H}")


# ----------------------------- 16×16 TEXT-ONLY AGENT -----------------------------

class AS66GuidedAgent(GuidedLLM):
    """
    16×16 downsample (codes-only). Observation + single tool call per turn.
    Now supports ACTION6 (click), mapping 16×16 cell indices to the center of
    the corresponding source-grid block (H×W; typically 64×64).
    """

    MAX_ACTIONS = 3000
    MODEL = "gpt-5-nano"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 20
    REASONING_EFFORT = "low"

    _transcript: str = ""
    _token_total: int = 0

    def _append(self, s: str) -> None:
        self._transcript = (self._transcript + s.rstrip() + "\n")[-8000:]

    def _observation(self, latest_frame: FrameData) -> tuple[str, int]:
        client = OpenAI()
        ds16 = downsample_4x4(latest_frame.frame, take_last_grid=True, round_to_int=True)
        sys_msg = build_observation_system_text(self.game_id)
        user_msg = build_observation_user_text(ds16, latest_frame.score, len(self.frames), self.game_id)
        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            reasoning_effort=self.REASONING_EFFORT,
        )
        text = (resp.choices[0].message.content or "").strip()
        if resp.choices[0].message.tool_calls:
            text = "(observation only; tool call suppressed)"
        for bad in ("{\"id\":\"ACTION", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"):
            if bad in text and "Rationale" not in text:
                text = "(observation only; action-like content suppressed)"
        used = getattr(resp.usage, "total_tokens", 0) or 0
        return text, used

    def _action(self, latest_frame: FrameData, last_obs: str) -> tuple[GameAction, int, Optional[str]]:
        """
        Return (action, token_used, mapping_note).
        mapping_note is a string describing ACTION6 mapping (if any) for reasoning logs.
        """
        client = OpenAI()
        ds16 = downsample_4x4(latest_frame.frame, take_last_grid=True, round_to_int=True)
        sys_msg = build_action_system_text(self.game_id)
        user_msg = build_action_user_text(ds16, last_obs, self.game_id)
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
            act = GameAction.ACTION5  # fallback "no-op / spacebar"
            act.reasoning = {"error": "model did not call a tool"}
            return act, used, None

        tc = m.tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}

        # Build action
        act = GameAction.from_name(name)

        mapping_note: Optional[str] = None
        if act is GameAction.ACTION6:
            # Accept 16×16 indices or absolute source coords; map as needed.
            x_in = args.get("x", args.get("cx"))
            y_in = args.get("y", args.get("cy"))
            x64, y64, mapping_note = _map_click_to_source_xy(latest_frame, x_in, y_in)
            act.set_data({"x": x64, "y": y64})
        else:
            act.set_data(args)

        return act, used, mapping_note

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            log.info("▶️  RESET required (state=%s)", latest_frame.state.name)
            return GameAction.RESET

        step = self.action_counter + 1
        log.info("──── Turn %d | Score=%d ────", step, latest_frame.score)

        # Observation
        obs_text, obs_tokens = self._observation(latest_frame)
        self._token_total += obs_tokens
        self._append("Observation agent message:\n" + obs_text + "\n")
        log.info("OBS (%d tok, total %d):\n%s", obs_tokens, self._token_total, obs_text)

        # Action (+ click mapping)
        act, act_tokens, mapping_note = self._action(latest_frame, obs_text)
        self._token_total += act_tokens
        self._append(f"Action agent choice: {act.name}\n")
        log.info("ACT  (%d tok, total %d): %s", act_tokens, self._token_total, act.name)
        if mapping_note:
            log.info("ACT  (click-mapping): %s", mapping_note)

        # Attach reasoning
        act.reasoning = {
            "agent": "as66guidedagent",
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "tokens_this_turn": {"observation": obs_tokens, "action": act_tokens},
            "tokens_total": self._token_total,
            "transcript_tail": self._transcript[-2000:],
        }
        if mapping_note:
            act.reasoning["click_mapping"] = mapping_note

        return act


# ----------------------------- VISUAL GUIDED AGENT (click mapping too) -----------------------------

class AS66VisualGuidedAgent(VisualGuidedLLM):
    """
    Visual guided agent (colors-only prompts). Supports ACTION6 (click) with the same
    16×16→H×W center mapping as the text agent.
    """

    MAX_ACTIONS = 80
    MODEL = "gpt-5"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 14
    REASONING_EFFORT = "high"

    _transcript: str = ""
    _token_total: int = 0

    def _append(self, s: str) -> None:
        self._transcript = (self._transcript + s.rstrip() + "\n")[-8000:]

    def build_game_context_prompt(self) -> str:
        return build_visual_context_header()

    def _observation(self, latest_frame: FrameData) -> tuple[str, int]:
        client = OpenAI()
        sys_msg = build_observation_system_visual()
        user_msg = build_observation_user_visual(latest_frame.state.name, latest_frame.score)
        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            reasoning_effort=self.REASONING_EFFORT,
        )
        text = (resp.choices[0].message.content or "").strip()
        if resp.choices[0].message.tool_calls:
            text = "(observation only; tool call suppressed)"
        used = getattr(resp.usage, "total_tokens", 0) or 0
        return text, used

    def _action(self, latest_frame: FrameData, last_obs: str) -> tuple[GameAction, int, Optional[str]]:
        client = OpenAI()
        sys_msg = build_action_system_visual()
        user_msg = build_action_user_visual()
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
            return act, used, None

        tc = m.tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}

        act = GameAction.from_name(name)
        mapping_note: Optional[str] = None
        if act is GameAction.ACTION6:
            x_in = args.get("x", args.get("cx"))
            y_in = args.get("y", args.get("cy"))
            x64, y64, mapping_note = _map_click_to_source_xy(latest_frame, x_in, y_in)
            act.set_data({"x": x64, "y": y64})
        else:
            act.set_data(args)
        return act, used, mapping_note

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            log.info("▶️  RESET required (state=%s)", latest_frame.state.name)
            return GameAction.RESET

        step = self.action_counter + 1
        log.info("──── Turn %d | Score=%d ────", step, latest_frame.score)

        obs_text, obs_tokens = self._observation(latest_frame)
        self._token_total += obs_tokens
        self._append("Observation agent message (visual):\n" + obs_text + "\n")
        log.info("OBS (%d tok, total %d):\n%s", obs_tokens, self._token_total, obs_text)

        act, act_tokens, mapping_note = self._action(latest_frame, obs_text)
        self._token_total += act_tokens
        self._append(f"Action agent choice: {act.name}\n")
        log.info("ACT  (%d tok, total %d): %s", act_tokens, self._token_total, act.name)
        if mapping_note:
            log.info("ACT  (click-mapping): %s", mapping_note)

        act.reasoning = {
            "agent": "as66visualguidedagent",
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "tokens_this_turn": {"observation": obs_tokens, "action": act_tokens},
            "tokens_total": self._token_total,
            "transcript_tail": self._transcript[-2000:],
        }
        if mapping_note:
            act.reasoning["click_mapping"] = mapping_note
        return act
