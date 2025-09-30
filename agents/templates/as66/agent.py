from __future__ import annotations
from typing import Any
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


# ----------------------------- 16×16 TEXT-ONLY AGENT -----------------------------

class AS66GuidedAgent(GuidedLLM):
    """
    16×16 downsample (codes-only). Prints observation/action & tokens each turn.
    """

    MAX_ACTIONS = 80
    MODEL = "gpt-5-mini"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 8
    REASONING_EFFORT = "low"

    _transcript: str = ""
    _token_total: int = 0

    def _append(self, s: str) -> None:
        self._transcript = (self._transcript + s.rstrip() + "\n")[-8000:]

    def _observation(self, latest_frame: FrameData) -> tuple[str, int]:
        client = OpenAI()
        ds16 = downsample_4x4(latest_frame.frame, take_last_grid=True, round_to_int=True)
        sys_msg = build_observation_system_text()
        user_msg = build_observation_user_text(ds16, latest_frame.score, len(self.frames))
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
        ds16 = downsample_4x4(latest_frame.frame, take_last_grid=True, round_to_int=True)
        sys_msg = build_action_system_text()
        user_msg = build_action_user_text(ds16, last_obs)
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
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            log.info("▶️  RESET required (state=%s)", latest_frame.state.name)
            return GameAction.RESET

        step = self.action_counter + 1
        log.info("──── Turn %d | Score=%d ────", step, latest_frame.score)

        obs_text, obs_tokens = self._observation(latest_frame)
        self._token_total += obs_tokens
        self._append("Observation agent message:\n" + obs_text + "\n")
        log.info("OBS (%d tok, total %d):\n%s", obs_tokens, self._token_total, obs_text)

        act, act_tokens = self._action(latest_frame, obs_text)
        self._token_total += act_tokens
        self._append(f"Action agent choice: {act.name}\n")
        log.info("ACT  (%d tok, total %d): %s", act_tokens, self._token_total, act.name)

        act.reasoning = {
            "agent": "as66guidedagent",
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "tokens_this_turn": {"observation": obs_tokens, "action": act_tokens},
            "tokens_total": self._token_total,
            "transcript_tail": self._transcript[-2000:],
        }
        return act


# ----------------------------- VISUAL GUIDED AGENT -----------------------------

class AS66VisualGuidedAgent(VisualGuidedLLM):
    """
    Visual guided agent (colors-only prompts). Prints observation/action & tokens.
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

    def _action(self, latest_frame: FrameData, last_obs: str) -> tuple[GameAction, int]:
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
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            log.info("▶️  RESET required (state=%s)", latest_frame.state.name)
            return GameAction.RESET

        step = self.action_counter + 1
        log.info("──── Turn %d | Score=%d ────", step, latest_frame.score)

        obs_text, obs_tokens = self._observation(latest_frame)
        self._token_total += obs_tokens
        self._append("Observation agent message (visual):\n" + obs_text + "\n")
        log.info("OBS (%d tok, total %d):\n%s", obs_tokens, self._token_total, obs_text)

        act, act_tokens = self._action(latest_frame, obs_text)
        self._token_total += act_tokens
        self._append(f"Action agent choice: {act.name}\n")
        log.info("ACT  (%d tok, total %d): %s", act_tokens, self._token_total, act.name)

        act.reasoning = {
            "agent": "as66visualguidedagent",
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "tokens_this_turn": {"observation": obs_tokens, "action": act_tokens},
            "tokens_total": self._token_total,
            "transcript_tail": self._transcript[-2000:],
        }
        return act
