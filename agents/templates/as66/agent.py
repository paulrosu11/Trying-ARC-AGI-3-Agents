"""
AS66 agents:
- AS66GuidedAgent: text-only (codes) prompts with 16×16 downsample. No colors in text.
- AS66VisualGuidedAgent: visual prompts (colors only). No numbers in text.
"""

from __future__ import annotations
from ..llm_agents import GuidedLLM, VisualGuidedLLM
from ...structs import FrameData
from .downsample import downsample_4x4, matrix16_to_lines
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


class AS66GuidedAgent(GuidedLLM):
    """
    Text-only guided agent using 16×16 codes. Strictly numeric prompts.
    """
    MAX_ACTIONS = 80
    MODEL = "gpt-5"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 8
    REASONING_EFFORT = "low"

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        ds16 = downsample_4x4(latest_frame.frame, take_last_grid=True, round_to_int=True)
        return (
            "# ACTION (codes only)\n"
            + build_action_system_text()
            + "\n\n"
            + build_action_user_text(ds16, "(observation above)")
        )

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        ds16 = downsample_4x4(latest_frame.frame, take_last_grid=True, round_to_int=True)
        return (
            "# OBSERVATION (codes only)\n"
            + build_observation_system_text()
            + "\n\n"
            + build_observation_user_text(ds16, latest_frame.score, len(self.frames))
        )


class AS66VisualGuidedAgent(VisualGuidedLLM):
    """
    Visual guided agent. Prompts never mention numbers; use only color words.
    (Images are attached by the parent class; this class only supplies text scaffolding.)
    """
    MAX_ACTIONS = 80
    MODEL = "gpt-5"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 14
    REASONING_EFFORT = "high"

    def build_game_context_prompt(self) -> str:
        return build_visual_context_header()

    def build_observation_system_prompt(self) -> str:
        return build_observation_system_visual()

    def build_observation_user_text(self, latest_frame: FrameData) -> str:
        return build_observation_user_visual(latest_frame.state.name, latest_frame.score)

    def build_action_system_prompt(self) -> str:
        return build_action_system_visual()

    def build_action_user_prompt(self) -> str:
        return build_action_user_visual()
