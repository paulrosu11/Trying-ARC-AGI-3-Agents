# agents/templates/as66/agent.py
from __future__ import annotations
from typing import Any, Optional, Tuple, List
import json
import logging
import base64
import os 
from openai import OpenAI

from ..llm_agents import GuidedLLM, VisualGuidedLLM
from ...structs import FrameData, GameAction, GameState
from .downsample import downsample_4x4, generate_numeric_grid_image_bytes
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


# ----------------------------- 16×16 TEXT-ONLY AGENT (NOW MULTIMODAL) -----------------------------

class AS66GuidedAgent(GuidedLLM):
    """
    16×16 downsample agent. Observation + single tool call per turn.
    Now supports multiple input modes ('text_only', 'image_only', 'text_and_image')
    via the `input_mode` init parameter.
    """

    MAX_ACTIONS = 3000
    MODEL = "gpt-5"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 1000
    REASONING_EFFORT = "low"

    _transcript: str = ""
    _token_total: int = 0

    def __init__(self, *args: Any, input_mode: str = "text_only", **kwargs: Any) -> None:
        """
        Initialize the agent with a specific input mode.
        :param input_mode: 'text_only' (default), 'image_only', or 'text_and_image'.
        """
        # --- FIX: Set input_mode FIRST ---
        # This ensures self.name property works correctly when super().__init__ calls it (via _open_transcript)
        self.input_mode = input_mode
        if self.input_mode not in ["text_only", "image_only", "text_and_image"]:
            log.warning(f"Invalid input_mode '{self.input_mode}', defaulting to 'text_only'.")
            self.input_mode = "text_only"

        # Now call super().__init__
        super().__init__(*args, **kwargs)
        
        # Model selection can happen after super()
        if self.input_mode != "text_only":
            self.MODEL = "gpt-5" # Assuming gpt-5 is the multimodal model

    @property
    def name(self) -> str:
        # Use the name from the base LLM class
        base_name = super(GuidedLLM, self).name
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        
        # Append input_mode if not the default "text_only"
        mode_tag = ""
        # We need to check if the attribute exists first during initialization
        input_mode = getattr(self, "input_mode", "text_only")
        if input_mode != "text_only":
            mode_tag = f".{input_mode}"

        name = f"{base_name}.{sanitized_model_name}.{obs}{mode_tag}"
        if self.REASONING_EFFORT:
            name += f".{self.REASONING_EFFORT}"
        return name

    def _append(self, s: str) -> None:
        self._transcript = (self._transcript + s.rstrip() + "\n")[-8000:]
        # --- FIX: Also write to the transcript file ---
        self._tw(s.rstrip())

    def _build_user_content(self, ds16: List[List[int]], user_prompt_text: str) -> List[Dict[str, Any]]:
        """
        Builds the 'content' array for the API call based on self.input_mode.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]
        
        if self.input_mode in ["image_only", "text_and_image"]:
            # Generate the numeric grid image
            try:
                png_bytes = generate_numeric_grid_image_bytes(ds16)
                b64_image = base64.b64encode(png_bytes).decode('utf-8')
                data_url = f"data:image/png;base64,{b64_image}"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "low" # Use "low" detail as requested
                    }
                })
            except Exception as e:
                log.error(f"Failed to generate numeric grid image: {e}")
                # If image gen fails, just proceed with text
                pass
                
        return content

    def _observation(self, latest_frame: FrameData) -> tuple[str, int]:
        client = OpenAI()
        ds16 = downsample_4x4(latest_frame.frame, take_last_grid=True, round_to_int=True)
        sys_msg = build_observation_system_text(self.game_id)
        
        # Build prompt text and clarification based on mode
        include_text = self.input_mode in ["text_only", "text_and_image"]
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = "The board state is provided as an attached image of the 16x16 grid."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both a textual matrix and an attached image of the 16x16 grid."
        elif self.input_mode == "text_only":
            format_clarification = "The board state is provided as a textual matrix."
            
        user_msg_text = build_observation_user_text(
            ds16, latest_frame.score, len(self.frames), self.game_id,
            format_clarification=format_clarification,
            include_text_matrix=include_text
        )
        
        # Build the final user message content (text, or text+image)
        user_content = self._build_user_content(ds16, user_msg_text)
        
        # --- Log the API call to the transcript ---
        messages_to_log = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]
        self._log_api_call(
            kind="observation",
            model=self.MODEL,
            messages=messages_to_log
        )
        
        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=messages_to_log, # Use the same messages list
            reasoning_effort=self.REASONING_EFFORT,
        )
        
        # --- Log the API response to the transcript ---
        self._log_api_response(resp)

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
        
        # Build prompt text and clarification based on mode
        include_text = self.input_mode in ["text_only", "text_and_image"]
        format_clarification = ""
        if self.input_mode == "image_only":
            format_clarification = "The board state is provided as an attached image of the 16x16 grid."
        elif self.input_mode == "text_and_image":
            format_clarification = "The board state is provided as both a textual matrix and an attached image of the 16x16 grid."
        elif self.input_mode == "text_only":
            format_clarification = "The board state is provided as a textual matrix."
            
        user_msg_text = build_action_user_text(
            ds16, last_obs, self.game_id,
            format_clarification=format_clarification,
            include_text_matrix=include_text
        )

        # Build the final user message content (text, or text+image)
        user_content = self._build_user_content(ds16, user_msg_text)
        tools = self.build_tools()

        # --- Log the API call to the transcript ---
        messages_to_log = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content}
        ]
        self._log_api_call(
            kind="action",
            model=self.MODEL,
            messages=messages_to_log,
            tools=tools,
            tool_choice="required"
        )
        
        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=messages_to_log, # Use the same messages list
            tools=tools,
            tool_choice="required",
            reasoning_effort=self.REASONING_EFFORT,
        )
        
        # --- Log the API response to the transcript ---
        self._log_api_response(resp)
        
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
        # --- Log the previous step's result to the transcript ---
        # This logs the 'tool_response' message
        if len(frames) > 0: # Avoid logging on the very first frame
            previous_action = latest_frame.action_input
            if previous_action.id.name != "RESET" or len(frames) > 1: # Don't log the *evaluator's* initial RESET
                tool_call_id = getattr(self, "_latest_tool_call_id", "call_12345") # Get ID from base class
                func_name = previous_action.id.name
                func_resp = self.build_func_resp_prompt(latest_frame)
                
                message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": str(func_resp),
                }
                # Use _log_api_response format to log this "message"
                self._tw("\nassistant_message:") # Simulates an assistant message
                self._tw(json.dumps(message, ensure_ascii=False, indent=2))
        
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            log.info("▶️  RESET required (state=%s)", latest_frame.state.name)
            
            # --- Log the RESET seed message ---
            if len(frames) > 0: # Only log if it's not the very first action
                self._log_seed_messages([
                    {"role": "user", "content": "Game over or not started. Must reset."},
                    {"role": "assistant", "tool_calls": [{"id": "call_reset", "type": "function", "function": {"name": "RESET", "arguments": "{}"}}]}
                ])
                self._latest_tool_call_id = "call_reset" # Set this for the next loop
            
            return GameAction.RESET

        step = self.action_counter + 1
        log.info("──── Turn %d | Score=%d ────", step, latest_frame.score)

        # Observation
        obs_text, obs_tokens = self._observation(latest_frame)
        self._token_total += obs_tokens
        self._append("Observation agent message:\n" + obs_text + "\n") # _append now writes to transcript
        log.info("OBS (%d tok, total %d):\n%s", obs_tokens, self._token_total, obs_text)

        # Action (+ click mapping)
        act, act_tokens, mapping_note = self._action(latest_frame, obs_text)
        self._token_total += act_tokens
        # We don't call _append for the action, as _log_api_response already wrote the tool call
        log.info("ACT  (%d tok, total %d): %s", act_tokens, self._token_total, act.name)
        if mapping_note:
            log.info("ACT  (click-mapping): %s", mapping_note)

        # Attach reasoning
        act.reasoning = {
            "agent": "as66guidedagent",
            "model": self.MODEL,
            "input_mode": self.input_mode, # Add the mode to reasoning
            "reasoning_effort": self.REASONING_EFFORT,
            "tokens_this_turn": {"observation": obs_tokens, "action": act_tokens},
            "tokens_total": self._token_total,
            "transcript_tail": self._transcript[-2000:],
        }
        if mapping_note:
            act.reasoning["click_mapping"] = mapping_note

        return act

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        """Override cleanup to log the correct data."""
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                # Recorder logic from base LLM
                meta = {
                    "llm_user_prompt": "N/A (AS66GuidedAgent uses custom prompt logic)",
                    "llm_tools": self.build_tools(),
                    "llm_tool_resp_prompt": "N/A (AS66GuidedAgent uses custom prompt logic)",
                }
                self.recorder.record(meta)
            
            # --- FIX: Log self._transcript instead of self.messages ---
            self._tw("\n=== FINAL CONTEXT (end of run) ===")
            self._tw(self._transcript)
            # --------------------------------------------------------
            
            # close transcript gracefully if open
            self._close_transcript()
            
            # Call the *Agent* (grandparent) cleanup, skipping LLM's
            super(GuidedLLM, self).cleanup(*args, **kwargs)


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
        # --- FIX: Also write to the transcript file ---
        self._tw(s.rstrip())

    def build_game_context_prompt(self) -> str:
        return build_visual_context_header()

    def _observation(self, latest_frame: FrameData) -> tuple[str, int]:
        client = OpenAI()
        sys_msg = build_observation_system_visual()
        user_msg = build_observation_user_visual(latest_frame.state.name, latest_frame.score)
        
        # --- Log the API call to the transcript ---
        messages_to_log = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        self._log_api_call(
            kind="observation",
            model=self.MODEL,
            messages=messages_to_log
        )

        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=messages_to_log,
            reasoning_effort=self.REASONING_EFFORT,
        )
        
        # --- Log the API response to the transcript ---
        self._log_api_response(resp)

        text = (resp.choices[0].message.content or "").strip()
        if resp.choices[0].message.tool_calls:
            text = "(observation only; tool call suppressed)"
        used = getattr(resp.usage, "total_tokens", 0) or 0
        return text, used

    def _action(self, latest_frame: FrameData, last_obs: str) -> tuple[GameAction, int, Optional[str]]:
        client = OpenAI()
        sys_msg = build_action_system_visual()
        user_msg = build_action_user_visual()
        tools = self.build_tools()

        # --- Log the API call to the transcript ---
        messages_to_log = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        self._log_api_call(
            kind="action",
            model=self.MODEL,
            messages=messages_to_log,
            tools=tools,
            tool_choice="required"
        )
        
        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=messages_to_log,
            tools=tools,
            tool_choice="required",
            reasoning_effort=self.REASONING_EFFORT,
        )
        
        # --- Log the API response to the transcript ---
        self._log_api_response(resp)

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
        # --- Log the previous step's result to the transcript ---
        if len(frames) > 0:
            previous_action = latest_frame.action_input
            if previous_action.id.name != "RESET" or len(frames) > 1:
                tool_call_id = getattr(self, "_latest_tool_call_id", "call_12345")
                func_name = previous_action.id.name
                func_resp = f"State: {latest_frame.state.name} | Score: {latest_frame.score}"
                
                message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": str(func_resp),
                }
                self._tw("\nassistant_message:")
                self._tw(json.dumps(message, ensure_ascii=False, indent=2))

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            log.info("▶️  RESET required (state=%s)", latest_frame.state.name)
            
            # --- Log the RESET seed message ---
            if len(frames) > 0:
                self._log_seed_messages([
                    {"role": "user", "content": "Game over or not started. Must reset."},
                    {"role": "assistant", "tool_calls": [{"id": "call_reset", "type": "function", "function": {"name": "RESET", "arguments": "{}"}}]}
                ])
                self._latest_tool_call_id = "call_reset"
                
            return GameAction.RESET

        step = self.action_counter + 1
        log.info("──── Turn %d | Score=%d ────", step, latest_frame.score)

        obs_text, obs_tokens = self._observation(latest_frame)
        self._token_total += obs_tokens
        self._append("Observation agent message (visual):\n" + obs_text + "\n") # _append now writes to transcript
        log.info("OBS (%d tok, total %d):\n%s", obs_tokens, self._token_total, obs_text)

        act, act_tokens, mapping_note = self._action(latest_frame, obs_text)
        self._token_total += act_tokens
        # We don't call _append for the action, as _log_api_response already wrote the tool call
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

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        """Override cleanup to log the correct data."""
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                # Recorder logic from base LLM
                meta = {
                    "llm_user_prompt": "N/A (AS66VisualGuidedAgent uses custom prompt logic)",
                    "llm_tools": self.build_tools(),
                    "llm_tool_resp_prompt": "N/A (AS66VisualGuidedAgent uses custom prompt logic)",
                }
                self.recorder.record(meta)
            
            # --- FIX: Log self._transcript instead of self.messages ---
            self._tw("\n=== FINAL CONTEXT (end of run) ===")
            self._tw(self._transcript)
            # --------------------------------------------------------
            
            # close transcript gracefully if open
            self._close_transcript()
            
            # Call the *Agent* (grandparent) cleanup, skipping LLM's
            super(GuidedLLM, self).cleanup(*args, **kwargs)

# --- Aliases for A/B/C testing ---

class AS66GuidedAgentImageOnly(AS66GuidedAgent):
    """Identical to AS66GuidedAgent but forces image-only input mode."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Force input_mode="image_only", pass other args through
        super().__init__(*args, input_mode="image_only", **kwargs)

class AS66GuidedAgentTextAndImage(AS66GuidedAgent):
    """Identical to AS66GuidedAgent but forces text+image input mode."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Force input_mode="text_and_image", pass other args through
        super().__init__(*args, input_mode="text_and_image", **kwargs)