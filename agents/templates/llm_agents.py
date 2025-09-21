import json
import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import numpy as np
import base64
import io
import hashlib   
from PIL import Image, ImageDraw


import openai
from openai import OpenAI as OpenAIClient

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger()


class LLM(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 80
    DO_OBSERVATION: bool = True
    REASONING_EFFORT: Optional[str] = None
    MODEL_REQUIRES_TOOLS: bool = False

    MESSAGE_LIMIT: int = 10
    MODEL: str = "gpt-4o-mini"
    messages: list[dict[str, Any] | Any]
    token_counter: int

    _latest_tool_call_id: str = "call_12345"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0
        # transcript plumbing (GuidedLLM enables it)
        self._transcript_enabled: bool = False
        self._transcript_file = None
        self._transcript_counter = 0
        self._transcript_path: Optional[Path] = None

    @property
    def name(self) -> str:
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        name = f"{super().name}.{sanitized_model_name}.{obs}"
        if self.REASONING_EFFORT:
            name += f".{self.REASONING_EFFORT}"
        return name

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment below to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

        functions = self.build_functions()
        tools = self.build_tools()

        # First prompt: seed reset
        if len(self.messages) == 0:
            user_prompt = self.build_user_prompt(latest_frame)
            message0 = {"role": "user", "content": user_prompt}
            self.push_message(message0)
            if self.MODEL_REQUIRES_TOOLS:
                message1 = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": self._latest_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": GameAction.RESET.name,
                                "arguments": json.dumps({}),
                            },
                        }
                    ],
                }
            else:
                message1 = {
                    "role": "assistant",
                    "function_call": {"name": "RESET", "arguments": json.dumps({})},  # type: ignore
                }
            self.push_message(message1)
            # log the exact seed messages the next API call will see
            self._log_seed_messages(self.messages)
            action = GameAction.RESET
            return action

        # Tool/function result back to the model as context
        function_name = latest_frame.action_input.id.name
        function_response = self.build_func_resp_prompt(latest_frame)
        if self.MODEL_REQUIRES_TOOLS:
            message2 = {
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": str(function_response),
            }
        else:
            message2 = {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        self.push_message(message2)

        # Optional observation turn
        if self.DO_OBSERVATION:
            logger.info("Sending to Assistant for observation...")
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self._coerce_messages_for_wire(self.messages),
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT

                # Log exactly what the model conditions on (messages only here)
                self._log_api_call(
                    kind="observation",
                    model=self.MODEL,
                    messages=self.messages,
                    tools=None,
                    functions=None,
                    tool_choice=None,
                )

                response = client.chat.completions.create(**create_kwargs)

                # Log exactly what the model returned as a message
                self._log_api_response(response)

            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e

            self.track_tokens(
                response.usage.total_tokens, response.choices[0].message.content
            )
            message3 = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            logger.info(f"Assistant: {response.choices[0].message.content}")
            self.push_message(message3)

        # Action turn (with tools/functions)
        user_prompt = self.build_user_prompt(latest_frame)
        message4 = {"role": "user", "content": user_prompt}
        self.push_message(message4)

        name = GameAction.ACTION5.name  # default action if LLM doesnt call one
        arguments = None
        message5 = None

        if self.MODEL_REQUIRES_TOOLS:
            logger.info("Sending to Assistant for action...")
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self._coerce_messages_for_wire(self.messages),
                    "tools": tools,
                    "tool_choice": "required",
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT

                # Log messages + tool schema (both are tokenized by the model)
                self._log_api_call(
                    kind="action",
                    model=self.MODEL,
                    messages=self.messages,
                    tools=tools,
                    functions=None,
                    tool_choice="required",
                )

                response = client.chat.completions.create(**create_kwargs)

                self._log_api_response(response)

            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e

            self.track_tokens(response.usage.total_tokens)
            message5 = response.choices[0].message
            logger.debug(f"... got response {message5}")
            tool_call = message5.tool_calls[0]
            self._latest_tool_call_id = tool_call.id
            logger.debug(
                f"Assistant: {tool_call.function.name} ({tool_call.id}) {tool_call.function.arguments}"
            )
            name = tool_call.function.name
            arguments = tool_call.function.arguments

            # sometimes the model will call multiple tools which isnt allowed
            extra_tools = message5.tool_calls[1:]
            for tc in extra_tools:
                logger.info(
                    "Error: assistant called more than one action, only using the first."
                )
                message_extra = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: assistant can only call one action (tool) at a time. default to only the first chosen action.",
                }
                self.push_message(message_extra)
        else:
            logger.info("Sending to Assistant for action...")
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self._coerce_messages_for_wire(self.messages),
                    "functions": functions,
                    "function_call": "auto",
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT

                # Log messages + function schema (both are tokenized by the model)
                self._log_api_call(
                    kind="action",
                    model=self.MODEL,
                    messages=self.messages,
                    tools=None,
                    functions=functions,
                    tool_choice=None,
                )

                response = client.chat.completions.create(**create_kwargs)

                self._log_api_response(response)

            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e

            self.track_tokens(response.usage.total_tokens)
            message5 = response.choices[0].message
            function_call = message5.function_call
            logger.debug(f"Assistant: {function_call.name} {function_call.arguments}")
            name = function_call.name
            arguments = function_call.arguments

        if message5:
            # Keep storing the SDK object as-is (don’t disrupt behavior)
            self.push_message(message5)

        action_id = name
        if arguments:
            try:
                data = json.loads(arguments) or {}
            except Exception as e:
                data = {}
                logger.warning(f"JSON parsing error on LLM function response: {e}")
        else:
            data = {}

        action = GameAction.from_name(action_id)
        action.set_data(data)
        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "assistant": message,
                }
            )
        logger.info(f"Received {tokens} tokens, new total {self.token_counter}")
        # handle tool to debug messages:
        # with open("messages.json", "w") as f:
        #     json.dump(
        #         [
        #             msg if isinstance(msg, dict) else msg.model_dump()
        #             for msg in self.messages
        #         ],
        #         f,
        #         indent=2,
        #     )

    def push_message(self, message: dict[str, Any] | Any) -> list[dict[str, Any] | Any]:
        """Push a message onto stack, store up to MESSAGE_LIMIT with FIFO."""
        self.messages.append(message)
        if len(self.messages) > self.MESSAGE_LIMIT:
            self.messages = self.messages[-self.MESSAGE_LIMIT :]
        if self.MODEL_REQUIRES_TOOLS:
            # cant clip the message list between tool
            # and tool_call else llm will error
            while True:
                first = self.messages[0]
                role = first.get("role") if isinstance(first, dict) else getattr(first, "role", None)
                if role == "tool":
                    self.messages.pop(0)
                else:
                    break
        return self.messages

    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions: list[dict[str, Any]] = [
            {
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# State:
{state}

# Score:
{score}

# Frame:
{latest_frame}

# TURN:
Reply with a several sentences/ paragraphs of plain-text strategy observation about the frame to inform your next action.
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.score,
                state=latest_frame.state.name,
            )
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                meta = {
                    "llm_user_prompt": self.build_user_prompt(self.frames[-1]),
                    "llm_tools": self.build_tools()
                    if self.MODEL_REQUIRES_TOOLS
                    else self.build_functions(),
                    "llm_tool_resp_prompt": self.build_func_resp_prompt(
                        self.frames[-1]
                    ),
                }
                self.recorder.record(meta)
            # Final snapshot of the exact context as JSON array
            self._log_final_context(self.messages)
        # close transcript gracefully if open
        self._close_transcript()
        super().cleanup(*args, **kwargs)

    # Transcript helpers (exact wire-style JSON)

    def _open_transcript(self) -> None:
        """Open a streaming transcript file if enabled."""
        if not self._transcript_enabled or self._transcript_file:
            return
        dir_path = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts"))
        dir_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        fname = f"{self.name}.{stamp}.transcript.txt"
        self._transcript_path = dir_path / fname
        self._transcript_file = open(self._transcript_path, "a", encoding="utf-8")
        self._tw(f"# transcript opened {stamp}Z")

    def _close_transcript(self) -> None:
        if self._transcript_file:
            self._tw("# end of transcript")
            try:
                self._transcript_file.flush()
                self._transcript_file.close()
            finally:
                self._transcript_file = None

    def _tw(self, s: str) -> None:
        """Write a line to transcript and flush."""
        if not self._transcript_file:
            return
        self._transcript_file.write(s if s.endswith("\n") else s + "\n")
        self._transcript_file.flush()

    def _coerce_message_to_dict(self, m: Any) -> dict[str, Any]:
        """Turn dict or ChatCompletionMessage into a plain dict for JSON logging / wire."""
        if isinstance(m, dict):
            return m
        # OpenAI SDK ChatCompletionMessage (or similar)
        out: dict[str, Any] = {}
        role = getattr(m, "role", None)
        if role:
            out["role"] = role
        # top-level content
        if hasattr(m, "content"):
            out["content"] = m.content
        # tool_calls
        tc_list = getattr(m, "tool_calls", None)
        if tc_list:
            tcs: list[dict[str, Any]] = []
            for tc in tc_list:
                d = {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": None,
                }
                fn = getattr(tc, "function", None)
                if fn is not None:
                    d["function"] = {
                        "name": getattr(fn, "name", None),
                        "arguments": getattr(fn, "arguments", None),
                    }
                tcs.append(d)
            out["tool_calls"] = tcs
        # function_call (deprecated format)
        fc = getattr(m, "function_call", None)
        if fc is not None:
            out["function_call"] = {
                "name": getattr(fc, "name", None),
                "arguments": getattr(fc, "arguments", None),
            }
        # name / tool_call_id if present on tool/function messages (rare on assistant objects)
        name = getattr(m, "name", None)
        if name:
            out["name"] = name
        tcid = getattr(m, "tool_call_id", None)
        if tcid:
            out["tool_call_id"] = tcid
        return out

    def _coerce_messages_for_wire(self, messages: list[dict[str, Any] | Any]) -> list[dict[str, Any]]:
        """Return a list of plain dict messages suitable for API call / exact logging."""
        return [self._coerce_message_to_dict(m) for m in messages]

    def _render_wire_json(self, messages: list[dict[str, Any] | Any]) -> str:
        """Render messages as exact JSON (what we send on the wire)."""
        try:
            coerced = self._coerce_messages_for_wire(messages)
            return json.dumps(coerced, ensure_ascii=False, indent=2)
        except Exception as e:
            return f'{{"error":"failed to render messages","detail":"{e}"}}'

    def _log_seed_messages(self, messages: list[dict[str, Any] | Any]) -> None:
        if not self._transcript_enabled:
            return
        self._open_transcript()
        self._tw("=== INIT / SEEDED CONTEXT ===")
        self._tw(self._render_wire_json(messages))

    def _log_api_call(
        self,
        kind: str,  # "observation" | "action"
        model: str,
        messages: list[dict[str, Any] | Any],
        tools: Optional[list[dict[str, Any]]] = None,
        functions: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> None:
        if not self._transcript_enabled:
            return
        self._open_transcript()
        self._transcript_counter += 1
        self._tw(f"\n=== PROMPT #{self._transcript_counter} [{kind}] model={model} ===")
        # Messages (exact)
        self._tw("messages:")
        self._tw(self._render_wire_json(messages))
        # Include tool/function schema verbatim (tokenized by the model)
        if tools:
            try:
                self._tw("tools:")
                self._tw(json.dumps(tools, ensure_ascii=False, indent=2))
            except Exception as e:
                self._tw(f'{{"error":"failed to render tools","detail":"{e}"}}')
        if functions:
            try:
                self._tw("functions:")
                self._tw(json.dumps(functions, ensure_ascii=False, indent=2))
            except Exception as e:
                self._tw(f'{{"error":"failed to render functions","detail":"{e}"}}')
        # Note: tool_choice is not part of the tokenized context; omit on purpose.

    def _log_api_response(self, response: Any) -> None:
        if not self._transcript_enabled:
            return
        try:
            msg = response.choices[0].message
            as_dict = self._coerce_message_to_dict(msg)
            self._tw("\nassistant_message:")
            self._tw(json.dumps(as_dict, ensure_ascii=False, indent=2))
        except Exception as e:
            self._tw(f'\nassistant_message: {{"error":"unavailable","detail":"{e}"}}')

    def _log_final_context(self, messages: list[dict[str, Any] | Any]) -> None:
        if not self._transcript_enabled:
            return
        self._open_transcript()
        self._tw("\n=== FINAL CONTEXT (end of run) ===")
        self._tw(self._render_wire_json(messages))


class ReasoningLLM(LLM, Agent):
    """An LLM agent that uses o4-mini and captures reasoning metadata in the action.reasoning field."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MODEL = "o4-mini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from reasoning models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For reasoning models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from {self.MODEL} response"
                )


class FastLLM(LLM, Agent):
    """Similar to LLM, but skips observations."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = False
    MODEL = "gpt-4o-mini"

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )


class GuidedLLM(LLM, Agent):
    """Similar to LLM, with explicit human-provided rules in the user prompt to increase success rate."""

    MAX_ACTIONS = 50
    DO_OBSERVATION = True
    MODEL = "gpt-5"  # switched from o3 to gpt-5
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 5
    REASONING_EFFORT = "high"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0
        # Always write a transcript for GuidedLLM
        self._transcript_enabled = True
        self._open_transcript()

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "agent_type": "guided_llm",
            "game_rules": "locksmith",
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from gpt-5 (reasoning) models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response."""
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from {self.MODEL} response"
                )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values. Each value represents a different color and usually implies a different object.

You are playing a game called LockSmith.

# CONTROLS
* RESET: start over
* ACTION1: move up
* ACTION2: move down
* ACTION3: move left
* ACTION4: move right
* ACTION5/ACTION6: no-ops **never do these as they do nothing!**

# ORIENTATION (read carefully)
* The global frame is fixed: the bottom-left KEY INDICATOR is always at the bottom-left; the EXIT/lock is elsewhere.
* ACTION1 (UP) moves you toward the area where 15s (move counters) are located; use this as a compass for “up.”
* ACTION4 (RIGHT) increases your distance from the bottom-left key indicator; ACTION3 (LEFT) decreases it.
* Legality checks with 4s (walls):
  - RIGHT is ILLEGAL if any 4s lie immediately to the right (“after” in row order) of any 12/9 cells of your 8×8 player.
  - LEFT is ILLEGAL if any 4s lie immediately to the left (“before” in row order) of any 12/9 cells of your 8×8 player.
  - UP is ILLEGAL if any 4s lie directly above the topmost 12 row of your player.
  - DOWN is ILLEGAL if any 4s lie directly below the bottommost 9 row of your player.
* Always orient carefully before moving: check where the 15s lie (for UP), how far you are from the bottom-left indicator (for LEFT/RIGHT), and whether 4s precede/follow your 12s/9s in the intended direction.

# GOAL
Find/generate a key that MATCHES the key pattern shown at the EXIT, then navigate to the EXIT to finish the level.
There are 6 levels total. Complete all levels to WIN.

# TILE MEANINGS (important ones)
* 0 = white cell (used in key patterns)
* 3 = walkable path / corridor marker (often outlines routes to key/exit)
* 4 = WALL (solid). You cannot enter 4s.
* 5 = EXIT indicator (presence of 5s marks/frames the exit area)!!! **avoid this area and anything which touches it till the end**
* 8 = LIFE token (count lives as the number of 2×2 blocks of 8s)
* 9 = blue cells (used in player lower body; also appear elsewhere in the map and in key logic)
* 12 = orange cells (used in player upper body)
* 15 = MOVE counter (number of 15s ≈ moves/energy left)

# PLAYER (sprite + movement legality)
* The player is an 8×8 sprite made ONLY of 12s (top two rows) over 9s (bottom six rows).
  - Concretely: rows 0–1 (relative to the sprite) are all 12s; rows 2–7 are all 9s.
  - Other scattered 9s on the board are NOT the player—only the exact 8×8 (2×8 of 12s stacked on 6×8 of 9s) is the player.
* Movement rules:
  - You may move only through non-4 cells (i.e., never overlap a 4 after a move).
  - If any part of your 8×8 sprite would overlap a 4 in the target direction (including cases where 4s lie “under” 9s), that move is ILLEGAL.
  - If your sprite is already touching 4s on a side, you cannot move further in that touching direction until you have clearance.
  - Practically: navigate corridors “between the 4s”.

# KEYS, EXIT, AND HOW TO FIND THEM
* Bottom-left KEY INDICATOR (non-interactive):
  - A small block of 0s and 9s COMPLETELY BOXED by 4s in the bottom-left of the entire grid.
  - This is NOT reachable and NOT a gameplay area. It only shows your CURRENT selected key.
* EXIT (the lock to open):
  - Elsewhere on the map, the lock is the target 0/9 pattern you must match, adjacent to some 5s (the 5s need not fully surround it).
  - You may leave the level only when your current key EXACTLY matches this lock’s 0/9 distribution. **I say distribution because there are more int values sometimes dedicated to one over the other but the SHAPE is what matters.
* KEY GENERATOR (interactive, how to generate keys):
  - Look for a patch consisting of 0s and 9s that has a CLEAR path of 3s leading to it and around it, and is NOT bordered by 5s or boxed by 4s.
  - When any part of your 8×8 player steps ONTO this patch, it GENERATES a NEW candidate key and updates the bottom-left indicator.
  - Step OFF the patch and then back ON to generate yet another candidate key; repeat to cycle through the set of keys.
  - Continue cycling until the bottom-left indicator EXACTLY matches the lock pattern at the exit (the same 0/9 layout adjacent to 5s).
* Locating things quickly:
  - Generator: 0/9 patch accessible via 3s, with only 3s around it (no 5s), is almost certainly the generator.
  - Exit: a matching 0/9 pattern near 5s marks the lock/exit area.

# HUD / PROGRESSION
* Visual score on the second-to-last grid row (row 62): it contains several 4-cell “slots”.
  - Slots made of 3s represent UNFINISHED levels.
  - When a slot changes from 3s to another color, that level is COMPLETE.
* Track remaining MOVES/ENERGY by counting tiles with value 15.
* Track LIVES by counting distinct 2×2 blocks of 8s.

# GENERAL NOTES
* The grid is a bird’s-eye view.
* If the grid does not change after an action, you likely attempted an illegal move into 4s.
* Favor routes marked by 3s; use the generator (0/9 patch surrounded by 3s) to cycle keys; confirm the match by comparing the bottom-left indicator (boxed by 4s) to the 0/9 lock pattern found near 5s; then proceed to the exit.

# EXAMPLE STRATEGY OBSERVATION
The 8×8 player (top rows 12s, bottom rows 9s) is touching 4s on the left, so LEFT is blocked. I see a 0/9 patch with only 3s around it to the right—that’s likely the generator. I should move RIGHT to step on/off it until the bottom-left indicator (boxed by 4s) matches the lock pattern I see near the 5s, then head to the exit.
**again if the key indicator EVER matches the exit, head DIRECTLY to the exit avoiding the key generator. COMPARE THE KEY INDICATOR AND THE EXIT EVERYTIME FIRST THING**
**Be smart and understand the positioning of the grids and where the 4s are**


**AGAIN ALWAYS ALWAYS FIRST BEFORE ALL ELSE CHECK IF THE WIN CONDITION OF THE MATCHING KEYS IS SATISFIED FIRST. DON'T UNDERESTIMATE THIS, FREQUENTLY THE DISTRIBUTIONS OF THE 0S AND 9S DO MATCH SO BE SMART**

Here's an example of a matching key and generator:


```
Grid 0:
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 8 8 4 4 8 8 4 4 8 8 4 4
  4 4 3 4 3 4 3 4 3 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 15 4 4 4 4 4 4 4 8 8 4 4 8 8 4 4 8 8 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 5 5 5 5 5 5 5 5 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 5 5 5 5 5 5 5 5 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 5 5 9 9 5 5 5 5 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 5 5 9 9 5 5 5 5 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 5 5 5 5 0 0 5 5 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 5 5 5 5 0 0 5 5 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 5 5 0 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 5 5 0 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 12 12 12 12 12 12 12 12 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 12 12 12 12 12 12 12 12 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 9 9 9 9 9 9 9 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 9 9 9 9 9 9 9 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 9 9 9 9 9 9 9 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 9 9 9 9 9 9 9 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 9 9 9 9 9 9 9 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 9 9 9 9 9 9 9 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 9 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 9 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 9 9 9 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 9 9 9 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4
  4 9 9 9 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 0 0 0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 0 0 0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 0 0 0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 0 0 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 0 0 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 0 0 0 0 0 0 0 0 0 4 4 3 3 3 3 4 3 3 3 3 4 3 3 3 3 4 3 3 3 3 4 3 3 3 3 4 3 3 3 3 4 3 3 3 3 4 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
```

Note how the 0s form a sort of tetris like T-block shape, both in the bottom left and the gate in the middle top. 

# TURN:
Call exactly one action.

        """.format()
        )


# Example of a custom LLM agent
class MyCustomLLM(LLM):
    """Template for creating your own custom LLM agent."""

    MAX_ACTIONS = 80
    MODEL = "gpt-"
    DO_OBSERVATION = True

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Customize this method to provide instructions to the LLM."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# CUSTOM INSTRUCTIONS:
Add your game instructions and strategy here.
For example, explain the game rules, objectives, and optimal strategies.

# TURN:
Call exactly one action.
        """.format()
        )
class VisualGuidedLLM(GuidedLLM, Agent):
    """
    Visual-only Guided LLM with phase-specific prompts:
      - Observation phase: paragraph analysis (no tool call)
      - Action phase: call exactly one tool
    Puts the observation directive into a SYSTEM message + repeats it in the USER
    text that accompanies the image (so it can't be ignored). Increases MESSAGE_LIMIT
    so the directive won't be clipped. Never sends the textual matrix; only a ≤64×64 PNG.
    Emits 'reasoning' into recorder so the UI shows a Reasoning Log.
    """

    # --- palette 0..15 ---
    KEY_COLORS = {
        0: "#FFFFFF", 1: "#CCCCCC", 2: "#999999",
        3: "#666666", 4: "#333333", 5: "#000000",
        6: "#E53AA3", 7: "#FF7BCC", 8: "#F93C31",
        9: "#1E93FF", 10: "#88D8F1", 11: "#FFDC00",
        12: "#FF851B", 13: "#921231", 14: "#4FCC30",
        15: "#A356D6",
    }

    # defaults (overridable via __init__)
    MODEL = "gpt-5"
    REASONING_EFFORT = "high"
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 14  # larger so phase/system directives don't get clipped

    def __init__(
        self,
        *args,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        do_observation: Optional[bool] = None,
        model_requires_tools: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # explicit kw override
        if model is not None:
            self.MODEL = model
        # env var fallback if kw not provided
        if model is None:
            import os
            env_model = (
                os.getenv("ARC_VLM_MODEL")
                or os.getenv("VLM_MODEL")
                or os.getenv("OPENAI_VLM_MODEL")
            )
            if env_model:
                self.MODEL = env_model
        if reasoning_effort is not None:
            self.REASONING_EFFORT = reasoning_effort
        if do_observation is not None:
            self.DO_OBSERVATION = do_observation
        if model_requires_tools is not None:
            self.MODEL_REQUIRES_TOOLS = model_requires_tools

        # state for image saving / de-dupe / logs
        self._last_digest: Optional[str] = None
        self._last_score: Optional[int] = None
        self._img_counter: int = 0
        self._last_obs_text: str = ""
        self._last_obs_digest: Optional[str] = None
        self._last_act_digest: Optional[str] = None

        # transcript + per-run images dir
        self._open_transcript()
        base_dir = Path("transcripts/images")
        run_stem = (self._transcript_path.stem if self._transcript_path else self.name)
        self._images_dir = base_dir / run_stem
        self._images_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- prompts (split by phase) -----------------

    def build_game_context_prompt(self) -> str:
        """Shared context fed to BOTH phases (no action directive)."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
colored tiles. Each color represents a different object or rule.

You are playing a game called LockSmith.

# CONTROLS
* RESET: start over
* ACTION1: move up
* ACTION2: move down
* ACTION3: move left
* ACTION4: move right
* ACTION5/ACTION6: no-ops

# ORIENTATION (read carefully)
* The global frame is fixed: the bottom-left KEY INDICATOR is always at the bottom-left; the EXIT/lock is elsewhere.
* ACTION1 (UP) moves you toward the lives and moves indicators and most importantly toward the upper wall.
* ACTION4 (RIGHT) increases your distance from the bottom-left key indicator; ACTION3 (LEFT) decreases it.
* Legality checks with dark gray walls:
  - RIGHT is ILLEGAL if any dark gray tiles lie immediately to the right (“after” in row order) of any orange/blue cells of your 8×8 player.
  - LEFT is ILLEGAL if any dark gray tiles lie immediately to the left (“before” in row order) of any orange/blue cells of your 8×8 player.
  - UP is ILLEGAL if any dark gray tiles lie directly above the topmost orange row of your player.
  - DOWN is ILLEGAL if any dark gray tiles lie directly below the bottommost blue row of your player.
* Always orient carefully before moving: check where the purple tiles lie (for UP), how far you are from the bottom-left indicator (for LEFT/RIGHT), and whether dark gray walls precede/follow your orange/blue rows in the intended direction.

# GOAL
Find/generate a key that MATCHES the key pattern shown at the EXIT, then navigate to the EXIT to finish the level.
There are 6 levels total. Complete all levels to WIN.

# TILE MEANINGS (important ones)
* white = key pattern cells
* light gray = walkable path / corridor marker (often outlines routes to key/exit)
* dark gray = WALL (solid). You cannot enter dark gray.
* pure black = EXIT indicator (presence of pure black tiles marks/frames the exit area)
* red = LIFE token (count lives as the number of 2×2 blocks of red)
* blue = player lower-body color and also used in key patterns
* orange = player upper-body color
* purple = MOVE/energy counter (number of purple tiles ≈ moves/energy left)

# PLAYER (sprite + movement legality)
* The player is an 8×8 sprite: the top two rows are all orange; the bottom six rows are all blue.
  - Other scattered blue tiles on the board are NOT the player—only the exact 8×8 (2×8 of orange stacked on 6×8 of blue) is the player.
* Movement rules:
  - You may move only through non–dark gray tiles (i.e., never overlap a dark gray wall after a move).
  - If any part of your 8×8 sprite would overlap dark gray in the target direction (including cases where dark gray lies “under” blue), that move is ILLEGAL.
  - If your sprite is already touching dark gray on a side, you cannot move further in that touching direction until you have clearance.
  - Practically: navigate corridors “between the dark gray walls.”

# KEYS, EXIT, AND HOW TO FIND THEM
* Bottom-left KEY INDICATOR (non-interactive):
  - A small block of white and blue COMPLETELY BOXED by dark gray in the bottom-left of the entire grid.
  - This is NOT reachable and NOT a gameplay area. It only shows your CURRENT selected key.
* EXIT (the lock to open):
  - Elsewhere on the map, the lock is the target white/blue pattern you must match, adjacent to some pure black tiles (they need not fully surround it).
  - You may leave the level only when your current key EXACTLY matches this lock’s white/blue distribution.
* KEY GENERATOR (interactive, how to generate keys):
  - Look for a patch consisting of white and blue that has a CLEAR path of light gray leading to it and around it, and is NOT bordered by pure black or boxed by dark gray.
  - When any part of your 8×8 player steps ONTO this patch, it GENERATES a NEW candidate key and updates the bottom-left indicator (boxed by dark gray).
  - Step OFF the patch and then back ON to generate yet another candidate key; repeat to cycle through the set of keys.
  - Continue cycling until the bottom-left indicator EXACTLY matches the lock pattern at the exit (the same white/blue layout adjacent to pure black).
* Locating things quickly:
  - Generator: a white/blue patch accessible via light gray, with only light gray around it (no pure black), is almost certainly the generator.
  - Exit: a matching white/blue pattern near pure black tiles marks the lock/exit area.

# HUD / PROGRESSION
* Visual score on the second-to-last grid row: it contains several 4-tile “slots.”
  - Slots made of light gray represent UNFINISHED levels.
  - When a slot changes from light gray to another color, that level is COMPLETE.
* Track remaining MOVES/ENERGY by counting purple tiles. **DO PAY ATTENTION TO THIS AND BE CAREFUL!**
* Track LIVES by counting distinct 2×2 blocks of red.

# GENERAL NOTES
* The grid is a bird’s-eye view.
* If the grid does not change after an action, you likely attempted an illegal move into dark gray.
* Favor routes marked by light gray; use the generator (white/blue patch surrounded by light gray) to cycle keys; confirm the match by comparing the bottom-left indicator (boxed by dark gray) to the white/blue lock pattern found near pure black; then proceed to the exit.

# EXAMPLE STRATEGY OBSERVATION
The 8×8 player (top rows orange, bottom rows blue) is touching dark gray on the left, so LEFT is blocked. I see a white/blue patch with only light gray around it to the right—that’s likely the generator. I should move RIGHT to step on/off it until the bottom-left indicator (boxed by dark gray) matches the lock pattern I see near the pure black tiles, then head to the exit.



            """
        )

    def build_observation_system_prompt(self) -> str:
        """Hard phase guard so the model doesn't pick an action here."""
        return textwrap.dedent(
        """
You are in the OBSERVATION PHASE.
Write ONE clear paragraph (5–8 sentences) analyzing the attached image ONLY.
Use color names (white, blue, orange, light gray, dark gray, pure black, red, purple)—no numbers.
Do NOT choose or execute an action. Do NOT call tools. Do NOT answer with a single word.

You MUST do ALL of the following in your paragraph:
1) Compare the bottom-left key INDICATOR (white/blue boxed by dark gray; non-interactive) to the EXIT/lock (white/blue pattern adjacent to pure black). Explicitly state whether they MATCH. Note: the blue portion is fixed in both places; only the white distribution must match—double-check this carefully.
2) Report the player’s location (the 8×8 sprite with orange top rows over blue bottom rows) relative to the board (e.g., quadrant/near which edge) AND the proximity of dark gray walls on all four sides (are walls close above, below, left, right?).
3) For each direction (UP, DOWN, LEFT, RIGHT), state what would happen if you attempted that move, and whether it would be ILLEGAL due to overlapping dark gray walls.
4) Identify the key GENERATOR if visible (white/blue patch surrounded by light gray, not boxed by dark gray and not near pure black) and the EXIT/lock if visible (white/blue near pure black).
5) Mention purple (moves/energy) or red (lives) only if immediately relevant (e.g., low moves or scarce lives).
If the indicator MATCHES the exit, explicitly begin planning to go to the exit via light gray paths, optionally touching purple tiles en route to recharge if needed.

End with a one-sentence directional INTENT in prose only (e.g., “I intend to move right next if legal…”). Do NOT call a tool.
        """
    )


    def build_observation_user_text(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
        f"""
# OBSERVE (image attached):
Analyze the attached image of the CURRENT 64×64 grid using ONLY color terms (white, blue, orange, light gray, dark gray, pure black, red, purple). No numbers.

Include ALL of the following in ONE paragraph (5–8 sentences):
1) Compare the bottom-left key INDICATOR (white/blue boxed by dark gray; non-interactive) with the EXIT/lock (white/blue adjacent to pure black). Say explicitly if they MATCH. Remember: the blue portion is fixed; only the white distribution must match.
2) Describe the player’s location (orange-over-blue 8×8) relative to the board and report whether dark gray walls are close in each direction (up/down/left/right).
3) Identify the GENERATOR (white/blue patch surrounded by light gray, no pure black nearby) and the EXIT/lock if visible.
4) For UP, DOWN, LEFT, and RIGHT, state what would happen if moved and whether the move is ILLEGAL due to overlapping dark gray.
5) Mention purple (moves) or red (lives) only if crucial now.
6) If the indicator matches the lock, begin planning a route to the EXIT via light gray paths and optionally recharge on purple tiles if needed.

Finish with a one-sentence directional INTENT in prose only (no tool call).

State: {latest_frame.state.name} | Score: {latest_frame.score}
        """
    )


    def build_action_system_prompt(self) -> str:
        """Hard phase guard for the action call."""
        return textwrap.dedent(
            """
You are in the ACTION PHASE.
Call exactly ONE tool (RESET/ACTION1..ACTION6). No paragraphs. No multiple tools.
            """
        )

    def build_action_user_prompt(self) -> str:
        """Short, action-only user task."""
        return textwrap.dedent(
            """
Use the attached image to decide exactly one action (RESET or ACTION1..ACTION4; avoid ACTION5/6 unless required).
            """
        )

    # ----------------- rendering / saving / embedding -----------------

    def _png_digest(self, png_bytes: bytes) -> str:
        return hashlib.sha256(png_bytes).hexdigest()

    def _grid_from_frame(self, frame: FrameData) -> Optional[list[list[int]]]:
        if not frame or not frame.frame: return None
        grids = frame.frame
        grid = grids[-1] if grids and grids[-1] else grids[0]
        if not grid or not grid[0]: return None
        return grid

    def _render_grid_png(self, grid: list[list[int]]) -> bytes:
        h, w = len(grid), len(grid[0])
        img = Image.new("RGB", (w, h), color="#000000")
        draw = ImageDraw.Draw(img)
        for y in range(h):
            row = grid[y]
            for x in range(w):
                draw.point((x, y), fill=self.KEY_COLORS.get(row[x], "#888888"))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _save_png_if_useful(self, png: bytes, latest_frame: FrameData, tag: str) -> tuple[Optional[str], Optional[str]]:
        digest = self._png_digest(png)
        force = (self._last_score is None) or (latest_frame.score != self._last_score)

        # skip uniform except at score changes (fixes “pure black” spam; still force on level up)
        im = Image.open(io.BytesIO(png))
        colors = im.getcolors(maxcolors=256*256)
        if colors and len(colors) == 1 and not force:
            return (None, None)

        if (self._last_digest == digest) and not force:
            return (None, None)

        self._img_counter += 1
        fname = f"{latest_frame.score:02d}-{self._img_counter:05d}-{tag}-{digest[:8]}.png"
        path = self._images_dir / fname
        with open(path, "wb") as f:
            f.write(png)
        self._tw(f"# saved image: {str(path.resolve())}")

        self._last_digest = digest
        self._last_score = latest_frame.score
        return (str(path.resolve()), digest)

    def _frame_to_png_dataurl(self, frame: FrameData, tag: str) -> tuple[Optional[str], Optional[str]]:
        grid = self._grid_from_frame(frame)
        if grid is None: return (None, None)
        png = self._render_grid_png(grid)
        _abs, digest = self._save_png_if_useful(png, frame, tag)
        if _abs is None: return (None, None)
        b64 = base64.b64encode(png).decode("ascii")
        return (f"data:image/png;base64,{b64}", digest)

    # ----------------- reasoning log helper -----------------

    def _record_reasoning(self, phase: str, text: str, **extra: Any) -> None:
        if hasattr(self, "recorder") and not self.is_playback:
            payload = {"phase": phase, "reasoning": text}
            payload.update(extra)
            # also mirror to keys some UIs expect
            payload["assistant"] = text
            payload["reasoning_log"] = text
            self.recorder.record(payload)

    # ----------------- main loop (phase-separated messaging) -----------------

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

        functions = self.build_functions()
        tools = self.build_tools()

        # First turn: seed with shared context + RESET
        if len(self.messages) == 0:
            # Shared game context (no “choose one action” language)
            self.push_message({"role": "system", "content": self.build_game_context_prompt()})
            # Kick off with RESET
            if self.MODEL_REQUIRES_TOOLS:
                self.push_message({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": self._latest_tool_call_id,
                        "type": "function",
                        "function": {"name": GameAction.RESET.name, "arguments": json.dumps({})},
                    }],
                })
            else:
                self.push_message({"role": "assistant", "function_call": {"name": "RESET", "arguments": json.dumps({})}})
            self._log_seed_messages(self.messages)
            return GameAction.RESET

        # Tool result sets up the observation turn
        function_name = latest_frame.action_input.id.name
        # Keep function/tool reply minimal; add big instruction in system+user below
        obs_header = f"State: {latest_frame.state.name} | Score: {latest_frame.score}"
        if self.MODEL_REQUIRES_TOOLS:
            self.push_message({"role": "tool", "tool_call_id": self._latest_tool_call_id, "content": obs_header})
        else:
            self.push_message({"role": "function", "name": function_name, "content": obs_header})

        # -------- OBSERVATION PHASE --------
        self._last_obs_text = ""
        self._last_obs_digest = None
        if self.DO_OBSERVATION:
            # Phase guard as SYSTEM + rich USER text + image in SAME user turn
            self.push_message({"role": "system", "content": self.build_observation_system_prompt()})
            data_url, digest = self._frame_to_png_dataurl(latest_frame, tag="obs")
            self._last_obs_digest = digest
            obs_text = self.build_observation_user_text(latest_frame)
            if data_url:
                content = [
                    {"type": "text", "text": obs_text},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ]
                self.push_message({"role": "user", "content": content})
                self._tw(f"# [obs] embedded digest={digest}")
            else:
                # Always send the checklist text even without an image (prevents empty observation).
                self.push_message({"role": "user", "content": [{"type": "text", "text": obs_text}]})
                self._tw("# [obs] no image embedded; sent text-only checklist")
            

            logger.info("Sending to Assistant for observation (with image)...")
            try:
                create_kwargs = {"model": self.MODEL, "messages": self._coerce_messages_for_wire(self.messages)}
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                self._log_api_call("observation", self.MODEL, self.messages)
                response = client.chat.completions.create(**create_kwargs)
                self.capture_reasoning_from_response(response)
                self._log_api_response(response)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e

            obs_text_out = response.choices[0].message.content or ""
            self._last_obs_text = obs_text_out
            self.track_tokens(response.usage.total_tokens, obs_text_out)
            logger.info(f"[OBSERVATION] {obs_text_out}")
            self.push_message({"role": "assistant", "content": obs_text_out})
            self._record_reasoning("observation", obs_text_out, image_digest=self._last_obs_digest)

        # -------- ACTION PHASE --------
        # Action directives as SYSTEM + short USER text + image; now we provide tools
        self.push_message({"role": "system", "content": self.build_action_system_prompt()})
        self.push_message({"role": "user", "content": self.build_action_user_prompt()})
        data_url_act, digest_act = self._frame_to_png_dataurl(latest_frame, tag="act")
        self._last_act_digest = digest_act
        if data_url_act:
            self.push_message({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_url_act, "detail": "high"}}],
            })
            self._tw(f"# [act] embedded digest={digest_act}")
        else:
            self._tw("# [act] skipped embedding image (empty/duplicate/uniform)")

        chosen_name = GameAction.ACTION5.name
        arguments = None
        assistant_msg = None

        if self.MODEL_REQUIRES_TOOLS:
            logger.info("Sending to Assistant for action (with image)...")
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self._coerce_messages_for_wire(self.messages),
                    "tools": tools,
                    "tool_choice": "required",
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                self._log_api_call("action", self.MODEL, self.messages, tools=tools, tool_choice="required")
                response = client.chat.completions.create(**create_kwargs)
                self.capture_reasoning_from_response(response)
                self._log_api_response(response)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e

            self.track_tokens(response.usage.total_tokens)
            assistant_msg = response.choices[0].message
            tool_call = assistant_msg.tool_calls[0]
            self._latest_tool_call_id = tool_call.id
            chosen_name = tool_call.function.name
            arguments = tool_call.function.arguments
            logger.info(f"[ACTION] {chosen_name} {arguments}")

            for tc in assistant_msg.tool_calls[1:]:
                self.push_message({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: assistant can only call one action at a time. Using the first.",
                })
        else:
            logger.info("Sending to Assistant for action (with image)...")
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self._coerce_messages_for_wire(self.messages),
                    "functions": functions,
                    "function_call": "auto",
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                self._log_api_call("action", self.MODEL, self.messages, functions=functions)
                response = client.chat.completions.create(**create_kwargs)
                self.capture_reasoning_from_response(response)
                self._log_api_response(response)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e

            self.track_tokens(response.usage.total_tokens)
            assistant_msg = response.choices[0].message
            fc = assistant_msg.function_call
            chosen_name = fc.name
            arguments = fc.arguments
            logger.info(f"[ACTION] {chosen_name} {arguments}")

        if assistant_msg:
            self.push_message(assistant_msg)

        # parse args
        try:
            data = json.loads(arguments) if arguments else {}
        except Exception as e:
            logger.warning(f"JSON parsing error on function response: {e}")
            data = {}

        action = GameAction.from_name(chosen_name)
        action.set_data(data)

        # attach rich reasoning + mirror to recorder
        action.reasoning = {
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "action_chosen": action.name,
            "agent_type": "visual_guided_llm",
            "game_rules": "locksmith",
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "observation": self._last_obs_text,
            "observation_image_digest": self._last_obs_digest,
            "action_image_digest": self._last_act_digest,
            "reasoning_tokens": getattr(self, "_last_reasoning_tokens", 0),
            "total_reasoning_tokens": getattr(self, "_total_reasoning_tokens", 0),
        }
        self._record_reasoning(
            "action",
            self._last_obs_text,
            chosen_action=action.name,
            observation_image_digest=self._last_obs_digest,
            action_image_digest=self._last_act_digest,
        )
        return action

class BimodalGuidedChecklistLLM(VisualGuidedLLM, Agent):
    """
    Visual-first guided VLM with a strict 10-item observation checklist.
    Optional 'bimodal' mode appends a numbers→colors mapping and (optionally) the textual grid,
    clearly labeled as the EXACT same state as the image.

    Differences from VisualGuidedLLM:
      - Observation: forced structured checklist (10 items) + "Summary:" line.
      - Bimodal gating: numeric mapping/grid only when enabled.
      - Model override supported (kwarg or env).
    """

    BIMODAL_TEXT: bool = False
    INCLUDE_TEXTUAL_GRID: bool = False

    INT_TO_COLOR_NAME: dict[int, str] = {
        0: "white", 1: "light gray", 2: "gray", 3: "dim gray",
        4: "dark gray", 5: "pure black", 6: "magenta", 7: "pink",
        8: "orange-red", 9: "blue", 10: "light blue", 11: "yellow",
        12: "orange", 13: "maroon", 14: "green", 15: "purple",
    }

    def __init__(
        self,
        *args,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        do_observation: Optional[bool] = None,
        model_requires_tools: Optional[bool] = None,
        bimodal: Optional[bool] = None,
        include_textual_grid: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            model=model,
            reasoning_effort=reasoning_effort,
            do_observation=do_observation,
            model_requires_tools=model_requires_tools,
            **kwargs,
        )
        if bimodal is not None:
            self.BIMODAL_TEXT = bool(bimodal)
        if include_textual_grid is not None:
            self.INCLUDE_TEXTUAL_GRID = bool(include_textual_grid)

    

    def build_observation_system_prompt(self) -> str:
        return textwrap.dedent(
            """
You are in the OBSERVATION PHASE (CHECKLIST). Do NOT call tools. No code blocks.
Output must contain exactly 10 numbered items followed by a final line beginning with "Summary:".

Checklist:
1) Player position & nearby walls (8×8: orange top rows over blue bottom rows). Where on the board? Any dark gray walls close above/below/left/right?
2) Bottom-left key indicator (boxed by dark gray): describe the white/blue layout and its orientation.
3) Exit/lock (near pure black): describe its white/blue layout and orientation.
4) Match status: “Match” or “No match” + justification (white/blue distribution AND orientation), reall be careful, oftentimes they do actually match, and the distribution of the blue squares almost always do!
5) Key generator: identify candidate patch(es) (white/blue surrounded by light gray, not boxed by dark gray, not near pure black) + short justification.
6) Moves/energy: status inferred from purple tiles (LOW/OK) and whether recharging is relevant in this position.
7) Lives: count distinct red 2×2 clusters; note if relevant risk.
8) Direction outcomes (UP/DOWN/LEFT/RIGHT): for EACH direction state LEGAL/ILLEGAL (overlapping dark gray?), and whether it moves you closer/farther/same to the generator and to the lock (with a brief reason).
9) Best immediate waypoint: name it (generator, lock, or corridor node) and give a 1–2 step light‑gray route.
10) Hazards & constraints: narrow corridors, dead-ends, wall contact, energy risks.

Summary: One sentence with intended next direction and the reason (no tool call).
            """
        )

    def _numbers_to_colors_map_text(self) -> str:
        lines = []
        for i in range(16):
            hexcol = self.KEY_COLORS.get(i, "#888888")
            name = self.INT_TO_COLOR_NAME.get(i, "unknown")
            lines.append(f"{i:>2} → {name} ({hexcol})")
        return "\n".join(lines)

    def _bimodal_tail(self, latest_frame: FrameData) -> str:
        mapping = self._numbers_to_colors_map_text()
        tail = [
            "",
            "# ADDITIONAL TEXTUAL CONTEXT (EXACT SAME STATE AS IMAGE)",
            "The textual representation encodes the same frame you see in the image.",
            "Use the following mapping from numbers to colors:",
            mapping,
        ]
        if self.INCLUDE_TEXTUAL_GRID:
            grid_text = self.pretty_print_3d(latest_frame.frame)
            tail.extend(["", "# TEXTUAL GRID", grid_text])
        return "\n".join(tail)

    def build_observation_user_text(self, latest_frame: FrameData) -> str:
        head = textwrap.dedent(
            f"""
# OBSERVE:
Fill the 10‑item checklist

State: {latest_frame.state.name} | Score: {latest_frame.score}
            """
        )
        if self.BIMODAL_TEXT:
            return head + "\n" + self._bimodal_tail(latest_frame)
        return head
