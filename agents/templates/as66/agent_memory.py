# agents/templates/as66/agent_memory.py
from __future__ import annotations
from typing import Any, List, Tuple, Optional, Dict
import json
import logging
import uuid
from pathlib import Path
import os
from openai import OpenAI

from ..llm_agents import GuidedLLM
from ...structs import FrameData, GameAction, GameState, ActionInput
from .agent import _map_click_to_source_xy
from .downsample import downsample_4x4, matrix16_to_lines
from .prompts_memory import (
    build_initial_hypotheses_system_prompt,
    build_initial_hypotheses_user_prompt,
    build_update_hypotheses_system_prompt,
    build_update_hypotheses_user_prompt,
    build_observation_system_prompt,
    build_observation_user_prompt,
    build_action_selection_system_prompt,
    build_action_selection_user_prompt,
)

log = logging.getLogger(__name__)

class AS66MemoryAgent(GuidedLLM):
    """
    An agent that uses an external memory file to manage context,
    including move history and dynamic hypotheses about game mechanics.
    This version uses a three-call process: Hypothesis Update, Observation, and Action Selection.
    """
    MAX_ACTIONS = 200
    MODEL = os.getenv("AGENT_MODEL_OVERRIDE", "gpt-5")
    REASONING_EFFORT = os.getenv("AGENT_REASONING_EFFORT", "low") 
    DOWNSAMPLE = True
    MEMORY_DIR = Path("memory")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.MEMORY_DIR.mkdir(exist_ok=True)
        self.memory_file_path = self.MEMORY_DIR / f"{self.game_id}_{uuid.uuid4()}.md"
        self.seen_state_actions = set()
        self.move_history_content = "## Move History\n\n(No moves recorded yet.)"
        self.hypotheses_content = "## Hypotheses\n\n(No hypotheses generated yet.)"
        self._is_initialized = False
        self._token_total = 0
        
        # --- NEW: Ablation settings from environment ---
        self.INCLUDE_TEXT_DIFF = os.getenv("INCLUDE_TEXT_DIFF", "true").lower() == "true"
        self.CONTEXT_LENGTH_LIMIT = int(os.getenv("CONTEXT_LENGTH_LIMIT", "-1")) # -1 for unlimited

    def _get_token_count(self, text: str) -> int:
        """A lightweight proxy for token counting."""
        # Using a simple 4 chars/token proxy as discussed
        return len(text) // 4

    def _get_truncated_history(self) -> str:
        """
        Applies context length limit with a sliding window that
        preserves high-information (level up / game over) entries.
        """
        if self.CONTEXT_LENGTH_LIMIT == -1:
            return self.move_history_content # No limit
            
        # Split history into header and entries
        try:
            header, all_entries_str = self.move_history_content.split("\n\n", 1)
            entries = all_entries_str.split("\n---\n\n")
        except ValueError:
            # Not enough content to split, just return as-is
            return self.move_history_content

        high_info_entries = []
        regular_entries = []
        
        for i, entry in enumerate(entries):
            # Store with original index to re-sort later
            if "⭐ LEVEL UP!" in entry or "☠️ GAME OVER!" in entry:
                high_info_entries.append((i, entry, self._get_token_count(entry)))
            else:
                regular_entries.append((i, entry, self._get_token_count(entry)))

        high_info_tokens = sum(tokens for _, _, tokens in high_info_entries)
        remaining_budget = self.CONTEXT_LENGTH_LIMIT - high_info_tokens

        kept_entries = [(i, entry) for i, entry, _ in high_info_entries]
        
        if remaining_budget > 0:
            # Fill remaining budget with most recent regular entries
            kept_regular_tokens = 0
            # Iterate in reverse (most recent first)
            for i, entry, tokens in reversed(regular_entries):
                if (kept_regular_tokens + tokens) <= remaining_budget:
                    kept_entries.append((i, entry))
                    kept_regular_tokens += tokens
                else:
                    # Budget is full
                    break
        
        # Re-sort all kept entries by their original index to maintain timeline order
        kept_entries.sort(key=lambda x: x[0])
        
        # Re-assemble the history content
        final_entries_str = "\n---\n\n".join([entry for _, entry in kept_entries])
        return f"{header}\n\n{final_entries_str}"


    def _read_memory(self) -> str:
        
        move_history = self._get_truncated_history()
        return f"{move_history}\n\n{self.hypotheses_content}"

    def _write_memory(self) -> None:
       
        content = self._read_memory()
        self.memory_file_path.write_text(content, encoding="utf-8")

    def _get_state_hash(self, grid: List[List[int]]) -> str:
        return json.dumps(grid, sort_keys=True)

    def _calculate_diff(self, grid1: List[List[int]], grid2: List[List[int]]) -> str:
        diffs = []
        h = len(grid1)
        w = len(grid1[0]) if h > 0 else 0
        if len(grid2) != h or (h > 0 and len(grid2[0]) != w):
            return "Error: Grids have different dimensions."

        for r in range(h):
            for c in range(w):
                if grid1[r][c] != grid2[r][c]:
                    diffs.append(f"- Cell ({r}, {c}): {grid1[r][c]} -> {grid2[r][c]}")
        
        if not diffs:
            return "No change in board state."
        return "\n".join(diffs)

    def _initialize_memory(self, initial_frame: FrameData) -> None:
        """API Call 1 (Setup): Generates the initial set of hypotheses."""
        log.info(f"[{self.game_id}] Initializing memory and hypotheses...")
        client = OpenAI()
        
        grid_3d = initial_frame.frame
        grid = downsample_4x4(grid_3d, take_last_grid=True, round_to_int=True) if self.DOWNSAMPLE else (grid_3d[-1] if grid_3d else [])

        sys_prompt = build_initial_hypotheses_system_prompt()
        user_prompt = build_initial_hypotheses_user_prompt(grid)
        
        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            reasoning_effort=self.REASONING_EFFORT,
        )
        
        self.hypotheses_content = f"## Hypotheses\n\n{(resp.choices[0].message.content or '').strip()}"
        self._token_total += getattr(resp.usage, "total_tokens", 0)
        self._write_memory()
        self._is_initialized = True
        log.info(f"[{self.game_id}] Initial hypotheses generated and saved to {self.memory_file_path}")

    def _update_memory(self, prev_frame: FrameData, action: ActionInput, new_frame: FrameData) -> bool:
        """
        Updates the memory file with the latest move and revised hypotheses.
        Returns True if the state-action pair was a repeat, False otherwise.
        """
        client = OpenAI()
        
        prev_grid_3d = prev_frame.frame
        new_grid_3d = new_frame.frame
        
        prev_grid = downsample_4x4(prev_grid_3d, take_last_grid=True, round_to_int=True) if self.DOWNSAMPLE else (prev_grid_3d[-1] if prev_grid_3d else [])
        new_grid = downsample_4x4(new_grid_3d, take_last_grid=True, round_to_int=True) if self.DOWNSAMPLE else (new_grid_3d[-1] if new_grid_3d else [])

        state_hash = self._get_state_hash(prev_grid)
        
        action_id_enum = action.id
        action_identifier = action_id_enum.name
        if action_id_enum == GameAction.ACTION6 and action.reasoning and "original_click" in action.reasoning:
            original_click = action.reasoning["original_click"]
            x = original_click.get('x', '?')
            y = original_click.get('y', '?')
            action_identifier = f"{action_id_enum.name}(x={x}, y={y})"
        elif action_id_enum == GameAction.ACTION6 and action.data:
            x = action.data.get('x', '?')
            y = action.data.get('y', '?')
            action_identifier = f"{action_id_enum.name}(x={x}, y={y}) [mapped]"

        state_action_tuple = (state_hash, action_identifier)

        if state_action_tuple in self.seen_state_actions:
            log.warning(f"[{self.game_id}] Repeated state-action pair detected: {action_identifier}. Applying penalty.")
            return True # Signal that this was a repeat

        self.seen_state_actions.add(state_action_tuple)

        is_level_up = new_frame.score > prev_frame.score
        diff = self._calculate_diff(prev_grid, new_grid)
        
        entry_header = f"### Turn {len(self.seen_state_actions)}\n\n"
        
       
        entry_parts_list = [
            f"**Action:** `{action_identifier}`\n\n",
            "**State Before:**\n",
            "```\n",
            f"{matrix16_to_lines(prev_grid)}\n",
            "```\n\n",
            "**Resulting State:**\n",
            "```\n",
            f"{matrix16_to_lines(new_grid)}\n",
            "```\n"
        ]
        
        if self.INCLUDE_TEXT_DIFF:
            entry_parts_list.extend([
                "\n**Resulting Diff:**\n",
                "```\n",
                f"{diff}\n",
                "```\n"
            ])
        else:
            entry_parts_list.append("\n") # Add a trailing newline if no diff

        entry_body = "".join(entry_parts_list)
      
        
        if is_level_up:
            history_entry = f"> **⭐ LEVEL UP! A new level begins below.**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        elif new_frame.state == GameState.GAME_OVER:
            history_entry = f"> **☠️ GAME OVER!**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        else:
            history_entry = f"{entry_header}{entry_body}\n---\n\n"

        if "(No moves recorded yet.)" in self.move_history_content:
            self.move_history_content = "## Move History\n\n" + history_entry
        else:
            self.move_history_content += history_entry

        log.info(f"[{self.game_id}] Updating hypotheses based on new move...")
        sys_prompt = build_update_hypotheses_system_prompt()
        
        user_prompt = build_update_hypotheses_user_prompt(self._read_memory())

        if is_level_up:
            special_instruction = (
                "**IMPORTANT CONTEXT: A LEVEL UP just occurred.** The game board has changed for the new level, though some rules and mechanics may be similar or have only one or two changes. "
                "Your primary task now is to **re-validate your current hypotheses against this new environment.** "
                "Meticulously check what has changed. Are your old hypotheses still valid, or do they need to be adapted or discarded entirely? "
                "Generate a new set of five hypotheses that reflect your understanding of this new level.\n\n"
            )
            user_prompt = special_instruction + user_prompt

        resp = client.chat.completions.create(model=self.MODEL, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], reasoning_effort=self.REASONING_EFFORT)
        self._token_total += getattr(resp.usage, "total_tokens", 0)
        
        new_hypotheses = (resp.choices[0].message.content or "").strip()
        self.hypotheses_content = "## Hypotheses\n\n" + (new_hypotheses if new_hypotheses else self.hypotheses_content.split("\n\n", 1)[1])
        
        self._write_memory()
        log.info(f"[{self.game_id}] Memory file updated.")
        return False # Signal that this was not a repeat

    def _get_observation_text(self, memory_content: str, ds16: List[List[int]], score: int, step: int) -> str:
        """API Call 2: Calls the LLM to get a text-based observation and rationale."""
        client = OpenAI()
        sys_prompt = build_observation_system_prompt()
        user_prompt = build_observation_user_prompt(memory_content, ds16, score, step)

        resp = client.chat.completions.create(model=self.MODEL, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], reasoning_effort=self.REASONING_EFFORT)
        self._token_total += getattr(resp.usage, "total_tokens", 0)
        
        observation = (resp.choices[0].message.content or "No observation generated.").strip()
        log.info(f"[{self.game_id} | Step {step}] Observation Rationale: {observation}")
        return observation
        
    def _select_action(self, observation_text: str, latest_frame: FrameData) -> Tuple[GameAction, Optional[Dict[str, Any]]]:
        """API Call 3: Takes observation text and forces a valid tool call, mapping clicks correctly."""
        client = OpenAI()
        sys_prompt = build_action_selection_system_prompt()
        user_prompt = build_action_selection_user_prompt(observation_text)
        
        tools = self.build_tools()
        click_reasoning = None

        resp = client.chat.completions.create(model=self.MODEL, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], tools=tools, tool_choice="required", reasoning_effort=self.REASONING_EFFORT)
        self._token_total += getattr(resp.usage, "total_tokens", 0)

        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            log.error("Action selection model failed to call a tool. Defaulting to ACTION5.")
            return GameAction.from_name("ACTION5"), None

        tool_call = tool_calls[0]
        action_name = tool_call.function.name
        
        try:
            arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except json.JSONDecodeError:
            log.error(f"Failed to parse arguments for {action_name}. Defaulting to no arguments.")
            arguments = {}

        action = GameAction.from_name(action_name)
        
        if action == GameAction.ACTION6:
            x_in = arguments.get("x")
            y_in = arguments.get("y")
            x64, y64, mapping_note = _map_click_to_source_xy(latest_frame, x_in, y_in)
            action.set_data({"x": x64, "y": y64})
            
            click_reasoning = {"original_click": {"x": x_in, "y": y_in}}
            log.info(f"[{self.game_id}] Recommended Action: {action_name} with mapped args {{'x': {x64}, 'y': {y64}}}. Note: {mapping_note}")
        elif arguments:
            action.set_data(arguments)
            log.info(f"[{self.game_id}] Recommended Action: {action_name} with args {arguments}")
        else:
            log.info(f"[{self.game_id}] Recommended Action: {action_name}")
            
        return action, click_reasoning

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Main logic loop for a single turn, orchestrating the three API calls."""
        was_repeated = False
        if not self._is_initialized:
            self._initialize_memory(latest_frame)
        elif len(frames) > 1:
            previous_frame = frames[-2]
            previous_action = latest_frame.action_input
            was_repeated = self._update_memory(previous_frame, previous_action, latest_frame)

        if latest_frame.state == GameState.GAME_OVER:
            log.info(f"[{self.game_id}] Game over detected. Returning RESET to try again.")
            return GameAction.RESET

        memory_content = self._read_memory()
        grid_3d = latest_frame.frame
        grid = downsample_4x4(grid_3d, take_last_grid=True, round_to_int=True) if self.DOWNSAMPLE else (grid_3d[-1] if grid_3d else [])

        observation_text = self._get_observation_text(memory_content, grid, latest_frame.score, len(frames))
        
        # **NEW LOGIC**: Prepend a penalty message if a repeated action was detected.
        if was_repeated:
            penalty_message = (
                "**PENALTY**: Your previous action was a repeat of an action you've already taken in this exact state, "
                "and it was ignored. You MUST choose a different action this turn to avoid getting stuck in a loop. "
                "Analyze your hypotheses and choose a different experiment or a move that has not been tried from this state before.\n\n---\n\n"
            )
            observation_text = penalty_message + observation_text
        
        action, click_reasoning = self._select_action(observation_text, latest_frame)
        
        reasoning_data = {
            "agent": self.__class__.__name__,
            "model": self.MODEL,
            "rationale": observation_text,
            "memory_file": str(self.memory_file_path.resolve()),
            "total_tokens": self._token_total,
        }
        if click_reasoning:
            reasoning_data.update(click_reasoning)
        
        action.reasoning = reasoning_data
        return action