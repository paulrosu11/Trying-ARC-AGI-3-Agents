# agents/templates/as66/agent_visual_memory.py
from __future__ import annotations
from typing import Any, List, Tuple, Optional, Dict
import json
import logging
import uuid
from pathlib import Path
import os
import base64
from openai import OpenAI

from ...structs import FrameData, GameAction, GameState, ActionInput
from .agent_memory import AS66MemoryAgent
from .downsample import downsample_4x4, render_grid_to_png_bytes 

# Import the visual prompt builders
from .prompts_visual_memory import (
    build_initial_hypotheses_system_prompt,
    build_initial_hypotheses_user_content,
    build_update_hypotheses_system_prompt,
    build_observation_system_prompt,
    build_action_selection_system_prompt,
    build_action_selection_user_prompt,
)

log = logging.getLogger(__name__)


class TurnData:
    """Stores complete data for a single turn to enable image regeneration."""
    def __init__(
        self, 
        turn_number: int,
        action_str: str,
        before_grid: List[List[int]],
        after_grid: List[List[int]],
        diff_str: str,
        is_level_up: bool = False,
        is_game_over: bool = False
    ):
        self.turn_number = turn_number
        self.action_str = action_str
        self.before_grid = before_grid
        self.after_grid = after_grid
        self.diff_str = diff_str
        self.is_level_up = is_level_up
        self.is_game_over = is_game_over


class AS66VisualMemoryAgent(AS66MemoryAgent):
    """
    An agent that uses multimodal context (images + text diffs) to manage
    hypotheses. Properly implements interleaved image+text history that scales
    with the number of turns.
    """
    MODEL = os.getenv("AGENT_MODEL_OVERRIDE", "gpt-5")  # Requires a vision model
    IMAGE_DIR = Path("memory") / "images"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        # Ensure the per-game image directory exists
        self.game_image_dir = self.IMAGE_DIR / self.game_id
        self.game_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Store structured turn data for regenerating images
        self.turn_history: List[TurnData] = []
        
        # Vision-specific settings
        self.DOWNSAMPLE = os.getenv("DOWNSAMPLE_IMAGES", "true").lower() == "true"
        self.IMAGE_DETAIL_LEVEL = os.getenv("IMAGE_DETAIL_LEVEL", "low").lower()
        self.PIXELS_PER_CELL = int(os.getenv("IMAGE_PIXELS_PER_CELL", "24"))

    def _get_grid_from_frame(self, frame_3d: List[List[List[int]]]) -> List[List[int]]:
        """Helper to get the correct grid (16x16 or 64x64) based on DOWNSAMPLE setting."""
        if self.DOWNSAMPLE:
            return downsample_4x4(frame_3d, take_last_grid=True, round_to_int=True)
        else:
            return frame_3d[-1] if frame_3d else []

    def _grid_to_base64(self, grid: List[List[int]]) -> str:
        """Convert a grid to base64 PNG data (without the data URL prefix)."""
        try:
            png_bytes = render_grid_to_png_bytes(grid, cell=self.PIXELS_PER_CELL)
            if not png_bytes:
                raise ValueError("Rendered empty PNG bytes")
            return base64.b64encode(png_bytes).decode('utf-8')
        except Exception as e:
            log.error(f"[{self.game_id}] Failed to generate image: {e}")
            return ""

    def _build_turn_multimodal_content(self, turn: TurnData) -> List[Dict[str, Any]]:
        """
        Builds interleaved multimodal content for a single turn.
        Returns a list of content items (text + images).
        """
        content = []
        
        # Turn header and action
        header_text = f"### Turn {turn.turn_number}\n\n**Action:** `{turn.action_str}`\n\n"
        
        if turn.is_level_up:
            header_text = f"> **⭐ LEVEL UP! A new level begins below.**\n>\n> {header_text}"
        elif turn.is_game_over:
            header_text = f"> **☠️ GAME OVER!**\n>\n> {header_text}"
        
        content.append({"type": "text", "text": header_text + "**State Before:**"})
        
        # Before image
        before_b64 = self._grid_to_base64(turn.before_grid)
        if before_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{before_b64}",
                    "detail": self.IMAGE_DETAIL_LEVEL
                }
            })
        else:
            content.append({"type": "text", "text": "\n*(Image generation failed)*"})
        
        content.append({"type": "text", "text": "\n\n**State After:**"})
        
        # After image
        after_b64 = self._grid_to_base64(turn.after_grid)
        if after_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{after_b64}",
                    "detail": self.IMAGE_DETAIL_LEVEL
                }
            })
        else:
            content.append({"type": "text", "text": "\n*(Image generation failed)*"})
        
        # Optional text diff
        if self.INCLUDE_TEXT_DIFF:
            content.append({
                "type": "text",
                "text": f"\n\n**Resulting Textual Diff:**\n```\n{turn.diff_str}\n```\n\n---\n\n"
            })
        else:
            content.append({"type": "text", "text": "\n\n---\n\n"})
        
        return content

    def _get_turns_within_context_limit(self) -> List[TurnData]:
        """
        Returns a subset of turn_history that fits within CONTEXT_LENGTH_LIMIT,
        preserving high-information turns (level ups, game overs).
        """
        if self.CONTEXT_LENGTH_LIMIT == -1:
            return self.turn_history  # No limit
        
        if not self.turn_history:
            return []
        
        # Separate high-info vs regular turns
        high_info_turns = []
        regular_turns = []
        
        for turn in self.turn_history:
            if turn.is_level_up or turn.is_game_over:
                high_info_turns.append(turn)
            else:
                regular_turns.append(turn)
        
        # Estimate tokens per turn (rough heuristic: ~150 tokens per turn for text + images)
        # Images with detail="low" are ~85 tokens each (OpenAI uses 85 tokens for low detail)
        # So ~150 tokens text + 170 tokens for 2 images = ~320 tokens per turn
        TOKENS_PER_TURN = 320
        
        max_turns = max(1, self.CONTEXT_LENGTH_LIMIT // TOKENS_PER_TURN)
        
        # Always include high-info turns
        kept_turns = high_info_turns[:]
        remaining_slots = max_turns - len(high_info_turns)
        
        if remaining_slots > 0 and regular_turns:
            # Fill remaining slots with most recent regular turns
            kept_turns.extend(regular_turns[-remaining_slots:])
        
        # Sort by turn number to maintain chronological order
        kept_turns.sort(key=lambda t: t.turn_number)
        
        return kept_turns

    def _build_full_history_multimodal_content(self) -> List[Dict[str, Any]]:
        """
        Builds complete interleaved multimodal content for all turns within context limit.
        Returns a list ready to be included in user message content.
        """
        turns_to_include = self._get_turns_within_context_limit()
        
        if not turns_to_include:
            return [{"type": "text", "text": "## Move History\n\n(No moves recorded yet.)\n\n"}]
        
        content = [{"type": "text", "text": "## Move History\n\n"}]
        
        for turn in turns_to_include:
            content.extend(self._build_turn_multimodal_content(turn))
        
        return content

    def _initialize_memory(self, initial_frame: FrameData) -> None:
        """API Call 1 (Setup): Generates initial hypotheses using an image."""
        log.info(f"[{self.game_id}] Initializing visual memory and hypotheses...")
        client = OpenAI()
        
        grid_3d = initial_frame.frame
        grid = self._get_grid_from_frame(grid_3d)
        
        img_b64 = self._grid_to_base64(grid)
        if not img_b64:
            log.error(f"[{self.game_id}] Cannot initialize memory, image generation failed.")
            self.hypotheses_content = "## Hypotheses\n\n(ERROR: Initial image generation failed.)"
            self._is_initialized = True
            return

        sys_prompt = build_initial_hypotheses_system_prompt()
        user_content = build_initial_hypotheses_user_content(self.game_id, img_b64, detail=self.IMAGE_DETAIL_LEVEL)
        
        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            reasoning_effort=self.REASONING_EFFORT,
        )
        
        self.hypotheses_content = f"## Hypotheses\n\n{(resp.choices[0].message.content or '').strip()}"
        self._token_total += getattr(resp.usage, "total_tokens", 0)
        
        # We don't need to store the initial state in turn_history since it's just setup
        self.move_history_content = "## Move History\n\n"
        
        self._write_memory()
        self._is_initialized = True
        log.info(f"[{self.game_id}] Initial visual hypotheses generated.")

    def _update_memory(self, prev_frame: FrameData, action: ActionInput, new_frame: FrameData) -> bool:
        """
        Updates the memory with the latest move and revises hypotheses using
        multimodal context (all past images + new images).
        """
        client = OpenAI()
        
        prev_grid_3d = prev_frame.frame
        new_grid_3d = new_frame.frame
        
        prev_grid = self._get_grid_from_frame(prev_grid_3d)
        new_grid = self._get_grid_from_frame(new_grid_3d)

        state_hash = self._get_state_hash(prev_grid)
        
        # Build action identifier
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

        # Check for repeated state-action
        if state_action_tuple in self.seen_state_actions:
            log.warning(f"[{self.game_id}] Repeated state-action pair detected: {action_identifier}. Applying penalty.")
            return True  # Signal that this was a repeat

        self.seen_state_actions.add(state_action_tuple)

        is_level_up = new_frame.score > prev_frame.score
        diff = self._calculate_diff(prev_grid, new_grid)
        
        # Store this turn's data
        turn_number = len(self.seen_state_actions)
        turn_data = TurnData(
            turn_number=turn_number,
            action_str=action_identifier,
            before_grid=prev_grid,
            after_grid=new_grid,
            diff_str=diff,
            is_level_up=is_level_up,
            is_game_over=(new_frame.state == GameState.GAME_OVER)
        )
        self.turn_history.append(turn_data)
        
        # Update text-based memory file (for disk persistence)
        # This is kept for reference but the actual prompts use multimodal content
        self._update_text_memory(turn_data)

        # Build multimodal content for hypothesis update
        log.info(f"[{self.game_id}] Updating hypotheses based on visual move history...")
        sys_prompt = build_update_hypotheses_system_prompt()
        
        # Build user content: hypotheses text + full interleaved history
        user_content = [
            {"type": "text", "text": "Here is the game memory so far, including your prior hypotheses.\n\n"},
            {"type": "text", "text": self.hypotheses_content + "\n\n"},
        ]
        
        # Add full interleaved history
        user_content.extend(self._build_full_history_multimodal_content())
        
        if is_level_up:
            special_instruction = (
                "**IMPORTANT CONTEXT: A LEVEL UP just occurred.** The game board has changed for the new level. "
                "Your primary task now is to **re-validate your current hypotheses against this new environment.** "
                "Generate a new set of five hypotheses that reflect your understanding of this new level.\n\n"
            )
            user_content.insert(0, {"type": "text", "text": special_instruction})

        user_content.append({
            "type": "text", 
            "text": "\nAnalyze this evidence and provide an updated list of five refined hypotheses."
        })

        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            reasoning_effort=self.REASONING_EFFORT,
        )
        self._token_total += getattr(resp.usage, "total_tokens", 0)
        
        new_hypotheses = (resp.choices[0].message.content or "").strip()
        self.hypotheses_content = "## Hypotheses\n\n" + (new_hypotheses if new_hypotheses else self.hypotheses_content.split("\n\n", 1)[1])
        
        self._write_memory()
        log.info(f"[{self.game_id}] Visual memory and hypotheses updated.")
        return False

    def _update_text_memory(self, turn: TurnData) -> None:
        """Updates the text-based memory file (for disk persistence)."""
        entry_header = f"### Turn {turn.turn_number}\n\n"
        entry_parts = [
            f"**Action:** `{turn.action_str}`\n\n",
            "**State Before (Image):** [Image stored]\n\n",
            "**State After (Image):** [Image stored]\n",
        ]
        
        if self.INCLUDE_TEXT_DIFF:
            entry_parts.append(f"\n**Resulting Textual Diff:**\n```\n{turn.diff_str}\n```\n")
        else:
            entry_parts.append("\n")
        
        entry_body = "".join(entry_parts)
        
        if turn.is_level_up:
            history_entry = f"> **⭐ LEVEL UP! A new level begins below.**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        elif turn.is_game_over:
            history_entry = f"> **☠️ GAME OVER!**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        else:
            history_entry = f"{entry_header}{entry_body}\n---\n\n"
        
        self.move_history_content += history_entry

    def _get_observation_text(self, memory_content: str, ds16_grid: List[List[int]], score: int, step: int) -> str:
        """
        API Call 2: Calls the LLM with complete multimodal context (all past images + current image)
        to get a text observation.
        """
        client = OpenAI()
        
        # Generate current state image
        current_img_b64 = self._grid_to_base64(ds16_grid)
        if not current_img_b64:
            log.error(f"[{self.game_id}] Cannot get observation, image generation failed.")
            return "ERROR: Could not generate current state image for observation."

        sys_prompt = build_observation_system_prompt()
        
        # Build user content: status + current state + hypotheses + full history
        user_content = [
            {"type": "text", "text": f"**Current Game Status:**\n- Step: {step}\n- Score: {score}\n\n**Current Board State (Image):**"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_img_b64}", "detail": self.IMAGE_DETAIL_LEVEL}},
            {"type": "text", "text": "\n\n" + self.hypotheses_content + "\n\n"},
        ]
        
        # Add full interleaved history (all past images + text)
        user_content.extend(self._build_full_history_multimodal_content())
        
        user_content.append({
            "type": "text",
            "text": "\nFollow your reasoning process and provide a detailed text analysis, concluding with your recommended action. Be precise with all coordinates."
        })

        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            reasoning_effort=self.REASONING_EFFORT,
        )
        self._token_total += getattr(resp.usage, "total_tokens", 0)
        
        observation = (resp.choices[0].message.content or "No observation generated.").strip()
        log.info(f"[{self.game_id} | Step {step}] Visual Observation Rationale generated.")
        return observation
    
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Main logic loop. Calls the overridden multimodal methods.
        """
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

        memory_content = self._read_memory()  # For logging/debugging
        grid_3d = latest_frame.frame
        grid = self._get_grid_from_frame(grid_3d)

        # Get observation with full multimodal history
        observation_text = self._get_observation_text(memory_content, grid, latest_frame.score, len(frames))
        
        if was_repeated:
            penalty_message = (
                "**PENALTY**: Your previous action was a repeat of an action you've already taken in this exact state, "
                "and it was ignored. You MUST choose a different action this turn to avoid getting stuck in a loop. "
                "Analyze your hypotheses and choose a different experiment or a move that has not been tried from this state before.\n\n---\n\n"
            )
            observation_text = penalty_message + observation_text
        
        # This function is inherited and works as-is (text-in, tool-out)
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