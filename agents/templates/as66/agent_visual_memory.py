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
from .downsample import downsample_4x4, ds16_png_bytes

# Import the new visual prompt builders
from .prompts_visual_memory import (
    build_initial_hypotheses_system_prompt,
    build_initial_hypotheses_user_content,
    build_update_hypotheses_system_prompt,
    build_update_hypotheses_user_content,
    build_observation_system_prompt,
    build_observation_user_content,
    build_action_selection_system_prompt,
    build_action_selection_user_prompt,
)

log = logging.getLogger(__name__)

class AS66VisualMemoryAgent(AS66MemoryAgent):
    """
    An agent that uses multimodal context (images + text diffs) to manage
    hypotheses, mirroring the AS66MemoryAgent's three-call structure.

    Overrides:
    - `_initialize_memory`: Uses multimodal prompt (text+image)
    - `_update_memory`: Generates/saves images, updates memory file with image
      paths, and calls hypothesis update with multimodal context.
    - `_get_observation_text`: Renamed to `_get_observation_content` and
      made multimodal.
    - `choose_action`: Orchestrates the new multimodal calls.
    """
    MODEL = os.getenv("AGENT_MODEL_OVERRIDE", "gpt-5") # Requires a vision model
    IMAGE_DIR = Path("memory") / "images"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Ensure the per-game image directory exists
        self.game_image_dir = self.IMAGE_DIR / self.game_id
        self.game_image_dir.mkdir(parents=True, exist_ok=True)
        # memory/images/ is covered by memory/ in .gitignore

    def _generate_and_save_image(self, grid: List[List[int]]) -> Tuple[str, str]:
        """
        Generates a 16x16 color PNG, saves it, and returns its path and b64 data.

        Returns:
            (relative_path, b64_data_url)
        """
        try:
            png_bytes = ds16_png_bytes(grid)
            b64_data = base64.b64encode(png_bytes).decode('utf-8')
            
            # Create a unique filename for this state
            img_filename = f"{uuid.uuid4()}.png"
            relative_path = Path("images") / self.game_id / img_filename
            full_path = self.IMAGE_DIR / self.game_id / img_filename
            
            full_path.write_bytes(png_bytes)
            
            data_url = f"data:image/png;base64,{b64_data}"
            return str(relative_path), data_url
        except Exception as e:
            log.error(f"[{self.game_id}] Failed to generate/save image: {e}")
            return "ERROR_GENERATING_IMAGE", ""

    def _initialize_memory(self, initial_frame: FrameData) -> None:
        """API Call 1 (Setup): Generates initial hypotheses using an image."""
        log.info(f"[{self.game_id}] Initializing visual memory and hypotheses...")
        client = OpenAI()
        
        grid_3d = initial_frame.frame
        grid = downsample_4x4(grid_3d, take_last_grid=True, round_to_int=True) if self.DOWNSAMPLE else (grid_3d[-1] if grid_3d else [])

        # Generate the initial image
        img_path, img_b64_url = self._generate_and_save_image(grid)
        if not img_b64_url:
            log.error(f"[{self.game_id}] Cannot initialize memory, image generation failed.")
            self.hypotheses_content = "## Hypotheses\n\n(ERROR: Initial image generation failed.)"
            self._is_initialized = True
            return

        img_b64 = img_b64_url.split(",", 1)[1] # Get just the data

        sys_prompt = build_initial_hypotheses_system_prompt()
        user_content = build_initial_hypotheses_user_content(self.game_id, img_b64)
        
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
        
        # Add a note about the first image to the move history
        self.move_history_content = (
            "## Move History\n\n"
            f"**Initial State (Image):**\n"
            f"![Initial State]({img_path})\n\n"
            "---\n\n"
        )
        
        self._write_memory() # Saves both hypotheses and history
        self._is_initialized = True
        log.info(f"[{self.game_id}] Initial visual hypotheses generated and saved to {self.memory_file_path}")

    def _update_memory(self, prev_frame: FrameData, action: ActionInput, new_frame: FrameData) -> bool:
        """
        Updates the memory file with the latest move (images + diff) and
        revised hypotheses (using multimodal call).
        """
        client = OpenAI()
        
        prev_grid_3d = prev_frame.frame
        new_grid_3d = new_frame.frame
        
        prev_grid = downsample_4x4(prev_grid_3d, take_last_grid=True, round_to_int=True) if self.DOWNSAMPLE else (prev_grid_3d[-1] if prev_grid_3d else [])
        new_grid = downsample_4x4(new_grid_3d, take_last_grid=True, round_to_int=True) if self.DOWNSAMPLE else (new_grid_3d[-1] if new_grid_3d else [])

        state_hash = self._get_state_hash(prev_grid)
        
        # --- (Re-using action identifier logic from parent) ---
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
        # --- (End of re-used logic) ---

        state_action_tuple = (state_hash, action_identifier)

        if state_action_tuple in self.seen_state_actions:
            log.warning(f"[{self.game_id}] Repeated state-action pair detected: {action_identifier}. Applying penalty.")
            return True # Signal that this was a repeat

        self.seen_state_actions.add(state_action_tuple)

        is_level_up = new_frame.score > prev_frame.score
        diff = self._calculate_diff(prev_grid, new_grid)
        
        # Generate and save images for this step
        prev_img_path, prev_img_b64_url = self._generate_and_save_image(prev_grid)
        new_img_path, new_img_b64_url = self._generate_and_save_image(new_grid)

        # Build the text block for the markdown file
        entry_header = f"### Turn {len(self.seen_state_actions)}\n\n"
        entry_body = (
            f"**Action:** `{action_identifier}`\n\n"
            "**State Before (Image):**\n"
            f"![Before]({prev_img_path})\n\n"
            "**State After (Image):**\n"
            f"![After]({new_img_path})\n\n"
            "**Resulting Textual Diff:**\n"
            "```\n"
            f"{diff}\n"
            "```\n"
        )
        
        if is_level_up:
            history_entry = f"> **⭐ LEVEL UP! A new level begins below.**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        elif new_frame.state == GameState.GAME_OVER:
            history_entry = f"> **☠️ GAME OVER!**\n>\n{'> '.join((entry_header + entry_body).splitlines(True))}\n---\n\n"
        else:
            history_entry = f"{entry_header}{entry_body}\n---\n\n"

        if "(No moves recorded yet.)" in self.move_history_content or "Initial State" in self.move_history_content:
            self.move_history_content = "## Move History\n\n" + history_entry
        else:
            self.move_history_content += history_entry
            
        # This is the multimodal content for the *last move only*
        # We need the raw b64 data for the API call
        prev_img_b64 = prev_img_b64_url.split(",", 1)[1]
        new_img_b64 = new_img_b64_url.split(",", 1)[1]
        
        last_move_block_content = [
            {"type": "text", "text": f"**Action:** `{action_identifier}`\n\n**State Before (Image):**"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_img_b64}", "detail": "low"}},
            {"type": "text", "text": "\n\n**State After (Image):**"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{new_img_b64}", "detail": "low"}},
            {"type": "text", "text": f"\n\n**Resulting Textual Diff:**\n```\n{diff}\n```\n"},
        ]

        log.info(f"[{self.game_id}] Updating hypotheses based on new visual move...")
        sys_prompt = build_update_hypotheses_system_prompt()
        
        # Get the text-only history (markdown with file paths)
        text_memory_content = self._read_memory()
        
        user_content = build_update_hypotheses_user_content(text_memory_content, last_move_block_content)

        if is_level_up:
             special_instruction = (
                 "**IMPORTANT CONTEXT: A LEVEL UP just occurred.** The game board has changed for the new level. "
                 "Your primary task now is to **re-validate your current hypotheses against this new environment.** "
                 "Generate a new set of five hypotheses that reflect your understanding of this new level.\n\n"
             )
             # Prepend the special instruction
             user_content.insert(0, {"type": "text", "text": special_instruction})


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
        log.info(f"[{self.game_id}] Visual memory file updated.")
        return False # Signal that this was not a repeat

    def _get_observation_text(self, memory_content: str, ds16_grid: List[List[int]], score: int, step: int) -> str:
        """
        API Call 2: Calls the LLM with multimodal context to get a text observation.
        Overrides parent method.
        """
        client = OpenAI()
        
        # Generate the *current* state image
        img_path, img_b64_url = self._generate_and_save_image(ds16_grid)
        if not img_b64_url:
            log.error(f"[{self.game_id}] Cannot get observation, image generation failed.")
            return "ERROR: Could not generate current state image for observation."
        
        current_img_b64 = img_b64_url.split(",", 1)[1]

        sys_prompt = build_observation_system_prompt()
        # memory_content already contains the markdown history (with image paths)
        user_content = build_observation_user_content(memory_content, current_img_b64, score, step)

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
        log.info(f"[{self.game_id} | Step {step}] Visual Observation Rationale: {observation}")
        return observation
    
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Main logic loop, identical to parent but calls the overridden (multimodal)
        _initialize_memory and _update_memory, and the new
        _get_observation_content.
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

        memory_content = self._read_memory()
        grid_3d = latest_frame.frame
        grid = downsample_4x4(grid_3d, take_last_grid=True, round_to_int=True) if self.DOWNSAMPLE else (grid_3d[-1] if grid_3d else [])

        # Call our new multimodal observation function
        observation_text = self._get_observation_text(memory_content, grid, latest_frame.score, len(frames))
        
        if was_repeated:
            penalty_message = (
                "**PENALTY**: Your previous action was a repeat of an action you've already taken in this exact state, "
                "and it was ignored. You MUST choose a different action this turn to avoid getting stuck in a loop. "
                "Analyze your hypotheses and choose a different experiment or a move that has not been tried from this state before.\n\n---\n\n"
            )
            observation_text = penalty_message + observation_text
        
        # This function is inherited from the parent and works as-is (text-in, tool-out)
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