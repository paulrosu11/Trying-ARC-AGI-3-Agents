# agents/templates/as66/prompts_visual_memory.py
"""
Wraps the prompts from prompts_memory.py to create a new multimodal (image-based)
agent. It prepends a new system prompt header that defines the color mapping
and the new multimodal context format (images + text diffs).
"""
from __future__ import annotations
from typing import List, Optional

# Import the original text-based prompt builders
from .prompts_memory import (
    build_initial_hypotheses_system_prompt as original_initial_system,
    build_initial_hypotheses_user_prompt as original_initial_user,
    build_update_hypotheses_system_prompt as original_update_system,
    build_update_hypotheses_user_prompt as original_update_user,
    build_observation_system_prompt as original_observation_system,
    build_observation_user_prompt as original_observation_user,
    build_action_selection_system_prompt as original_action_system,
    build_action_selection_user_prompt as original_action_user,
)

# This is the color mapping from downsample.py, which generates the images
COLOR_MAPPING_HEADER = """
You are an expert game analyst. You will be playing a game represented by 16x16 color images and a textual matrix of ints.
Your goal is to form and test hypotheses about the game's rules. **you are not to make any tool calls**

**IMAGE AND COLOR MAPPING:**
All game states are provided as 16x16 images, where each color corresponds to a specific integer.
This mapping is static:
- 0: #FFFFFF (White)
- 1: #CCCCCC (Light Gray)
- 2: #999999 (Gray)
- 3: #666666 (Dim Gray)
- 4: #000000 (Black)
- 5: #202020 (Near Black)
- 6: #1E93FF (Blue)
- 7: #F93C31 (Red)
- 8: #FF851B (Orange)
- 9: #921231 (Maroon / Dark Red)
- 10: #88D8F1 (Light Blue)
- 11: #FFDC00 (Yellow)
- 12: #FF7BCC (Pink)
- 13: #4FCC30 (Light Green)
- 14: #2ECC71 (Green)
- 15: #7F3FBF (Purple)

**CONTEXT FORMAT:**
You will receive context as a mix of text and images.
- The **current state** is provided as a 16x16 image.
- **Move History** is provided as a sequence of:
    1. The Action taken (e.g., `ACTION3`).
    2. The 'State Before' image.
    3. The 'State After' image.
    4. A textual 'Resulting Diff', which lists changes in the underlying *integers*.

**TEXTUAL DIFFS:**
The textual diffs refer to cell coordinates. The coordinate system is **(row, column)**,
where (0, 0) is the **top-left** corner.
Example: "- Cell (8, 5): 6 -> 15" means the cell at row 8, column 5 changed
from integer 6 (Blue) to integer 15 (Purple).

Your analysis, hypotheses, and rationales must be based on this multimodal information. **AGAIN RESPOND WITH SIMPLE TEXT AND USE NO TOOLS**
---
"""

# --- Wrapped Initial Hypothesis Generation ---

def build_initial_hypotheses_system_prompt() -> str:
    """System prompt for generating the first set of hypotheses."""
    return COLOR_MAPPING_HEADER + original_initial_system()

def build_initial_hypotheses_user_content(game_id: str, initial_image_b64: str, detail: str = "low") -> List[dict]:
    """User prompt for the first hypotheses, now with an image."""
    return [
        {
            "type": "text",
            "text": f"Here is the initial state of the game board (game_id: {game_id}). Please generate five initial hypotheses about the game's rules, structure, and objectives, each with a concrete test case."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{initial_image_b64}",
                "detail": detail 
            }
        }
    ]

# --- Wrapped Hypothesis Update ---

def build_update_hypotheses_system_prompt() -> str:
    """System prompt for the hypothesis update step."""
    return COLOR_MAPPING_HEADER + original_update_system()

def build_update_hypotheses_user_content(memory_content: str, last_move_block: List[dict]) -> List[dict]:
    """
    User prompt for the hypothesis update step.
    'memory_content' is the text-based history (markdown).
    'last_move_block' is the multimodal content for the *most recent* turn.
    """
    content = [
        {
            "type": "text",
            "text": (
                "Here is the game memory so far, including your prior hypotheses.\n\n"
                f"{memory_content}\n\n"
                "---\n\n"
                "And here is the detailed breakdown of the **most recent turn**:\n"
            )
        }
    ]
    # Add the [Action text, Before Image, After Image, Diff text]
    # Note: The 'detail' level is already set in last_move_block by the agent
    content.extend(last_move_block)
    content.append(
        {
            "type": "text",
            "text": "\nAnalyze this new evidence and provide an updated list of five refined hypotheses."
        }
    )
    return content

# --- Wrapped Observation Step ---

def build_observation_system_prompt() -> str:
    """System prompt for the observation step."""
    return COLOR_MAPPING_HEADER + original_observation_system()

def build_observation_user_content(memory_content: str, current_image_b64: str, score: int, step: int, detail: str = "low") -> List[dict]:
    """User prompt for the observation step, with image."""
    return [
        {
            "type": "text",
            "text": (
                f"**Current Game Status:**\n"
                f"- Step: {step}\n"
                f"- Score: {score}\n\n"
                "**Current Board State (Image):**"
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{current_image_b64}",
                "detail": detail 
            }
        },
        {
            "type": "text",
            "text": (
                "\n\n**Full Game Memory (Text History):**\n"
                f"{memory_content}\n\n"
                "Follow your reasoning process and provide a detailed text analysis, concluding with your recommended action. Be precise with all coordinates. **you are not to make any tool calls** "
            )
        }
    ]

# --- Wrapped Action Selection Step ---

def build_action_selection_system_prompt() -> str:
    """System prompt for the final, tool-constrained action selection step."""
    # This prompt doesn't need the header, as it just executes the text rationale.
    return original_action_system()

def build_action_selection_user_prompt(observation_text: str) -> str:
    """User prompt providing the rationale to the action selection model."""
    # This is text-in, tool-out. No images needed.
    return original_action_user(observation_text)