# agents/templates/as66/prompts_configurable.py
from __future__ import annotations
from typing import List, Optional, Dict
import os

# Import the utility to convert a matrix to an ASCII string
from .downsample import matrix16_to_lines

"""
This file is an adaptation of prompts_text.py.

It provides configurable prompt builders for an agent that can receive
board state as:
1.  Text Only: (identical to AS66GuidedAgent)
2.  Image Only: (prompt text omits the ASCII grid)
3.  Text and Image: (prompt text includes ASCII grid, and also notes the image)

The system prompts are identical across all modes, as requested.
The user prompts are modified only to remove/add the ASCII
grid and clarify the input format.
"""

# -------------------------
# SYSTEM PROMPTS (Identical for all modes)
# -------------------------

# This is the full, detailed system prompt from your 'prompts_text.py'
_AS66_OBS_SYSTEM = (
    "You are playing a game which is represented by a 16×16 matrix of integer codes. "
    "This matrix is the current state after a reset or after your most recent move. "
    "Your task is to observe the position and analyze the potential benefits and drawbacks of each possible move.\n\n"
    "Movement model:\n"
    "• There is one main movable integer. Its specific value may vary across levels, but it will be unique among the surrounding integers (it could even be a small square of 8/9s). "
    "  There may also be multiple movable integers; they remain distinct from the rest of the board and move together under the same command.\n"
    "• When you choose a direction (Up, Down, Left, Right), each movable integer slides as far as possible in that direction. "
    "  Sliding wraps across the board edges when unobstructed (after passing beyond one edge, it reappears from the opposite edge and continues). "
    "  Sliding stops as soon as an obstacle blocks further motion.\n"
    "If you move in a direction with no obstacles ever, you move back to where you started. For example, if one moves up and the entire wraparound has no obstacles, the game state does not change due to your action and you remain stationary.\n \n"
    "Obstacles and board semantics:\n"
    "• 4 are walls. Sliding stops adjacent to a 4 and cannot overlap 4s.\n"
    "• 15 are background cells that constitute the playable area (free to traverse and occupy).\n"
    "• The board is typically delimited by boundaries such as 1 or 14. You can generally localize the playable field by the region filled with 15s.\n"
    "• 0 are target cells. They form a U shape that can be viewed as a 2×3 rectangle with the center cell removed. "
    "  Your objective is to navigate the movable integer(s) into this space to complete the U by filling its cavity. "
    "  The 0 region also interrupts sliding (you stop upon reaching it when the motion would otherwise continue).\n"
    "• You may observe an increasing number of 6 near the top; this indicates the consumption of available moves in this downsampled 16×16 view.\n\n"
    "Hostile entity (avoid at all costs but note frequently there is not one. They are large 3 by 3 and easy to spot):\n"
    "• Larger composite blocks consisting of 8 and 9 indicate an enemy. If any movable integer collides with this enemy, it is game over.\n"
    "• The position of 9 within that 8/9 block indicates the direction in which this enemy will step per your move (one tile, row, or column, potentially diagonally depending on level behavior). "
    "  If the 9 is centered and the entity is stationary, it remains stationary. Use history to infer whether it moves.\n\n"
    "Multiple movers:\n"
    "• If multiple movable integers exist, they all move together in the same chosen direction. Avoid any collision with the hostile entity while advancing the objective of completing the U.\n\n"
    "Target matching with multiple movers:\n"
    "• On later levels, there may be multiple target U regions that expect specific movers. The 0 U may include a single cell of the intended mover’s code to indicate which mover should fill that particular U.\n\n"
    "What to produce during observation (rationale only):\n"
    "• Identify the locations of the movable integer(s) and all relevant structures (0 region, 4 walls, 15 background, 1/14 boundaries, any 8/9 enemy mass). "
    "• For each direction (Up, Down, Left, Right), reason carefully about full wrap-around sliding: what blocking elements will be met, what will be the final resting locations, and how these outcomes change proximity/alignment to the U cavity. "
    "• Consider the enemy’s response (8/9), including whether a move would cause immediate collision or a forced collision on the subsequent step. "
    "• Conclude which direction best progresses toward completing the 2×3 cavity in the 0 region while avoiding risk. "
    "This is a text-only analysis turn; do not name or call an action tool here."
    "**THE MOST IMPORTANT THING TO KEEP IN MIND IS THE RESULTS OF YOUR PAST ACTIONS AND PREVIOUSLY WHAT STATE CHANGE CAME FROM THEM, DO NOT REPREAT ACTIONS THAT CHANGED NOTHING!"
)

# This is the action-phase system prompt from your 'prompts_text.py'
_AS66_ACT_SYSTEM = (
    "Select exactly one move by calling a single tool. Do not include prose.\n"
    "Available tools:\n"
    "• ACTION1 = Up\n"
    "• ACTION2 = Down\n"
    "• ACTION3 = Left\n"
    "• ACTION4 = Right"
)

def build_observation_system_configurable() -> str:
    """Returns the identical, word-for-word system prompt."""
    return _AS66_OBS_SYSTEM

def build_action_system_configurable() -> str:
    """Returns the identical, word-for-word system prompt."""
    return _AS66_ACT_SYSTEM

# -------------------------
# USER PROMPTS (Configurable)
# -------------------------

def build_observation_user_configurable(
    ds16: List[List[int]],
    score: int,
    step: int,
    input_mode: str,
) -> str:
    """
    Builds the user prompt for the observation phase based on the input mode.
    """
    
    # Base prompt text (identical to the original)
    rationale_prompt = (
        "Rationale:\n"
        "  • Identify the movable integer(s) and relevant structures (0/4/15/1/14 and any 8/9 enemy mass).\n"
        "  • For Up, Down, Left, Right: fully simulate wrap-around sliding, state blockers, and final landing positions.\n"
        "  • Explain how each landing affects progress toward completing the 2×3 U cavity (0 region) and whether the enemy’s response threatens collision.\n"
        "  • Conclude which direction is best and why. Do not output an action here."
    )
    
    header = f"Score: {score}\nStep: {step}\n"
    
    if input_mode == "text_only":
        # Mode 1: Text Only (Identical to AS66GuidedAgent)
        matrix = matrix16_to_lines(ds16)
        return (
            f"{header}"
            "Matrix 16x16 (integer codes):\n"
            f"{matrix}\n\n"
            f"{rationale_prompt}"
        )
        
    elif input_mode == "image_only":
        # Mode 2: Image Only
        # Omits the ASCII matrix, clarifies input format.
        return (
            f"{header}\n"
            "The 16x16 board state is provided as an attached image.\n\n"
            f"{rationale_prompt}"
        )
        
    elif input_mode == "text_and_image":
        # Mode 3: Text and Image
        # Includes ASCII matrix, and clarifies image is also attached.
        matrix = matrix16_to_lines(ds16)
        return (
            f"{header}"
            "Matrix 16x16 (integer codes):\n"
            f"{matrix}\n\n"
            "The board state is also provided as an attached image for visual reference.\n\n"
            f"{rationale_prompt}"
        )
    
    else:
        # Fallback to text_only
        matrix = matrix16_to_lines(ds16)
        return (
            f"{header}"
            "Matrix 16x16 (integer codes):\n"
            f"{matrix}\n\n"
            f"{rationale_prompt}"
        )


def build_action_user_configurable(
    ds16: List[List[int]],
    last_observation_text: str,
    input_mode: str,
) -> str:
    """
    Builds the user prompt for the action phase based on the input mode.
    """
    
    header = "Choose the best single move as a function call.\n"
    
    if input_mode == "text_only":
        # Mode 1: Text Only (Identical to AS66GuidedAgent)
        matrix = matrix16_to_lines(ds16)
        return (
            f"{header}"
            "Matrix 16x16 (integer codes):\n"
            f"{matrix}\n\n"
            "Previous observation summary:\n"
            f"{last_observation_text}\n"
        )
        
    elif input_mode == "image_only":
        # Mode 2: Image Only
        # Omits the ASCII matrix, clarifies input format.
        return (
            f"{header}\n"
            "The 16x16 board state is provided as an attached image.\n\n"
            "Previous observation summary:\n"
            f"{last_observation_text}\n"
        )
        
    elif input_mode == "text_and_image":
        # Mode 3: Text and Image
        # Includes ASCII matrix, and clarifies image is also attached.
        matrix = matrix16_to_lines(ds16)
        return (
            f"{header}"
            "Matrix 16x16 (integer codes):\n"
            f"{matrix}\n\n"
            "The board state is also provided as an attached image for visual reference.\n\n"
            "Previous observation summary:\n"
            f"{last_observation_text}\n"
        )
        
    else:
        # Fallback to text_only
        matrix = matrix16_to_lines(ds16)
        return (
            f"{header}"
            "Matrix 16x16 (integer codes):\n"
            f"{matrix}\n\n"
            "Previous observation summary:\n"
            f"{last_observation_text}\n"
        )