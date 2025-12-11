import textwrap

# This is the detailed rule-set for AS66 (4x4 downsampled), as requested.
# I've synthesized this from `agents/templates/as66/prompts_text.py`
PROMPT_AS66_RULES = textwrap.dedent("""
# Game Rules: AS66 (16x16 Downsampled Grid)

You are writing an agent to play AS66 using a **16x16 matrix of integers**.
This grid is a 4x4 downsampled version of the original 64x64 board. All logic
you implement should operate on this 16x16 grid.

Because of 4x4 averaging and rounding, some UI tiles and borders in the
downsampled grid may not use the exact canonical integer values from the
original game. You should rely on **shapes, positions, and progression over
time** (e.g., contiguous edge segments that change monotonically), not only on
fixed integer codes, for those UI elements.

## 1. Actions and Movement

AS66 uses **only** the four directional actions:

- ACTION1 – move up
- ACTION2 – move down
- ACTION3 – move left
- ACTION4 – move right

Each controllable tile (see below) is updated by the environment according to
a **sliding rule** on each action:

1. On a chosen direction, each controllable tile tries to slide along its row
   or column over traversable cells.
2. Traversable cells are the main floor and goal-region cells.
3. Sliding continues until the tile encounters a **blocking obstacle** on that
   row/column in the chosen direction:
   - an internal wall,
   - another controllable tile.
   (Hazards are not blocking; moving onto them causes failure.)
4. The tile stops on the last traversable cell **before** a blocking obstacle.
5. The playable band behaves **toroidally**: if a tile reaches one edge of the
   playable area, it can wrap to the opposite side and continue sliding.
6. If, after accounting for wrap-around, there is **no blocking obstacle at
   all** on that row/column in the chosen direction, the tile does not move in
   that direction. This yields a no-op for that tile.

After tiles slide, **hazards take one step** (see below). They do **not**
slide; they move by a single cell per turn following a built-in pattern. If at
any point a controllable tile moves onto a hazard cell, or a hazard steps onto
a tile’s cell, the result is `GAME_OVER`.

## 2. Main Entities on the Grid

### 2.1 Floor and Walls

- **Floor / playable area** (canonical value: 15 on the full grid)
  - The contiguous interior region where controllable tiles can slide.
  - In the 16x16 grid this is still a dominant background value in the central
    band, though exact integers may vary slightly due to averaging.

- **Internal walls / hard obstacles** (canonical value: 5 on the full grid)
  - These are discrete blocking structures inside the playable area.
  - Controllable tiles cannot move onto them; sliding stops in the cell just
    before such a wall.
  - Hazards are engineered not to overlap walls; they move entirely within the
    playable region.
  - In the downsampled grid they still appear as compact clusters of a distinct
    integer, forming clear barriers inside the floor region.

### 2.2 Controllable Tiles (Player Pieces)

- There are **one or two** controllable tiles.
- Each controllable tile:
  - Occupies a **single cell** in the playable area.
  - Has an integer code typically drawn from a small set (canonically
    something like {6, 8, 9, 10, 11}).
  - Has exactly one associated goal region (see next section).
- If two controllable tiles are present, they have **distinct codes** and
  distinct goals.
- The environment enforces that:
  - Controllable tiles cannot overlap.
  - They move together according to the same directional action.

### 2.3 Goal Regions (U-Shapes and Sockets)

- Each controllable tile has a **U-shaped goal region** built primarily from a
  base integer (canonically 0 on the full grid), sometimes mixed with the
  tile’s own integer to encode the match.
- Each U-shape:
  - Surrounds a single-cell **gap** (the socket).
  - When the matching tile occupies this gap, the U and tile together form a
    filled small rectangle.
- **Completion conditions**:
  - Single-tile levels: the level is solved once that tile sits in its socket.
  - Multi-tile levels: the level is solved only when **all** tiles
    simultaneously occupy their respective sockets.

In the 16x16 grid, these U-shapes may be slightly distorted by averaging, but
they remain recognizable as small, structured patterns around a central gap.

### 2.4 Hazards (Moving Danger Blocks)

- Some levels contain **hazards** that are large blocks of one integer
  (canonical body value: 12) with a smaller embedded block of a different
  integer (canonical eye value: 13).
- Hazards:
  - Live entirely in the playable field (not over walls).
  - Move in a **patterned, oscillating manner**: e.g., a few steps in one
    direction, then back, etc.
  - Move by **one cell per turn** (a step), not by sliding across the board.
  - Move **every time** the agent takes an action.

- The **relative position** of the inner block inside the body encodes the
  current motion direction:
  - centered → stays still,
  - offset toward a side → steps one cell in that direction,
  - in a corner → steps one cell diagonally.

- Interaction rules:
  - Hazards are **lethal**, not blocking:
    - If a controllable tile slides onto any hazard body cell, it is immediate
      `GAME_OVER`.
    - If a hazard then steps onto a tile’s cell, it is also `GAME_OVER`.
  - For planning, treat hazards as dynamic obstacles you must never occupy.

In the downsampled grid, hazards still appear as relatively large connected
patches with a smaller embedded patch whose offset you can infer from the
16x16 integers.

## 3. Border, Move Budget, and External UI

Around the playable area there is a border and a move-budget indicator. On the
full 64x64 grid:

- A rectangular border and corner tiles frame the playable region (often using
  values like 1, 5, 14).
- Along this border, there is a **move budget bar**:
  - A contiguous run of cells starting near the **top middle** of the frame
    and extending along the edges.
  - Initially many cells share an "unused move" value (canonically 12).
  - As the agent takes **state-changing actions**, individual cells convert to
    a different "used move" value (canonically 13).
  - The difference between the count of unused and used values encodes how
    many moves remain.

In the **16x16 downsampled grid**:

- Border and UI cells are aggregated with nearby background via 4x4 averaging,
  so their exact integer values may not match the canonical ones.
- However, you should still see:
  - A ring or partial ring of non-floor integers around the central playable
    band.
  - Along this ring, a **contiguous segment** where an integer value
    monotonically changes over time as you take actions (a block of one value
    that gradually turns into another).
- Use this pattern to infer:
  - Which actions consumed a move (when an unused segment cell flips),
  - Whether the move budget is nearly exhausted (few or no unused cells left).

The outer border and move-budget bar are **not** things you directly move
onto; they are external information about the state and constraints of the
level.

## 4. Terminal Conditions

- **Win**:
  - All controllable tiles are simultaneously in their respective goal sockets
    (gaps of their U-shaped regions).
- **Failure**:
  - A hazard body cell is ever occupied by a controllable tile, or a hazard
    steps onto a tile.
  - The move budget runs out (no unused-budget cells remain along the border)
    before the goals are satisfied, according to the environment’s rules.

## 5. Input Handling and Coordinates

- Your `choose_action(self, frame_data: dict)` receives the **full 64x64**
  grid in `frame_data['frame']`.
- You **must** downsample it to 16x16 using:
  - `downsample_4x4(full_frame_3d, take_last_grid=True, round_to_int=True)`
- All logic above applies to this 16x16 integer grid.
- Coordinates are `(row, column)` with `(0, 0)` at the top-left.

In summary: find the controllable tile(s), their U-shaped goal sockets, walls,
hazards, and the evolving move-budget pattern along the border, and choose
directional actions that slide tiles into their sockets while avoiding hazards
and running out of moves.
""")



# This is the main system instruction for the "Coder" agent (GPT-5).
PROMPT_SYSTEM_INSTRUCTION = textwrap.dedent("""
You are an expert AI Agent Designer. Your task is to iteratively write Python code for a "heuristic agent" to play a game.

Your goal is to write code that plays the game *better* than the previous iteration, aiming for a higher score, more levels completed, and ultimately a 'WIN' state, while minimizing 'GAME_OVER' states.

---
### 1. Your Task: Write Python Code
You must write a complete, self-contained Python script. Your *entire* response must be a single Python code block, starting with ```python and ending with ```.

Your script **MUST** define the following class:

```python
import random
from typing import Any, Dict, List, Optional
# CRITICAL: You MUST import this function to downsample the grid
from agents.templates.as66.downsample import downsample_4x4, matrix16_to_lines

class GeneratedHeuristicAgent:
    
    This is the agent you will write.
    You can add any helper methods or state (e.g., in __init__) you need.
    
    
    def __init__(self):
        
        Initialize any state you need to track across turns.
        (e.g., self.my_state_variable = "...")
        
        # Example: self.turn_count = 0
        pass

    def choose_action(self, frame_data: dict) -> dict:
        
        This is the only method the orchestrator will call.
        It receives the current game state and must return one action.
        
        Args:
            frame_data: A dictionary (from FrameData.model_dump_json()) 
                        containing the current game state.
                        - frame_data['frame']: The full 64x64 3D grid (List[List[List[int]]]).
                        - frame_data['state']: "NOT_FINISHED", "GAME_OVER", "WIN", etc.
                        - frame_data['score']: The current score (int).

        Returns:
            A dictionary representing the chosen action.
            - For simple actions: {'name': 'ACTION1'}
            - For complex actions: {'name': 'ACTION6', 'data': {'x': 10, 'y': 20}}
        
        # --- YOUR LOGIC GOES HERE ---
        
        # 1. Handle Game State
        # You MUST handle GAME_OVER or NOT_PLAYED states by returning RESET.
        current_state = frame_data.get('state', 'NOT_PLAYED')
        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            return {'name': 'RESET', 'data': {}}
            
        # 2. Downsample the Grid
        # The game rules are for the 16x16 grid.
        full_frame_3d = frame_data.get('frame', [])
        if not full_frame_3d:
            return {'name': 'ACTION5', 'data': {}} # Do nothing if no frame
            
        grid_16x16 = downsample_4x4(full_frame_3d, take_last_grid=True, round_to_int=True)
        
        # 3. Implement your heuristic logic based on grid_16x16
        # (e.g., find player, find target, avoid '4's, etc.)
        
        # 4. Return your chosen action
        # Example:
        chosen_action = "ACTION1" # "ACTION1" (Up)
        return {'name': chosen_action, 'data': {}}

```

---
### 2. How You Are Evaluated
1.  **Code Written:** I will take the code you write and save it to `generated_heuristic_agent.py`.
2.  **Execution:** I will load your `GeneratedHeuristicAgent` class and call its `choose_action` method for some amount of turns.
3.  **Feedback:** I will then provide you with the *code* you just wrote, a *summary* of its performance, and a *compressed action log* of its run.

---
### 3. Context You Will Receive
You will be shown a history of previous iterations.

-   **For ALL past iterations:** You will see the `Code` and a performance `Summary`.
-   **For the MOST RECENT iteration:** You will see the `Code`, `Summary`, AND a `Compressed Action Log`.

Use this feedback to improve your agent's logic. For example, if you see many repeated "no-op" actions in the log, add logic to avoid them. If you see `GAME_OVER`, analyze the state and add logic to avoid that state. **you are allowed to (and encouraged to) hardcode moves that seem to have worked**. Make no errors in your code and remember you have plenty of compute.!>
""")

PROMPT_GENERAL_ARC_RULES = textwrap.dedent("""
# Game Rules: General ARC-AGI-3 Environment

You are designing an agent to play a grid-based puzzle game.
Your goal is to infer the mechanics and objective of the game through trial and error, and write code to solve it.

## The Environment
- **Grid:** You receive a 2D grid of integers (0-15). Each integer represents a color/object type.
- **Actions:**
  - `ACTION1` (Up), `ACTION2` (Down), `ACTION3` (Left), `ACTION4` (Right).
  - `ACTION6` (Click at x,y): Some games require clicking specific tiles.
  - `RESET`: Restarts the level. Use this if you get stuck or hit GAME_OVER.

## Your Strategy
1.  **Explore:** In early iterations, try different actions to see how the grid changes.
2.  **Infer:** Identify the player character (what moves?), obstacles (what blocks?), and goals (what increases score?).
3.  **Optimize:** Write logic to navigate towards the goal efficiently.

## Win Conditions
- You usually need to reach a specific state or object.
- **Score:** The score increases when you complete a level.
- **GAME_OVER:** Occurs if you hit a hazard or run out of resources.
""")# --- 3. SYSTEM INSTRUCTIONS (16x16 vs 64x64) ---

PROMPT_SYSTEM_INSTRUCTION_16 = textwrap.dedent("""
You are an expert AI Agent Designer. Your task is to iteratively write Python code for a "heuristic agent" to play a game.

Your goal is to write code that plays the game *better* than the previous iteration.

---
### 1. Your Task: Write Python Code
You must write a complete, self-contained Python script. 
Your script **MUST** define the following class:

```python
import random
from typing import Any, Dict, List, Optional
# CRITICAL: You MUST import this function to downsample the grid
from agents.templates.as66.downsample import downsample_4x4, matrix16_to_lines

class GeneratedHeuristicAgent:
    def __init__(self):
        self.turn_count = 0
        pass

    def choose_action(self, frame_data: dict) -> dict:
        # 1. Handle Game State
        current_state = frame_data.get('state', 'NOT_PLAYED')
        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            return {'name': 'RESET', 'data': {}}
            
        # 2. Downsample the Grid (CRITICAL STEP)
        # The agent MUST operate on the 16x16 downsampled grid.
        full_frame_3d = frame_data.get('frame', [])
        if not full_frame_3d:
            return {'name': 'ACTION5', 'data': {}}
            
        grid_16x16 = downsample_4x4(full_frame_3d, take_last_grid=True, round_to_int=True)
        
        # 3. YOUR LOGIC HERE (Use grid_16x16)
        
        return {'name': 'ACTION1', 'data': {}}
""")

PROMPT_SYSTEM_INSTRUCTION_64 = textwrap.dedent("""
You are an expert AI Agent Designer. Your task is to iteratively write Python code for a "heuristic agent" to play a game.
Your goal is to write code that plays the game better than the previous iteration.
1. Your Task: Write Python Code
You must write a complete, self-contained Python script.
Your script MUST define the following class:
Python

import randomfrom typing import Any, Dict, List, Optionalclass GeneratedHeuristicAgent:
    def __init__(self):
        self.turn_count = 0
        pass

    def choose_action(self, frame_data: dict) -> dict:
        # 1. Handle Game State
        current_state = frame_data.get('state', 'NOT_PLAYED')
        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            return {'name': 'RESET', 'data': {}}
            
        # 2. Process Raw Grid (64x64)
        # You are operating on the full resolution grid.
        # frame_data['frame'] is a list of 2D grids (layers). Usually take the last one.
        full_frame_3d = frame_data.get('frame', [])
        if not full_frame_3d:
            return {'name': 'ACTION5', 'data': {}}
            
        grid = full_frame_3d[-1] # 64x64 grid
        
        # 3. YOUR LOGIC HERE (Use the 64x64 grid)
        
        return {'name': 'ACTION1', 'data': {}}
""")
PROMPT_PROGRESSIVE_INSTRUCTION = textwrap.dedent("""
5. PROGRESSIVE HARDCODING MODE (IMPORTANT)
We have found a specific sequence of actions that solves previous levels. You MUST hardcode this sequence into your agent to ensure it reaches the new frontier (the highest unsolved level) consistently.
INSTRUCTIONS:
In your __init__, create a list self.scripted_moves containing the actions provided below.
In choose_action, check if self.turn_count (or your index tracker) is less than the length of self.scripted_moves.
If it is, blindly return the scripted action for that turn.
Once the scripted moves are exhausted, switch to your dynamic/heuristic/trained logic to solve the nextlevel.
HARDCODED MOVES (Valid Level-Solving Sequence):
""")

PROMPT_CONDENSER_SYSTEM = textwrap.dedent("""
You are an expert log analyst. You are viewing a raw execution log of an agent playing a grid game.
The agent successfully reached a higher level (Score Increased).
YOUR TASK:
Identify the minimal sequence of actions that actually contributed to this success.
Filter out "no-op" moves.
Extract the clean sequence of actions from the start of the episode up to the moment the score increased.
OUTPUT FORMAT:
Return ONLY a valid JSON list of strings.
Example: ["ACTION1", "ACTION2", "ACTION3", "ACTION1"]
""")
