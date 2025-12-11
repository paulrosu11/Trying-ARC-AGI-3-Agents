import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from agents.templates.as66.downsample import downsample_4x4, matrix16_to_lines


class GeneratedHeuristicAgent:
    """
    Iteration 5 – Persistent tile tracking + sliding heuristic

    Incremental improvements over Iteration 4:

    - Persistently tracks discovered controllable-tile values across turns:
        * Once a tile value is detected by the rarity heuristic, we remember
          it in `self.known_tile_values`.
        * On subsequent turns we *first* search directly for these values in
          the interior, even if their global frequency has changed due to
          level transitions, hazards, or UI changes.
    - Falls back to the earlier rarity-based detection when no known-value
      tiles are visible, so new levels / new tile types can still be found.
    - Keeps all previous improvements:
        * Sliding simulator with toroidal wrap-around.
        * Hazard avoidance (11, 12, 13 blocked).
        * Basic stall/oscillation avoidance.
    """

    def __init__(self):
        self.turn_count: int = 0

        # For stall detection
        self.last_action: Optional[str] = None
        self.last_tile_pos: Optional[Tuple[int, int]] = None
        self.repeat_count: int = 0

        # Values that have been identified as controllable tiles within this
        # episode (1–2 occurrences, not walls/UI/hazards/floor/goals).
        self.known_tile_values: set[int] = set()

    # ---------- Utility helpers on 16x16 grids ----------

    @staticmethod
    def _find_floor_value(grid: List[List[int]]) -> int:
        """
        Estimate main floor value as the most frequent value
        in a central window (to avoid borders/UI).
        """
        vals: List[int] = []
        for r in range(2, 14):
            for c in range(2, 14):
                vals.append(grid[r][c])
        if not vals:
            flat = [v for row in grid for v in row]
            return Counter(flat).most_common(1)[0][0]
        return Counter(vals).most_common(1)[0][0]

    @staticmethod
    def _find_goal_cells(grid: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Identify potential goal-region cells as value 0 in interior.
        """
        goals: List[Tuple[int, int]] = []
        for r in range(2, 14):
            for c in range(2, 14):
                if grid[r][c] == 0:
                    goals.append((r, c))
        return goals

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ----- Tile detection helpers -----

    def _find_tiles_by_known_values(
        self,
        grid: List[List[int]],
    ) -> List[Tuple[int, int, int]]:
        """
        First-pass tile detection: look explicitly for any cell in the interior
        whose value is one of the remembered `known_tile_values`.

        Returns list of (row, col, value).
        """
        if not self.known_tile_values:
            return []

        tiles: List[Tuple[int, int, int]] = []
        for r in range(2, 14):
            for c in range(2, 14):
                v = grid[r][c]
                if v in self.known_tile_values:
                    tiles.append((r, c, v))
        return tiles

    def _find_controllable_tiles_by_rarity(
        self,
        grid: List[List[int]],
        floor_val: int,
    ) -> List[Tuple[int, int, int]]:
        """
        Second-pass (fallback) tile detection using rarity.

        Heuristic:
        - Ignore clearly non-playable / UI / wall / hazard values.
        - Among remaining interior cells, look for values that appear only a
          small number of times (1–2). These are likely player tiles.
        Returns list of (row, col, value).
        """
        # Values we consider *not* tiles:
        # - 0: goal region
        # - floor_val: main floor
        # - 1,3,4,5,6,7,14: borders, walls, UI bars
        # - 11,12,13: hazards (eye/body variants)
        non_tile_vals = {0, floor_val, 1, 3, 4, 5, 6, 7, 11, 12, 13, 14}

        counts: Counter = Counter()
        coords_by_val: Dict[int, List[Tuple[int, int]]] = {}

        for r in range(2, 14):
            for c in range(2, 14):
                v = grid[r][c]
                if v in non_tile_vals:
                    continue
                counts[v] += 1
                coords_by_val.setdefault(v, []).append((r, c))

        tiles: List[Tuple[int, int, int]] = []
        for v, cnt in counts.items():
            # We expect 1–2 occurrences per tile type; higher counts are
            # probably decorative regions, hazards, or part of goals.
            if 1 <= cnt <= 2:
                for (r, c) in coords_by_val[v]:
                    tiles.append((r, c, v))

        # Update persistent knowledge of tile values
        for _, _, v in tiles:
            self.known_tile_values.add(v)

        return tiles

    @staticmethod
    def _build_traversable_mask(
        grid: List[List[int]],
        floor_val: int,
    ) -> List[List[bool]]:
        """
        Traversable cells: main floor and goal region (0).
        Hazards and walls are treated as non-traversable.
        """
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        traversable: List[List[bool]] = [[False] * cols for _ in range(rows)]

        # Values we explicitly disallow walking through:
        # walls/UI: 1,3,4,5,6,7,14
        # hazards (eye + body): 11,12,13
        blocked = {1, 3, 4, 5, 6, 7, 11, 12, 13, 14}

        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v in blocked:
                    traversable[r][c] = False
                elif v == floor_val or v == 0:
                    traversable[r][c] = True
                else:
                    # Other interior values (possible tiles) are *not*
                    # traversable for sliding purposes (they block).
                    traversable[r][c] = False

        return traversable

    @staticmethod
    def _simulate_slide(
        start: Tuple[int, int],
        direction: Tuple[int, int],
        traversable: List[List[bool]],
    ) -> Tuple[int, int]:
        """
        Simulate AS66-style sliding with toroidal wrap-around.

        - Slides along the given direction over traversable cells.
        - Stops just before a non-traversable cell (wall, other tile, hazard).
        - Wraps around edges (toroidal).
        - If after a full wrap-around there is no blocking obstacle at all
          on that line, returns the original position (no-op).
        """
        rows = len(traversable)
        cols = len(traversable[0]) if rows else 0
        r, c = start
        dr, dc = direction
        steps = 0

        while True:
            nr = (r + dr) % rows
            nc = (c + dc) % cols
            if not traversable[nr][nc]:
                # Next cell is non-traversable -> stop before it
                break
            r, c = nr, nc
            steps += 1
            if steps > max(rows, cols):
                # Pure wrap-around with no blocker -> treated as no-op
                return start
        return (r, c)

    # ---------- Main decision function ----------

    def choose_action(self, frame_data: dict) -> dict:
        # 1. Handle game state / episode reset
        current_state = frame_data.get("state", "NOT_PLAYED")
        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            self.turn_count = 0
            self.last_action = None
            self.last_tile_pos = None
            self.repeat_count = 0
            self.known_tile_values.clear()
            return {"name": "RESET", "data": {}}

        self.turn_count += 1

        # 2. Downsample 64x64 -> 16x16
        full_frame_3d = frame_data.get("frame", [])
        if not full_frame_3d:
            return {
                "name": random.choice(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]),
                "data": {},
            }

        try:
            grid16 = downsample_4x4(
                full_frame_3d, take_last_grid=True, round_to_int=True
            )
        except Exception:
            # If we can't see the board, at least keep moving
            return {
                "name": random.choice(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]),
                "data": {},
            }

        # 3. Infer floor, goals, and controllable tiles
        floor_val = self._find_floor_value(grid16)
        goals = self._find_goal_cells(grid16)
        traversable = self._build_traversable_mask(grid16, floor_val)

        # 3a. Try to find tiles by previously known-value first
        tiles = self._find_tiles_by_known_values(grid16)

        # 3b. If none found, fall back to rarity-based discovery
        if not tiles:
            tiles = self._find_controllable_tiles_by_rarity(grid16, floor_val)

        if not tiles:
            # No visible controllable tiles – either level is in transition
            # or our detection failed. Use mild exploration to avoid
            # pathological oscillation but keep acting.
            cycle = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
            if self.last_action in cycle:
                idx = (cycle.index(self.last_action) + 1) % 4
                action_name = cycle[idx]
            else:
                action_name = random.choice(cycle)
            self.last_action = action_name
            self.last_tile_pos = None
            self.repeat_count = 0
            return {"name": action_name, "data": {}}

        # Use first detected tile as primary (most levels start with 1 tile)
        tile_r, tile_c, _ = tiles[0]
        tile_pos = (tile_r, tile_c)

        if not goals:
            # No clear goals; just explore in a gentle cycle
            cycle = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
            if self.last_action in cycle:
                idx = (cycle.index(self.last_action) + 1) % 4
                action_name = cycle[idx]
            else:
                action_name = random.choice(cycle)
            self.last_action = action_name
            self.last_tile_pos = tile_pos
            self.repeat_count = 0
            return {"name": action_name, "data": {}}

        # 4. Choose direction that best reduces distance to nearest goal
        nearest_goal = min(goals, key=lambda g: self._manhattan(tile_pos, g))
        base_dist = self._manhattan(tile_pos, nearest_goal)

        directions = {
            "ACTION1": (-1, 0),  # up
            "ACTION2": (1, 0),   # down
            "ACTION3": (0, -1),  # left
            "ACTION4": (0, 1),   # right
        }

        best_actions: List[str] = []
        best_dist = base_dist

        # Stall-avoidance: if we keep repeating the same action without
        # effective movement, temporarily forbid that action.
        forbidden_action: Optional[str] = None
        if self.repeat_count >= 4 and self.last_action in directions:
            forbidden_action = self.last_action

        for action_name, (dr, dc) in directions.items():
            if action_name == forbidden_action:
                continue

            end_pos = self._simulate_slide(tile_pos, (dr, dc), traversable)
            new_dist = self._manhattan(end_pos, nearest_goal)

            if new_dist < best_dist:
                best_dist = new_dist
                best_actions = [action_name]
            elif new_dist == best_dist:
                best_actions.append(action_name)

        if best_actions:
            chosen = random.choice(best_actions)
        else:
            # No direction improves distance (or all were forbidden);
            # choose any non-forbidden direction.
            candidate_actions = list(directions.keys())
            if forbidden_action in candidate_actions and len(candidate_actions) > 1:
                candidate_actions.remove(forbidden_action)
            chosen = random.choice(candidate_actions)

        # 5. Update stall tracking
        end_pos = self._simulate_slide(tile_pos, directions[chosen], traversable)
        if self.last_tile_pos == end_pos and self.last_action == chosen:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_tile_pos = end_pos
        self.last_action = chosen

        return {"name": chosen, "data": {}}