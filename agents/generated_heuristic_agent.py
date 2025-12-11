import random
from typing import Any, Dict, List, Optional, Tuple

# CRITICAL: You MUST import this function to downsample the grid
from agents.templates.as66.downsample import downsample_4x4, matrix16_to_lines

Coord = Tuple[int, int]


class GeneratedHeuristicAgent:
    """
    Progressive-hardcoded + improved goal-chasing / hazard-avoiding agent.

    - First: blindly executes the provided scripted sequence at the start of
      each episode (PROGRESSIVE HARDCODING MODE).

    - After the script:
        * Downsamples the grid to 16x16.
        * Infers the floor value (usually 15).
        * Detects the controllable tile(s) by RARE value patterns among
          {6, 8, 9, 10, 11} instead of treating all of them as players.
        * Detects the goal/socket region from 0-valued cells.
        * Simulates sliding moves toward the goal and scores directions by:
              - must not slide through/onto hazard values (12/13),
              - must change position (filter no-ops),
              - reduce distance to goal,
              - and keep a good distance from hazards (avoid standing next to
                hazard clusters when possible).
        * When no good direction is found, falls back to a mild
          anti-hammering random exploration.
    """

    def __init__(self):
        # Turn counter within current episode
        self.turn_count: int = 0

        # REQUIRED: exact scripted sequence (progressive hardcoding mode)
        self.scripted_moves: List[str] = [
            "ACTION2", "ACTION3", "ACTION2", "ACTION3",
            "ACTION3", "ACTION2", "ACTION3", "ACTION1",
            "ACTION4", "ACTION2", "ACTION3", "ACTION3",
            "ACTION3", "ACTION2", "ACTION3", "ACTION1",
            "ACTION4", "ACTION2", "ACTION3", "ACTION1",
            "ACTION3", "ACTION3", "ACTION2", "ACTION3",
            "ACTION1", "ACTION4", "ACTION3", "ACTION3",
            "ACTION3", "ACTION3", "ACTION2", "ACTION3",
            "ACTION3", "ACTION3", "ACTION3", "ACTION3",
            "ACTION3", "ACTION1", "ACTION4",
        ]

        # Last action and repetition tracking (for anti-hammering)
        self.last_action_name: Optional[str] = None
        self.last_action_repeat_count: int = 0

        # RNG for post-script exploration
        self.rng = random.Random(2026)

    # ---------------------------------------------------------------------
    # Reset helpers
    # ---------------------------------------------------------------------

    def _reset_episode_state(self) -> None:
        """Reset all state that should be cleared on RESET / new episode."""
        self.turn_count = 0
        self.last_action_name = None
        self.last_action_repeat_count = 0

    # ---------------------------------------------------------------------
    # Grid analysis helpers
    # ---------------------------------------------------------------------

    def _infer_floor_value(self, grid: List[List[int]]) -> Optional[int]:
        """
        Infer the dominant floor value from the interior of the grid.

        - Prefer 15 if present; otherwise, take the most frequent interior value.
        """
        if not grid or not grid[0]:
            return None
        n, m = len(grid), len(grid[0])

        counts: Dict[int, int] = {}
        for r in range(1, n - 1):
            for c in range(1, m - 1):
                v = grid[r][c]
                counts[v] = counts.get(v, 0) + 1

        if not counts:
            return None

        if 15 in counts:
            return 15
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _find_players_by_value(
        self,
        grid: List[List[int]],
        floor_value: Optional[int],
    ) -> List[Coord]:
        """
        Detect controllable tile(s) by value, using a *rarity-based* heuristic.

        - Candidate values: {6, 8, 9, 10, 11}.
        - We count occurrences of each candidate in the interior.
          Values that occur very frequently are likely UI / decorative,
          while the true player pieces are usually rare small clusters.
        - We select cells belonging to the *rarest* candidate value(s).
        """
        if not grid or not grid[0]:
            return []
        n, m = len(grid), len(grid[0])

        candidate_values = {6, 8, 9, 10, 11}
        counts: Dict[int, int] = {}

        # First pass: count interior occurrences of candidate values
        for r in range(1, n - 1):
            for c in range(1, m - 1):
                v = grid[r][c]
                if v in candidate_values and v != floor_value:
                    counts[v] = counts.get(v, 0) + 1

        if not counts:
            return []

        # Find minimal frequency among candidates
        min_count = min(counts.values())

        # Allow a small band above min_count so we don't overfit to a single cell
        # (if min_count is large, this still groups comparable frequencies).
        rare_threshold = min_count + 1

        rare_values = {v for v, cnt in counts.items() if cnt <= rare_threshold}
        if not rare_values:
            rare_values = set(counts.keys())

        result: List[Coord] = []
        for r in range(1, n - 1):
            for c in range(1, m - 1):
                v = grid[r][c]
                if v in rare_values and v != floor_value:
                    result.append((r, c))

        return result

    def _player_centroid(self, players: List[Coord], grid: List[List[int]]) -> Coord:
        """
        Get a single representative player position, as integer (row, col).
        If no players given, fall back to rough board center.
        """
        n, m = len(grid), len(grid[0])
        if not players:
            return (n // 2, m // 2)

        sr = sum(r for r, _ in players)
        sc = sum(c for _, c in players)
        cr = int(round(sr / len(players)))
        cc = int(round(sc / len(players)))
        cr = max(0, min(n - 1, cr))
        cc = max(0, min(m - 1, cc))
        return (cr, cc)

    def _find_goal_centroid(self, grid: List[List[int]]) -> Optional[Tuple[float, float]]:
        """
        Approximate goal/socket location as centroid of all 0-valued interior cells.
        If no zeros exist, return None.
        """
        if not grid or not grid[0]:
            return None
        n, m = len(grid), len(grid[0])

        rows: List[int] = []
        cols: List[int] = []
        for r in range(1, n - 1):
            for c in range(1, m - 1):
                if grid[r][c] == 0:
                    rows.append(r)
                    cols.append(c)

        if not rows:
            return None
        return float(sum(rows)) / len(rows), float(sum(cols)) / len(cols)

    # ---------------------------------------------------------------------
    # Movement & hazard modeling
    # ---------------------------------------------------------------------

    def _is_hazard(self, v: int) -> bool:
        """
        Treat canonical hazard body/eye values 12 and 13 as lethal.
        """
        return v in (12, 13)

    def _collect_hazard_cells(self, grid: List[List[int]]) -> List[Coord]:
        """Return list of hazard cell coordinates in the interior."""
        if not grid or not grid[0]:
            return []
        n, m = len(grid), len(grid[0])
        hazards: List[Coord] = []
        for r in range(1, n - 1):
            for c in range(1, m - 1):
                if self._is_hazard(grid[r][c]):
                    hazards.append((r, c))
        return hazards

    def _nearest_hazard_distance_sq(
        self,
        pos: Coord,
        hazards: List[Coord],
    ) -> int:
        """
        Squared distance to the nearest hazard cell.
        If no hazards, return a large number (effectively "safe").
        """
        if not hazards:
            return 10_000  # bigger than any possible 16x16 distance^2
        pr, pc = pos
        best = 10_000
        for hr, hc in hazards:
            d2 = (hr - pr) ** 2 + (hc - pc) ** 2
            if d2 < best:
                best = d2
        return best

    def _simulate_slide(
        self,
        grid: List[List[int]],
        start: Coord,
        direction: str,
        floor_value: Optional[int],
    ) -> Tuple[Coord, bool]:
        """
        Simulate sliding from `start` in `direction` on the 16x16 grid.

        Traversable values:
          - floor_value,
          - floor_value ± 1,
          - 0 (goal area, U-shapes).

        Blocking:
          - anything not traversable and not hazard.

        Hazard:
          - cells with values 12 or 13 are lethal. We mark the move as hazardous
            if we would enter such a cell while sliding.

        Returns:
            (end_coord, hit_hazard)
        """
        if not grid or not grid[0]:
            return start, False

        n, m = len(grid), len(grid[0])
        r, c = start

        if direction == "ACTION1":   # up
            dr, dc = -1, 0
        elif direction == "ACTION2":  # down
            dr, dc = 1, 0
        elif direction == "ACTION3":  # left
            dr, dc = 0, -1
        elif direction == "ACTION4":  # right
            dr, dc = 0, 1
        else:
            return start, False

        traversable: set = {0}
        if floor_value is not None:
            traversable.add(floor_value)
            traversable.add(max(0, floor_value - 1))
            traversable.add(floor_value + 1)

        cur_r, cur_c = r, c
        hit_hazard = False

        while True:
            nr, nc = cur_r + dr, cur_c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= m:
                break

            v = grid[nr][nc]

            if self._is_hazard(v):
                hit_hazard = True
                cur_r, cur_c = nr, nc
                break

            if v in traversable:
                cur_r, cur_c = nr, nc
                continue

            # Non-traversable, non-hazard -> blocking obstacle; stop before it.
            break

        return (cur_r, cur_c), hit_hazard

    def _choose_goal_directed_action(
        self,
        grid: List[List[int]],
        player_pos: Coord,
        goal_pos: Tuple[float, float],
        floor_value: Optional[int],
        hazards: List[Coord],
    ) -> Optional[str]:
        """
        Choose a direction that:
          - does not slide into a hazard,
          - actually changes the player's position,
          - reduces squared distance to the goal centroid,
          - and prefers standing farther from hazards (maximize distance).

        If no direction both moves and avoids hazards, falls back to any
        non-hazard move that changes position. Returns None if nothing seems
        useful.
        """
        pr, pc = player_pos
        tr, tc = goal_pos
        base_dirs = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

        # Prioritize directions based on relative offset to goal
        dr = tr - pr
        dc = tc - pc
        dirs_priority: List[str] = []

        if abs(dr) >= abs(dc):
            dirs_priority.append("ACTION1" if dr < 0 else "ACTION2")
            dirs_priority.append("ACTION3" if dc < 0 else "ACTION4")
        else:
            dirs_priority.append("ACTION3" if dc < 0 else "ACTION4")
            dirs_priority.append("ACTION1" if dr < 0 else "ACTION2")

        for d in base_dirs:
            if d not in dirs_priority:
                dirs_priority.append(d)

        best_candidates: List[Tuple[float, float, str]] = []

        for d in dirs_priority:
            end_pos, hazard = self._simulate_slide(grid, player_pos, d, floor_value)
            if hazard:
                continue
            if end_pos == player_pos:
                continue
            er, ec = end_pos
            dist2_goal = (er - tr) ** 2 + (ec - tc) ** 2
            dist2_hazard = self._nearest_hazard_distance_sq(end_pos, hazards)
            # We want to MINIMIZE dist2_goal and MAXIMIZE dist2_hazard,
            # so we use (dist2_goal, -dist2_hazard) as sort key.
            best_candidates.append((dist2_goal, -dist2_hazard, d))

        if best_candidates:
            best_candidates.sort(key=lambda x: (x[0], x[1]))
            # Prefer not to repeat the exact same action if possible
            for _, _, d in best_candidates:
                if d != self.last_action_name:
                    return d
            return best_candidates[0][2]

        # If everything was no-op or hazardous, attempt any non-hazard move
        safe_nonnoop: List[str] = []
        for d in dirs_priority:
            end_pos, hazard = self._simulate_slide(grid, player_pos, d, floor_value)
            if not hazard and end_pos != player_pos:
                safe_nonnoop.append(d)

        if safe_nonnoop:
            for d in safe_nonnoop:
                if d != self.last_action_name:
                    return d
            return safe_nonnoop[0]

        return None

    def _pick_exploratory_action(self) -> str:
        """
        Simple exploratory policy:
        - Uniformly random among 4 directions.
        - Avoid repeating the same direction too many times in a row.
        """
        directions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

        # If we've repeated the same move 3+ times, force a different one.
        if self.last_action_name is not None and self.last_action_repeat_count >= 3:
            alternatives = [d for d in directions if d != self.last_action_name]
            return self.rng.choice(alternatives)

        # Otherwise, random with slight bias against immediate repeat.
        if self.last_action_name is not None and self.rng.random() < 0.75:
            alternatives = [d for d in directions if d != self.last_action_name]
            return self.rng.choice(alternatives)

        return self.rng.choice(directions)

    # ---------------------------------------------------------------------
    # Main API
    # ---------------------------------------------------------------------

    def choose_action(self, frame_data: dict) -> dict:
        # 1. Handle terminal / not-started states
        current_state = frame_data.get("state", "NOT_PLAYED")

        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            # Reset per the spec, and clear our internal episode state
            self._reset_episode_state()
            return {"name": "RESET", "data": {}}

        # 2. Progressive hardcoded script – execute blindly at the start
        if self.turn_count < len(self.scripted_moves):
            action_name = self.scripted_moves[self.turn_count]
            self.turn_count += 1
        else:
            # 3. Heuristic / goal-chasing mode after the script
            full_frame_3d = frame_data.get("frame", [])
            if not full_frame_3d:
                # No frame available: fall back to exploration
                action_name = self._pick_exploratory_action()
                self.turn_count += 1
            else:
                try:
                    grid = downsample_4x4(
                        full_frame_3d,
                        take_last_grid=True,
                        round_to_int=True,
                    )
                except Exception:
                    grid = None

                if grid is None:
                    action_name = self._pick_exploratory_action()
                    self.turn_count += 1
                else:
                    floor_value = self._infer_floor_value(grid)
                    players = self._find_players_by_value(grid, floor_value)
                    player_pos = self._player_centroid(players, grid)
                    goal_centroid = self._find_goal_centroid(grid)
                    hazards = self._collect_hazard_cells(grid)

                    chosen: Optional[str] = None
                    if goal_centroid is not None:
                        chosen = self._choose_goal_directed_action(
                            grid,
                            player_pos,
                            goal_centroid,
                            floor_value,
                            hazards,
                        )

                    if chosen is None:
                        action_name = self._pick_exploratory_action()
                    else:
                        action_name = chosen

                    self.turn_count += 1

        # Update anti-hammering counters
        if action_name == self.last_action_name:
            self.last_action_repeat_count += 1
        else:
            self.last_action_repeat_count = 1
            self.last_action_name = action_name

        return {"name": action_name, "data": {}}