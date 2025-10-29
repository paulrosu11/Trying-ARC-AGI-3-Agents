# evaluation/metrics.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

@dataclass
class LevelMetrics:
    """Metrics for a single attempt at a specific level within a game run."""
    level_number: int
    actions: int = 0
    duration_seconds: float = 0.0
    state_changes: int = 0
    game_overs: int = 0
    status: str = "IN_PROGRESS" # IN_PROGRESS, COMPLETED, TIMEOUT, ERROR

@dataclass
class GameMetrics:
    """Metrics collected for a single game evaluation run."""
    # --- Identification ---
    game_id: str
    agent_name: str
    run_index: int = 1
    guid: Optional[str] = None 

    # --- Overall Run Performance ---
    total_actions_taken: int = 0 # Renamed from actions_taken
    final_score: int = 0
    highest_level_reached: int = 1 # Added: Track max level started
    run_status: str = "PENDING" # Renamed from status (PENDING, IN_PROGRESS, COMPLETED_RUN, TIMEOUT, ERROR)

    # --- Detailed Level Data ---
    # Stores metrics for each level attempt
    level_metrics: Dict[int, LevelMetrics] = field(default_factory=dict) 

    # --- Process Details (Aggregated from levels) ---
    total_state_changes: int = 0 # Renamed from state_changes
    total_game_overs: int = 0 # Renamed from game_overs

    # --- Timing ---
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    run_duration_seconds: float = 0.0 # Renamed from duration_seconds

    # --- Replay ---
    replay_url: Optional[str] = None 

    # Removed actions_per_level (now part of level_metrics)
    # Removed state_change_percentage property (can calculate in report if needed)

# OverallMetrics remains just a container
@dataclass
class OverallMetrics:
    """Stores all individual run metrics for later aggregation."""
    results: List[GameMetrics] = field(default_factory=list)

    def add_game_result(self, result: GameMetrics):
        self.results.append(result)