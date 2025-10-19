# evaluation/metrics.py

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class GameMetrics:
    """Metrics collected for a single game evaluation."""
    game_id: str
    agent_name: str
    actions_taken: int = 0
    state_changes: int = 0
    game_overs: int = 0
    final_score: int = 0
    status: str = "IN_PROGRESS"  # e.g., COMPLETED, TIMEOUT, GAME_OVER
    actions_per_level: Dict[int, int] = field(default_factory=dict)

    @property
    def state_change_percentage(self) -> float:
        """Calculate the percentage of actions that resulted in a state change."""
        if self.actions_taken == 0:
            return 0.0
        return (self.state_changes / self.actions_taken) * 100

@dataclass
class OverallMetrics:
    """Aggregated metrics across all games in an evaluation suite."""
    results_per_game: Dict[str, GameMetrics] = field(default_factory=dict)

    def add_game_result(self, result: GameMetrics):
        self.results_per_game[result.game_id] = result

    @property
    def total_games_completed(self) -> int:
        return sum(1 for r in self.results_per_game.values() if r.status == "COMPLETED")

    @property
    def completion_rate(self) -> float:
        if not self.results_per_game:
            return 0.0
        return (self.total_games_completed / len(self.results_per_game)) * 100

    @property
    def average_actions_per_game(self) -> float:
        if not self.results_per_game:
            return 0.0
        total_actions = sum(r.actions_taken for r in self.results_per_game.values())
        return total_actions / len(self.results_per_game)