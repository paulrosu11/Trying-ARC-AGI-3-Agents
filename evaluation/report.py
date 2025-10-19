# evaluation/report.py

import textwrap
from .metrics import OverallMetrics

# evaluation/report.py

def generate_report(metrics: OverallMetrics, suite_name: str):
    ...
    # --- Per-Game Details Table ---
    print("🕹️ Per-Game Details")
    print("-----------------------------------------------------------------------------------------------------")
    # Updated header
    print(f"{'Game ID':<25} | {'Status':<12} | {'Score':>5} | {'Actions (per Level)':>25} | {'State Changes (%)':>20} | {'Game Overs':>10}")
    print("-----------------------------------------------------------------------------------------------------")
    
    if not metrics.results_per_game:
        print("No game results to display.")
    else:
        for game_id, result in sorted(metrics.results_per_game.items()):
            game_id_short = textwrap.shorten(game_id, width=25, placeholder="...")
            status = result.status
            score = result.final_score
            
            # --- Format the actions_per_level dict into a string ---
            if result.actions_per_level:
                actions_str = ", ".join([f"L{lvl}: {count}" for lvl, count in result.actions_per_level.items()])
            else:
                actions_str = str(result.actions_taken)

            state_change_pct = f"{result.state_change_percentage:.1f}%"
            game_overs = result.game_overs
            
            # Updated print statement with new actions_str
            print(f"{game_id_short:<25} | {status:<12} | {score:>5} | {actions_str:<25} | {state_change_pct:>20} | {game_overs:>10}")
    
    print("-----------------------------------------------------------------------------------------------------\n")