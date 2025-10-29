# evaluation/report.py

import json
import textwrap
from collections import defaultdict
from pathlib import Path 
from typing import Dict, List, Tuple, Any 
from datetime import datetime
import statistics # Added for potential stdev calculations

# Keep GameMetrics import for potential type hinting if needed elsewhere
from .metrics import GameMetrics, LevelMetrics # Import LevelMetrics

def calculate_stats(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Calculates per-game, per-level, and overall summary statistics.
    Accepts a list of dictionaries (parsed from JSONL).
    """
    # Structure: game_stats[game_id][level_number][stat_name] -> list of values across runs
    game_level_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # Structure: game_summary_stats[game_id][stat_name] -> list of values across runs
    game_summary_stats = defaultdict(lambda: defaultdict(list))
    
    total_runs_all = 0
    total_completed_all = 0
    total_duration_all = 0.0

    # --- Pass 1: Collect raw data per game and per level ---
    for res in results:
        game_id = res['game_id']
        total_runs_all += 1
        total_duration_all += res.get('run_duration_seconds', 0.0)
        
        # Store run-level summary stats
        game_summary_stats[game_id]['run_status'].append(res.get('run_status', 'UNKNOWN'))
        game_summary_stats[game_id]['final_score'].append(res.get('final_score', 0))
        game_summary_stats[game_id]['total_actions'].append(res.get('total_actions_taken', 0))
        game_summary_stats[game_id]['run_duration'].append(res.get('run_duration_seconds', 0.0))
        game_summary_stats[game_id]['total_game_overs'].append(res.get('total_game_overs', 0))
        game_summary_stats[game_id]['highest_level'].append(res.get('highest_level_reached', 1))

        if res.get('run_status') == "COMPLETED_RUN":
            total_completed_all += 1
        
        # Store level-specific stats
        level_metrics_dict = res.get('level_metrics', {})
        # Ensure keys are integers if they were stringified in JSONL
        level_metrics_processed = {int(k): v for k, v in level_metrics_dict.items()}

        for level_num, level_data in level_metrics_processed.items():
            if not isinstance(level_data, dict): continue # Skip if data format is wrong
            
            game_level_stats[game_id][level_num]['actions'].append(level_data.get('actions', 0))
            game_level_stats[game_id][level_num]['duration'].append(level_data.get('duration_seconds', 0.0))
            game_level_stats[game_id][level_num]['state_changes'].append(level_data.get('state_changes', 0))
            game_level_stats[game_id][level_num]['game_overs'].append(level_data.get('game_overs', 0))
            game_level_stats[game_id][level_num]['status'].append(level_data.get('status', 'UNKNOWN'))

    # --- Pass 2: Calculate aggregated statistics ---
    overall_summary = {
        "total_runs": total_runs_all,
        "total_completed": total_completed_all,
        "overall_completion_rate": (float(total_completed_all) / total_runs_all * 100.0) if total_runs_all else 0.0,
        "average_duration_all": total_duration_all / total_runs_all if total_runs_all else 0.0,
    }

    processed_game_stats = {}
    all_game_ids = set(game_summary_stats.keys()) | set(game_level_stats.keys())

    for game_id in all_game_ids:
        summary_data = game_summary_stats.get(game_id, defaultdict(list))
        level_data_raw = game_level_stats.get(game_id, defaultdict(lambda: defaultdict(list)))
        
        runs = len(summary_data['run_status'])
        completed_runs = summary_data['run_status'].count("COMPLETED_RUN")
        
        # Calculate run summary stats
        avg_final_score = statistics.mean(summary_data['final_score']) if summary_data['final_score'] else 0.0
        avg_total_actions = statistics.mean(summary_data['total_actions']) if summary_data['total_actions'] else 0.0
        avg_run_duration = statistics.mean(summary_data['run_duration']) if summary_data['run_duration'] else 0.0
        avg_total_game_overs = statistics.mean(summary_data['total_game_overs']) if summary_data['total_game_overs'] else 0.0
        avg_highest_level = statistics.mean(summary_data['highest_level']) if summary_data['highest_level'] else 1.0
        completion_rate = (float(completed_runs) / runs * 100.0) if runs else 0.0

        # Calculate aggregated level stats
        aggregated_levels = {}
        if level_data_raw:
             max_level_num = max(level_data_raw.keys()) if level_data_raw else 0
             for level_num in range(1, max_level_num + 1):
                 level_stats = level_data_raw.get(level_num, defaultdict(list))
                 
                 attempt_count = len(level_stats['status'])
                 completed_count = level_stats['status'].count("COMPLETED")
                 
                 avg_actions = statistics.mean(level_stats['actions']) if level_stats['actions'] else 0.0
                 avg_duration = statistics.mean(level_stats['duration']) if level_stats['duration'] else 0.0
                 avg_state_changes = statistics.mean(level_stats['state_changes']) if level_stats['state_changes'] else 0.0
                 avg_game_overs = statistics.mean(level_stats['game_overs']) if level_stats['game_overs'] else 0.0
                 level_completion_rate = (float(completed_count) / attempt_count * 100.0) if attempt_count else 0.0 # Rate of completing this level attempt

                 aggregated_levels[level_num] = {
                     "attempts": attempt_count,
                     "completed": completed_count,
                     "avg_actions": avg_actions,
                     "avg_duration": avg_duration,
                     "avg_state_changes": avg_state_changes,
                     "avg_game_overs": avg_game_overs,
                     "completion_rate": level_completion_rate,
                 }

        processed_game_stats[game_id] = {
            # Run Summary Stats
            "num_runs": runs,
            "completed_runs": completed_runs,
            "avg_final_score": avg_final_score,
            "avg_total_actions": avg_total_actions,
            "avg_run_duration": avg_run_duration,
            "run_completion_rate": completion_rate,
            "avg_total_game_overs": avg_total_game_overs,
            "avg_highest_level": avg_highest_level,
            # Level Breakdown Stats
            "level_stats": aggregated_levels, 
        }

    return processed_game_stats, overall_summary


# --- Console Report Function ---
def generate_console_report(results_data: List[Dict[str, Any]], suite_name: str, agent_name: str, num_runs_per_game: int):
    """Generates and prints a report to the console with level details."""

    if not results_data:
        print("No results found to generate a console report.")
        return
        
    game_stats, overall_summary = calculate_stats(results_data)

    print("\n--- Evaluation Report (Console) ---")
    print(f"Agent: {agent_name}")
    print(f"Suite: {suite_name}")
    print(f"Requested Runs per Game: {num_runs_per_game}")
    print("-----------------------------------")

    # Overall Summary
    print("\n## 📈 Overall Summary")
    print("--------------------------------------------------")
    # ... (overall summary print - same as before) ...
    print(f"Total Runs Attempted: {overall_summary['total_runs']}")
    print(f"Total Runs Completed: {overall_summary['total_completed']}")
    print(f"Overall Completion Rate: {overall_summary['overall_completion_rate']:.1f}%")
    print(f"Average Duration (all runs): {overall_summary['average_duration_all']:.2f}s")
    print("--------------------------------------------------")


    # Per-Game Summary (Averaged) - Now includes highest level
    print("\n## 🎮 Per-Game Summary (Averaged)")
    print("-" * 140) # Adjusted width
    header = f"{'Game ID':<25} | {'Avg Final Score':>16} | {'Avg Highest Lvl':>16} | {'Avg Actions (Run)':>18} | {'Avg Duration (Run)':>18} | {'Run Cmpl Rate':>15} | {'Runs (Done/Total)':>18}"
    print(header)
    print("-" * 140)
    
    if not game_stats:
        print("No game results to display.")
    else:
        for game_id, stats in sorted(game_stats.items()):
            game_id_short = textwrap.shorten(game_id, width=25, placeholder="...")
            avg_score_str = f"{stats['avg_final_score']:.1f}"
            avg_lvl_str = f"{stats['avg_highest_level']:.1f}"
            avg_actions_str = f"{stats['avg_total_actions']:.1f}"
            avg_duration_str = f"{stats['avg_run_duration']:.2f}s"
            completion_rate_str = f"{stats['run_completion_rate']:.1f}%"
            runs_str = f"{stats['completed_runs']}/{stats['num_runs']}"
            
            print(f"{game_id_short:<25} | {avg_score_str:>16} | {avg_lvl_str:>16} | {avg_actions_str:>18} | {avg_duration_str:>18} | {completion_rate_str:>15} | {runs_str:>18}")
            
            # --- Print Averaged Level Stats ---
            if stats['level_stats']:
                 print("  Level Stats (Averaged):")
                 lvl_header = f"    {'Lvl':>3} | {'Avg Actions':>12} | {'Avg Duration':>13} | {'Avg State ∆':>12} | {'Avg Game Overs':>14} | {'Complete Rate':>15} | {'Attempts':>10}"
                 print(lvl_header)
                 print("    " + "-" * (len(lvl_header) - 4))
                 for lvl_num, lvl_stat in sorted(stats['level_stats'].items()):
                     print(f"    {lvl_num:>3} | {lvl_stat['avg_actions']:>12.1f} | {lvl_stat['avg_duration']:>12.2f}s | {lvl_stat['avg_state_changes']:>12.1f} | {lvl_stat['avg_game_overs']:>14.1f} | {lvl_stat['completion_rate']:>14.1f}% | {lvl_stat['attempts']:>10}")
                 print("-" * 140) # Separator after each game's level stats
            else:
                 print("  No level stats collected.")
                 print("-" * 140)


    # Detailed Per-Run Results (Console) - Includes replay URL
    print("\n## 📊 Detailed Per-Run Results")
    print("-" * 150) 
    # Added Highest Lvl column
    detail_header = f"{'Game ID':<25} | {'Run':>5} | {'Status':<15} | {'Score':>7} | {'Highest Lvl':>12} | {'Actions':>9} | {'Duration (s)':>14} | {'Game Overs':>12} | {'Replay URL'}" 
    print(detail_header)
    print("-" * 150)

    if not results_data:
         print("No detailed results to display.")
    else:
        sorted_results = sorted(results_data, key=lambda r: (r.get('game_id', ''), r.get('run_index', 0)))
        for res in sorted_results:
            game_id_short = textwrap.shorten(res.get('game_id', 'N/A'), width=25, placeholder="...")
            run_idx = res.get('run_index', 'N/A')
            status = res.get('run_status', 'N/A') # Use run_status
            score = res.get('final_score', 'N/A')
            highest_lvl = res.get('highest_level_reached', 'N/A')
            actions = res.get('total_actions_taken', 'N/A') # Use total_actions_taken
            duration = f"{res.get('run_duration_seconds', 0.0):.2f}s" # Use run_duration_seconds
            game_overs = res.get('total_game_overs', 'N/A') # Use total_game_overs
            replay = res.get('replay_url') or 'N/A' 
            print(f"{game_id_short:<25} | {run_idx:>5} | {status:<15} | {score:>7} | {highest_lvl:>12} | {actions:>9} | {duration:>14} | {game_overs:>12} | {replay}") 

    print("-" * 150)
    print("\n--- End of Console Report ---")


# --- Summary Text File Function ---
def save_summary_report(
    filepath: str, 
    game_stats: Dict[str, Dict[str, Any]], 
    overall_summary: Dict[str, Any], 
    results_data: List[Dict[str, Any]],
    agent_name: str,
    suite_name: str,
    num_runs_per_game: int
):
    """Formats and saves a comprehensive summary report to a text file."""
    
    report_lines = []
    
    report_lines.append("--- Evaluation Summary Report ---")
    report_lines.append(f"Agent: {agent_name}")
    report_lines.append(f"Suite: {suite_name}")
    report_lines.append(f"Requested Runs per Game: {num_runs_per_game}")
    report_lines.append(f"Generated At: {datetime.now().isoformat()}")
    report_lines.append("---------------------------------")

    # Overall Summary
    report_lines.append("\n## Overall Summary")
    report_lines.append("--------------------------------------------------")
    report_lines.append(f"Total Runs Attempted: {overall_summary['total_runs']}")
    report_lines.append(f"Total Runs Completed (Full Game Win): {overall_summary['total_completed']}")
    report_lines.append(f"Overall Game Completion Rate: {overall_summary['overall_completion_rate']:.1f}%")
    report_lines.append(f"Average Run Duration (all runs): {overall_summary['average_duration_all']:.2f}s")
    report_lines.append("--------------------------------------------------")

    # Per-Game Summary with Level Stats
    report_lines.append("\n## Per-Game Summary (Averaged)")
    
    if not game_stats:
        report_lines.append("No game results to display.")
    else:
        for game_id, stats in sorted(game_stats.items()):
            report_lines.append("\n" + "=" * 80)
            report_lines.append(f"Game ID: {game_id}")
            report_lines.append("=" * 80)
            
            # Run Summary for this game
            report_lines.append("  Run Summary:")
            report_lines.append(f"    Total Runs: {stats['num_runs']}")
            report_lines.append(f"    Completed Runs (Full Game Win): {stats['completed_runs']} ({stats['run_completion_rate']:.1f}%)")
            report_lines.append(f"    Avg Final Score (All Runs): {stats['avg_final_score']:.1f}")
            report_lines.append(f"    Avg Highest Level Reached: {stats['avg_highest_level']:.1f}")
            report_lines.append(f"    Avg Total Actions (Per Run): {stats['avg_total_actions']:.1f}")
            report_lines.append(f"    Avg Run Duration: {stats['avg_run_duration']:.2f}s")
            report_lines.append(f"    Avg Total Game Overs (Per Run): {stats['avg_total_game_overs']:.1f}")
            
            # Level Breakdown for this game
            if stats['level_stats']:
                 report_lines.append("\n  Level Statistics (Averaged per Attempt):")
                 lvl_header = f"    {'Lvl':>3} | {'Avg Actions':>12} | {'Avg Duration':>13} | {'Avg State ∆':>12} | {'Avg GOs':>9} | {'Cmpl Rate':>11} | {'Attempts':>10}"
                 report_lines.append("    " + "-" * (len(lvl_header) -4))
                 report_lines.append(lvl_header)
                 report_lines.append("    " + "-" * (len(lvl_header) - 4))
                 for lvl_num, lvl_stat in sorted(stats['level_stats'].items()):
                     report_lines.append(f"    {lvl_num:>3} | {lvl_stat['avg_actions']:>12.1f} | {lvl_stat['avg_duration']:>12.2f}s | {lvl_stat['avg_state_changes']:>12.1f} | {lvl_stat['avg_game_overs']:>9.1f} | {lvl_stat['completion_rate']:>10.1f}% | {lvl_stat['attempts']:>10}")
            else:
                 report_lines.append("\n  No detailed level statistics collected for this game.")

    # Detailed Run List with Replay Links
    report_lines.append("\n" + "=" * 80)
    report_lines.append("\n## Detailed Run List & Replay Links")
    report_lines.append("-" * 80)
    if not results_data:
        report_lines.append("No runs recorded.")
    else:
        sorted_results = sorted(results_data, key=lambda r: (r.get('game_id', ''), r.get('run_index', 0)))
        current_game_id = None
        for res in sorted_results:
            game_id = res.get('game_id', 'N/A')
            # Print game ID header when it changes
            if game_id != current_game_id:
                report_lines.append(f"\nGame: {game_id}")
                current_game_id = game_id
            
            run_idx = res.get('run_index', '?')
            status = res.get('run_status', 'N/A') # Use run_status
            score = res.get('final_score', 'N/A')
            highest_lvl = res.get('highest_level_reached', 'N/A')
            actions = res.get('total_actions_taken', 'N/A') # Use total_actions_taken
            duration = f"{res.get('run_duration_seconds', 0.0):.2f}s" # Use run_duration_seconds
            replay = res.get('replay_url') or 'N/A'
            report_lines.append(f"  Run {run_idx:>2}: {status:<15} Score={score:>4}, HighestLvl={highest_lvl:>2}, Actions={actions:>4}, Dur={duration:>8} -> {replay}")
            
    report_lines.append("-" * 80)
    report_lines.append("\n--- End of Summary Report ---")

    # Write to file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
    except Exception as e:
        print(f"Error writing summary report to {filepath}: {e}")