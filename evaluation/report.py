# evaluation/report.py

import json
import textwrap
from collections import defaultdict
from pathlib import Path 
from typing import Dict, List, Tuple, Any 
from datetime import datetime
import statistics 

# Import the new metrics structures
from .metrics import GameMetrics, LevelMetrics, AttemptMetrics

def calculate_stats(results: List[GameMetrics]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Calculates per-game, per-level, and overall summary statistics.
    Accepts a list of GameMetrics objects.
    """
    # Structure: game_level_stats[game_id][level_number][stat_name] -> list of values across runs
    game_level_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # Structure: game_summary_stats[game_id][stat_name] -> list of values across runs
    game_summary_stats = defaultdict(lambda: defaultdict(list))
    
    total_runs_all = 0
    total_completed_all = 0
    total_duration_all = 0.0

    # --- Pass 1: Collect raw data per game and per level ---
    for res in results:
        game_id = res.game_id
        total_runs_all += 1
        total_duration_all += res.run_duration_seconds
        
        # Store run-level summary stats
        game_summary_stats[game_id]['status'].append(res.status)
        game_summary_stats[game_id]['final_score'].append(res.final_score)
        game_summary_stats[game_id]['run_total_actions'].append(res.run_total_actions)
        game_summary_stats[game_id]['run_duration'].append(res.run_duration_seconds)
        game_summary_stats[game_id]['total_game_overs'].append(res.total_game_overs_across_run)
        game_summary_stats[game_id]['highest_level'].append(res.highest_level_reached)

        if res.status == "COMPLETED_RUN":
            total_completed_all += 1
        
        # Store level-specific stats
        for level_num, level_data in res.level_metrics.items():
            game_level_stats[game_id][level_num]['total_actions'].append(level_data.total_actions)
            game_level_stats[game_id][level_num]['total_game_overs'].append(level_data.total_game_overs)
            game_level_stats[game_id][level_num]['total_state_changes'].append(level_data.total_state_changes)
            game_level_stats[game_id][level_num]['status'].append(level_data.status)
            
            # This will be None if the level was not completed
            success_actions = level_data.actions_in_successful_attempt
            if success_actions is not None:
                game_level_stats[game_id][level_num]['success_actions'].append(success_actions)
            
            # Collect durations for all attempts on this level across runs
            game_level_stats[game_id][level_num]['attempt_durations'].extend([a.duration_seconds for a in level_data.attempts])


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
        
        runs = len(summary_data['status'])
        completed_runs = summary_data['status'].count("COMPLETED_RUN")
        
        # Calculate run summary stats
        avg_final_score = statistics.mean(summary_data['final_score']) if summary_data['final_score'] else 0.0
        avg_run_total_actions = statistics.mean(summary_data['run_total_actions']) if summary_data['run_total_actions'] else 0.0
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
                
                # Avg actions for *successful* attempts only
                avg_success_actions = statistics.mean(level_stats['success_actions']) if level_stats['success_actions'] else 0.0
                # Avg actions across *all* attempts (the "34" number)
                avg_total_actions = statistics.mean(level_stats['total_actions']) if level_stats['total_actions'] else 0.0
                avg_duration = statistics.mean(level_stats['attempt_durations']) if level_stats['attempt_durations'] else 0.0
                avg_state_changes = statistics.mean(level_stats['total_state_changes']) if level_stats['total_state_changes'] else 0.0
                avg_game_overs = statistics.mean(level_stats['total_game_overs']) if level_stats['total_game_overs'] else 0.0
                level_completion_rate = (float(completed_count) / attempt_count * 100.0) if attempt_count else 0.0

                aggregated_levels[level_num] = {
                    "attempts": attempt_count,
                    "completed": completed_count,
                    "avg_total_actions": avg_total_actions, # Your "34"
                    "avg_success_actions": avg_success_actions, # Your "4"
                    "avg_duration_per_attempt": avg_duration,
                    "avg_total_state_changes": avg_state_changes,
                    "avg_total_game_overs": avg_game_overs,
                    "completion_rate": level_completion_rate,
                }

        processed_game_stats[game_id] = {
            # Run Summary Stats
            "num_runs": runs,
            "completed_runs": completed_runs,
            "avg_final_score": avg_final_score,
            "avg_run_total_actions": avg_run_total_actions,
            "avg_run_duration": avg_run_duration,
            "run_completion_rate": completion_rate,
            "avg_total_game_overs_per_run": avg_total_game_overs,
            "avg_highest_level": avg_highest_level,
            # Level Breakdown Stats
            "level_stats": aggregated_levels, 
        }

    return processed_game_stats, overall_summary


# --- Console Report Function ---
def generate_console_report(results_data: List[GameMetrics], suite_name: str, agent_name: str, num_runs_per_game: int):
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
    print("\n##  Overall Summary")
    print("--------------------------------------------------")
    print(f"Total Runs Attempted: {overall_summary['total_runs']}")
    print(f"Total Runs Completed (Full Game Win): {overall_summary['total_completed']}")
    print(f"Overall Game Completion Rate: {overall_summary['overall_completion_rate']:.1f}%")
    print(f"Average Run Duration (all runs): {overall_summary['average_duration_all']:.2f}s")
    print("--------------------------------------------------")


    # Per-Game Summary (Averaged)
    print("\n## Per-Game Summary (Averaged Across Runs)")
    print("-" * 140)
    header = f"{'Game ID':<25} | {'Avg Final Score':>16} | {'Avg Highest Lvl':>16} | {'Avg Total Actions':>18} | {'Avg Run Duration':>18} | {'Game Cmpl Rate':>15} | {'Runs (Done/Total)':>18}"
    print(header)
    print("-" * 140)
    
    if not game_stats:
        print("No game results to display.")
    else:
        for game_id, stats in sorted(game_stats.items()):
            game_id_short = textwrap.shorten(game_id, width=25, placeholder="...")
            avg_score_str = f"{stats['avg_final_score']:.1f}"
            avg_lvl_str = f"{stats['avg_highest_level']:.1f}"
            avg_actions_str = f"{stats['avg_run_total_actions']:.1f}"
            avg_duration_str = f"{stats['avg_run_duration']:.2f}s"
            completion_rate_str = f"{stats['run_completion_rate']:.1f}%"
            runs_str = f"{stats['completed_runs']}/{stats['num_runs']}"
            
            print(f"{game_id_short:<25} | {avg_score_str:>16} | {avg_lvl_str:>16} | {avg_actions_str:>18} | {avg_duration_str:>18} | {completion_rate_str:>15} | {runs_str:>18}")
            
            # --- Print Averaged Level Stats ---
            if stats['level_stats']:
                print("  Level Stats (Averaged Across Runs):")
                lvl_header = f"    {'Lvl':>3} | {'Avg Total Actions':>18} | {'Avg Success Actions':>20} | {'Avg Total GOs':>14} | {'Avg State ∆':>12} | {'Cmpl Rate':>11} | {'Attempts':>10}"
                print(lvl_header)
                print("    " + "-" * (len(lvl_header) - 4))
                for lvl_num, lvl_stat in sorted(stats['level_stats'].items()):
                    avg_total_act_str = f"{lvl_stat['avg_total_actions']:.1f}"
                    avg_success_act_str = f"{lvl_stat['avg_success_actions']:.1f}" if lvl_stat['avg_success_actions'] > 0 else "N/A"
                    print(f"    {lvl_num:>3} | {avg_total_act_str:>18} | {avg_success_act_str:>20} | {lvl_stat['avg_total_game_overs']:>14.1f} | {lvl_stat['avg_total_state_changes']:>12.1f} | {lvl_stat['completion_rate']:>10.1f}% | {lvl_stat['attempts']:>10}")
                print("-" * 140) # Separator after each game's level stats
            else:
                print("  No level stats collected.")
                print("-" * 140)


    # Detailed Per-Run Results (Console)
    print("\n##  Detailed Per-Run Results")
    # --- UPDATED HEADER ---
    detail_header = f"{'Game ID':<25} | {'Run':>5} | {'Status':<15} | {'Score':>7} | {'H-Lvl':>5} | {'Actions':>7} | {'Duration':>10} | {'GOs':>3} | {'Details (Replay URL / Error)'}"
    print(detail_header)
    print("-" * (len(detail_header) + 4)) # Auto-fit width

    if not results_data:
        print("No detailed results to display.")
    else:
        sorted_results = sorted(results_data, key=lambda r: (r.game_id, r.run_index))
        for res in sorted_results:
            game_id_short = textwrap.shorten(res.game_id, width=25, placeholder="...")
            run_idx = res.run_index
            status = res.status
            score = res.final_score
            highest_lvl = res.highest_level_reached
            actions = res.run_total_actions # Use new aggregated field
            duration = f"{res.run_duration_seconds:.2f}s"
            game_overs = res.total_game_overs_across_run # Use new aggregated field
            
            # --- NEW DETAIL LOGIC ---
            details = res.replay_url or 'N/A'
            if res.status == "ERROR" and res.error_message:
                # Shorten the error message to fit on one line
                details = f"ERROR: {textwrap.shorten(res.error_message.replace(chr(10), ' '), width=60, placeholder='...')}"
            
            print(f"{game_id_short:<25} | {run_idx:>5} | {status:<15} | {score:>7} | {highest_lvl:>5} | {actions:>7} | {duration:>10} | {game_overs:>3} | {details}") 

    print("-" * (len(detail_header) + 4))
    print("\n--- End of Console Report ---")


# --- Summary Text File Function ---
def save_summary_report(
    filepath: str, 
    game_stats: Dict[str, Dict[str, Any]], 
    overall_summary: Dict[str, Any], 
    results_data: List[GameMetrics], # Now expects GameMetrics objects
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
    report_lines.append("\n## Per-Game Summary (Averaged Across Runs)")
    
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
            report_lines.append(f"    Avg Total Actions (Per Run): {stats['avg_run_total_actions']:.1f}")
            report_lines.append(f"    Avg Run Duration: {stats['avg_run_duration']:.2f}s")
            report_lines.append(f"    Avg Total Game Overs (Per Run): {stats['avg_total_game_overs_per_run']:.1f}")
            
            # Level Breakdown for this game
            if stats['level_stats']:
                report_lines.append("\n  Level Statistics (Averaged Across Runs):")
                lvl_header = f"    {'Lvl':>3} | {'Avg Total Actions':>18} | {'Avg Success Actions':>20} | {'Avg Total GOs':>14} | {'Avg State ∆':>12} | {'Cmpl Rate':>11} | {'Attempts':>10}"
                report_lines.append("    " + "-" * (len(lvl_header) - 4))
                report_lines.append(lvl_header)
                report_lines.append("    " + "-" * (len(lvl_header) - 4))
                for lvl_num, lvl_stat in sorted(stats['level_stats'].items()):
                    avg_total_act_str = f"{lvl_stat['avg_total_actions']:.1f}"
                    avg_success_act_str = f"{lvl_stat['avg_success_actions']:.1f}" if lvl_stat['avg_success_actions'] > 0 else "N/A"
                    report_lines.append(f"    {lvl_num:>3} | {avg_total_act_str:>18} | {avg_success_act_str:>20} | {lvl_stat['avg_total_game_overs']:>14.1f} | {lvl_stat['avg_total_state_changes']:>12.1f} | {lvl_stat['completion_rate']:>10.1f}% | {lvl_stat['attempts']:>10}")
            else:
                report_lines.append("\n  No detailed level statistics collected for this game.")

    # Detailed Run List with Replay Links
    report_lines.append("\n" + "=" * 80)
    report_lines.append("\n## Detailed Run List & Replay Links")
    report_lines.append("-" * 80)
    if not results_data:
        report_lines.append("No runs recorded.")
    else:
        sorted_results = sorted(results_data, key=lambda r: (r.game_id, r.run_index))
        current_game_id = None
        for res in sorted_results:
            game_id = res.game_id
            # Print game ID header when it changes
            if game_id != current_game_id:
                report_lines.append(f"\nGame: {game_id}")
                current_game_id = game_id
            
            run_idx = res.run_index
            status = res.status
            score = res.final_score
            highest_lvl = res.highest_level_reached
            actions = res.run_total_actions
            duration = f"{res.run_duration_seconds:.2f}s"
            game_overs = res.total_game_overs_across_run
            
            # --- NEW DETAIL LOGIC ---
            details = f"-> {res.replay_url or 'N/A'}"
            if res.status == "ERROR" and res.error_message:
                # Shorten the error message
                details = f"-> ERROR: {textwrap.shorten(res.error_message.replace(chr(10), ' '), width=70, placeholder='...')}"

            report_lines.append(f"  Run {run_idx:>2}: {status:<15} Score={score:>4}, HighestLvl={highest_lvl:>2}, Actions={actions:>4}, Dur={duration:>8}, GOs={game_overs:>3} {details}")
            
    report_lines.append("-" * 80)
    report_lines.append("\n--- End of Summary Report ---")

    # Write to file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
    except Exception as e:
        print(f"Error writing summary report to {filepath}: {e}")

