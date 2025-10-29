# evaluation/evaluate.py

import argparse
import logging
import os
import sys
import time
import threading
import json
import dataclasses 
from datetime import datetime, timezone 
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Type, List 
from concurrent.futures import ThreadPoolExecutor, as_completed 

import requests
from requests.cookies import RequestsCookieJar
from dotenv import load_dotenv 

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file explicitly
load_dotenv(dotenv_path=project_root / ".env.example")
load_dotenv(dotenv_path=project_root / ".env", override=True)

from agents import AVAILABLE_AGENTS
from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
from evaluation.config import EVALUATION_GAMES
# Import LevelMetrics as well
from evaluation.metrics import GameMetrics, LevelMetrics 
# Import the specific functions needed from report
from evaluation.report import generate_console_report, save_summary_report, calculate_stats 

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")

# --- Function to run a single game attempt ---
def evaluate_single_game(
    agent: Agent,
    game_id: str,
    max_actions_per_game: int,
    run_index: int 
) -> GameMetrics:
    """Runs a single game evaluation, tracking per-level metrics."""
    
    run_metrics = GameMetrics(
        game_id=game_id, 
        agent_name=agent.agent_name, 
        run_index=run_index, 
        start_time=time.time() 
    )
    run_metrics.run_status = "IN_PROGRESS"
    
    # --- Level Tracking State ---
    current_level_number = 1
    current_level_metrics = LevelMetrics(level_number=current_level_number)
    level_start_time = time.time()
    # --------------------------

    max_score = 0
    total_actions_this_run = 0 

    try:
        frame = agent.take_action(GameAction.RESET)
        if not frame:
            raise ConnectionError(f"Failed to RESET game {game_id} (Run {run_index}).")
        
        agent.append_frame(frame) 
        run_metrics.start_time = time.time() # Refined start time
        level_start_time = run_metrics.start_time # Start level 1 timer

        while total_actions_this_run < max_actions_per_game:
            action = agent.choose_action(agent.frames, agent.frames[-1])
            previous_frame = agent.frames[-1]
            latest_frame = agent.take_action(action)
            
            # Action completed, update counters BEFORE checking for errors/completion
            agent.action_counter += 1 
            total_actions_this_run += 1
            current_level_metrics.actions += 1 # Increment current level actions

            if not latest_frame:
                log.error(f"[{game_id} Run {run_index}] Agent failed to get a valid frame after action {action.name}. Stopping.")
                current_level_metrics.status = "ERROR"
                run_metrics.run_status = "ERROR"
                break
            
            agent.append_frame(latest_frame)
            max_score = max(max_score, latest_frame.score)
            run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, current_level_number)

            # --- Update Level Metrics ---
            if latest_frame.frame != previous_frame.frame:
                 current_level_metrics.state_changes += 1

            # --- Handle Level Completion ---
            level_completed = (latest_frame.score > previous_frame.score and 
                               latest_frame.state != GameState.WIN and 
                               latest_frame.state != GameState.GAME_OVER) # Ensure score increase isn't due to reset after win/loss
                               
            if level_completed:
                level_end_time = time.time()
                current_level_metrics.duration_seconds = level_end_time - level_start_time
                current_level_metrics.status = "COMPLETED"
                run_metrics.level_metrics[current_level_number] = current_level_metrics # Store completed level metrics
                
                log.info(f"[{game_id} Run {run_index}] Level {current_level_number} COMPLETED. Actions: {current_level_metrics.actions}, Duration: {current_level_metrics.duration_seconds:.2f}s. Score: {latest_frame.score}.")

                # Start next level
                current_level_number += 1
                run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, current_level_number)
                current_level_metrics = LevelMetrics(level_number=current_level_number) # Reset for next level
                level_start_time = level_end_time # Start timer for the new level

            # --- Handle Game Over ---
            if latest_frame.state == GameState.GAME_OVER:
                current_level_metrics.game_overs += 1 # Count game over for the current level
                run_metrics.total_game_overs += 1 # Count for the whole run
                log.warning(f"[{game_id} Run {run_index}] Game Over on Level {current_level_number}. Issuing RESET.")
                
                # Finalize current level metrics before reset (as TIMEOUT/ERROR state)
                level_end_time = time.time()
                current_level_metrics.duration_seconds = level_end_time - level_start_time
                current_level_metrics.status = "GAME_OVER" 
                run_metrics.level_metrics[current_level_number] = current_level_metrics

                # Issue reset
                reset_action = GameAction.RESET 
                latest_frame = agent.take_action(reset_action)
                if not latest_frame:
                    log.error(f"[{game_id} Run {run_index}] Failed to RESET after GAME_OVER. Stopping.")
                    run_metrics.run_status = "ERROR"
                    break # Stop run on failed reset
                agent.append_frame(latest_frame)
                
                # Start the level over
                log.info(f"[{game_id} Run {run_index}] Restarting Level {current_level_number} after GAME_OVER.")
                current_level_metrics = LevelMetrics(level_number=current_level_number) 
                level_start_time = time.time() 

            # --- Handle Win Condition ---
            if latest_frame.state == GameState.WIN:
                level_end_time = time.time()
                current_level_metrics.duration_seconds = level_end_time - level_start_time
                current_level_metrics.status = "COMPLETED" # Final level completed
                run_metrics.level_metrics[current_level_number] = current_level_metrics 

                run_metrics.run_status = "COMPLETED_RUN" # Mark the whole run as complete
                log.info(f"[{game_id} Run {run_index}] Game COMPLETED successfully! Final Score: {latest_frame.score}")
                break # Exit loop on win

        # --- Handle Timeout Condition ---
        else: 
            run_metrics.run_status = "TIMEOUT"
            current_level_metrics.status = "TIMEOUT" # Mark current level as timed out
            log.warning(f"[{game_id} Run {run_index}] Game TIMEOUT after {total_actions_this_run} actions.")

    # --- Handle Errors ---
    except Exception as e:
        run_metrics.run_status = "ERROR"
        current_level_metrics.status = "ERROR" # Mark current level as errored
        log.error(f"[{game_id} Run {run_index}] Exception occurred: {e}", exc_info=True)
    
    # --- Finalize Metrics ---
    finally:
        run_metrics.end_time = time.time()
        run_metrics.run_duration_seconds = run_metrics.end_time - run_metrics.start_time
        
        # Store the metrics for the last level if it wasn't already stored (e.g., timeout/error)
        if current_level_number not in run_metrics.level_metrics:
            # Ensure duration is calculated if not already set
            if current_level_metrics.duration_seconds == 0.0:
                 current_level_metrics.duration_seconds = run_metrics.end_time - level_start_time
            run_metrics.level_metrics[current_level_number] = current_level_metrics

        # Aggregate totals from level metrics
        run_metrics.total_state_changes = sum(lm.state_changes for lm in run_metrics.level_metrics.values())
        run_metrics.total_game_overs = sum(lm.game_overs for lm in run_metrics.level_metrics.values()) # Recalculate based on level data

        run_metrics.total_actions_taken = total_actions_this_run
        run_metrics.final_score = max_score # Final score is max score achieved during run
        
        # Capture GUID and Construct CORRECT Replay URL
        run_metrics.guid = agent.guid 
        if run_metrics.guid and agent.game_id: # Use agent's game_id
            # Correct URL Format: /replay/{game_id}/{guid}
            run_metrics.replay_url = f"{ROOT_URL}/replay/{agent.game_id}/{run_metrics.guid}" 
            log.debug(f"[{game_id} Run {run_index}] Captured guid: {run_metrics.guid}, Replay URL: {run_metrics.replay_url}")
        else:
            log.warning(f"[{game_id} Run {run_index}] Could not capture GUID for replay link.")
            run_metrics.replay_url = None

    return run_metrics


# --- Task Wrapper for Parallel Execution (Unchanged) ---
def run_evaluation_task(
    game_id: str, 
    run_index: int, 
    agent_class: Type[Agent], 
    agent_name_cli: str, 
    card_id: str, 
    cookies: RequestsCookieJar, 
    max_actions: int
) -> GameMetrics:
    """Creates an agent and runs evaluate_single_game for one task."""
    
    log.debug(f"Task starting: Game {game_id}, Run {run_index}")
    agent_instance = agent_class(
        card_id=card_id,
        game_id=game_id,
        agent_name=agent_name_cli, 
        ROOT_URL=ROOT_URL,
        record=False,  
        cookies=deepcopy(cookies) 
    )
    metrics = evaluate_single_game( 
        agent=agent_instance,
        game_id=game_id,
        max_actions_per_game=max_actions,
        run_index=run_index 
    )
    log.debug(f"Task finished: Game {game_id}, Run {run_index} -> Status: {metrics.run_status}")
    return metrics


# --- Main Function (Mostly Unchanged, calls updated report functions) ---
def main():
    parser = argparse.ArgumentParser(description="Run parallel agent evaluation with reruns.")
    parser.add_argument("--agent", required=True, choices=list(AVAILABLE_AGENTS.keys()), help="The name of the agent to evaluate.")
    parser.add_argument("--suite", required=True, choices=list(EVALUATION_GAMES.keys()), help="The evaluation suite to run.")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum actions per game run before timeout.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times to run each game.") 
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of parallel workers.") 
    args = parser.parse_args()

    agent_name_cli = args.agent 
    agent_class = AVAILABLE_AGENTS[agent_name_cli]
    game_ids = EVALUATION_GAMES[args.suite]
    num_runs = args.num_runs
    max_workers = args.max_workers
    
    log.info(f"Agent: '{agent_name_cli}', Suite: '{args.suite}', Games: {len(game_ids)}, Runs per game: {num_runs}, Max workers: {max_workers}")

    card_id: Optional[str] = None
    api_key = os.getenv("ARC_API_KEY", "") 
    if not api_key:
        log.error("ARC_API_KEY environment variable not found. Please set it in your .env file.")
        sys.exit(1) 
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    cookies: Optional[RequestsCookieJar] = None
    
    # Setup Results Files
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") 
    base_filename = f"{agent_name_cli}_{args.suite}_runs{num_runs}_{timestamp}"
    results_filepath_jsonl = results_dir / f"{base_filename}.jsonl"
    results_filepath_txt = results_dir / f"{base_filename}.summary.txt" 
    file_lock = threading.Lock()
    log.info(f"Detailed results (JSONL): {results_filepath_jsonl}")
    log.info(f"Summary report (TXT): {results_filepath_txt}")

    overall_start_time = time.time()
    results_data: List[Dict[str, Any]] = [] 

    try:
        # Open Scorecard
        with requests.Session() as s:
            s.headers.update(headers)
            tags = [f"eval-{agent_name_cli}", args.suite, f"runs-{num_runs}", f"workers-{max_workers}"]
            log.info(f"Attempting to open scorecard with URL: {ROOT_URL}/api/scorecard/open")
            r = s.post(f"{ROOT_URL}/api/scorecard/open", json={"tags": tags}, timeout=30)
            log.info(f"Scorecard open response status: {r.status_code}") 
            if r.status_code == 401:
                 log.error(f"Authentication failed (401 Unauthorized). Check if ARC_API_KEY is correct and valid: {api_key[:4]}...{api_key[-4:]}")
                 sys.exit(1) 
            r.raise_for_status() 
            response_data = r.json()
            card_id = response_data.get("card_id")
            if not card_id:
                 raise ValueError("API did not return a card_id when opening scorecard.")
            cookies = s.cookies 
            log.info(f"Scorecard '{card_id}' opened for evaluation run.")

        # Create Task List
        tasks_to_run = [
            (game_id, run_idx) 
            for run_idx in range(1, num_runs + 1) 
            for game_id in game_ids
        ]
        total_tasks = len(tasks_to_run)
        log.info(f"Total evaluation tasks to run: {total_tasks}")
        
        # Execute in Parallel
        completed_tasks = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    run_evaluation_task,
                    game_id, run_index, agent_class, agent_name_cli, card_id, cookies, args.max_actions
                ): (game_id, run_index)
                for game_id, run_index in tasks_to_run
            }

            for future in as_completed(future_to_task):
                game_id, run_index = future_to_task[future]
                try:
                    result_metrics: GameMetrics = future.result()
                    metrics_dict = dataclasses.asdict(result_metrics)
                    results_data.append(metrics_dict) 
                    
                    # Incremental Saving to JSONL
                    with file_lock:
                        with open(results_filepath_jsonl, "a", encoding="utf-8") as f:
                            # Convert nested LevelMetrics properly for JSON
                            metrics_dict_serializable = deepcopy(metrics_dict)
                            metrics_dict_serializable['level_metrics'] = {
                                str(k): dataclasses.asdict(v) 
                                for k, v in result_metrics.level_metrics.items()
                            }
                            metrics_dict_serializable['start_time_iso'] = datetime.fromtimestamp(metrics_dict['start_time'], timezone.utc).isoformat()
                            metrics_dict_serializable['end_time_iso'] = datetime.fromtimestamp(metrics_dict['end_time'], timezone.utc).isoformat()
                            f.write(json.dumps(metrics_dict_serializable) + "\n")
                    
                    completed_tasks += 1
                    log.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed.")

                except Exception as exc:
                    log.error(f"Task {game_id} (Run {run_index}) generated an exception: {exc}", exc_info=True)
                    error_metric_obj = GameMetrics( 
                        game_id=game_id, agent_name=agent_name_cli, run_index=run_index, 
                        run_status="ERROR", start_time=time.time() 
                    )
                    error_metric_obj.end_time = time.time()
                    error_metric_obj.run_duration_seconds = error_metric_obj.end_time - error_metric_obj.start_time 
                    err_dict = dataclasses.asdict(error_metric_obj)
                    results_data.append(err_dict) 

                    with file_lock:
                        with open(results_filepath_jsonl, "a", encoding="utf-8") as f:
                           err_dict['start_time_iso'] = datetime.fromtimestamp(err_dict['start_time'], timezone.utc).isoformat()
                           err_dict['end_time_iso'] = datetime.fromtimestamp(err_dict['end_time'], timezone.utc).isoformat()
                           f.write(json.dumps(err_dict) + "\n")
                           
                    completed_tasks += 1 
                    log.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed (including errors).")

    except KeyboardInterrupt:
        log.warning("Keyboard interrupt. Shutting down. Results saved so far are in JSONL.")
    except requests.exceptions.HTTPError as http_err:
        log.error(f"HTTP error during evaluation setup: {http_err}") 
    except Exception as e:
         log.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        # Close Scorecard
        if card_id:
            log.info(f"Closing scorecard '{card_id}'.")
            try:
                with requests.Session() as s_close:
                    s_close.headers.update(headers) 
                    if cookies:
                         s_close.cookies.update(cookies)
                    close_response = s_close.post(f"{ROOT_URL}/api/scorecard/close", json={"card_id": card_id}, timeout=30)
                    if close_response.status_code != 200:
                         log.warning(f"Failed to close scorecard '{card_id}'. Status: {close_response.status_code}, Response: {close_response.text[:200]}")
                    else:
                         log.info(f"Scorecard '{card_id}' closed successfully.")
                         log.info(f"View final scorecard online: {ROOT_URL}/scorecards/{card_id}")
            except requests.exceptions.RequestException as close_err:
                 log.error(f"Network error closing scorecard '{card_id}': {close_err}")
            except Exception as generic_close_err:
                log.error(f"Unexpected error closing scorecard '{card_id}': {generic_close_err}")
        else:
             log.warning("No scorecard opened or card_id lost; skipping close.")

        # Generate Final Reports
        overall_end_time = time.time()
        total_duration = overall_end_time - overall_start_time
        log.info(f"Total evaluation time: {total_duration:.2f} seconds.")
        
        if results_data: 
             log.info("Calculating final statistics...")
             # Pass the raw list of dicts to calculate_stats
             game_stats, overall_summary = calculate_stats(results_data) 

             log.info(f"Generating console report...")
             try:
                 generate_console_report(results_data, args.suite, agent_name_cli, num_runs) 
             except Exception as report_err:
                 log.error(f"Failed to generate console report: {report_err}", exc_info=True)

             log.info(f"Saving summary report to: {results_filepath_txt}")
             try:
                 save_summary_report(
                     str(results_filepath_txt), 
                     game_stats, overall_summary, results_data, 
                     agent_name_cli, args.suite, num_runs
                 )
             except Exception as save_err:
                 log.error(f"Failed to save summary text report: {save_err}", exc_info=True)
        else: 
             log.error("No evaluation results were collected. Cannot generate reports.")
             # Print minimal summary if no data
             print("\n--- Evaluation Summary (No Results) ---")
             # ... (minimal summary print) ...

    log.info("Evaluation script finished.")

if __name__ == "__main__":
    main()