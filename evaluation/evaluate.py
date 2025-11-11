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
from typing import Any, Dict, Optional, Type, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import openai
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
# Import the new metrics structures
from evaluation.metrics import GameMetrics, LevelMetrics, AttemptMetrics
from evaluation.report import generate_console_report, save_summary_report, calculate_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")

# --- Agent Variant Helper ---
def _get_agent_tags(agent_name: str) -> List[str]:
    """
    Checks for known environment variables that create agent "variants"
    and returns a list of tags.
    """
    tags = []

    #
    if agent_name.startswith("as66"):
        # Check for the general prompts env var you mentioned
        if os.getenv("ARCGAME_GENERAL_PROMPTS", "0").strip().lower() in ("1", "true", "yes", "on"):
            tags.append("general")
    
    
    
    # Model/Reasoning
    model_override = os.getenv("AGENT_MODEL_OVERRIDE")
    if model_override:
        # Sanitize model name for filename
        sanitized_model = model_override.split('/')[-1].replace('.', '_')
        tags.append(sanitized_model)
        
    reasoning_effort = os.getenv("AGENT_REASONING_EFFORT")
    if reasoning_effort:
        tags.append(f"reason-{reasoning_effort}")

    # Text Diff
    include_text_diff = os.getenv("INCLUDE_TEXT_DIFF", "true").lower() == "true"
    if not include_text_diff:
        tags.append("noDiff")

    # Context Limit
    context_limit = int(os.getenv("CONTEXT_LENGTH_LIMIT", "-1"))
    if context_limit != -1:
        tags.append(f"ctx{context_limit // 1000}k")

    # Downsample
    downsample_images = os.getenv("DOWNSAMPLE_IMAGES", "true").lower() == "true"
    if not downsample_images and "as66visualmemoryagent" in agent_name:
        tags.append("64x64")

    # Image Detail
    image_detail = os.getenv("IMAGE_DETAIL_LEVEL", "low").lower()
    if image_detail != "low" and "as66visualmemoryagent" in agent_name:
        tags.append(f"detail-{image_detail}")

    # Pixels Per Cell
    pixels_per_cell = int(os.getenv("IMAGE_PIXELS_PER_CELL", "24"))
    if pixels_per_cell != 24 and "as66visualmemoryagent" in agent_name:
        tags.append(f"cell{pixels_per_cell}")

    return tags

# --- Retry Helper ---
MAX_RETRIES = 5
INITIAL_BACKOFF = 1 # in seconds

def _run_with_retries(func_to_run: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Runs a function with exponential backoff for specific retriable API errors.
    """
    retries = 0
    backoff = INITIAL_BACKOFF
    while True:
        try:
            return func_to_run(*args, **kwargs)
        except (openai.RateLimitError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retries >= MAX_RETRIES:
                log.error(f"Final attempt failed for {func_to_run.__name__} after {retries} retries. Raising error.")
                raise e # Re-raise the error to be caught by the main try/except
            
            log.warning(f"API error detected for {func_to_run.__name__}: {type(e).__name__}. Retrying in {backoff}s... (Attempt {retries + 1}/{MAX_RETRIES})")
            time.sleep(backoff)
            retries += 1
            backoff *= 2 # Exponential backoff
        # Any other exception (e.g., logic error, validation error) should not be retried and will be caught by the main handler.
        except Exception as e:
            log.error(f"Non-retriable error in {func_to_run.__name__}: {e}", exc_info=False) # Log minimal info
            raise e # Re-raise

# --- Function to run a single game attempt ---
def evaluate_single_game(
    agent: Agent,
    game_id: str,
    max_actions_per_game: int,
    run_index: int 
) -> GameMetrics:
    """Runs a single game evaluation, tracking per-level and per-attempt metrics."""
    
    run_metrics = GameMetrics(
        game_id=game_id, 
        agent_name=agent.agent_name, 
        run_index=run_index, 
        start_time=time.time() 
    )
    run_metrics.status = "IN_PROGRESS"
    
    # --- Level & Attempt Tracking State ---
    current_level_number = 1
    current_level_metrics = LevelMetrics(level_number=current_level_number)
    current_attempt_number = 1
    current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
    level_start_time = time.time() # This will reset for the first attempt of each level
    attempt_start_time = level_start_time
    # ------------------------------------

    max_score = 0
    total_actions_this_run = 0 
    loop_exited_normally = False # Flag to track if loop timed out

    try:
        # Use retry helper for initial RESET
        frame = _run_with_retries(agent.take_action, GameAction.RESET)
        if not frame:
            # This check remains, as _run_with_retries might return None if the underlying function does
            raise ConnectionError(f"Failed to RESET game {game_id} (Run {run_index}).")
        
        agent.append_frame(frame) 
        run_metrics.start_time = time.time() # Refined start time
        level_start_time = run_metrics.start_time # Start level 1 timer
        attempt_start_time = run_metrics.start_time # Start attempt 1 timer

        while total_actions_this_run < max_actions_per_game:
            # Use retry helper for choose_action (LLM call)
            action = _run_with_retries(agent.choose_action, agent.frames, agent.frames[-1])
            previous_frame = agent.frames[-1]
            # Use retry helper for take_action (Game server call)
            latest_frame = _run_with_retries(agent.take_action, action)
            
            # Action completed, update counters BEFORE checking for errors/completion
            agent.action_counter += 1 
            total_actions_this_run += 1
            current_attempt_metrics.actions += 1 # Increment current attempt actions

            if not latest_frame:
                log.error(f"[{game_id} Run {run_index}] Agent failed to get a valid frame after action {action.name}. Stopping.")
                current_attempt_metrics.status = "ERROR"
                current_level_metrics.status = "ERROR"
                run_metrics.status = "ERROR"
                run_metrics.error_message = "Agent failed to get a valid frame." # <-- STORE ERROR
                break
            
            agent.append_frame(latest_frame)
            max_score = max(max_score, latest_frame.score)
            run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, current_level_number)

            # --- Update Attempt Metrics ---
            if latest_frame.frame != previous_frame.frame:
                current_attempt_metrics.state_changes += 1

            # --- Handle Level Completion ---
            level_completed = (latest_frame.score > previous_frame.score and 
                               latest_frame.state != GameState.WIN and 
                               latest_frame.state != GameState.GAME_OVER)
                                
            if level_completed:
                attempt_end_time = time.time()
                current_attempt_metrics.duration_seconds = attempt_end_time - attempt_start_time
                current_attempt_metrics.status = "COMPLETED"
                current_level_metrics.attempts.append(current_attempt_metrics) # Store final attempt
                
                current_level_metrics.status = "COMPLETED"
                run_metrics.level_metrics[current_level_number] = current_level_metrics # Store completed level metrics
                
                log.info(f"[{game_id} Run {run_index}] Level {current_level_number} COMPLETED. Successful Attempt: {current_attempt_number} ({current_attempt_metrics.actions} actions). Total Level Actions: {current_level_metrics.total_actions}. Score: {latest_frame.score}.")

                # Start next level
                current_level_number += 1
                run_metrics.highest_level_reached = max(run_metrics.highest_level_reached, current_level_number)
                
                # Reset for new level
                current_level_metrics = LevelMetrics(level_number=current_level_number)
                current_attempt_number = 1
                current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number)
                level_start_time = attempt_end_time
                attempt_start_time = attempt_end_time

            # --- Handle Game Over ---
            elif latest_frame.state == GameState.GAME_OVER:
                attempt_end_time = time.time()
                current_attempt_metrics.duration_seconds = attempt_end_time - attempt_start_time
                current_attempt_metrics.status = "GAME_OVER"
                current_attempt_metrics.game_overs = 1 # This attempt ended in a game over
                current_level_metrics.attempts.append(current_attempt_metrics) # Store failed attempt
                
                log.warning(f"[{game_id} Run {run_index}] Game Over on Level {current_level_number}, Attempt {current_attempt_number}. Actions this attempt: {current_attempt_metrics.actions}. Total Level Actions: {current_level_metrics.total_actions}.")
                
                # Issue reset
                reset_action = GameAction.RESET 
                # Use retry helper for the RESET action
                latest_frame = _run_with_retries(agent.take_action, reset_action)
                if not latest_frame:
                    log.error(f"[{game_id} Run {run_index}] Failed to RESET after GAME_OVER. Stopping.")
                    current_level_metrics.status = "ERROR"
                    run_metrics.status = "ERROR"
                    run_metrics.error_message = "Failed to RESET after GAME_OVER." # <-- STORE ERROR
                    break # Stop run on failed reset
                agent.append_frame(latest_frame)
                
                # Start the next attempt for the same level
                current_attempt_number += 1
                current_attempt_metrics = AttemptMetrics(attempt_number=current_attempt_number) 
                attempt_start_time = time.time() 

            # --- Handle Win Condition ---
            elif latest_frame.state == GameState.WIN:
                attempt_end_time = time.time()
                current_attempt_metrics.duration_seconds = attempt_end_time - attempt_start_time
                current_attempt_metrics.status = "COMPLETED"
                current_level_metrics.attempts.append(current_attempt_metrics) # Store final successful attempt
                
                current_level_metrics.status = "COMPLETED"
                run_metrics.level_metrics[current_level_number] = current_level_metrics 
                
                run_metrics.status = "COMPLETED_RUN" # Mark the whole run as complete
                log.info(f"[{game_id} Run {run_index}] Game COMPLETED successfully! Final Level {current_level_number} actions: {current_attempt_metrics.actions}. Final Score: {latest_frame.score}")
                break # Exit loop on win

            # --- Handle Timeout Condition (Loop Exited Normally) ---
            else: 
                loop_exited_normally = True # Set flag

    # --- Handle Errors ---
    except Exception as e:
        run_metrics.status = "ERROR"
        run_metrics.error_message = str(e) # <-- STORE THE EXCEPTION TEXT
        current_attempt_metrics.status = "ERROR"
        current_level_metrics.status = "ERROR"
        log.error(f"[{game_id} Run {run_index}] Exception occurred: {e}", exc_info=True)
    
    # --- Finalize Metrics ---
    finally:
        run_metrics.end_time = time.time()
        run_metrics.run_duration_seconds = run_metrics.end_time - run_metrics.start_time
        
        # --- FIX: Robustly finalize status and last attempt ---
        
        # Determine the final status of the last attempt
        final_attempt_status = "UNKNOWN"
        if run_metrics.status == "COMPLETED_RUN":
            final_attempt_status = "COMPLETED" # This was set in the WIN block
        elif loop_exited_normally:
            final_attempt_status = "TIMEOUT"
            log.warning(f"[{game_id} Run {run_index}] Game TIMEOUT after {total_actions_this_run} actions on Level {current_level_number}.")
        elif run_metrics.status == "ERROR":
            final_attempt_status = "ERROR"
        elif run_metrics.status == "IN_PROGRESS": # Should only happen if loop broke unexpectedly
            log.error(f"[{game_id} Run {run_index}] Run ended with IN_PROGRESS status. Marking as ERROR.")
            final_attempt_status = "ERROR"
            run_metrics.status = "ERROR"
            if not run_metrics.error_message: # <-- STORE ERROR
                run_metrics.error_message = "Run ended unexpectedly in IN_PROGRESS state."

        # Update the last attempt *if it's still marked IN_PROGRESS*
        if current_attempt_metrics.status == "IN_PROGRESS":
            current_attempt_metrics.duration_seconds = run_metrics.end_time - attempt_start_time
            current_attempt_metrics.status = final_attempt_status
        
        # Append the last attempt if it hasn't been appended yet
        if not current_level_metrics.attempts or current_level_metrics.attempts[-1].attempt_number != current_attempt_metrics.attempt_number:
            current_level_metrics.attempts.append(current_attempt_metrics)
        
        # Finalize the last level's status
        if current_level_metrics.status == "IN_PROGRESS":
            current_level_metrics.status = final_attempt_status

        # Store the last level's metrics
        if current_level_number not in run_metrics.level_metrics:
            run_metrics.level_metrics[current_level_number] = current_level_metrics

        # Set the final run status (if it's still IN_PROGRESS, set it to the final attempt status)
        if run_metrics.status == "IN_PROGRESS":
            run_metrics.status = final_attempt_status
        # --- End of FIX ---

        # Aggregate totals from the new LevelMetrics properties
        run_metrics.run_total_actions = sum(lm.total_actions for lm in run_metrics.level_metrics.values())
        run_metrics.total_game_overs_across_run = sum(lm.total_game_overs for lm in run_metrics.level_metrics.values())
        run_metrics.total_state_changes_across_run = sum(lm.total_state_changes for lm in run_metrics.level_metrics.values())
        
        # This now correctly reflects the *true* total actions from the loop
        run_metrics.total_actions_taken = total_actions_this_run 
        
        # This check is crucial: if the loop counter (e.g. 20) doesn't match the sum (e.g. 19), log it.
        # This can happen if an error occurs *during* the final action, where total_actions_this_run is 20
        # but the final attempt's action count is still 19. We trust the loop counter.
        if run_metrics.run_total_actions != total_actions_this_run and run_metrics.status != "ERROR":
            log.warning(f"[{game_id} Run {run_index}] Mismatch! Loop counter `total_actions_this_run` is {total_actions_this_run}, but summed `run_total_actions` is {run_metrics.run_total_actions}. Using loop counter.")
            # Correct the metric to reflect the max_actions timeout
            run_metrics.run_total_actions = total_actions_this_run
        elif run_metrics.status == "ERROR":
            # If an error happened, the loop counter is the source of truth
            run_metrics.run_total_actions = total_actions_this_run


        run_metrics.final_score = max_score
        
        run_metrics.guid = agent.guid 
        if run_metrics.guid and agent.game_id:
            run_metrics.replay_url = f"{ROOT_URL}/replay/{agent.game_id}/{run_metrics.guid}" 
            log.debug(f"[{game_id} Run {run_index}] Captured guid: {run_metrics.guid}, Replay URL: {run_metrics.replay_url}")
        else:
            log.warning(f"[{game_id} Run {run_index}] Could not capture GUID for replay link.")
            run_metrics.replay_url = None
        
        
        # Call cleanup on the agent
        agent.cleanup() 
    

    return run_metrics


# --- Task Wrapper for Parallel Execution ---
def run_evaluation_task(
    game_id: str, 
    run_index: int, 
    agent_class: Type[Agent], 
    agent_name_cli: str, 
    card_id: str, 
    cookies: RequestsCookieJar, 
    max_actions: int,
   
    agent_name_with_variant: str,
    env_vars_to_set: Dict[str, str]
) -> GameMetrics:
    """Creates an agent and runs evaluate_single_game for one task."""
    
    log.debug(f"Task starting: Game {game_id}, Run {run_index}")
    
    # Set environment variables for this thread/task 
    # This ensures the agent's __init__ method reads the correct settings
    for k, v in env_vars_to_set.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k] # Unset if default was None
    
    agent_instance = agent_class(
        card_id=card_id,
        game_id=game_id,
        agent_name=agent_name_with_variant, # Pass the full name with all tags
        ROOT_URL=ROOT_URL,
        record=False, 
        cookies=deepcopy(cookies) 
    )
    # The GameMetrics object will now also store this full name
    metrics = evaluate_single_game( 
        agent=agent_instance,
        game_id=game_id,
        max_actions_per_game=max_actions,
        run_index=run_index 
    )
    log.debug(f"Task finished: Game {game_id}, Run {run_index} -> Status: {metrics.status}")
    return metrics


# --- Main Function ---
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
    
    log.info(f"Agent: '{agent_name_cli}', Suite: '{args.suite}', Games: {len(game_ids)}, Runs per game: {num_runs}, Max workers: {max_workers}, Max Actions: {args.max_actions}")

    card_id: Optional[str] = None
    api_key = os.getenv("ARC_API_KEY", "") 
    if not api_key:
        log.error("ARC_API_KEY environment variable not found. Please set it in your .env file.")
        sys.exit(1) 
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    cookies: Optional[RequestsCookieJar] = None
    
    # --- Setup Results Files (with new naming convention) ---
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") 
    
    #  Get all tags for filename AND for passing to threads 
    agent_tags = [agent_name_cli] + _get_agent_tags(agent_name_cli)
    agent_name_with_variant = "-".join(agent_tags)
    
    #  Prepare dict of env vars to pass to tasks 
    env_vars_to_set = {
        "AGENT_MODEL_OVERRIDE": os.getenv("AGENT_MODEL_OVERRIDE"),
        "AGENT_REASONING_EFFORT": os.getenv("AGENT_REASONING_EFFORT"),
        "INCLUDE_TEXT_DIFF": os.getenv("INCLUDE_TEXT_DIFF", "true"),
        "CONTEXT_LENGTH_LIMIT": os.getenv("CONTEXT_LENGTH_LIMIT", "-1"),
        "DOWNSAMPLE_IMAGES": os.getenv("DOWNSAMPLE_IMAGES", "true"),
        "IMAGE_DETAIL_LEVEL": os.getenv("IMAGE_DETAIL_LEVEL", "low"),
        "IMAGE_PIXELS_PER_CELL": os.getenv("IMAGE_PIXELS_PER_CELL", "24"),
        "ARCGAME_GENERAL_PROMPTS": os.getenv("ARCGAME_GENERAL_PROMPTS", "0"),
    }
    
    # Build comprehensive filename
    base_filename = f"{agent_name_with_variant}_{args.suite}_runs{num_runs}_max{args.max_actions}_{timestamp}"
    
    results_filepath_jsonl = results_dir / f"{base_filename}.jsonl"
    results_filepath_txt = results_dir / f"{base_filename}.summary.txt" 
    file_lock = threading.Lock()
    log.info(f"Detailed results (JSONL): {results_filepath_jsonl}")
    log.info(f"Summary report (TXT): {results_filepath_txt}")

    overall_start_time = time.time()
    results_data: List[Dict[str, Any]] = [] # Store raw dicts for JSONL

    try:
        # Open Scorecard
        with requests.Session() as s:
            s.headers.update(headers)
            # Use the full agent name with variants as the primary tag
            tags = [f"eval-{agent_name_with_variant}", args.suite, f"runs-{num_runs}", f"workers-{max_workers}", f"max_actions-{args.max_actions}"]
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
        
        # This list will hold the GameMetrics objects for final reporting
        game_metrics_objects_list: List[GameMetrics] = []

        # Execute in Parallel
        completed_tasks = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    run_evaluation_task,
                    game_id, 
                    run_index, 
                    agent_class, 
                    agent_name_cli, # Pass the base name
                    card_id, 
                    cookies, 
                    args.max_actions,
                    agent_name_with_variant, # Pass the full name
                    env_vars_to_set          # Pass the env var dict
                ): (game_id, run_index)
                for game_id, run_index in tasks_to_run
            }

            for future in as_completed(future_to_task):
                game_id, run_index = future_to_task[future]
                try:
                    result_metrics: GameMetrics = future.result()
                    # Store the object for final stats
                    game_metrics_objects_list.append(result_metrics) 
                    
                    # Convert to dict for JSONL saving
                    metrics_dict = dataclasses.asdict(result_metrics)
                    
                    # Incremental Saving to JSONL
                    with file_lock:
                        with open(results_filepath_jsonl, "a", encoding="utf-8") as f:
                            # Convert nested metrics properly for JSON
                            metrics_dict_serializable = deepcopy(metrics_dict)
                            metrics_dict_serializable['level_metrics'] = {
                                str(k): dataclasses.asdict(v) 
                                for k, v in result_metrics.level_metrics.items()
                            }
                            # Convert LevelMetrics.attempts
                            for k_lm, v_lm in metrics_dict_serializable['level_metrics'].items():
                                v_lm['attempts'] = [dataclasses.asdict(att) for att in result_metrics.level_metrics[int(k_lm)].attempts]
                                # Pop derived properties, they will be recalculated on load
                                v_lm.pop('total_actions', None)
                                v_lm.pop('total_game_overs', None)
                                v_lm.pop('total_state_changes', None)
                                v_lm.pop('actions_in_successful_attempt', None)
                                v_lm.pop('state_change_percentage', None)

                            metrics_dict_serializable['start_time_iso'] = datetime.fromtimestamp(metrics_dict['start_time'], timezone.utc).isoformat()
                            metrics_dict_serializable['end_time_iso'] = datetime.fromtimestamp(metrics_dict['end_time'], timezone.utc).isoformat()
                            f.write(json.dumps(metrics_dict_serializable) + "\n")
                            
                    completed_tasks += 1
                    log.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed.")

                except Exception as exc:
                    log.error(f"Task {game_id} (Run {run_index}) generated an exception: {exc}", exc_info=True)
                    error_metric_obj = GameMetrics( 
                        game_id=game_id, agent_name=agent_name_with_variant, run_index=run_index, 
                        status="ERROR", start_time=time.time(),
                        error_message=str(exc) # <-- STORE THE EXCEPTION TEXT
                    )
                    error_metric_obj.end_time = time.time()
                    error_metric_obj.run_duration_seconds = error_metric_obj.end_time - error_metric_obj.start_time 
                    
                    # Store object for final stats
                    game_metrics_objects_list.append(error_metric_obj)
                    
                    err_dict = dataclasses.asdict(error_metric_obj)

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
        
        if game_metrics_objects_list: 
            log.info("Calculating final statistics...")
            
            # Pass the list of GameMetrics objects
            game_stats, overall_summary = calculate_stats(game_metrics_objects_list) 

            log.info(f"Generating console report...")
            try:
                # Pass the list of GameMetrics objects
                generate_console_report(game_metrics_objects_list, args.suite, agent_name_with_variant, num_runs) 
            except Exception as report_err:
                log.error(f"Failed to generate console report: {report_err}", exc_info=True)

            log.info(f"Saving summary report to: {results_filepath_txt}")
            try:
                save_summary_report(
                    str(results_filepath_txt), 
                    game_stats, overall_summary, game_metrics_objects_list, # Pass objects
                    agent_name_with_variant, args.suite, num_runs
                )
            except Exception as save_err:
                log.error(f"Failed to save summary text report: {save_err}", exc_info=True)
        else: 
            log.error("No evaluation results were collected. Cannot generate reports.")
            print("\n--- Evaluation Summary (No Results) ---")
            print(f"Agent: {agent_name_with_variant}")
            print(f"Suite: {args.suite}")
            print(f"Total Runs Attempted: 0")
            print(f"Total Duration: {total_duration:.2f}s")
            print("---------------------------------------")

    log.info("Evaluation script finished.")

if __name__ == "__main__":
    main()