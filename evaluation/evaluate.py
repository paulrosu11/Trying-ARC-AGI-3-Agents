# evaluation/evaluate.py

import argparse
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.cookies import RequestsCookieJar

# Add project root to sys.path to allow imports from agents
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents import AVAILABLE_AGENTS
from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
from evaluation.config import EVALUATION_GAMES
from evaluation.metrics import GameMetrics, OverallMetrics
from evaluation.report import generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT_URL = os.environ.get("ROOT_URL", "https://three.arcprize.org")

def evaluate_single_game(
    agent: Agent,
    game_id: str,
    card_id: str,
    headers: Dict[str, str],
    max_actions_per_game: int
) -> GameMetrics:
    """Runs a single game evaluation and returns the collected metrics."""
    metrics = GameMetrics(game_id=game_id, agent_name=agent.agent_name)
    
    current_level = 1
    actions_this_level = 0
    max_score = 0
    
    # --- CORRECTED LOOP LOGIC ---
    # The evaluation loop must manage its own action counter.
    actions_taken_so_far = 0

    try:
        frame = agent.take_action(GameAction.RESET)
        if not frame:
            raise ConnectionError("Failed to RESET and start the game.")
        
        agent.append_frame(frame)
        
        # The loop condition now correctly checks its own counter.
        while actions_taken_so_far < max_actions_per_game:
            actions_taken_so_far += 1
            
            action = agent.choose_action(agent.frames, agent.frames[-1])
            
            previous_frame = agent.frames[-1]
            latest_frame = agent.take_action(action)

            if not latest_frame:
                log.error(f"[{game_id}] Agent failed to get a valid frame. Stopping.")
                metrics.status = "ERROR"
                break
            
            agent.append_frame(latest_frame)
            
            max_score = max(max_score, latest_frame.score)
            actions_this_level += 1

            if latest_frame.model_dump() != previous_frame.model_dump():
                metrics.state_changes += 1

            if latest_frame.score > previous_frame.score:
                log.info(f"Level {current_level} completed in {actions_this_level} actions for game {game_id}.")
                metrics.actions_per_level[current_level] = actions_this_level
                current_level += 1
                actions_this_level = 0
            
            if latest_frame.state == GameState.GAME_OVER:
                metrics.game_overs += 1
                # Ask the agent for a RESET action.
                reset_action = agent.choose_action(agent.frames, latest_frame)
                if reset_action.name == 'RESET':
                    latest_frame = agent.take_action(reset_action)
                    agent.append_frame(latest_frame)
                else:
                    log.error(f"[{game_id}] Agent did not issue RESET after GAME_OVER. Stopping.")
                    metrics.status = "ERROR"
                    break

            if latest_frame.state == GameState.WIN:
                metrics.status = "COMPLETED"
                log.info(f"Game '{game_id}' completed successfully!")
                if actions_this_level > 0:
                    metrics.actions_per_level[current_level] = actions_this_level
                break
        else: 
            metrics.status = "TIMEOUT"
            log.warning(f"Game '{game_id}' timed out after {actions_taken_so_far} actions.")
    # --- END CORRECTION ---

    except Exception as e:
        metrics.status = "ERROR"
        log.error(f"An exception occurred during evaluation of game '{game_id}': {e}", exc_info=True)
    
    if actions_this_level > 0 and current_level not in metrics.actions_per_level:
        metrics.actions_per_level[current_level] = actions_this_level

    metrics.actions_taken = actions_taken_so_far
    metrics.final_score = max_score
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run a sequential agent evaluation.")
    parser.add_argument("--agent", required=True, choices=list(AVAILABLE_AGENTS.keys()), help="The name of the agent to evaluate.")
    parser.add_argument("--suite", required=True, choices=list(EVALUATION_GAMES.keys()), help="The evaluation suite to run.")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum actions per game before timeout.")
    args = parser.parse_args()

    agent_name = args.agent
    agent_class = AVAILABLE_AGENTS[agent_name]
    game_ids = EVALUATION_GAMES[args.suite]
    
    log.info(f"Preparing to test agent '{agent_name}' on {len(game_ids)} games from suite '{args.suite}'.")
    log.info("Running sequentially to ensure correctness and avoid rate-limiting.")

    overall_metrics = OverallMetrics()
    card_id: Optional[str] = None
    headers = {"X-API-Key": os.getenv("ARC_API_KEY", ""), "Accept": "application/json"}
    
    try:
        with requests.Session() as s:
            s.headers.update(headers)
            r = s.post(f"{ROOT_URL}/api/scorecard/open", json={"tags": [f"eval-{agent_name}", args.suite]})
            r.raise_for_status()
            card_id = r.json()["card_id"]
            cookies = s.cookies
            log.info(f"Scorecard '{card_id}' opened for the evaluation run.")

        for game_id in game_ids:
            log.info(f"--- Now evaluating game: {game_id} ---")
            agent_instance: Agent = agent_class(
                card_id=card_id,
                game_id=game_id,
                agent_name=agent_name,
                ROOT_URL=ROOT_URL,
                record=False,
                cookies=cookies
            )
            
            metrics = evaluate_single_game(
                agent=agent_instance,
                game_id=game_id,
                card_id=card_id,
                headers=headers,
                max_actions_per_game=args.max_actions
            )
            overall_metrics.add_game_result(metrics)
            log.info(f"--- Finished evaluation for {game_id} (Status: {metrics.status}) ---")
            
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt received. Shutting down.")
    finally:
        if card_id:
            log.info(f"All games finished. Closing scorecard '{card_id}'.")
            with requests.Session() as s:
                s.headers.update(headers)
                s.post(f"{ROOT_URL}/api/scorecard/close", json={"card_id": card_id})
            log.info(f"View your final scorecard online: {ROOT_URL}/scorecards/{card_id}")
    
    generate_report(overall_metrics, args.suite)
    log.info("Evaluation complete.")

if __name__ == "__main__":
    main()