# ruff: noqa: E402
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import textwrap
import time
import statistics
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI
from requests.cookies import RequestsCookieJar

# --- Add project root to sys.path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Load Environment & Imports ---
load_dotenv(dotenv_path=project_root / ".env.example")
load_dotenv(dotenv_path=project_root / ".env", override=True)

from agents.structs import GameAction
from agents.templates.as66.downsample import downsample_4x4, matrix16_to_lines
from agents.templates.meta_agent_prompts import (
    PROMPT_AS66_RULES,
    PROMPT_SYSTEM_INSTRUCTION,
    PROMPT_PROGRESSIVE_INSTRUCTION,
    PROMPT_CONDENSER_SYSTEM
)

# --- REUSE EXISTING EVALUATION MODULES ---
from evaluation.metrics import GameMetrics, LevelMetrics, AttemptMetrics
from evaluation.report import save_summary_report, calculate_stats

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)



GAME_ID = "as66-821a4dcad9c2"
META_AGENT_NAME = "MetaAgent_AS66_Iterative"

# --- Coder Settings ---
CODER_MODEL = "gpt-5.1"             # e.g., "gpt-5", "o1-preview", "gpt-4o"
REASONING_EFFORT = "low"         # "low", "medium", "high", or None
CODER_TEMPERATURE = 0.2           # Temperature for generation
MAX_META_ITERATIONS = 30          

# --- Evaluation Settings ---
EPISODES_PER_ITERATION = 2        
ACTIONS_PER_EPISODE = 50          

# --- Action Limit Polynomial Settings ---
ACTION_POLY_A = 12.0
ACTION_POLY_B = 1.5
ACTION_POLY_C = 15.0

# --- Context Management ---
SLIDING_WINDOW_SIZE = 3           
CONTEXT_TOKEN_LIMIT = 50000       

# --- Progressive Mode Settings ---
PROGRESSIVE_MODE = True           
CONDENSER_MODEL = "gpt-5.1"         
CONDENSER_REASONING = "medium"
CONDENSER_MAX_RETRIES = 10         
ACTION_GROWTH_FACTOR = 1.5        
STUCK_PATIENCE = 2                



GENERATED_AGENT_PATH = project_root / "agents" / "generated_heuristic_agent.py"
META_MEMORY_DIR = project_root / "evaluation_results" / "meta_agent_logs"

# --- BOOTSTRAP AGENT CODE (The Seed) ---
BOOTSTRAP_AGENT_CODE = r'''import random
from typing import Any, Dict, List, Optional
import logging

# Get a logger instance
log = logging.getLogger(__name__)

# --- CRITICAL IMPORT ---

from agents.templates.as66.downsample import downsample_4x4, matrix16_to_lines



class GeneratedHeuristicAgent:
    """
    This is a bootstrap "random" agent.
    """

    def __init__(self):
        self.turn_count = 0
        # Example of hardcoded moves list (populated by prompts in progressive mode)
        self.scripted_moves = [] 
        log.info("Bootstrap Heuristic Agent (Random) initialized.")

    def choose_action(self, frame_data: dict) -> dict:
        self.turn_count += 1
        current_state = frame_data.get('state', 'NOT_PLAYED')
        
        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            self.turn_count = 0
            return {'name': 'RESET', 'data': {}}
        
        # Progressive Mode: Execute scripted moves first
        if self.turn_count <= len(self.scripted_moves):
            action_name = self.scripted_moves[self.turn_count - 1]
            return {'name': action_name, 'data': {}}

        # Example downsample usage
        full_frame_3d = frame_data.get('frame', [])
        if full_frame_3d:
            try:
                _ = downsample_4x4(full_frame_3d, take_last_grid=True, round_to_int=True)
            except Exception:
                pass

        possible_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        chosen_action = random.choice(possible_actions)
        
        return {'name': chosen_action, 'data': {}}
'''

# --- ARC API Client (Minimal) ---
class _EnvClient:
    def __init__(self, root_url: str, game_id: str, card_id: str, headers: Dict[str, str], cookies: RequestsCookieJar):
        self.root = root_url
        self.game_id = game_id
        self.card_id = card_id
        self.sess = requests.Session()
        self.sess.headers.update(headers)
        self.sess.cookies = deepcopy(cookies)
        self.guid: Optional[str] = None

    def _post(self, cmd: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = self.sess.post(f"{self.root}/api/cmd/{cmd}", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "guid" in data and data["guid"]:
                self.guid = data["guid"]
            return data
        except requests.exceptions.RequestException as e:
            log.error(f"API Error during {cmd}: {e}")
            return {"state": "ERROR", "score": 0, "frame": [], "error": str(e)}

    def reset(self) -> Dict[str, Any]:
        self.guid = None
        return self._post("RESET", {"card_id": self.card_id, "game_id": self.game_id})

    def act(self, action: GameAction) -> Dict[str, Any]:
        if not self.guid:
            log.error("Cannot act without a GUID. Call reset() first.")
            return {"state": "ERROR", "score": 0, "frame": [], "error": "No GUID"}
        
        payload = action.action_data.model_dump()
        payload["game_id"] = self.game_id
        payload["guid"] = self.guid
        if action.reasoning:
            payload["reasoning"] = action.reasoning

        return self._post(action.name, payload)


# --- Orchestrator Class ---
class MetaAgentOrchestrator:
    def __init__(self, agent_name: str, game_id: str):
        self.agent_name = agent_name
        self.game_id = game_id
        
        # API and Scorecard Setup
        self.root_url = self._get_root_url()
        self.api_key = os.getenv("ARC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ARC_API_KEY not found in environment.")
        self.headers = {"X-API-Key": self.api_key, "Accept": "application/json"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.card_id: Optional[str] = None

        # OpenAI Client
        self.openai_client = OpenAI()

        # Logging and Memory
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{self.agent_name}_{timestamp}"
        self.log_dir = META_MEMORY_DIR / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.meta_memory_log_path = self.log_dir / "meta_memory.jsonl"
        self.meta_memory: List[Dict[str, Any]] = []
        
        # Progressive Mode State
        self.best_level_solved = 0 
        self.stuck_counter = 0
        self.stuck_multiplier_level = 0 
        self.current_max_actions = 0 
        self.hardcoded_moves: List[str] = [] # The cumulative list of moves to get to best_level_solved

        # Initialize action limit for Level 1
        self.update_action_limit(level=1)

        log.info(f"Meta-Agent run starting. Logs: {self.log_dir}")
        log.info(f"Config: Model={CODER_MODEL}, Reasoning={REASONING_EFFORT}, Runs={EPISODES_PER_ITERATION}, Actions={ACTIONS_PER_EPISODE}")
        log.info(f"Actions Polynomial: {ACTION_POLY_A}*(L^{ACTION_POLY_B}) + {ACTION_POLY_C}")

    def _get_root_url(self) -> str:
        scheme = os.environ.get("SCHEME", "https")
        host = os.environ.get("HOST", "three.arcprize.org")
        port = os.environ.get("PORT", "443")
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            return f"{scheme}://{host}"
        return f"{scheme}://{host}:{port}"

    def calculate_action_limit(self, level: int) -> int:
        """Calculates max actions based on level polynomial and stuck multiplier."""
        base_actions = (ACTION_POLY_A * (level ** ACTION_POLY_B)) + ACTION_POLY_C
        stuck_factor = ACTION_GROWTH_FACTOR ** self.stuck_multiplier_level
        limit = int(base_actions * stuck_factor)
        return max(limit, 1)

    def update_action_limit(self, level: int):
        """Updates self.current_max_actions based on the provided level."""
        self.current_max_actions = self.calculate_action_limit(level)
        log.info(f"Action Limit Updated for Level {level} (Stuck x{self.stuck_multiplier_level}): {self.current_max_actions}")

    def open_scorecard(self) -> Optional[str]:
        tags = ["meta-agent", self.agent_name, self.run_id, f"game-{self.game_id}"]
        if PROGRESSIVE_MODE:
            tags.append("progressive")
        
        try:
            r = self.session.post(f"{self.root_url}/api/scorecard/open", json={"tags": tags}, timeout=30)
            r.raise_for_status()
            data = r.json()
            self.card_id = data.get("card_id")
            if self.card_id:
                log.info(f"Scorecard opened: {self.card_id}")
                return self.card_id
            else:
                log.error(f"Failed to open scorecard. Response: {data}")
                return None
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to open scorecard: {e}")
            return None

    def close_scorecard(self) -> None:
        if not self.card_id: return
        try:
            self.session.post(f"{self.root_url}/api/scorecard/close", json={"card_id": self.card_id}, timeout=30)
            log.info(f"Scorecard closed. View at: {self.root_url}/scorecards/{self.card_id}")
        except Exception as e:
            log.error(f"Failed to close scorecard: {e}")

    def load_agent_from_file(self, iteration: int) -> Any:
        try:
            module_name = "generated_heuristic_agent"
            spec = importlib.util.spec_from_file_location(module_name, GENERATED_AGENT_PATH)
            if not spec or not spec.loader:
                raise ImportError(f"Could not create module spec from {GENERATED_AGENT_PATH}")
            
            gen_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = gen_module 
            spec.loader.exec_module(gen_module)
            
            AgentClass = getattr(gen_module, "GeneratedHeuristicAgent")
            return AgentClass()
        except Exception as e:
            log.error(f"[Iteration {iteration}] Failed to load agent: {e}")
            raise

    def hash_frame(self, frame_dict: Dict[str, Any]) -> str:
        frame_json = json.dumps(frame_dict.get('frame', []))
        return hashlib.md5(frame_json.encode()).hexdigest()

    def group_consecutive_turns(self, turns: List[int]) -> str:
        if not turns: return ""
        groups = []
        start = turns[0]
        end = turns[0]
        for turn in turns[1:]:
            if turn == end + 1:
                end = turn
            else:
                groups.append(f"{start}-{end}" if start != end else str(start))
                start = end = turn
        groups.append(f"{start}-{end}" if start != end else str(start))
        return ", ".join(groups)

    def _estimate_entry_tokens(self, entry: Dict[str, Any]) -> int:
        return len(json.dumps(entry)) // 4

    def filter_action_log(self, action_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if CONTEXT_TOKEN_LIMIT < 0:
            return action_log 

        important_indices = set()
        current_level = 0
        
        for i, entry in enumerate(action_log):
            entry_level = entry.get("level_at_turn", 1)
            if entry_level > current_level:
                important_indices.add(i) 
                current_level = entry_level
            if entry.get("is_game_over"):
                important_indices.add(i)

        total_tokens = 0
        for idx in important_indices:
            total_tokens += self._estimate_entry_tokens(action_log[idx])

        remaining_budget = CONTEXT_TOKEN_LIMIT - total_tokens
        
        filtered_indices = set(important_indices)
        for i in range(len(action_log) - 1, -1, -1):
            if i in filtered_indices: continue
            cost = self._estimate_entry_tokens(action_log[i])
            if cost <= remaining_budget:
                filtered_indices.add(i)
                remaining_budget -= cost
            else:
                break

        sorted_indices = sorted(list(filtered_indices))
        return [action_log[i] for i in sorted_indices]

    def compress_action_log(self, action_log: List[Dict[str, Any]], seen_states: Dict[str, List[int]]) -> str:
        filtered_log = self.filter_action_log(action_log)
        compressed_lines = []
        processed_turns = set()

        if len(filtered_log) < len(action_log):
            compressed_lines.append(f"NOTE: Context limited to {CONTEXT_TOKEN_LIMIT} tokens. Showing {len(filtered_log)}/{len(action_log)} turns.")

        for entry in filtered_log:
            turn = entry["turn"]
            if turn in processed_turns: continue

            state_action_key = entry["state_action_key"]
            all_turns_for_key = seen_states[state_action_key]
            turn_group_str = self.group_consecutive_turns(all_turns_for_key)
            
            line = (
                f"Turns [{turn_group_str}]:\n"
                f"  State (16x16):\n{textwrap.indent(entry['grid_16x16'], '    ')}\n"
                f"  Action: {json.dumps(entry['action'])}\n"
                f"  Result: State={entry['state']}, Score={entry['score']}"
            )
            compressed_lines.append(line)
            processed_turns.update(all_turns_for_key)
        
        return "\n".join(compressed_lines)

    def call_responses_api(self, model: str, instructions: str, input_content: Any, reasoning_effort: Optional[str] = None, temperature: float = 1.0, previous_response_id: Optional[str] = None) -> Tuple[str, str]:
        """Calls OpenAI Responses API and extracts the output text."""
        kwargs = {
            "model": model,
            "instructions": instructions,
            "input": input_content
        }

        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        if reasoning_effort and ("o1" in model or "gpt-5" in model):
            kwargs["reasoning"] = {"effort": reasoning_effort}
        else:
            kwargs["temperature"] = temperature

        log.info(f"Calling Responses API ({model})...")
        try:
            response = self.openai_client.responses.create(**kwargs)
        except Exception as e:
            log.error(f"API Request Failed: {e}")
            raise e
        
        try:
            if hasattr(response, 'output_text'):
                 return response.output_text, response.id

            output_items = response.output
            if not output_items:
                return "", response.id
            
            message_text = ""
            for item in output_items:
                if item.type == 'message':
                    content_parts = item.content
                    text_parts = [p.text for p in content_parts if p.type == 'output_text']
                    message_text = "".join(text_parts)
                    break 
            
            return message_text, response.id

        except Exception as e:
            log.error(f"Failed to parse Responses API output: {e}")
            log.error(f"Raw response object: {response}")
            raise e

    def run_condenser_loop(self, action_log: List[Dict[str, Any]], start_score: int, target_score: int) -> List[str]:
        """
        Iterative Condenser: Extracts moves for ONE level segment (start_score -> target_score).
        """
        log.info(f"Starting Iterative Condenser Loop. Segment: Score {start_score} -> {target_score}")
        
        # 1. Filter log to just the relevant segment
        # We want entries where score matches start_score (solving the level)
        # OR the exact moment it hits target_score (the level up action)
        
        segment_log = []
        found_start = False
        
        for entry in action_log:
            score = entry['score']
            
            # Simple heuristic: If score < start, skip (previous levels)
            if score < start_score:
                continue
            
            # If score matches start_score, we are IN the level we want to solve
            if score == start_score:
                segment_log.append(entry)
            
            # If score reaches target_score, this action CAUSED the level up. Include it and stop.
            if score >= target_score:
                segment_log.append(entry)
                break
                
        log_text_lines = []
        for entry in segment_log:
             # --- FIX: Include StateHash so model can detect no-ops ---
             state_hash = entry['state_action_key'].split('|')[0]
             log_text_lines.append(
                f"Turn {entry['turn']}: Action={entry['action']['name']}, StateHash={state_hash}, Score={entry['score']}"
             )
             # --- END FIX ---
        log_content = "\n".join(log_text_lines)

        # Initial input for the conversation
        current_input = (
            f"Here is the log segment for the level we just solved (Score {start_score} to {target_score}).\n"
            f"Identify the sequence of actions that leads from Score {start_score} to the level up event (Score {target_score}).\n\n"
            f"{log_content}"
        )
        
        last_response_id = None

        for attempt in range(1, CONDENSER_MAX_RETRIES + 1):
            log.info(f"Condenser Attempt {attempt}/{CONDENSER_MAX_RETRIES}")
            
            try:
                content, resp_id = self.call_responses_api(
                    model=CONDENSER_MODEL,
                    instructions=PROMPT_CONDENSER_SYSTEM,
                    input_content=current_input,
                    reasoning_effort=CONDENSER_REASONING,
                    temperature=0.2,
                    previous_response_id=last_response_id
                )
                
                last_response_id = resp_id
                log.info(f"Raw Condenser Output: {content}")
                
                if not content: content = "[]"

                extracted_moves = []
                if "[" in content and "]" in content:
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    json_str = content[start:end]
                    try:
                        extracted_moves = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass 

                if not extracted_moves or not isinstance(extracted_moves, list):
                    feedback = f"Invalid JSON output: {content}. Please output strictly a JSON list of strings like [\"ACTION1\", ...]."
                    log.warning(f"Condenser Error: {feedback}")
                    current_input = feedback
                    continue
                
                log.info(f"Extracted {len(extracted_moves)} moves for segment. Validating...")

                # 2. Iterative Validation
                # We test: Cumulative Hardcoded Moves + New Extracted Moves
                full_test_sequence = self.hardcoded_moves + extracted_moves
                valid, valid_score, fail_reason = self.validate_sequence(full_test_sequence)
                
                # Check if we actually reached the NEW target score
                if valid and valid_score >= target_score:
                    log.info(f"SUCCESS: Segment validated! (Reached Score {valid_score})")
                    return extracted_moves
                
                feedback = (
                    f"Validation Failed. When appending your sequence to the previous known moves, the run resulted in: {fail_reason}. "
                    f"Final Score: {valid_score}. Expected Score: >= {target_score}. "
                    "Please analyze the log segment again. Ensure you include ALL steps needed to pass this specific level."
                )
                log.warning(f"Condenser Validation Failed: {feedback}")
                current_input = feedback
                
            except Exception as e:
                log.error(f"Condenser Loop Exception: {e}")
                feedback = f"System Error during generation: {e}. Please try again."
                current_input = feedback

        log.warning(f"Condenser failed to find a valid sequence after {CONDENSER_MAX_RETRIES} retries. Continuing without updating hardcoded moves.")
        return []

    def validate_sequence(self, moves: List[str]) -> Tuple[bool, int, str]:
        client = _EnvClient(self.root_url, self.game_id, self.card_id, self.headers, self.session.cookies)
        frame = client.reset()
        
        for i, move_name in enumerate(moves):
            try:
                action = GameAction.from_name(move_name)
                frame = client.act(action)
                if frame['state'] in ('GAME_OVER', 'ERROR'):
                    return False, frame.get('score', 0), f"State became {frame['state']} at step {i} ({move_name})"
            except Exception as e:
                return False, 0, f"Exception at step {i}: {e}"

        final_score = frame.get("score", 0)
        return True, final_score, "Completed sequence successfully"

    def run_single_episode(self, agent: Any, iteration: int, episode_num: int) -> Tuple[GameMetrics, List[Dict[str, Any]], Dict[str, List[int]]]:
        log.info(f"[Iter {iteration} | Ep {episode_num}] Starting episode with Limit={self.current_max_actions}")
        client = _EnvClient(self.root_url, self.game_id, self.card_id, self.headers, self.session.cookies)
        frame_dict = client.reset()

        start_time = time.time()
        metrics = GameMetrics(
            game_id=self.game_id,
            agent_name=f"{self.agent_name}_Iter{iteration}",
            run_index=episode_num,
            start_time=start_time
        )
        metrics.status = "IN_PROGRESS"

        action_log_entries = []
        seen_state_actions: Dict[str, List[int]] = {}
        
        current_level = 1
        metrics.level_metrics[1] = LevelMetrics(level_number=1)
        current_attempt = AttemptMetrics(attempt_number=1)
        metrics.level_metrics[1].attempts.append(current_attempt)

        if hasattr(agent, "scripted_moves") and agent.scripted_moves:
            log.info(f"Agent has {len(agent.scripted_moves)} scripted moves.")

        try:
            for j in range(self.current_max_actions):
                full_frame_3d = frame_dict.get('frame', [])
                grid_16x16 = downsample_4x4(full_frame_3d, take_last_grid=True, round_to_int=True)
                grid_16x16_str = matrix16_to_lines(grid_16x16)

                try:
                    action_dict = agent.choose_action(frame_dict)
                except Exception as e:
                    raise RuntimeError(f"Agent logic crashed: {e}")

                action_name = action_dict.get("name", "ACTION5")
                action_data = action_dict.get("data", {})
                
                state_hash = self.hash_frame(frame_dict)
                action_hash = f"{action_name}:{json.dumps(action_data, sort_keys=True)}"
                state_action_key = f"{state_hash}|{action_hash}"
                
                log_entry = {
                    "turn": j + 1,
                    "action": action_dict,
                    "score": frame_dict['score'],
                    "state": frame_dict['state'],
                    "grid_16x16": grid_16x16_str,
                    "state_action_key": state_action_key,
                    "level_at_turn": current_level,
                    "is_game_over": False
                }
                
                action = GameAction.from_name(action_name)
                action.set_data(action_data)
                
                prev_frame_hash = state_hash
                frame_dict = client.act(action)
                
                current_attempt.actions += 1
                if self.hash_frame(frame_dict) != prev_frame_hash:
                    current_attempt.state_changes += 1
                
                metrics.run_total_actions += 1
                metrics.final_score = max(metrics.final_score, frame_dict.get('score', 0))

                log_entry['is_game_over'] = (frame_dict['state'] == 'GAME_OVER')
                action_log_entries.append(log_entry)
                seen_state_actions.setdefault(state_action_key, []).append(j + 1)

                if frame_dict['state'] == 'WIN':
                    metrics.status = "COMPLETED_RUN"
                    metrics.level_metrics[current_level].status = "COMPLETED"
                    current_attempt.status = "COMPLETED"
                    break 
                
                if frame_dict['state'] == 'GAME_OVER':
                    current_attempt.status = "GAME_OVER"
                    current_attempt.game_overs = 1
                    metrics.total_game_overs_across_run += 1
                    frame_dict = client.reset() 
                    new_attempt_num = len(metrics.level_metrics[current_level].attempts) + 1
                    current_attempt = AttemptMetrics(attempt_number=new_attempt_num)
                    metrics.level_metrics[current_level].attempts.append(current_attempt)
                    continue 
                
                new_score = frame_dict.get('score', 0)
                new_level_calc = new_score + 1
                
                if new_level_calc > current_level:
                    metrics.level_metrics[current_level].status = "COMPLETED"
                    current_attempt.status = "COMPLETED"
                    current_level = new_level_calc
                    metrics.highest_level_reached = max(metrics.highest_level_reached, current_level)
                    metrics.level_metrics[current_level] = LevelMetrics(level_number=current_level)
                    current_attempt = AttemptMetrics(attempt_number=1)
                    metrics.level_metrics[current_level].attempts.append(current_attempt)

            if metrics.status == "IN_PROGRESS":
                metrics.status = "TIMEOUT"

        except Exception as e:
            metrics.status = "ERROR"
            metrics.error_message = str(e)
            raise

        finally:
            metrics.end_time = time.time()
            metrics.run_duration_seconds = metrics.end_time - metrics.start_time
        
        log.info(f"[Iter {iteration} | Ep {episode_num}] Done. Score: {metrics.final_score}, Status: {metrics.status}")
        return metrics, action_log_entries, seen_state_actions

    def execute_sub_agent_batch(self, iteration: int) -> Tuple[Dict[str, Any], str, List[GameMetrics]]:
        agent = self.load_agent_from_file(iteration)
        metrics_list = []
        all_logs = [] 
        
        try:
            for i in range(EPISODES_PER_ITERATION):
                metrics, log_entries, seen_states = self.run_single_episode(agent, iteration, i+1)
                metrics_list.append(metrics)
                
                if i == 0:
                    first_run_compressed_log = self.compress_action_log(log_entries, seen_states)
                    self.last_raw_log = log_entries

            game_stats, overall_summary = calculate_stats(metrics_list)
            
            report_path = self.log_dir / f"report_iter_{iteration}.txt"
            save_summary_report(
                str(report_path),
                game_stats, 
                overall_summary, 
                metrics_list,
                f"{self.agent_name}_Iter{iteration}",
                f"MetaRun_{self.run_id}",
                EPISODES_PER_ITERATION
            )
            
            summary_for_llm = {
                "status": "SUCCESS",
                "avg_score": overall_summary.get("avg_final_score", 0), 
                "max_level_reached": overall_summary.get("avg_highest_level", 1), 
                "completion_rate": overall_summary.get("overall_completion_rate", 0),
                "report_file": str(report_path)
            }
            
            if PROGRESSIVE_MODE and metrics_list:
                best_run_score = max(m.final_score for m in metrics_list)
                
                if best_run_score > self.best_level_solved:
                    log.info(f"Progressive: New level solved! (Score {best_run_score} > {self.best_level_solved})")
                    
                    # Call Iterative Condenser (Pass previous best and new best)
                    condensed_moves_segment = self.run_condenser_loop(
                        self.last_raw_log, 
                        start_score=self.best_level_solved, 
                        target_score=best_run_score
                    )
                    
                    if condensed_moves_segment:
                        log.info(f"Progressive: Sequence validated! Appending {len(condensed_moves_segment)} moves.")
                        self.hardcoded_moves.extend(condensed_moves_segment)
                        self.best_level_solved = best_run_score
                        self.stuck_counter = 0
                        self.stuck_multiplier_level = 0
                        self.update_action_limit(level=self.best_level_solved + 1)
                    else:
                        log.warning("Progressive: Condenser returned empty moves, skipping update.")

                else:
                    self.stuck_counter += 1
                    log.info(f"Progressive: Stuck counter = {self.stuck_counter}")
                    if self.stuck_counter >= STUCK_PATIENCE:
                        self.stuck_multiplier_level += 1
                        log.info(f"Progressive: Stuck patience exceeded. Increasing multiplier to {self.stuck_multiplier_level}")
                        self.stuck_counter = 0
                        self.update_action_limit(level=self.best_level_solved + 1)

            return summary_for_llm, first_run_compressed_log, metrics_list

        except Exception as e:
            log.error(f"Batch execution failed for iteration {iteration}: {e}")
            failure_summary = {
                "status": "ERROR",
                "error_message": str(e)
            }
            return failure_summary, "", []

    def build_gpt5_prompt(self) -> List[Dict[str, Any]]:
        system_prompt = PROMPT_SYSTEM_INSTRUCTION + "\n\n" + PROMPT_AS66_RULES
        messages = [{"role": "system", "content": system_prompt}]

        if PROGRESSIVE_MODE and self.hardcoded_moves:
            moves_str = json.dumps(self.hardcoded_moves)
            prog_msg = PROMPT_PROGRESSIVE_INSTRUCTION + f"\n```json\n{moves_str}\n```"
            messages.append({"role": "system", "content": prog_msg})
            log.info(f"Injected {len(self.hardcoded_moves)} hardcoded moves into instructions.")

        if not self.meta_memory:
            return messages

        window_start = max(0, len(self.meta_memory) - SLIDING_WINDOW_SIZE)
        memory_window = self.meta_memory[window_start:]
        
        for i, entry in enumerate(memory_window):
            is_last_entry = (i == len(memory_window) - 1)
            
            try:
                code_that_ran = Path(entry["code_file_path"]).read_text(encoding="utf-8")
            except Exception as e:
                code_that_ran = f"# Error reading code: {e}"

            if entry.get("status") == "ERROR":
                context_parts = [
                    f"--- Iteration {entry['iteration']} (FAILED) ---",
                    f"Code:\n```python\n{code_that_ran}\n```",
                    f"ERROR REPORT:\n{entry['error_message']}",
                    "The previous agent crashed. Please fix the code."
                ]
            else:
                summary_str = json.dumps(entry["summary"], indent=2)
                context_parts = [
                    f"--- Iteration {entry['iteration']} ---",
                    f"Code:\n```python\n{code_that_ran}\n```",
                    f"Summary:\n```json\n{summary_str}\n```"
                ]
                if is_last_entry and "action_log" in entry and entry["action_log"]:
                    context_parts.append(f"Detailed Log (Context Limited to {CONTEXT_TOKEN_LIMIT} tokens):\n{entry['action_log']}")
            
            messages.append({"role": "user", "content": "\n".join(context_parts)})
        
        messages.append({"role": "user", "content": "Based on the history, write the Python code for the next agent. Output ONLY valid Python code inside a markdown block."})
        return messages

    def call_coder_model(self, messages: List[Dict[str, Any]]) -> str:
        try:
            log.info("--- Sending prompt to Coder (Responses API) ---")
            log.info(f"Instructions length: {len(messages[0]['content'])}")
            
            # Separate system instruction (first message) from user context
            instructions = messages[0]['content']
            user_messages = messages[1:]
            input_content = "\n\n".join([m['content'] for m in user_messages])
            log.info(f"Input length: {len(input_content)}")
            
            code_string, _ = self.call_responses_api(
                model=CODER_MODEL,
                instructions=instructions,
                input_content=input_content,
                reasoning_effort=REASONING_EFFORT,
                temperature=CODER_TEMPERATURE
            )
            
            if "```python" in code_string:
                code_string = code_string.split("```python", 1)[1] 
            if "```" in code_string:
                code_string = code_string.split("```", 1)[0]
            
            code_string = code_string.strip()
            if code_string.lower().startswith("python"):
                code_string = code_string[6:].strip() 
            
            code_string = textwrap.dedent(code_string) 
            
            if not code_string or "GeneratedHeuristicAgent" not in code_string:
                raise RuntimeError("Model response missing 'GeneratedHeuristicAgent' class.")
                
            return code_string

        except Exception as e:
            log.error(f"Coder Model Failed: {e}")
            raise RuntimeError(f"Coder failed: {e}") 

    def write_agent_code(self, code_string: str) -> str:
        try:
            GENERATED_AGENT_PATH.write_text(code_string, encoding="utf-8")
            return str(GENERATED_AGENT_PATH)
        except Exception as e:
            log.error(f"Write failed: {e}")
            raise

    def reset_to_bootstrap(self):
        log.info("Resetting to bootstrap code...")
        self.write_agent_code(BOOTSTRAP_AGENT_CODE)

    def run(self):
        if not self.open_scorecard(): return

        try:
            self.reset_to_bootstrap()

            log.info("--- Iteration 0: Bootstrap ---")
            bs_summary, bs_log, _ = self.execute_sub_agent_batch(iteration=0)
            
            bs_entry = {
                "iteration": 0,
                "code_file_path": str(GENERATED_AGENT_PATH),
                "summary": bs_summary,
                "action_log": bs_log,
                "status": bs_summary.get("status", "SUCCESS")
            }
            self.meta_memory.append(bs_entry)

            for i in range(1, MAX_META_ITERATIONS + 1):
                log.info(f"--- Iteration {i}: Planning ---")
                
                prompt = self.build_gpt5_prompt()
                try:
                    new_code = self.call_coder_model(prompt)
                except Exception as e:
                    log.error(f"Iteration {i} Coder failed: {e}. Skipping.")
                    continue
                
                iter_path = self.log_dir / f"generated_agent_iter_{i}.py"
                iter_path.write_text(new_code, encoding="utf-8")
                self.write_agent_code(new_code)
                
                summary, action_log, _ = self.execute_sub_agent_batch(iteration=i)
                
                entry = {
                    "iteration": i,
                    "code_file_path": str(iter_path),
                    "summary": summary,
                    "action_log": action_log,
                    "status": summary.get("status", "SUCCESS"),
                    "error_message": summary.get("error_message", None)
                }
                self.meta_memory.append(entry)
                
                with open(self.meta_memory_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

        except Exception as e:
            log.critical(f"Run failed: {e}", exc_info=True)
        finally:
            self.close_scorecard()

if __name__ == "__main__":
    MetaAgentOrchestrator(META_AGENT_NAME, GAME_ID).run()