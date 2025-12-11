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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
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
    PROMPT_GENERAL_ARC_RULES,
    PROMPT_SYSTEM_INSTRUCTION,
    PROMPT_SYSTEM_INSTRUCTION_16,
    PROMPT_SYSTEM_INSTRUCTION_64,
    PROMPT_PROGRESSIVE_INSTRUCTION,
    PROMPT_CONDENSER_SYSTEM
)

from evaluation.metrics import GameMetrics, LevelMetrics, AttemptMetrics
from evaluation.report import save_summary_report, calculate_stats
from evaluation.config import EVALUATION_GAMES

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# --- Default Constants ---
CODER_MODEL = "gpt-5.1"
REASONING_EFFORT = "low"

DEFAULT_MAX_ITERATIONS = 30
EPISODES_PER_ITERATION = 5  # Changed from 2 to 5
ACTIONS_PER_EPISODE = 50

# --- Action Limit Polynomial Settings ---
ACTION_POLY_A = 12.0
ACTION_POLY_B = 1.5
ACTION_POLY_C = 15.0

# --- Context Management ---
SLIDING_WINDOW_SIZE = 3
CONTEXT_TOKEN_LIMIT = 50000

# --- Progressive Mode Defaults ---
CONDENSER_MODEL = "gpt-5.1"
CONDENSER_REASONING = "medium"
CONDENSER_MAX_RETRIES = 10
ACTION_GROWTH_FACTOR = 1.5
STUCK_PATIENCE = 2

META_MEMORY_DIR = project_root / "evaluation_results" / "meta_agent_logs"


def get_bootstrap_code(use_64x64: bool) -> str:
    """Returns the initial random agent code, adapted for the grid resolution."""
    if use_64x64:
        return r'''import random
from typing import Any, Dict, List, Optional
import logging

log = logging.getLogger(__name__)

class GeneratedHeuristicAgent:
    """Bootstrap Random Agent (64x64 Mode)"""
    def __init__(self):
        self.turn_count = 0
        self.scripted_moves = [] 
        log.info("Bootstrap Heuristic Agent (Random 64x64) initialized.")

    def choose_action(self, frame_data: dict) -> dict:
        self.turn_count += 1
        current_state = frame_data.get('state', 'NOT_PLAYED')
        
        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            self.turn_count = 0
            return {'name': 'RESET', 'data': {}}
        
        if self.turn_count <= len(self.scripted_moves):
            return {'name': self.scripted_moves[self.turn_count - 1], 'data': {}}

        # Random fallback
        possible_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        return {'name': random.choice(possible_actions), 'data': {}}
'''
    else:
        return r'''import random
from typing import Any, Dict, List, Optional
import logging
from agents.templates.as66.downsample import downsample_4x4, matrix16_to_lines

log = logging.getLogger(__name__)

class GeneratedHeuristicAgent:
    """Bootstrap Random Agent (16x16 Mode)"""
    def __init__(self):
        self.turn_count = 0
        self.scripted_moves = [] 
        log.info("Bootstrap Heuristic Agent (Random 16x16) initialized.")

    def choose_action(self, frame_data: dict) -> dict:
        self.turn_count += 1
        current_state = frame_data.get('state', 'NOT_PLAYED')
        
        if current_state in ("GAME_OVER", "NOT_PLAYED"):
            self.turn_count = 0
            return {'name': 'RESET', 'data': {}}
        
        if self.turn_count <= len(self.scripted_moves):
            return {'name': self.scripted_moves[self.turn_count - 1], 'data': {}}

        # Downsample check (to ensure imports work)
        full_frame_3d = frame_data.get('frame', [])
        if full_frame_3d:
            try:
                _ = downsample_4x4(full_frame_3d, take_last_grid=True, round_to_int=True)
            except Exception: pass

        possible_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        return {'name': random.choice(possible_actions), 'data': {}}
'''


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
            return {"state": "ERROR", "score": 0, "frame": [], "error": "No GUID"}
        
        payload = action.action_data.model_dump()
        payload["game_id"] = self.game_id
        payload["guid"] = self.guid
        if action.reasoning:
            payload["reasoning"] = action.reasoning

        return self._post(action.name, payload)


class MetaAgentOrchestrator:
    def __init__(self, game_id: str, config: argparse.Namespace):
        self.game_id = game_id
        self.config = config
        
        # Build a descriptive name for the log folder
        variant_tags = []
        if config.use_64x64: variant_tags.append("64x64")
        else: variant_tags.append("16x16")
        
        if config.general: variant_tags.append("Gen")
        else: variant_tags.append("Specific")
        
        if config.no_progressive: variant_tags.append("NoProg")
        else: variant_tags.append("Prog")
        
        self.agent_name = f"MetaAgent_{'_'.join(variant_tags)}_{game_id.split('-')[0]}"
        
        self.root_url = self._get_root_url()
        self.api_key = os.getenv("ARC_API_KEY", "")
        self.headers = {"X-API-Key": self.api_key, "Accept": "application/json"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.card_id: Optional[str] = None
        self.openai_client = OpenAI()

        # Add microseconds and process ID to ensure uniqueness
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        microseconds = datetime.now(timezone.utc).strftime("%f")
        pid = os.getpid()
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        
        self.run_id = f"{self.agent_name}_{timestamp}_{microseconds}_{slurm_job_id}_{pid}"
        self.log_dir = META_MEMORY_DIR / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: Each orchestrator gets its own agent file path
        self.generated_agent_path = self.log_dir / "current_agent.py"
        self.module_name = f"generated_agent_{slurm_job_id}_{pid}_{timestamp}"
        
        self.meta_memory_log_path = self.log_dir / "meta_memory.jsonl"
        self.meta_memory: List[Dict[str, Any]] = []
        
        # Progressive state
        self.best_level_solved = 0 
        self.stuck_counter = 0
        self.stuck_multiplier_level = 0 
        self.current_max_actions = 0 
        self.hardcoded_moves: List[str] = []

        self.update_action_limit(level=1)

        # --- Select Prompts Based on Config ---
        if self.game_id.startswith("as66") and not self.config.general:
            self.rules_prompt = PROMPT_AS66_RULES
        else:
            self.rules_prompt = PROMPT_GENERAL_ARC_RULES
            
        if self.config.use_64x64:
            self.system_instruction = PROMPT_SYSTEM_INSTRUCTION_64
        else:
            try:
                self.system_instruction = PROMPT_SYSTEM_INSTRUCTION_16
            except NameError:
                self.system_instruction = PROMPT_SYSTEM_INSTRUCTION

        log.info(f"Meta-Agent Config: {self.agent_name}")
        log.info(f"Run ID: {self.run_id}")
        log.info(f"Log Directory: {self.log_dir}")
        log.info(f"Agent File: {self.generated_agent_path}")
        log.info(f"Progressive: {not self.config.no_progressive}, 64x64: {self.config.use_64x64}")
        log.info(f"Episodes per iteration: {EPISODES_PER_ITERATION}")

    def _get_root_url(self) -> str:
        scheme = os.environ.get("SCHEME", "https")
        host = os.environ.get("HOST", "three.arcprize.org")
        port = os.environ.get("PORT", "443")
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            return f"{scheme}://{host}"
        return f"{scheme}://{host}:{port}"

    def calculate_action_limit(self, level: int) -> int:
        base_actions = (ACTION_POLY_A * (level ** ACTION_POLY_B)) + ACTION_POLY_C
        stuck_factor = ACTION_GROWTH_FACTOR ** self.stuck_multiplier_level
        limit = int(base_actions * stuck_factor)
        return max(limit, 1)

    def update_action_limit(self, level: int):
        self.current_max_actions = self.calculate_action_limit(level)
        log.info(f"Action limit updated for level {level}: {self.current_max_actions} actions")

    def open_scorecard(self) -> Optional[str]:
        tags = ["meta-agent", self.agent_name, self.run_id, f"game-{self.game_id}"]
        if not self.config.no_progressive:
            tags.append("progressive")
        
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            tags.append(f"slurm-{slurm_job_id}")
        
        try:
            r = self.session.post(f"{self.root_url}/api/scorecard/open", json={"tags": tags}, timeout=30)
            r.raise_for_status()
            data = r.json()
            self.card_id = data.get("card_id")
            if self.card_id:
                log.info(f"Scorecard opened: {self.card_id}")
                return self.card_id
            return None
        except Exception as e:
            log.error(f"Failed to open scorecard: {e}")
            return None

    def close_scorecard(self) -> None:
        if not self.card_id: return
        try:
            self.session.post(f"{self.root_url}/api/scorecard/close", json={"card_id": self.card_id}, timeout=30)
            log.info(f"Scorecard closed: {self.card_id}")
        except Exception: pass

    def load_agent_from_file(self) -> Any:
        """Load agent with unique module name to avoid caching conflicts."""
        try:
            # Use unique module name that includes iteration counter to force reload
            module_name = f"{self.module_name}_{len(self.meta_memory)}"
            
            # Remove old module from cache if exists
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            spec = importlib.util.spec_from_file_location(module_name, self.generated_agent_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not create module spec from {self.generated_agent_path}")
            
            gen_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = gen_module 
            spec.loader.exec_module(gen_module)
            
            AgentClass = getattr(gen_module, "GeneratedHeuristicAgent")
            return AgentClass()
        except Exception as e:
            log.error(f"Failed to load agent: {e}")
            raise

    def hash_frame(self, frame_dict: Dict[str, Any]) -> str:
        frame_json = json.dumps(frame_dict.get('frame', []))
        return hashlib.md5(frame_json.encode()).hexdigest()

    def _estimate_entry_tokens(self, entry: Dict[str, Any]) -> int:
        return len(json.dumps(entry)) // 4

    def filter_action_log(self, action_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if CONTEXT_TOKEN_LIMIT < 0: return action_log 

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
            compressed_lines.append(f"NOTE: Context limited. Showing {len(filtered_log)}/{len(action_log)} turns.")

        for entry in filtered_log:
            turn = entry["turn"]
            if turn in processed_turns: continue

            state_action_key = entry["state_action_key"]
            all_turns_for_key = seen_states.get(state_action_key, [turn])
            
            turn_group_str = f"{min(all_turns_for_key)}-{max(all_turns_for_key)}" if len(all_turns_for_key) > 1 else str(turn)
            
            line = (
                f"Turns [{turn_group_str}]:\n"
                f"  State (16x16 Representation):\n{textwrap.indent(entry['grid_view'], '    ')}\n"
                f"  Action: {json.dumps(entry['action'])}\n"
                f"  Result: State={entry['state']}, Score={entry['score']}"
            )
            compressed_lines.append(line)
            processed_turns.update(all_turns_for_key)
        
        return "\n".join(compressed_lines)

    def run_condenser_loop(self, action_log: List[Dict[str, Any]], start_score: int, target_score: int) -> List[str]:
        """Extract minimal action sequence for the solved level segment."""
        log.info(f"Condenser: Extracting moves for Score {start_score} -> {target_score}")
        
        segment_log = []
        for entry in action_log:
            if entry['score'] == start_score: segment_log.append(entry)
            if entry['score'] >= target_score:
                segment_log.append(entry)
                break
                
        if not segment_log: return []

        log_content = "\n".join([
            f"Turn {e['turn']}: {e['action']['name']} (StateHash: {e['state_action_key'].split('|')[0]})" 
            for e in segment_log
        ])

        current_input = (
            f"Segment: Score {start_score} -> {target_score}.\n"
            f"Identify the minimal sequence of actions to reach the level up.\n\n"
            f"{log_content}"
        )
        
        for attempt in range(1, 4):
            try:
                response = self.openai_client.chat.completions.create(
                    model=CONDENSER_MODEL,
                    messages=[
                        {"role": "system", "content": PROMPT_CONDENSER_SYSTEM},
                        {"role": "user", "content": current_input}
                    ],
                )
                
                content = response.choices[0].message.content or "[]"
                if "[" in content and "]" in content:
                    json_str = content[content.find("["):content.rfind("]")+1]
                    moves = json.loads(json_str)
                    
                    full_seq = self.hardcoded_moves + moves
                    valid, v_score, _ = self.validate_sequence(full_seq)
                    
                    if valid and v_score >= target_score:
                        log.info(f"Condenser: Validated sequence of {len(moves)} moves.")
                        return moves
                    
                    current_input = f"Sequence failed validation (Score {v_score} < {target_score}). Try again."
            except Exception as e:
                log.error(f"Condenser error: {e}")
        
        return []

    def validate_sequence(self, moves: List[str]) -> Tuple[bool, int, str]:
        client = _EnvClient(self.root_url, self.game_id, self.card_id, self.headers, self.session.cookies)
        frame = client.reset()
        for i, m in enumerate(moves):
            try:
                frame = client.act(GameAction.from_name(m))
                if frame['state'] in ('GAME_OVER', 'ERROR'):
                    return False, frame.get('score', 0), "Game Over/Error"
            except: return False, 0, "Exception"
        return True, frame.get('score', 0), "Success"

    def run_single_episode(self, agent: Any, iteration: int, episode_num: int) -> Tuple[GameMetrics, List[Dict[str, Any]], Dict[str, List[int]]]:
        """Run a single episode with COMPLETE metrics tracking."""
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
        
        # CRITICAL: Initialize level 1 metrics properly
        current_level = 1
        metrics.level_metrics[1] = LevelMetrics(level_number=1)
        current_attempt = AttemptMetrics(attempt_number=1)
        metrics.level_metrics[1].attempts.append(current_attempt)
        metrics.highest_level_reached = 1

        try:
            for j in range(self.current_max_actions):
                # Get grid representation
                full_frame_3d = frame_dict.get('frame', [])
                try:
                    grid_16x16 = downsample_4x4(full_frame_3d, take_last_grid=True, round_to_int=True)
                    grid_16x16_str = matrix16_to_lines(grid_16x16)
                except Exception:
                    grid_16x16_str = "(Grid Downsample Error)"

                # Get action from agent
                try:
                    action_dict = agent.choose_action(frame_dict)
                except Exception as e:
                    log.error(f"Agent crashed: {e}")
                    metrics.status = "ERROR"
                    metrics.error_message = str(e)
                    break

                action_name = action_dict.get("name", "ACTION5")
                action_data = action_dict.get("data", {})
                
                # Create state hash
                state_hash = self.hash_frame(frame_dict)
                action_hash = f"{action_name}:{json.dumps(action_data, sort_keys=True)}"
                state_action_key = f"{state_hash}|{action_hash}"
                
                # Log entry for meta-agent analysis
                log_entry = {
                    "turn": j + 1,
                    "action": action_dict,
                    "score": frame_dict['score'],
                    "state": frame_dict['state'],
                    "grid_view": grid_16x16_str,
                    "state_action_key": state_action_key,
                    "level_at_turn": current_level,
                    "is_game_over": False
                }
                
                # Execute action
                action = GameAction.from_name(action_name)
                action.set_data(action_data)
                
                prev_frame_hash = state_hash
                frame_dict = client.act(action)
                
                # CRITICAL: Update attempt metrics
                current_attempt.actions += 1
                if self.hash_frame(frame_dict) != prev_frame_hash:
                    current_attempt.state_changes += 1
                
                # CRITICAL: Update run-level metrics
                metrics.run_total_actions += 1
                metrics.final_score = max(metrics.final_score, frame_dict.get('score', 0))

                # Update log entry with game over status
                log_entry['is_game_over'] = (frame_dict['state'] == 'GAME_OVER')
                action_log_entries.append(log_entry)
                seen_state_actions.setdefault(state_action_key, []).append(j + 1)

                # Check for WIN condition
                if frame_dict['state'] == 'WIN':
                    metrics.status = "COMPLETED_RUN"
                    metrics.level_metrics[current_level].status = "COMPLETED"
                    current_attempt.status = "COMPLETED"
                    log.info(f"[Iter {iteration} | Ep {episode_num}] WIN! Final Score: {metrics.final_score}")
                    break 
                
                # Handle GAME_OVER
                if frame_dict['state'] == 'GAME_OVER':
                    current_attempt.status = "GAME_OVER"
                    current_attempt.game_overs = 1
                    metrics.total_game_overs_across_run += 1
                    
                    log.info(f"[Iter {iteration} | Ep {episode_num}] Game Over at turn {j+1}, Level {current_level}")
                    
                    # Reset and start new attempt
                    frame_dict = client.reset()
                    new_attempt_num = len(metrics.level_metrics[current_level].attempts) + 1
                    current_attempt = AttemptMetrics(attempt_number=new_attempt_num)
                    metrics.level_metrics[current_level].attempts.append(current_attempt)
                    continue 
                
                # Check for level progression
                new_score = frame_dict.get('score', 0)
                new_level_calc = new_score + 1
                
                if new_level_calc > current_level:
                    # Completed current level
                    metrics.level_metrics[current_level].status = "COMPLETED"
                    current_attempt.status = "COMPLETED"
                    
                    log.info(f"[Iter {iteration} | Ep {episode_num}] Level {current_level} -> {new_level_calc}")
                    
                    # Initialize next level
                    current_level = new_level_calc
                    metrics.highest_level_reached = max(metrics.highest_level_reached, current_level)
                    metrics.level_metrics[current_level] = LevelMetrics(level_number=current_level)
                    current_attempt = AttemptMetrics(attempt_number=1)
                    metrics.level_metrics[current_level].attempts.append(current_attempt)

            # If we exhausted all actions without winning
            if metrics.status == "IN_PROGRESS":
                metrics.status = "TIMEOUT"
                log.info(f"[Iter {iteration} | Ep {episode_num}] Timeout after {self.current_max_actions} actions")

        except Exception as e:
            log.error(f"Episode error: {e}", exc_info=True)
            metrics.status = "ERROR"
            metrics.error_message = str(e)

        finally:
            metrics.end_time = time.time()
            metrics.run_duration_seconds = metrics.end_time - metrics.start_time
        
        log.info(f"[Iter {iteration} | Ep {episode_num}] Done. Score: {metrics.final_score}, "
                f"Status: {metrics.status}, Actions: {metrics.run_total_actions}, "
                f"GameOvers: {metrics.total_game_overs_across_run}")
        
        return metrics, action_log_entries, seen_state_actions

    def execute_sub_agent_batch(self, iteration: int) -> Tuple[Dict[str, Any], str, List[GameMetrics]]:
        """Execute multiple episodes and generate reports."""
        log.info(f"[Iter {iteration}] Loading agent from {self.generated_agent_path}")
        agent = self.load_agent_from_file()
        
        metrics_list = []
        last_raw_log = []
        first_run_compressed_log = ""

        for i in range(EPISODES_PER_ITERATION):
            try:
                metrics, log_entries, seen_states = self.run_single_episode(agent, iteration, i+1)
                metrics_list.append(metrics)
                
                if i == 0:
                    first_run_compressed_log = self.compress_action_log(log_entries, seen_states)
                    last_raw_log = log_entries
            except Exception as e:
                log.error(f"[Iter {iteration}] Episode {i+1} failed: {e}")
                # Create error metrics
                error_metrics = GameMetrics(
                    game_id=self.game_id,
                    agent_name=f"{self.agent_name}_Iter{iteration}",
                    run_index=i+1,
                    start_time=time.time()
                )
                error_metrics.status = "ERROR"
                error_metrics.error_message = str(e)
                error_metrics.end_time = time.time()
                metrics_list.append(error_metrics)

        # Calculate statistics
        game_stats, overall_summary = calculate_stats(metrics_list)
        
        # Generate report
        report_path = self.log_dir / f"report_iter_{iteration}.txt"
        try:
            save_summary_report(
                str(report_path),
                game_stats,
                overall_summary, 
                metrics_list,
                f"{self.agent_name}_Iter{iteration}",
                f"MetaRun_{self.run_id}",
                EPISODES_PER_ITERATION
            )
            log.info(f"[Iter {iteration}] Report saved to {report_path}")
        except Exception as e:
            log.error(f"[Iter {iteration}] Failed to save report: {e}")
        
        # Progressive Logic
        if not self.config.no_progressive and metrics_list:
            best_run_score = max(m.final_score for m in metrics_list)
            
            if best_run_score > self.best_level_solved:
                log.info(f"[Iter {iteration}] New best score: {best_run_score} (previous: {self.best_level_solved})")
                
                moves = self.run_condenser_loop(last_raw_log, self.best_level_solved, best_run_score)
                if moves:
                    self.hardcoded_moves.extend(moves)
                    self.best_level_solved = best_run_score
                    self.stuck_counter = 0
                    self.stuck_multiplier_level = 0
                    self.update_action_limit(self.best_level_solved + 1)
                    log.info(f"[Iter {iteration}] Updated hardcoded moves: {len(self.hardcoded_moves)} total")
                else:
                    log.warning(f"[Iter {iteration}] Condenser failed to extract moves")
            else:
                self.stuck_counter += 1
                log.info(f"[Iter {iteration}] Stuck counter: {self.stuck_counter}/{STUCK_PATIENCE}")
                
                if self.stuck_counter >= STUCK_PATIENCE:
                    self.stuck_multiplier_level += 1
                    self.stuck_counter = 0
                    self.update_action_limit(self.best_level_solved + 1)
                    log.info(f"[Iter {iteration}] Increased stuck multiplier to {self.stuck_multiplier_level}")

        summary_for_llm = {
            "status": "SUCCESS",
            "avg_score": overall_summary.get("avg_final_score", 0),
            "max_score": max(m.final_score for m in metrics_list),
            "avg_actions": overall_summary.get("avg_total_actions", 0),
            "status_counts": list({m.status for m in metrics_list})
        }
        
        return summary_for_llm, first_run_compressed_log, metrics_list

    def build_gpt5_prompt(self) -> List[Dict[str, Any]]:
        system_prompt = self.system_instruction + "\n\n" + self.rules_prompt
        messages = [{"role": "system", "content": system_prompt}]

        if not self.config.no_progressive and self.hardcoded_moves:
            prog_msg = PROMPT_PROGRESSIVE_INSTRUCTION + f"\n```json\n{json.dumps(self.hardcoded_moves)}\n```"
            messages.append({"role": "system", "content": prog_msg})

        window = self.meta_memory[-SLIDING_WINDOW_SIZE:] if self.meta_memory else []
        for i, entry in enumerate(window):
            try:
                code = Path(entry["code_file_path"]).read_text(encoding="utf-8")
            except: code = "# Code not found"
            
            log_section = ""
            if i == len(window) - 1:
                log_section = f"\nExecution Log:\n{entry['action_log']}"

            content = f"--- Iteration {entry['iteration']} ---\nCode:\n```python\n{code}\n```\nResult: {json.dumps(entry['summary'])}{log_section}"
            messages.append({"role": "user", "content": content})
        
        messages.append({"role": "user", "content": "Analyze the log. Write the next iteration of the agent code."})
        return messages

    def call_coder_model(self, messages: List[Dict[str, Any]]) -> str:
        instructions = messages[0]['content']
        history = "\n".join([m['content'] for m in messages[1:]])
        
        try:
            response = self.openai_client.responses.create(
                model=CODER_MODEL,
                instructions=instructions,
                input=history,
                reasoning={"effort": REASONING_EFFORT},
            )
            if hasattr(response, 'output_text'): return response.output_text
            if response.output:
                for item in response.output:
                    if item.type == 'message':
                        return "".join([p.text for p in item.content if p.type == 'output_text'])
            return ""
        except Exception as e:
            log.error(f"Coder failed: {e}")
            raise e

    def run(self):
        if not self.open_scorecard(): 
            log.error("Failed to open scorecard, aborting run")
            return

        try:
            # Bootstrap
            bs_code = get_bootstrap_code(self.config.use_64x64)
            self.generated_agent_path.write_text(bs_code, encoding="utf-8")
            log.info(f"Bootstrap code written to {self.generated_agent_path}")
            
            # Iteration 0
            log.info("--- Iteration 0: Bootstrap Evaluation ---")
            summary, logs, _ = self.execute_sub_agent_batch(0)
            
            self.meta_memory.append({
                "iteration": 0,
                "code_file_path": str(self.generated_agent_path),
                "summary": summary,
                "action_log": logs,
                "status": "SUCCESS"
            })

            # Main loop
            for i in range(1, self.config.max_iterations + 1):
                log.info(f"--- Iteration {i}/{self.config.max_iterations} ---")
                
                prompt = self.build_gpt5_prompt()
                try:
                    new_code = self.call_coder_model(prompt)
                    
                    # Clean code block
                    if "```python" in new_code:
                        new_code = new_code.split("```python")[1].split("```")[0]
                    elif "```" in new_code:
                        new_code = new_code.split("```")[1].split("```")[0]
                    new_code = textwrap.dedent(new_code).strip()
                    
                    # Validate it has the class
                    if "GeneratedHeuristicAgent" not in new_code:
                        log.error(f"Iteration {i}: Generated code missing GeneratedHeuristicAgent class")
                        continue
                    
                    # Save iteration copy
                    iter_path = self.log_dir / f"generated_agent_iter_{i}.py"
                    iter_path.write_text(new_code, encoding="utf-8")
                    
                    # Update current agent
                    self.generated_agent_path.write_text(new_code, encoding="utf-8")
                    log.info(f"Iteration {i}: New agent code written")
                    
                    # Execute
                    summary, logs, _ = self.execute_sub_agent_batch(i)
                    
                    entry = {
                        "iteration": i,
                        "code_file_path": str(iter_path),
                        "summary": summary,
                        "action_log": logs,
                        "status": "SUCCESS"
                    }
                    self.meta_memory.append(entry)
                    
                    # Append to JSONL log
                    with open(self.meta_memory_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                        
                except Exception as e:
                    log.error(f"Iteration {i} failed: {e}", exc_info=True)
                    
                    error_entry = {
                        "iteration": i,
                        "code_file_path": str(self.generated_agent_path),
                        "summary": {"status": "ERROR", "error": str(e)},
                        "action_log": "",
                        "status": "ERROR",
                        "error_message": str(e)
                    }
                    self.meta_memory.append(error_entry)
                    
                    with open(self.meta_memory_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_entry) + "\n")

        except Exception as e:
            log.critical(f"Run failed catastrophically: {e}", exc_info=True)
        finally:
            self.close_scorecard()
            log.info(f"=== Run Complete ===")
            log.info(f"Logs saved to: {self.log_dir}")
            log.info(f"Total iterations: {len(self.meta_memory)}")
            if self.hardcoded_moves:
                log.info(f"Final hardcoded moves: {len(self.hardcoded_moves)}")
                log.info(f"Best level solved: {self.best_level_solved}")

def main():
    parser = argparse.ArgumentParser(description="Meta-Agent Evaluation System")
    parser.add_argument("--game", type=str, help="Single Game ID to run")
    parser.add_argument("--suite", type=str, choices=list(EVALUATION_GAMES.keys()), help="Game suite to run")
    parser.add_argument("--no-progressive", action="store_true", help="Disable progressive hardcoding")
    parser.add_argument("--general", action="store_true", help="Use general ARC prompts")
    parser.add_argument("--use_64x64", action="store_true", help="Use 64x64 grid resolution")
    parser.add_argument("--max_iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="Max iterations")
    args = parser.parse_args()

    games = []
    if args.game:
        games = [args.game]
    elif args.suite:
        games = EVALUATION_GAMES[args.suite]
    else:
        games = ["as66-821a4dcad9c2"]

    for game in games:
        log.info(f"========================================")
        log.info(f"Starting Meta-Agent for {game}")
        log.info(f"========================================")
        orchestrator = MetaAgentOrchestrator(game, args)
        orchestrator.run()
        log.info(f"Completed Meta-Agent for {game}")
        log.info(f"========================================")

if __name__ == "__main__":
    main()