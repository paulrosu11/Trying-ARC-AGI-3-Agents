import requests
import os
import itertools
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from copy import deepcopy
import sys
import time
import ast # Import the ast module
from dotenv import load_dotenv # Import dotenv

# --- Load .env file ---
# Attempt to find the project root and load the .env file
try:
    script_location = Path(__file__).resolve()
    # Traverse upwards looking for a known root indicator (e.g., '.git' or 'main.py')
    project_root = script_location.parent
    while not (project_root / '.git').exists() and not (project_root / 'main.py').exists() and project_root != project_root.parent:
        project_root = project_root.parent

    # Check if we successfully found a root marker
    if project_root == project_root.parent and not (project_root / '.git').exists() and not (project_root / 'main.py').exists():
         print("Warning: Could not reliably determine project root. Assuming script parent's parent's parent's parent.")
         # Fallback to the previous calculation if markers aren't found
         project_root = script_location.parent.parents[3]


    dotenv_path = project_root / '.env'
    print(f"Attempting to load .env file from: {dotenv_path}") # Debug print

    loaded_dotenv = False
    if dotenv_path.exists():
        loaded_dotenv = load_dotenv(dotenv_path=dotenv_path, override=True)
        if loaded_dotenv:
            print(f"Successfully loaded environment variables from: {dotenv_path}")
        else:
             print(f"Warning: Found {dotenv_path} but failed to load variables.")
    elif (project_root / '.env.example').exists():
         loaded_dotenv = load_dotenv(dotenv_path=(project_root / '.env.example'), override=True)
         if loaded_dotenv:
             print(f"Loaded environment variables from: {project_root / '.env.example'} (using example as fallback)")
         else:
            print(f"Warning: Found {(project_root / '.env.example')} but failed to load variables.")
    else:
        print(f"Warning: .env file not found at calculated project root: {project_root}. Relying on system environment variables.")

except Exception as e:
    print(f"Error during .env file search/load: {e}")
    # Continue, relying on system environment

# Check immediately if the key is loaded
api_key_value = os.getenv("ARC_API_KEY")
if not api_key_value:
    print("\nError: ARC_API_KEY not found after attempting to load .env file.")
    print("Please ensure ARC_API_KEY is set correctly in your .env file or system environment.")
    sys.exit(1)
else:
    # Optional: Print confirmation that the key was found (masked for security)
    print(f"ARC_API_KEY loaded successfully (value starts with: {api_key_value[:4]}...)")


# --- Configuration ---
# Assuming this script is run from the project root
MANUAL_SCRIPT_PATH = Path("agents/templates/as66/manual_script.py")
GAME_ID = "as66-821a4dcad9c2" # Replace if needed

# --- Load WAYS_BY_LEVEL dynamically using AST ---
def load_ways_from_script(script_path: Path) -> Tuple[Dict[str, List[List[str]]], List[str]]:
    """Loads WAYS_BY_LEVEL and LEVEL_ORDER from the manual_script.py file using AST."""
    # Resolve the script path relative to the project root found earlier
    absolute_script_path = project_root / script_path
    if not absolute_script_path.exists():
        print(f"Error: Manual script not found at calculated path: {absolute_script_path}")
        # Try the original relative path as a fallback
        if not script_path.exists():
             print(f"Error: Manual script also not found at relative path: {script_path}")
             sys.exit(1)
        else:
             print(f"Warning: Using manual script from relative path: {script_path}")
             absolute_script_path = script_path # Use the relative path if found

    script_content = absolute_script_path.read_text(encoding="utf-8")
    ways_match = None
    order_match = None

    try:
        tree = ast.parse(script_content)
        for node in ast.walk(tree):
            # Check for assignments like WAYS_BY_LEVEL = {...} or WAYS_BY_LEVEL: type = {...}
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = None
                value_node = None
                if isinstance(node, ast.Assign):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        target = node.targets[0].id
                        value_node = node.value
                elif isinstance(node, ast.AnnAssign):
                     if isinstance(node.target, ast.Name):
                         target = node.target.id
                         value_node = node.value # Can be None if only annotation

                if value_node is None:
                    continue

                if target == "WAYS_BY_LEVEL":
                    try:
                        # Safely evaluate the literal dictionary
                        ways_match = ast.literal_eval(value_node)
                        if not isinstance(ways_match, dict):
                            print("Error: Parsed WAYS_BY_LEVEL is not a dictionary.")
                            ways_match = None # Reset if type mismatch
                    except ValueError:
                        print("Error: Could not safely evaluate WAYS_BY_LEVEL literal.")
                        ways_match = None
                elif target == "LEVEL_ORDER":
                    try:
                        # Safely evaluate the literal list
                        order_match = ast.literal_eval(value_node)
                        if not isinstance(order_match, list):
                            print("Error: Parsed LEVEL_ORDER is not a list.")
                            order_match = None # Reset if type mismatch
                    except ValueError:
                        print("Error: Could not safely evaluate LEVEL_ORDER literal.")
                        order_match = None

            # Stop searching if both found
            if ways_match is not None and order_match is not None:
                break

    except Exception as e:
        print(f"Error parsing script with AST: {e}")
        sys.exit(1)


    # Final check
    if not isinstance(ways_match, dict) or not isinstance(order_match, list):
        print("Error: Failed to extract WAYS_BY_LEVEL or LEVEL_ORDER using AST.")
        print("Please ensure they are defined correctly as literals in agents/templates/as66/manual_script.py")
        sys.exit(1)

    return ways_match, order_match

WAYS_BY_LEVEL, LEVEL_ORDER = load_ways_from_script(MANUAL_SCRIPT_PATH)
print("Successfully loaded WAYS_BY_LEVEL and LEVEL_ORDER.")

# --- API Interaction ---
def _root_url() -> str:
    scheme = os.getenv("SCHEME", "https")
    host = os.getenv("HOST", "three.arcprize.org")
    port = os.getenv("PORT", "443")
    if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"

def _headers() -> Dict[str, str]:
    # No longer raises ValueError immediately, relies on check after load_dotenv
    key = os.getenv("ARC_API_KEY", "").strip()
    if not key:
         # This should ideally not be reached if load_dotenv worked
         print("Critical Error: ARC_API_KEY is missing even after checking .env.")
         sys.exit(1)
    return {"X-API-Key": key, "Accept": "application/json", "Content-Type": "application/json"}

class APIClient:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.root_url = _root_url()
        self.headers = _headers() # Headers are now fetched after dotenv load attempt
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.guid: Optional[str] = None
        self.card_id: Optional[str] = None

    def open_card(self):
        try:
            r = self.session.post(f"{self.root_url}/api/scorecard/open", json={"tags": ["verify_routes"]}, timeout=20)
            r.raise_for_status()
            self.card_id = r.json()["card_id"]
            print(f"Opened scorecard: {self.card_id}")
        except requests.RequestException as e:
            print(f"Error opening scorecard: {e}")
            raise

    def close_card(self):
         if self.card_id:
            try:
                self.session.post(f"{self.root_url}/api/scorecard/close", json={"card_id": self.card_id}, timeout=20)
                print(f"Closed scorecard: {self.card_id}")
            except requests.RequestException as e:
                print(f"Warning: Error closing scorecard {self.card_id}: {e}")
            finally:
                self.card_id = None


    def _post_cmd(self, action_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        full_payload = deepcopy(payload)
        if self.guid:
            full_payload["guid"] = self.guid
        full_payload["game_id"] = self.game_id
        if self.card_id and action_name.upper() == "RESET":
             full_payload["card_id"] = self.card_id

        # Minimal logging for verification script
        # print(f"POST /api/cmd/{action_name} Payload: { {k:v for k,v in full_payload.items() if k != 'game_id'} }")
        try:
            response = self.session.post(f"{self.root_url}/api/cmd/{action_name}", json=full_payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("guid"):
                self.guid = data["guid"]
            # print(f"  Response: State={data.get('state')}, Score={data.get('score')}")
            return data
        except requests.RequestException as e:
            print(f"API Error during {action_name}: {e}")
            if e.response is not None:
                print(f"Response body: {e.response.text}")
            return {"error": str(e), "state": "ERROR", "score": -1} # Indicate error

    def reset(self) -> Dict[str, Any]:
        self.guid = None # Reset guid before calling RESET
        return self._post_cmd("RESET", {})

    def move(self, direction: str) -> Dict[str, Any]:
        action_map = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
        action_name = action_map.get(direction.lower())
        if not action_name:
            print(f"Invalid direction: {direction}")
            return {"error": "Invalid direction", "state": "ERROR", "score": -1}
        return self._post_cmd(action_name, {})

# --- Verification Logic ---
def verify_routes():
    """Generates combinations and runs them against the API."""
    # APIClient init moved here, after load_dotenv has run
    try:
        api_client = APIClient(GAME_ID)
        api_client.open_card()
    except ValueError as e: # Catch API key error during client init
        print(f"\nError: {e}")
        print("Please ensure ARC_API_KEY is correctly set in your .env file or system environment.")
        return # Exit the function
    except requests.exceptions.RequestException as e:
         print(f"\nAPI Error during initialization or opening scorecard: {e}")
         return


    level_ways = [WAYS_BY_LEVEL[lvl] for lvl in LEVEL_ORDER]
    index_products = list(itertools.product(*[range(len(ws)) for ws in level_ways])) # Make it a list

    total_combos = len(index_products)
    success_count = 0
    failures = []

    print(f"Verifying {total_combos} route combinations...")

    for i, idx_tuple in enumerate(index_products):
        parts = []
        full_move_sequence = []
        for lvl_idx, (lvl_name, way_idx) in enumerate(zip(LEVEL_ORDER, idx_tuple)):
            way = WAYS_BY_LEVEL[lvl_name][way_idx]
            full_move_sequence.extend(way)
            parts.append(f"{lvl_name}w{way_idx+1}")
        combo_key = "_".join(parts)

        print(f"\n[{i+1}/{total_combos}] Testing combo: {combo_key}")

        # Use a fresh API client (and thus session/GUID) for each combo for isolation
        try:
            combo_api_client = APIClient(GAME_ID)
             # Re-use the same scorecard ID for all tests in this run
            combo_api_client.card_id = api_client.card_id
        except ValueError as e: # Catch API key error during combo client init
             print(f"  ERROR: Could not initialize API Client for combo (API Key issue?): {e}")
             failures.append(combo_key + " (API Client Init failed)")
             continue
        except requests.exceptions.RequestException as e:
             print(f"  API Error during combo initialization: {e}")
             failures.append(combo_key + " (API Client Init failed - connection)")
             continue


        final_state = combo_api_client.reset()
        if final_state.get("error"):
            print(f"  ERROR: Failed to RESET.")
            failures.append(combo_key + " (RESET failed)")
            continue # Skip to next combo if RESET fails

        current_score = final_state.get("score", 0)
        last_move_failed = False
        result = final_state # Initialize result with the state after reset

        for move_idx, move_dir in enumerate(full_move_sequence):
             # Add a small delay between moves to avoid overwhelming the server
            time.sleep(0.05) # 50 milliseconds

            result = combo_api_client.move(move_dir) # Use combo client
            if result.get("error"):
                print(f"  ERROR on move {move_idx+1} ({move_dir}). Details: {result.get('error')}")
                failures.append(f"{combo_key} (failed on move {move_idx+1}: {move_dir})")
                last_move_failed = True
                break # Stop processing moves for this combo on error
            current_score = result.get("score", current_score)
            if result.get("state") == "GAME_OVER":
                 print(f"  GAME OVER on move {move_idx+1} ({move_dir}).")
                 failures.append(f"{combo_key} (GAME OVER on move {move_idx+1}: {move_dir})")
                 last_move_failed = True
                 break # Stop if game over
            elif result.get("state") == "WIN":
                 print(f"  WIN state reached on move {move_idx+1} ({move_dir}).")
                 # Don't break here, let the loop finish in case score condition matters
                 # but record this state.

        if not last_move_failed:
            # Check success condition (reaching Level 4 usually means score >= 3)
            # A WIN state would also be success.
            # Use the 'result' from the last successful move or reset
            final_actual_state = result.get("state")
            is_win_state = final_actual_state == "WIN"
            reached_level_4 = current_score >= 4

            if is_win_state or reached_level_4:
                print(f"  SUCCESS: Reached final score {current_score} / state {final_actual_state}")
                success_count += 1
            else:
                print(f"  FAILURE: Finished with score {current_score} / state {final_actual_state} (expected score >= 4 or WIN)")
                failures.append(f"{combo_key} (final score {current_score} / state {final_actual_state})")

        # No need to close combo_api_client's card, it shares the main one

    api_client.close_card() # Close the main scorecard once

    # --- Final Report ---
    print("\n" + "="*30)
    print("Verification Summary")
    print("="*30)
    print(f"Total combinations tested: {total_combos}")
    print(f"Successful routes (reached score >= 4 or WIN): {success_count}")
    print(f"Failed routes: {len(failures)}")

    if failures:
        print("\nFailed combinations:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nAll combinations successfully reached the target score/state!")

    print("="*30)

if __name__ == "__main__":
    try:
        verify_routes()
    # Removed specific ValueError catch here as APIClient init is now inside verify_routes
    except Exception as e:
        print(f"\nAn unexpected error occurred outside the main loop: {e}")

