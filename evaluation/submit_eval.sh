#!/usr/bin/env bash
#SBATCH --job-name=arc-3-eval
#SBATCH --partition=compsci-gpu          # Use a CPU partition (adjust if needed)
#SBATCH --cpus-per-task=16           # Request 16 CPUs (adjust based on max_workers + overhead)
#SBATCH --mem=64G                    # Request 64GB memory (adjust if needed)
#SBATCH --time=24:00:00              # Time limit: 24 hours
#SBATCH --output=/usr/xtmp/%u/slurm_logs/%x-%j.out  # Use Slurm %u for username in output path
#SBATCH --error=/usr/xtmp/%u/slurm_logs/%x-%j.err   # Use Slurm %u for username in error path

set -euo pipefail # Exit on error, undefined variable, or pipe failure

# --- Environment Setup ---

# Create user-specific slurm log directory if it doesn't exist
# Slurm might create the base /usr/xtmp/%u, but maybe not slurm_logs
mkdir -p "/usr/xtmp/$USER/slurm_logs"

# Determine Repository Root (copied from your training script for consistency)
if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(readlink -f "$REPO_ROOT")"
else
  _SCRIPT_PATH="${BASH_SOURCE[0]}"
  [[ "$_SCRIPT_PATH" != /* ]] && _SCRIPT_PATH="$(readlink -f "$PWD/$_SCRIPT_PATH")"
  _SCRIPT_DIR="$(dirname "$_SCRIPT_PATH")"
  # Assuming the script is directly in the project root or similar location
  REPO_ROOT="$(readlink -f "$_SCRIPT_DIR")"
  # Fallback if run from elsewhere using SLURM_SUBMIT_DIR
  if [[ ! -f "$REPO_ROOT/pyproject.toml" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(readlink -f "$SLURM_SUBMIT_DIR")"
  fi
fi
cd "$REPO_ROOT"
echo "[INFO] Changed directory to REPO_ROOT = $REPO_ROOT"

# Activate Python environment - Assuming .venv is in the REPO_ROOT
# If it's elsewhere (like $HOME/Arc), adjust the path accordingly.
VENV_PATH="$REPO_ROOT/.venv/bin/activate"
# Alternative if venv is consistently in user's home:
# VENV_PATH="$HOME/Arc/.venv/bin/activate" 

if [[ -f "$VENV_PATH" ]]; then
  echo "[INFO] Activating virtual environment: $VENV_PATH"
  source "$VENV_PATH"
else
  echo "[WARN] Virtual environment not found at '$VENV_PATH'. Trying to proceed, uv run might still work if uv is globally available or in PATH."
fi

# Ensure necessary environment variables are loaded (uv run usually handles .env from REPO_ROOT)
# No need to manually export ARC_API_KEY if it's in the .env file in REPO_ROOT

# --- Run the Evaluation Command ---

echo "[INFO] Starting evaluation command..."
uv run evaluation/evaluate.py \
    --agent as66guidedagent \
    --suite debug_suite \
    --num_runs 5 \
    --max_workers 10 \
    --max_actions 200

echo "[INFO] Evaluation command finished."

