#!/usr/bin/env bash
#SBATCH --job-name=eval-vis-yesDiff-ctx200k-16px-300actions
#SBATCH --partition=compsci-gpu         
#SBATCH --cpus-per-task=16          
#SBATCH --mem=64G                    
#SBATCH --time=24:00:00              
#SBATCH --output=/usr/xtmp/%u/slurm_logs/%x-%j.out  # Use Slurm %u for username in output path
#SBATCH --error=/usr/xtmp/%u/slurm_logs/%x-%j.err   # Use Slurm %u for username in error path

set -euo pipefail # Exit on error, undefined variable, or pipe failure

echo "[INFO] Starting sbatch job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"

mkdir -p "/usr/xtmp/$USER/slurm_logs"

# Determine Repository Root 
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
VENV_PATH="$REPO_ROOT/.venv/bin/activate"

if [[ -f "$VENV_PATH" ]]; then
  echo "[INFO] Activating virtual environment: $VENV_PATH"
  source "$VENV_PATH"
else
  echo "[WARN] Virtual environment not found at '$VENV_PATH'. Trying to proceed, uv run might still work if uv is globally available or in PATH."
fi


echo "[INFO] Setting Environment Variables..."
export AGENT_MODEL_OVERRIDE="gpt-5"
export AGENT_REASONING_EFFORT="low"
export INCLUDE_TEXT_DIFF="true"
export CONTEXT_LENGTH_LIMIT="200000"
export DOWNSAMPLE_IMAGES="true"
export IMAGE_DETAIL_LEVEL="low"
export IMAGE_PIXELS_PER_CELL="16"


echo "  AGENT_MODEL_OVERRIDE=$AGENT_MODEL_OVERRIDE"
echo "  AGENT_REASONING_EFFORT=$AGENT_REASONING_EFFORT"
echo "  INCLUDE_TEXT_DIFF=$INCLUDE_TEXT_DIFF"
echo "  CONTEXT_LENGTH_LIMIT=$CONTEXT_LENGTH_LIMIT"
echo "  DOWNSAMPLE_IMAGES=$DOWNSAMPLE_IMAGES"
echo "  IMAGE_DETAIL_LEVEL=$IMAGE_DETAIL_LEVEL"
echo "  IMAGE_PIXELS_PER_CELL=$IMAGE_PIXELS_PER_CELL"


echo "[INFO] Starting evaluation command..."
uv run evaluation/evaluate.py \
    --agent as66visualmemoryagent \
    --suite standard_suite \
    --num_runs 5 \
    --max_workers 30 \
    --max_actions 300

echo "[INFO] Evaluation command finished."