#!/usr/bin/env bash
#SBATCH --job-name=qwen-serve-and-eval
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/xtmp/par55/slurm_logs/%x-%j.out
#SBATCH --error=/usr/xtmp/par55/slurm_logs/%x-%j.err

set -euo pipefail

# This trap will kill the vLLM server on script exit 
trap 'echo "Evaluation finished, cleaning up vLLM server..."; kill $VLLM_PID' EXIT

#  ENV 
SCRATCH="/usr/xtmp/par55"
mkdir -p "$SCRATCH"/{slurm_logs,huggingface_cache,tmp,venvs}

export HF_HOME="$SCRATCH/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$SCRATCH/.cache"
mkdir -p "$XDG_CACHE_HOME"

export TMPDIR="$SCRATCH/tmp"
export TMP="$SCRATCH/tmp"
export TEMP="$SCRATCH/tmp"
export PYTHONPYCACHEPREFIX="$SCRATCH/.pycache"

# Root
if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(readlink -f "$REPO_ROOT")"
else
  _SCRIPT_PATH="${BASH_SOURCE[0]}"
  [[ "$_SCRIPT_PATH" != /* ]] && _SCRIPT_PATH="$(readlink -f "$PWD/$_SCRIPT_PATH")"
  _SCRIPT_DIR="$(dirname "$_SCRIPT_PATH")"
  REPO_ROOT="$(readlink -f "$_SCRIPT_DIR")"
  if [[ ! -f "$REPO_ROOT/main.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(readlink -f "$SLURM_SUBMIT_DIR")"
  fi
fi
cd "$REPO_ROOT"
echo "[INFO] REPO_ROOT = $REPO_ROOT"

# DEPS (NEW VENV) 
VENV_PATH="$SCRATCH/venvs/vllm_qwen_venv_v2" # Using a new venv path to be safe

# Check if python3.12 exists, otherwise fall back to python3
PYTHON_CMD="python3.12"
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "[WARN] python3.12 not found, falling back to python3"
    PYTHON_CMD="python3"
fi

if [ ! -d "$VENV_PATH" ]; then
  echo "[INFO] Creating new venv at $VENV_PATH using $PYTHON_CMD"
  $PYTHON_CMD -m venv $VENV_PATH
else
  echo "[INFO] Using existing venv at $VENV_PATH"
fi

echo "[INFO] Activating venv"
source "$VENV_PATH/bin/activate"

# Set cache dirs after activation
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

echo "[INFO] Installing/verifying dependencies in venv..."
# Install all deps for both serving AND evaluation
python -m pip install -U pip uv
uv pip install -U --no-cache \
  "transformers>=4.44" \
  "vllm>=0.5.0" \
  "openai>=1.20.0" \
  "accelerate" \
  "datasets" \
  "requests>=2.0.0" \
  "python-dotenv" # For evaluation script

echo "[INFO] Dependencies installed."

#  SERVER CONFIG 
export QWEN_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export PORT=${PORT:-8009}
export HOSTNAME=${SLURMD_NODENAME} # Get hostname from Slurm
export MAX_LEN=262144

echo "[INFO] Starting vLLM server in the background for $QWEN_MODEL on $HOSTNAME:$PORT"

#  LAUNCH SERVER (BACKGROUND) 
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port $PORT \
  --model "$QWEN_MODEL" \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len $MAX_LEN \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  --enforce-eager \
  & # Run in background

VLLM_PID=$! # Capture the server's Process ID
echo "[INFO] vLLM Server started in background with PID: $VLLM_PID"

#  WAIT FOR SERVER TO BE READY 
echo "[INFO] Waiting for vLLM server to be ready at http://$HOSTNAME:$PORT/health ..."
max_wait=3000 # 50 minutes
count=0
while ! curl -s --fail "http://$HOSTNAME:$PORT/health"; do
    sleep 5
    count=$((count+5))
    if [ $count -ge $max_wait ]; then
        echo "[ERROR] vLLM server failed to start after $max_wait seconds."
        # The trap will handle killing the PID
        exit 1
    fi
    echo "[INFO] Server not ready, waiting..."
done
echo "[INFO] vLLM Server is ready!"

#  EVALUATION ENV VARS 
export OPENAI_API_BASE="http://${HOSTNAME}:${PORT}/v1"
export OPENAI_API_KEY="EMPTY"
export AGENT_MODEL_OVERRIDE="Qwen/Qwen3-4B-Instruct-2507"

echo "[INFO] Set OPENAI_API_BASE=$OPENAI_API_BASE"
echo "[INFO] Set AGENT_MODEL_OVERRIDE=$AGENT_MODEL_OVERRIDE"

#  RUN EVALUATION (FOREGROUND) 
echo "[INFO] Starting parallel evaluation..."
python evaluation/evaluate.py \
    --agent as66memoryagent \
    --suite debug_suite \
    --num_runs 5 \
    --max_workers 10\
    --max_actions 5


echo "[INFO] Evaluation script finished."

# The 'trap' command at the top will now execute, killing the VLLM_PID