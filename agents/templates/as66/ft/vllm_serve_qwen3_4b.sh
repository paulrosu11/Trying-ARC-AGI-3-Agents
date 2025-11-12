#!/usr/bin/env bash
#SBATCH --job-name=qwen3-vl-serve-b50k
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=/usr/xtmp/par55/slurm_logs/%x-%j.out
#SBATCH --error=/usr/xtmp/par55/slurm_logs/%x-%j.err

set -euo pipefail

cd /usr/project/xtmp/par55/Trying-ARC-AGI-3-Agents

# Caches
export HF_HOME="/usr/xtmp/par55/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="/usr/xtmp/par55/torch_cache"
export XDG_CACHE_HOME="/usr/xtmp/par55/.cache"
export TMPDIR="/usr/xtmp/par55/tmp"

# Env
source .venv/bin/activate

trap 'kill ${VLLM_PID:-} 2>/dev/null || true' EXIT

MODEL="Qwen/Qwen3-VL-4B-Thinking"
SERVED="qwen3-vl-4b-thinking"
PORT=8009

echo "[INFO] Launching vLLM for $MODEL..."

# Notes:
# - No image-token-id / image-input-type needed; vLLM handles Qwen3-VL’s image preproc.

vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt image=32,video=0 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --disable-log-stats &

VLLM_PID=$!

echo "[INFO] Waiting for server..."
for i in {1..120}; do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null; then break; fi
  [[ $i -eq 120 ]] && echo "Server timeout" && exit 1
  sleep 5
done
echo "[INFO] Server ready."

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export AGENT_MODEL_OVERRIDE="$SERVED"

echo "[INFO] Running evaluation with VISUAL agent..."
python evaluation/evaluate.py \
    --agent as66visualmemoryagent \
    --suite standard_suite \
    --num_runs 5 \
    --max_workers 30 \
    --max_actions 500

echo "[INFO] Done."
