#!/usr/bin/env bash
#SBATCH --job-name=qwen-eval-a6000
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=/usr/xtmp/par55/slurm_logs/%x-%j.out
#SBATCH --error=/usr/xtmp/par55/slurm_logs/%x-%j.err

set -euo pipefail

cd /usr/project/xtmp/par55/Trying-ARC-AGI-3-Agents

# Redirect caches 
export HF_HOME="/usr/xtmp/par55/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="/usr/xtmp/par55/torch_cache"
export XDG_CACHE_HOME="/usr/xtmp/par55/.cache"
export TMPDIR="/usr/xtmp/par55/tmp"

# One venv with everything
source .venv/bin/activate

trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

MODEL="Qwen/Qwen3-4B-Instruct-2507"
SERVED="qwen3-4b-instruct"
PORT=8009

vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --port $PORT \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --gpu-memory-utilization 0.9 \
#  --max-model-len 20000 \
  --dtype auto \
  &

VLLM_PID=$!

echo "Waiting for server..."
for i in {1..120}; do
  curl -sf http://localhost:$PORT/health > /dev/null 2>&1 && break
  [ $i -eq 120 ] && echo "Server timeout" && exit 1
  sleep 5
done
echo "Server ready!"

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export AGENT_MODEL_OVERRIDE="$SERVED"

python evaluation/evaluate.py \
    --agent as66memoryagent \
    --suite standard_suite \
    --num_runs 5 \
    --max_workers 30 \
    --max_actions 500
