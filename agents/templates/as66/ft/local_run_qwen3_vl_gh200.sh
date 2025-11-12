set -euo pipefail

# Resolve repository root from this script's location
_SCRIPT_PATH="${BASH_SOURCE[0]}"
[[ "$_SCRIPT_PATH" != /* ]] && _SCRIPT_PATH="$(readlink -f "$PWD/$_SCRIPT_PATH")"
_REPO_ROOT="$(readlink -f "$(dirname "$_SCRIPT_PATH")/../../../..")"
cd "$_REPO_ROOT"

echo "[INFO] Repo root: $_REPO_ROOT"
echo "[INFO] PWD: $(pwd)"

# Activate local venv if present (kept minimal per your request)
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi



MODEL="Qwen/Qwen3-VL-4B-Thinking"
SERVED="qwen3-vl-4b-thinking"
PORT=8009

# Clean shutdown of the background server
VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    echo "[INFO] Stopping vLLM server (pid=$VLLM_PID)…"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[INFO] Launching vLLM server for $MODEL …"

vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 50000 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 30 \
  --enable-prefix-caching \
  --dtype auto \
  --disable-log-stats \
  --host 0.0.0.0 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 &

VLLM_PID=$!
echo "[INFO] vLLM PID: $VLLM_PID"

echo "[INFO] Waiting for server health on http://localhost:$PORT/health …"
for i in {1..120}; do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null; then
    echo "[INFO] Server is healthy."
    break
  fi
  [[ $i -eq 120 ]] && echo "[ERROR] Server failed to become healthy in time." && exit 1
  sleep 2
done

# OpenAI-compatible endpoint for your evaluation harness
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="EMPTY"
export AGENT_MODEL_OVERRIDE="$SERVED"

# Your agent’s expected ablation/env knobs (kept as in your submit script)
export AGENT_REASONING_EFFORT="low"
export INCLUDE_TEXT_DIFF="false"
export CONTEXT_LENGTH_LIMIT="50000"
export DOWNSAMPLE_IMAGES="false"
export IMAGE_DETAIL_LEVEL="high"
export IMAGE_PIXELS_PER_CELL="4"

echo "[INFO] Env configured:"
echo "  OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "  AGENT_MODEL_OVERRIDE=$AGENT_MODEL_OVERRIDE"
echo "  AGENT_REASONING_EFFORT=$AGENT_REASONING_EFFORT"
echo "  INCLUDE_TEXT_DIFF=$INCLUDE_TEXT_DIFF"
echo "  CONTEXT_LENGTH_LIMIT=$CONTEXT_LENGTH_LIMIT"
echo "  DOWNSAMPLE_IMAGES=$DOWNSAMPLE_IMAGES"
echo "  IMAGE_DETAIL_LEVEL=$IMAGE_DETAIL_LEVEL"
echo "  IMAGE_PIXELS_PER_CELL=$IMAGE_PIXELS_PER_CELL"

echo "[INFO] Starting evaluation (console output follows)…"
# Use python (venv) by default; switch to `uv run` if you prefer.
python evaluation/evaluate.py \
  --agent as66visualmemoryagent \
  --suite standard_suite \
  --num_runs 5 \
  --max_workers 30 \
  --max_actions 300

echo "[INFO] Evaluation finished."
