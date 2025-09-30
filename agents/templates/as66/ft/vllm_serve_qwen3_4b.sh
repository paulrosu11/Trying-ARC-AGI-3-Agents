#!/usr/bin/env bash
# Serve the fine-tuned model with vLLM (OpenAI-compatible server) + tool calling.
# vLLM supports function/tool calling through the Chat API. :contentReference[oaicite:14]{index=14}

set -euo pipefail

CKPT_DIR=${1:-/usr/xtmp/par55/huggingface_cache/as66_qwen3_4b_sft}
PORT=${PORT:-8009}
HOST=0.0.0.0

# Use static YaRN because vLLM uses a static factor; target ~131k if you need it. :contentReference[oaicite:15]{index=15}
EXTRA_ARGS="--rope-scaling type=yarn,factor=4.0"

# Start server
python -m vllm.entrypoints.openai.api_server \
  --host $HOST --port $PORT \
  --model "$CKPT_DIR" \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len ${MAX_MODEL_LEN:-65536} \
  --enable-auto-tool-choice \
  $EXTRA_ARGS
