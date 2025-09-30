#!/usr/bin/env bash
#SBATCH --job-name=as66-qwen3-4b-sft
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/usr/xtmp/par55/slurm_logs/%x-%j.out
#SBATCH --error=/usr/xtmp/par55/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /usr/xtmp/par55/slurm_logs
mkdir -p /usr/xtmp/par55/huggingface_cache/as66_manual_traces
mkdir -p /usr/xtmp/par55/huggingface_cache/as66_qwen3_4b_sft

# Caches (fast SSD)
export HF_HOME=/usr/xtmp/par55/huggingface_cache
export TRANSFORMERS_CACHE=/usr/xtmp/par55/huggingface_cache
export HF_DATASETS_CACHE=/usr/xtmp/par55/huggingface_cache/datasets
export HUGGINGFACE_HUB_CACHE=/usr/xtmp/par55/huggingface_cache

# Optional: activate your env here (uv / conda / venv)
# module load cuda/12.x  # if your cluster uses modules
# uv pip install -U "transformers>=4.44" "trl>=0.10" accelerate datasets bitsandbytes deepspeed "flash-attn>=2" liger-kernel

# Build dataset from transcripts (you can run this separately as needed)
uv run agents/templates/as66/ft/dataset_builder.py \
  --transcripts-root /home/users/par55/ARC-AGI-3-Agents/transcripts \
  --out-root /usr/xtmp/par55/huggingface_cache/as66_manual_traces \
  --val-ratio 0.02

# Train
uv run agents/templates/as66/ft/train_qwen3_sft.py \
  --model ${QWEN_MODEL:-Qwen/Qwen3-4B} \
  --train-jsonl /usr/xtmp/par55/huggingface_cache/as66_manual_traces/train.jsonl \
  --val-jsonl   /usr/xtmp/par55/huggingface_cache/as66_manual_traces/val.jsonl \
  --out-dir /usr/xtmp/par55/huggingface_cache/as66_qwen3_4b_sft \
  --max-seq-len ${MAX_SEQ_LEN:-32768} \
  --epochs ${EPOCHS:-1.0} \
  --lr ${LR:-1e-5} \
  --micro-batch ${MICRO_BATCH:-1} \
  --grad-accum ${GRAD_ACCUM:-8} \
  --deepspeed agents/templates/as66/ft/deepspeed_zero3.json \
  --bf16 \
  --flashattn \
  --liger
