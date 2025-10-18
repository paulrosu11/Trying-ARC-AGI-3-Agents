#!/usr/bin/env bash
#SBATCH --job-name=as66-qwen3-4b-sft
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/usr/xtmp/par55/slurm_logs/%x-%j.out
#SBATCH --error=/usr/xtmp/par55/slurm_logs/%x-%j.err

set -euo pipefail

# ----------------------- SCRATCH-ONLY ENV -----------------------
SCRATCH="/usr/xtmp/par55"
mkdir -p "$SCRATCH"/{slurm_logs,huggingface_cache,.cache/uv,.cache/pip,tmp}

# All caches/temp → SCRATCH (nothing in $HOME)
export HF_HOME="$SCRATCH/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

export XDG_CACHE_HOME="$SCRATCH/.cache"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TMPDIR="$SCRATCH/tmp"
export TMP="$SCRATCH/tmp"
export TEMP="$SCRATCH/tmp"
export PYTHONPYCACHEPREFIX="$SCRATCH/.pycache"

# ----------------------- REPO ROOT ------------------------------
if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(readlink -f "$REPO_ROOT")"
else
  _SCRIPT_PATH="${BASH_SOURCE[0]}"
  [[ "$_SCRIPT_PATH" != /* ]] && _SCRIPT_PATH="$(readlink -f "$PWD/$_SCRIPT_PATH")"
  _SCRIPT_DIR="$(dirname "$_SCRIPT_PATH")"
  REPO_ROOT="$(readlink -f "$_SCRIPT_DIR/../../../..")"
  if [[ ! -f "$REPO_ROOT/pyproject.toml" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(readlink -f "$SLURM_SUBMIT_DIR")"
  fi
fi
cd "$REPO_ROOT"
echo "[INFO] REPO_ROOT = $REPO_ROOT"

# ----------------------- DEPS (no flash-attn/liger) -------------
# Keep it light; everything caches to SCRATCH.
uv pip install -U \
  "transformers>=4.44" "trl>=0.10" accelerate datasets deepspeed bitsandbytes

# ----------------------- DATASET PATHS --------------------------
SAFE_DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/agents/templates/as66/ft/sft_data/manual_traces}"
LEGACY_DATA_ROOT="$REPO_ROOT/agents/templates/as66/ft/datasets/manual_traces"
TRANSCRIPTS_ROOT="${TRANSCRIPTS_ROOT:-$REPO_ROOT/transcripts}"
VAL_RATIO="${VAL_RATIO:-0.02}"

# Prefer SAFE path; fall back if legacy already populated
if [[ -s "$SAFE_DATA_ROOT/train.jsonl" && -s "$SAFE_DATA_ROOT/val.jsonl" ]]; then
  DATA_ROOT="$SAFE_DATA_ROOT"
elif [[ -s "$LEGACY_DATA_ROOT/train.jsonl" && -s "$LEGACY_DATA_ROOT/val.jsonl" ]]; then
  echo "[WARN] Using legacy data dir: $LEGACY_DATA_ROOT"
  DATA_ROOT="$LEGACY_DATA_ROOT"
else
  echo "[INFO] Building dataset → $SAFE_DATA_ROOT (from $TRANSCRIPTS_ROOT)"
  mkdir -p "$SAFE_DATA_ROOT"
  uv run -m agents.templates.as66.ft.dataset_builder \
    --transcripts-root "$TRANSCRIPTS_ROOT" \
    --out-root         "$SAFE_DATA_ROOT" \
    --val-ratio        "$VAL_RATIO"
  DATA_ROOT="$SAFE_DATA_ROOT"
fi

TRAIN_JSONL="$DATA_ROOT/train.jsonl"
VAL_JSONL="$DATA_ROOT/val.jsonl"
[[ -s "$TRAIN_JSONL" ]] || { echo "[ERROR] Missing $TRAIN_JSONL"; exit 1; }
[[ -s "$VAL_JSONL"   ]] || { echo "[ERROR] Missing $VAL_JSONL";   exit 1; }

# ----------------------- TRAINING KNOBS -------------------------
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
OUT_DIR="${OUT_DIR:-$SCRATCH/huggingface_cache/as66_qwen3_4b_sft}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
EPOCHS="${EPOCHS:-1.0}"
LR="${LR:-1e-5}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-agents/templates/as66/ft/deepspeed_zero3.json}"

echo "[INFO] Training config:"
echo "  model        = $QWEN_MODEL"
echo "  train.jsonl  = $TRAIN_JSONL"
echo "  val.jsonl    = $VAL_JSONL"
echo "  out-dir      = $OUT_DIR"
echo "  max-seq-len  = $MAX_SEQ_LEN"
echo "  epochs       = $EPOCHS"
echo "  lr           = $LR"
echo "  micro-batch  = $MICRO_BATCH"
echo "  grad-accum   = $GRAD_ACCUM"
echo "  deepspeed    = $DEEPSPEED_CFG"

# ----------------------- LAUNCH TRAINING ------------------------
# NOTE: no --flashattn and no --liger (avoids compiling/building in $HOME).
uv run -m agents.templates.as66.ft.train_qwen3_sft \
  --model "$QWEN_MODEL" \
  --train-jsonl "$TRAIN_JSONL" \
  --val-jsonl   "$VAL_JSONL" \
  --out-dir     "$OUT_DIR" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --epochs      "$EPOCHS" \
  --lr          "$LR" \
  --micro-batch "$MICRO_BATCH" \
  --grad-accum  "$GRAD_ACCUM" \
  --deepspeed   "$DEEPSPEED_CFG" \
  --bf16
