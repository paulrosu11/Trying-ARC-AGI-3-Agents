#!/usr/bin/env bash
#!/bin/bash
#SBATCH --job-name=kag
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --exclude=research-external-11,research-external-12
#SBATCH --output=slurm-%j.out

cd ..
source /home/jwang/Arc/.venv/bin/activate

export NCCL_DEBUG=INFO
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_PROJECT="arc-agents-sft"
export TRITON_CACHE_DIR="/data/junlin/.triton/autotune"
export NCCL_P2P_LEVEL=NVL


CONFIG_FILE="configs/default_train.yaml"
DEEPSPEED_CONFIG="deepspeed/ddp_config.yml"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG}"


CONFIG_SUMMARY=$(python3 - "$CONFIG_FILE" <<'PY'
import sys
try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {exc}")

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}

model = data.get("model_name_or_path", "<missing>")
train = data.get("train_file", "<missing>")
output = data.get("output_dir", "<missing>")
seed = data.get("seed", "<default>")

import os
ds_config = os.environ.get("DEEPSPEED_CONFIG", "<none>")

lines = [
    f"📄 Config: {path}",
    f"📦 Model: {model}",
    f"🗂️ Train file: {train}",
    f"💾 Output dir: {output}",
    f"⚙️ DeepSpeed config: {ds_config}",
    f"🌱 Seed: {seed}",
]

print("\n".join(lines))
PY
)

echo "🚀 Launching SFT training with DeepSpeed"
echo "${CONFIG_SUMMARY}"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch \
    --config_file "${DEEPSPEED_CONFIG}" \
    "train.py" \
    --config "${CONFIG_FILE}"

