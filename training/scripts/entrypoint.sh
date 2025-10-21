#!/usr/bin/env bash

cd ..
source /home/jwang/Arc/.venv/bin/activate

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_PROJECT="arc-agents"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/configs/default_train.yaml"}



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
ds = data.get("deepspeed_config", "<none>")
seed = data.get("seed", "<default>")

lines = [
    f"📄 Config: {path}",
    f"📦 Model: {model}",
    f"🗂️ Train file: {train}",
    f"💾 Output dir: {output}",
    f"⚙️ DeepSpeed config: {ds}",
    f"🌱 Seed: {seed}",
]

print("\n".join(lines))
PY
)

echo "🚀 Launching SFT training with DeepSpeed"
echo "${CONFIG_SUMMARY}"


accelerate launch \
    --config_file "deepspeed/ddp_config.yml" \
    "${SCRIPT_DIR}/train.py" \
    --config "${CONFIG_FILE}"

