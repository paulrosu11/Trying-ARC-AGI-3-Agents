#!/usr/bin/env bash

cd ../../evaluation
source /home/jwang/Arc/.venv/bin/activate

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_PROJECT="arc-agents"
export ARC_API_KEY=$ARC_API_KEY

# AVAILABLE_AGENTS.update({
#     "as66manualscripttext": AS66ManualScriptText,
#     "as66manualscriptvision": AS66ManualScriptVision,
#     "as66-manual-text": AS66ManualScriptText,
#     "as66-manual-vision": AS66ManualScriptVision,

#     "as66guidedagent": AS66GuidedAgent,
#     "as66-visual-guided": AS66VisualGuidedAgent,
#     "as66visualguidedagent": AS66VisualGuidedAgent,
#     "as66-guided": AS66GuidedAgent,
#     "as66guidedagent64": AS66GuidedAgent64,
# })
AGENT="as66guidedagent"
SUITE="debug_suite"



python3 evaluate.py \
    --agent $AGENT \
    --suite $SUITE \
    --max_actions 200

