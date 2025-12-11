#!/usr/bin/env bash
#SBATCH --job-name=meta-ablations-launcher
#SBATCH --partition=compsci-gpu         
#SBATCH --cpus-per-task=2          
#SBATCH --mem=8G                    
#SBATCH --time=00:10:00              
#SBATCH --output=/usr/xtmp/%u/slurm_logs/%x-%j.out
#SBATCH --error=/usr/xtmp/%u/slurm_logs/%x-%j.err

set -euo pipefail
echo "[INFO] Starting Meta-Agent Ablation Launcher: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
mkdir -p "/usr/xtmp/$USER/slurm_logs"

# Determine Repository Root 
if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(readlink -f "$REPO_ROOT")"
else
    _SCRIPT_PATH="${BASH_SOURCE[0]}"
    [[ "$_SCRIPT_PATH" != /* ]] && _SCRIPT_PATH="$(readlink -f "$PWD/$_SCRIPT_PATH")"
    _SCRIPT_DIR="$(dirname "$_SCRIPT_PATH")"
    REPO_ROOT="$(readlink -f "$_SCRIPT_DIR/..")"
    if [[ ! -f "$REPO_ROOT/pyproject.toml" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        REPO_ROOT="$(readlink -f "$SLURM_SUBMIT_DIR")"
    fi
fi

cd "$REPO_ROOT"
echo "[INFO] Repository Root: $REPO_ROOT"

# Activate Python environment
VENV_PATH="$REPO_ROOT/.venv/bin/activate"
if [[ -f "$VENV_PATH" ]]; then
    echo "[INFO] Virtual environment found: $VENV_PATH"
else
    echo "[ERROR] Virtual environment not found at: $VENV_PATH"
    exit 1
fi

# Configuration
GAME="as66-821a4dcad9c2"
MAX_ITERATIONS=30
PARTITION="compsci-gpu"
CPUS=16
MEM="64G"
TIME="24:00:00"

echo "[INFO] Launching 4 ablation experiments..."
echo "[INFO] Game: $GAME"
echo "[INFO] Max Iterations: $MAX_ITERATIONS"
echo "[INFO] Episodes Per Iteration: 5 (hardcoded in script)"
echo ""

# Define ablation configurations (resolution, rules)
declare -a CONFIGS=(
    "16x16:specific"
    "16x16:general"
    "64x64:specific"
    "64x64:general"
)

SUBMITTED_JOBS=()

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r RESOLUTION RULES <<< "$config"
    
    # Build job name
    JOB_NAME="meta-${RESOLUTION}-${RULES}"
    
    # Build command arguments
    CMD_ARGS="--game $GAME --max_iterations $MAX_ITERATIONS"
    
    if [[ "$RESOLUTION" == "64x64" ]]; then
        CMD_ARGS="$CMD_ARGS --use_64x64"
    fi
    
    if [[ "$RULES" == "general" ]]; then
        CMD_ARGS="$CMD_ARGS --general"
    fi
    
    echo "[INFO] Submitting: $JOB_NAME"
    echo "       Args: $CMD_ARGS"
    
    JOB_SCRIPT=$(cat <<EOF
#!/usr/bin/env bash
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=/usr/xtmp/$USER/slurm_logs/${JOB_NAME}-%j.out
#SBATCH --error=/usr/xtmp/$USER/slurm_logs/${JOB_NAME}-%j.err

set -euo pipefail

echo "[INFO] =========================================="
echo "[INFO] Meta-Agent Job: $JOB_NAME"
echo "[INFO] SLURM Job ID: \${SLURM_JOB_ID}"
echo "[INFO] Node: \${SLURM_NODELIST}"
echo "[INFO] =========================================="
echo ""

cd "$REPO_ROOT"
source "$VENV_PATH"

echo "[INFO] Starting Meta-Agent Evaluation..."
echo "[INFO] Command: uv run python evaluation/evaluate_meta_agent.py $CMD_ARGS"
echo ""

uv run python evaluation/evaluate_meta_agent.py $CMD_ARGS

EXIT_CODE=\$?

if [[ \$EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "[SUCCESS] Meta-Agent run completed successfully!"
else
    echo ""
    echo "[ERROR] Meta-Agent run failed with exit code: \$EXIT_CODE"
fi

echo "[INFO] Job $JOB_NAME finished."
exit \$EXIT_CODE
EOF
)
    
    # Submit the job and capture job ID
    SUBMIT_OUTPUT=$(echo "$JOB_SCRIPT" | sbatch --job-name="$JOB_NAME" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
        SUBMITTED_JOBS+=("$JOB_NAME:$JOB_ID")
        echo "       Submitted as Job ID: $JOB_ID"
    else
        echo "       [ERROR] Failed to submit: $SUBMIT_OUTPUT"
    fi
    
    echo ""
    
    # Small delay to ensure unique timestamps
    sleep 0.5
done

echo "[INFO] =========================================="
echo "[INFO] All jobs submitted!"
echo "[INFO] =========================================="
echo ""
echo "Submitted Jobs:"
for job_info in "${SUBMITTED_JOBS[@]}"; do
    echo "  - $job_info"
done
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Cancel all with: scancel -u $USER"
echo ""
echo "Results will be in: $REPO_ROOT/evaluation_results/meta_agent_logs/"
echo ""