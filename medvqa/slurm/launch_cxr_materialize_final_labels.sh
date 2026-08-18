#!/bin/bash
# Rematerialize released labels under results/.../final_labels/ (CPU, high memory).
#
# Loads full verify jsonl into memory — use 32G+ for MIMIC / CheXpert / ReX.
# No LLM API calls; deterministic merge from regex matches + verify jsonl.
#
# Examples:
#   DATASET=iuxray ./medvqa/slurm/launch_cxr_materialize_final_labels.sh
#   DATASET=mimiccxr MEM=64G TIME=2:00:00 ./medvqa/slurm/launch_cxr_materialize_final_labels.sh
#   STAGE=regex_backed DATASET=chexpert_plus ./medvqa/slurm/launch_cxr_materialize_final_labels.sh
#   DATASETS="iuxray mimiccxr chexpert_plus rexgradient" MEM=64G TIME=4:00:00 \
#     ./medvqa/slurm/launch_cxr_materialize_final_labels.sh
#
# Env overrides:
#   TIME MEM CPUS CONDA_ENV RUN_ID MODEL_NAME CONFIG_SLUG DATASET DATASETS STAGE TARGET_NODE

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TIME=${TIME:-"2:00:00"}
MEM=${MEM:-"32G"}
CPUS=${CPUS:-4}
CONDA_ENV=${CONDA_ENV:-"py313"}
RUN_ID=${RUN_ID:-"v1_k1200_iu400"}
MODEL_NAME=${MODEL_NAME:-"gemini-2.5-flash-lite"}
CONFIG_SLUG=${CONFIG_SLUG:-""}
DATASET=${DATASET:-""}
DATASETS=${DATASETS:-""}
STAGE=${STAGE:-"final"}
TARGET_NODE=${TARGET_NODE:-""}

if [ -z "$DATASET" ] && [ -z "$DATASETS" ]; then
  echo "Set DATASET=<id> or DATASETS=\"iuxray mimiccxr ...\"." >&2
  exit 1
fi

if [ -n "$DATASET" ] && [ -n "$DATASETS" ]; then
  echo "Use DATASET or DATASETS, not both." >&2
  exit 1
fi

if [ -n "$DATASET" ]; then
  DATASETS="$DATASET"
fi

NODE_DIRECTIVE=""
if [ -n "$TARGET_NODE" ]; then
  NODE_DIRECTIVE="#SBATCH --nodelist=$TARGET_NODE"
fi

JOB_NAME="cxr_materialize"
LOG_DIR="/mnt/workspace/$USER/slurm-out"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-%j.out"

TEMP_SCRIPT=$(mktemp)
cat << EOF > "$TEMP_SCRIPT"
#!/bin/bash
#SBATCH --job-name="$JOB_NAME"
#SBATCH --output="$LOG_FILE"
#SBATCH --partition=batch
#SBATCH --qos=batch
#SBATCH --time="$TIME"
#SBATCH --mem="$MEM"
#SBATCH --cpus-per-task="$CPUS"
$NODE_DIRECTIVE

echo "Job started on \$(hostname) at \$(date)"
echo "Repo: $REPO_ROOT"
echo "Resources: Time=$TIME Mem=$MEM CPUs=$CPUS Node=${TARGET_NODE:-Any}"
echo "RUN_ID=$RUN_ID MODEL_NAME=$MODEL_NAME STAGE=$STAGE DATASETS=$DATASETS"

module load conda
conda activate $CONDA_ENV

echo "------------------------------------------------"
echo "Python: \$(which python)"
echo "Python version: \$(python --version)"
echo "Conda prefix: \$CONDA_PREFIX"
echo "------------------------------------------------"

cd "$REPO_ROOT"

CONFIG_ARGS=()
if [ -n "$CONFIG_SLUG" ]; then
  CONFIG_ARGS=(--config_slug "$CONFIG_SLUG")
fi

for ds in $DATASETS; do
  echo "=== materialize --dataset \$ds --stage $STAGE ==="
  python -m medvqa.scripts.cxr_report_labeling.materialize_final_labels \\
    --run_id "$RUN_ID" \\
    --dataset "\$ds" \\
    --stage "$STAGE" \\
    --model_name "$MODEL_NAME" \\
    "\${CONFIG_ARGS[@]}"
done

echo "Job finished at \$(date)"
EOF

chmod +x "$TEMP_SCRIPT"
echo "Submitting: sbatch $TEMP_SCRIPT"
sbatch "$TEMP_SCRIPT"
rm -f "$TEMP_SCRIPT"
