#!/bin/bash
# Precompute full-dataset regex matches for CXR sampling strata (CPU-only).
#
# Writes one jsonl per dataset under:
#   $MEDVQA_WORKSPACE_DIR/cache/cxr_report_labeling/<RUN_ID>/full_regex_matches/<dataset>.jsonl
#
# Examples:
#   ./medvqa/slurm/launch_cxr_precompute_regex_matches.sh
#   CPUS=32 TIME=12:00:00 ./medvqa/slurm/launch_cxr_precompute_regex_matches.sh
#   DATASETS="iuxray" CPUS=8 TIME=1:00:00 ./medvqa/slurm/launch_cxr_precompute_regex_matches.sh  # smoke one corpus
#
# Env overrides:
#   TIME MEM CPUS CONDA_ENV RUN_ID DATASETS NUM_PROCESSES TARGET_NODE
#   DATASETS = space-separated subset of: mimiccxr chexpert_plus rexgradient iuxray

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TIME=${TIME:-"12:00:00"}
MEM=${MEM:-"64G"}
CPUS=${CPUS:-32}
CONDA_ENV=${CONDA_ENV:-"py313"}
RUN_ID=${RUN_ID:-"v1_k1200_iu400"}
DATASETS=${DATASETS:-"mimiccxr chexpert_plus rexgradient iuxray"}
NUM_PROCESSES=${NUM_PROCESSES:-$CPUS}
TARGET_NODE=${TARGET_NODE:-""}

NODE_DIRECTIVE=""
if [ -n "$TARGET_NODE" ]; then
    NODE_DIRECTIVE="#SBATCH --nodelist=$TARGET_NODE"
fi

JOB_NAME="cxr_regex_precompute"
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
echo "RUN_ID=$RUN_ID NUM_PROCESSES=$NUM_PROCESSES DATASETS=$DATASETS"

module load conda
conda activate $CONDA_ENV

echo "------------------------------------------------"
echo "Python: \$(which python)"
echo "Python version: \$(python --version)"
echo "Conda prefix: \$CONDA_PREFIX"
echo "------------------------------------------------"

cd "$REPO_ROOT"

for ds in $DATASETS; do
  echo "======== Applying regex to dataset=\$ds ========"
  python -m medvqa.scripts.cxr_report_labeling.apply_regex_to_dataset \\
      --run_id "$RUN_ID" \\
      --dataset "\$ds" \\
      --num_processes $NUM_PROCESSES
  echo "Finished \$ds at \$(date)"
done

echo "All datasets done at \$(date)"
EOF

echo "Submitting job: $JOB_NAME"
echo "Log file pattern: $LOG_FILE"
echo "Temporary script: $TEMP_SCRIPT"
sbatch "$TEMP_SCRIPT"
rm "$TEMP_SCRIPT"
