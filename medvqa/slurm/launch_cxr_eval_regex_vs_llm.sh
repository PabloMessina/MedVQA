#!/bin/bash
# Lightweight regex-vs-LLM metrics over sampled reports (CPU-only).
#
# Runs eval_regex_vs_llm with --lightweight (metrics_per_class.json only; no
# per-class FP/FN dumps). Matching uses batch_match_prompt_classes with
# NUM_PROCESSES workers (same Pool path as apply_regex_to_dataset).
#
# Default output:
#   <REPO>/.agent/regex_refine/_recall_scan_<stamp>/metrics_per_class.json
#
# Examples:
#   ./medvqa/slurm/launch_cxr_eval_regex_vs_llm.sh
#   CPUS=16 TIME=1:00:00 ./medvqa/slurm/launch_cxr_eval_regex_vs_llm.sh
#   LIGHTWEIGHT=0 OUT_DIR=.agent/regex_refine/_full_eval ./medvqa/slurm/launch_cxr_eval_regex_vs_llm.sh
#
# Env overrides:
#   TIME MEM CPUS CONDA_ENV RUN_ID MODEL_NAME NUM_PROCESSES LIGHTWEIGHT OUT_DIR
#   ANNOTATIONS_PATH SAMPLES_PATH CLASSES TARGET_NODE

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TIME=${TIME:-"1:00:00"}
MEM=${MEM:-"32G"}
CPUS=${CPUS:-16}
CONDA_ENV=${CONDA_ENV:-"py313"}
RUN_ID=${RUN_ID:-"v1_k1200_iu400"}
MODEL_NAME=${MODEL_NAME:-"gemini-2.5-flash-lite"}
NUM_PROCESSES=${NUM_PROCESSES:-$CPUS}
LIGHTWEIGHT=${LIGHTWEIGHT:-"1"}
OUT_DIR=${OUT_DIR:-""}
ANNOTATIONS_PATH=${ANNOTATIONS_PATH:-""}
SAMPLES_PATH=${SAMPLES_PATH:-""}
CLASSES=${CLASSES:-""}
TARGET_NODE=${TARGET_NODE:-""}

NODE_DIRECTIVE=""
if [ -n "$TARGET_NODE" ]; then
    NODE_DIRECTIVE="#SBATCH --nodelist=$TARGET_NODE"
fi

JOB_NAME="cxr_regex_vs_llm"
LOG_DIR="/mnt/workspace/$USER/slurm-out"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-%j.out"

TEMP_SCRIPT=$(mktemp)

# Expand OUT_DIR default inside the job so the stamp is job-time, not submit-time.
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
echo "RUN_ID=$RUN_ID MODEL_NAME=$MODEL_NAME NUM_PROCESSES=$NUM_PROCESSES LIGHTWEIGHT=$LIGHTWEIGHT"

module load conda
conda activate $CONDA_ENV

echo "------------------------------------------------"
echo "Python: \$(which python)"
echo "Python version: \$(python --version)"
echo "Conda prefix: \$CONDA_PREFIX"
echo "------------------------------------------------"

cd "$REPO_ROOT"

OUT_DIR="$OUT_DIR"
if [ -z "\$OUT_DIR" ]; then
  STAMP=\$(date +%Y%m%d_%H%M%S)
  OUT_DIR=".agent/regex_refine/_recall_scan_\$STAMP"
fi
mkdir -p "\$OUT_DIR"

EXTRA_ARGS=()
if [ "$LIGHTWEIGHT" = "1" ] || [ "$LIGHTWEIGHT" = "true" ] || [ "$LIGHTWEIGHT" = "yes" ]; then
  EXTRA_ARGS+=(--lightweight)
fi
if [ -n "$ANNOTATIONS_PATH" ]; then
  EXTRA_ARGS+=(--annotations_path "$ANNOTATIONS_PATH")
fi
if [ -n "$SAMPLES_PATH" ]; then
  EXTRA_ARGS+=(--samples_path "$SAMPLES_PATH")
fi
if [ -n "$CLASSES" ]; then
  # shellcheck disable=SC2206
  CLASS_ARR=($CLASSES)
  EXTRA_ARGS+=(--classes "\${CLASS_ARR[@]}")
fi

echo "OUT_DIR=\$OUT_DIR"
echo "EXTRA_ARGS=\${EXTRA_ARGS[*]:-}"

python -m medvqa.scripts.cxr_report_labeling.eval_regex_vs_llm \\
    --run_id "$RUN_ID" \\
    --model_name "$MODEL_NAME" \\
    --num_processes $NUM_PROCESSES \\
    --out_dir "\$OUT_DIR" \\
    "\${EXTRA_ARGS[@]}"

echo "Finished at \$(date)"
echo "Metrics: \$OUT_DIR/metrics_per_class.json"
EOF

echo "Submitting job: $JOB_NAME"
echo "Log file pattern: $LOG_FILE"
echo "Temporary script: $TEMP_SCRIPT"
sbatch "$TEMP_SCRIPT"
rm "$TEMP_SCRIPT"
