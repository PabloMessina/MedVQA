#!/bin/bash
# Launch Tier-1 Gemini Batch CXR annotation loop on Slurm (CPU-only).
#
# Examples:
#   ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh
#   TIME=3-00:00:00 CHUNK_SIZE=2000 MAX_TOKENS_PER_REQUEST=2048 ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh
#   JOB_SOURCE=regex_positives DATASETS=iuxray MAX_CHUNKS=1 TIME=4:00:00 ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh
#
# Env overrides:
#   TIME MEM CPUS CONDA_ENV RUN_ID SAMPLES_PATH MODEL_NAME API_KEY_NAME
#   CHUNK_SIZE MAX_CHUNKS MAX_TOKENS_PER_REQUEST TEMPERATURE
#   SKIP_ANNOTATION_PATHS  # space-separated prior jsonl(s) for leftover-only runs
#   CONFIG_EXTRA REASONING_EFFORT  # Gemini 3 thinking_level via OpenAI-compat reasoning_effort
#   JOB_SOURCE  # samples | regex_positives | no_finding
#   DATASETS    # required for regex_positives / no_finding (exactly one corpus, e.g. iuxray)
#   COMPACT_LABELS_PATH  # optional compact jsonl for job_source=no_finding
#   SUBMIT_BACKOFF_INITIAL_S SUBMIT_BACKOFF_MAX_S
#   POST_COLLECT_SLEEP_S BATCH_POLL_INTERVAL_S TARGET_NODE

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TIME=${TIME:-"3-00:00:00"}
MEM=${MEM:-"16G"}
CPUS=${CPUS:-4}
CONDA_ENV=${CONDA_ENV:-"py313"}
RUN_ID=${RUN_ID:-"v1_k1200_iu400"}
SAMPLES_PATH=${SAMPLES_PATH:-""}
MODEL_NAME=${MODEL_NAME:-"gemini-2.5-flash-lite"}
API_KEY_NAME=${API_KEY_NAME:-"GEMINI_API_KEY"}
CHUNK_SIZE=${CHUNK_SIZE:-2000}
MAX_CHUNKS=${MAX_CHUNKS:-""}
MAX_TOKENS_PER_REQUEST=${MAX_TOKENS_PER_REQUEST:-2048}
TEMPERATURE=${TEMPERATURE:-0}
SKIP_ANNOTATION_PATHS=${SKIP_ANNOTATION_PATHS:-""}
CONFIG_EXTRA=${CONFIG_EXTRA:-""}
REASONING_EFFORT=${REASONING_EFFORT:-""}
JOB_SOURCE=${JOB_SOURCE:-"samples"}
DATASETS=${DATASETS:-""}
COMPACT_LABELS_PATH=${COMPACT_LABELS_PATH:-""}
SUBMIT_BACKOFF_INITIAL_S=${SUBMIT_BACKOFF_INITIAL_S:-60}
SUBMIT_BACKOFF_MAX_S=${SUBMIT_BACKOFF_MAX_S:-900}
POST_COLLECT_SLEEP_S=${POST_COLLECT_SLEEP_S:-60}
BATCH_POLL_INTERVAL_S=${BATCH_POLL_INTERVAL_S:-30}
TARGET_NODE=${TARGET_NODE:-""}

NODE_DIRECTIVE=""
if [ -n "$TARGET_NODE" ]; then
    NODE_DIRECTIVE="#SBATCH --nodelist=$TARGET_NODE"
fi

JOB_NAME="cxr_annotate_loop"
if [ "$JOB_SOURCE" != "samples" ]; then
    JOB_NAME="cxr_annotate_${JOB_SOURCE}"
    if [ -n "$DATASETS" ]; then
        JOB_NAME="${JOB_NAME}_${DATASETS// /_}"
    fi
fi
LOG_DIR="/mnt/workspace/$USER/slurm-out"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-%j.out"

# Build optional CLI args for the inner script (escaped for embedding).
EXTRA_ARGS=""
if [ -n "$SAMPLES_PATH" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --samples_path $(printf '%q' "$SAMPLES_PATH")"
fi
if [ -n "$MAX_CHUNKS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_chunks $MAX_CHUNKS"
fi
if [ -n "$SKIP_ANNOTATION_PATHS" ]; then
    # shellcheck disable=SC2086
    for p in $SKIP_ANNOTATION_PATHS; do
        EXTRA_ARGS="$EXTRA_ARGS --skip_annotation_paths $(printf '%q' "$p")"
    done
fi
if [ -n "$CONFIG_EXTRA" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --config_extra $(printf '%q' "$CONFIG_EXTRA")"
fi
if [ -n "$REASONING_EFFORT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --reasoning_effort $(printf '%q' "$REASONING_EFFORT")"
fi
if [ -n "$JOB_SOURCE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --job_source $(printf '%q' "$JOB_SOURCE")"
fi
if [ -n "$DATASETS" ]; then
    # shellcheck disable=SC2086
    for d in $DATASETS; do
        EXTRA_ARGS="$EXTRA_ARGS --datasets $(printf '%q' "$d")"
    done
fi
if [ -n "$COMPACT_LABELS_PATH" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --compact_labels_path $(printf '%q' "$COMPACT_LABELS_PATH")"
fi

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
echo "Loop: RUN_ID=$RUN_ID MODEL=$MODEL_NAME JOB_SOURCE=$JOB_SOURCE DATASETS=${DATASETS:-samples} CHUNK_SIZE=$CHUNK_SIZE MAX_TOKENS_PER_REQUEST=$MAX_TOKENS_PER_REQUEST TEMPERATURE=$TEMPERATURE MAX_CHUNKS=${MAX_CHUNKS:-unlimited}"
echo "SKIP_ANNOTATION_PATHS=${SKIP_ANNOTATION_PATHS:-none}"
echo "REASONING_EFFORT=${REASONING_EFFORT:-default}"

module load conda
conda activate $CONDA_ENV

echo "------------------------------------------------"
echo "Python: \$(which python)"
echo "Python version: \$(python --version)"
echo "Conda prefix: \$CONDA_PREFIX"
echo "------------------------------------------------"

cd "$REPO_ROOT"

python -m medvqa.scripts.cxr_report_labeling.run_annotation_batch_loop \\
    --run_id "$RUN_ID" \\
    --model_name "$MODEL_NAME" \\
    --api_key_name "$API_KEY_NAME" \\
    --chunk_size $CHUNK_SIZE \\
    --max_tokens_per_request $MAX_TOKENS_PER_REQUEST \\
    --temperature $TEMPERATURE \\
    --submit_backoff_initial_s $SUBMIT_BACKOFF_INITIAL_S \\
    --submit_backoff_max_s $SUBMIT_BACKOFF_MAX_S \\
    --post_collect_sleep_s $POST_COLLECT_SLEEP_S \\
    --batch_poll_interval_s $BATCH_POLL_INTERVAL_S \\
    $EXTRA_ARGS

echo "Job finished at \$(date)"
EOF

echo "Submitting job: $JOB_NAME"
echo "Log file pattern: $LOG_FILE"
echo "Temporary script: $TEMP_SCRIPT"
sbatch "$TEMP_SCRIPT"
rm "$TEMP_SCRIPT"
