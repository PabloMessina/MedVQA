#!/bin/bash
# Usage: ./launch_eval_phrase_grounding__metrics.sh [python_args...]
# 
# Examples:
#
# 1. Default (No GPU, 20G, 2 Hours)
#    ./launch_eval_phrase_grounding__metrics.sh --dataset_name mscxr --predictions_and_gt_filepath /path/to/preds.pkl
#
# 2. Request High Memory
#    MEM=64G ./launch_eval_phrase_grounding__metrics.sh --dataset_name mscxr ...
#
# 3. Request a GPU (if needed for some specific metric computation)
#    GPUS=1 ./launch_eval_phrase_grounding__metrics.sh --dataset_name mscxr ...

# --- 1. Arguments ---
# We capture all arguments directly to pass to the python script
PYTHON_ARGS="$@"

if [ -z "$PYTHON_ARGS" ]; then
    echo "ERROR: No arguments provided."
    echo "Usage: $0 [python_args...]"
    exit 1
fi

# --- Configurable SLURM Settings (Env Vars with Defaults) ---
TIME=${TIME:-"0-02:00:00"}  # Default 2 hours
MEM=${MEM:-"20G"}           # Default 20 GB
CPUS=${CPUS:-2}             # Default 2 CPUs
GPUS=${GPUS:-0}             # Default 0 GPUs (Metrics usually CPU-only)
TARGET_NODE=${TARGET_NODE:-""} # Default: Any node
CONDA_ENV=${CONDA_ENV:-"py313"} # Default: py313

# Prepare dynamic directives
NODE_DIRECTIVE=""
if [ -n "$TARGET_NODE" ]; then
    NODE_DIRECTIVE="#SBATCH --nodelist=$TARGET_NODE"
fi

# Prepare GPU directive (only add if GPUS > 0)
GPU_DIRECTIVE=""
if [ "$GPUS" -gt 0 ]; then
    GPU_DIRECTIVE="#SBATCH --gres=gpu:$GPUS"
fi

JOB_NAME="pg_metrics"
LOG_DIR="/mnt/workspace/$USER/slurm-out/eval"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-%j.out"

# --- 2. Create Temporary SLURM Script ---
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
$GPU_DIRECTIVE
$NODE_DIRECTIVE

echo "Job started on \$(hostname) at \$(date)"
echo "Resources: Time=$TIME, Mem=$MEM, CPUs=$CPUS, GPUs=$GPUS, Node=${TARGET_NODE:-Any}"
echo "Python Args: $PYTHON_ARGS"

# --- Environment Setup ---
echo "Loading conda module..."
module load conda

echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV

# --- Verification ---
echo "------------------------------------------------"
echo "ENVIRONMENT VERIFICATION:"
echo "Python binary: \$(which python)"
echo "Python version: \$(python --version)"
echo "------------------------------------------------"

# Run Script
python /home/pamessina/medvqa/medvqa/scripts/evaluation_scripts/eval_phrase_grounding.py \\
    metrics \\
    $PYTHON_ARGS

echo "Job finished at \$(date)"
EOF

# --- 3. Submit Job ---
echo "Submitting metrics job: $JOB_NAME"
echo "Log file: $LOG_FILE"
echo "Python Args: $PYTHON_ARGS"
echo "Temporary script: $TEMP_SCRIPT"
# echo "Temporary script content: $(cat $TEMP_SCRIPT)"
echo "Running command: sbatch $TEMP_SCRIPT"

sbatch "$TEMP_SCRIPT"

# --- 4. Cleanup ---
rm "$TEMP_SCRIPT"