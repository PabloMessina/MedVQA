#!/bin/bash
# Usage: ./launch_eval_phrase_grounding.sh <path_to_config> [python_args...]
# 
# Examples:
#
# 1. Default (1 GPU, 32G, 4 Hours)
#    ./launch_eval_phrase_grounding.sh /path/to/eval_config.yaml
#
# 2. Override Memory and Time
#    MEM=48G TIME="8:00:00" ./launch_eval_phrase_grounding.sh /path/to/eval_config.yaml
#
# 3. Override Python Args (e.g. batch size or workers)
#    ./launch_eval_phrase_grounding.sh /path/to/eval_config.yaml --num_workers 8 --max_images_per_batch 16
#
# 4. Everything Custom
#    CONDA_ENV=myenv MEM=64G CPUS=8 ./launch_eval_phrase_grounding.sh /path/to/eval_config.yaml --use_amp

# --- 1. Arguments & Defaults ---
if [ -z "$1" ]; then
    echo "ERROR: Missing config path."
    echo "Usage: $0 <path_to_config_yaml> [python_args...]"
    exit 1
fi
CONFIG_PATH=$(realpath $1)
shift # Remove config path from arguments
EXTRA_PYTHON_ARGS="$@" # Capture all remaining arguments

# --- Configurable SLURM Settings (Env Vars with Defaults) ---
TIME=${TIME:-"0-04:00:00"}  # Default 4 hours
MEM=${MEM:-"32G"}           # Default 32 GB
CPUS=${CPUS:-4}             # Default 4 CPUs
GPUS=${GPUS:-1}             # Default 1 GPU
TARGET_NODE=${TARGET_NODE:-""} # Default: Any node
CONDA_ENV=${CONDA_ENV:-"py313"} # Default: py313

# Prepare dynamic directives
NODE_DIRECTIVE=""
if [ -n "$TARGET_NODE" ]; then
    NODE_DIRECTIVE="#SBATCH --nodelist=$TARGET_NODE"
fi

JOB_NAME="eval_$(basename "$CONFIG_PATH" .yaml)"
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
#SBATCH --gres=gpu:$GPUS
$NODE_DIRECTIVE

echo "Job started on \$(hostname) at \$(date)"
echo "Config: $CONFIG_PATH"
echo "Resources: Time=$TIME, Mem=$MEM, CPUs=$CPUS, Node=${TARGET_NODE:-Any}"
echo "Extra Python Args: $EXTRA_PYTHON_ARGS"

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
# Note: config_filepath is passed first, then the extra overrides
python /home/pamessina/medvqa/medvqa/scripts/evaluation_scripts/eval_phrase_grounding.py \\
    inference \\
    --config_filepath "$CONFIG_PATH" \\
    $EXTRA_PYTHON_ARGS

echo "Job finished at \$(date)"
EOF

# --- 3. Submit Job ---
echo "Submitting job: $JOB_NAME"
echo "Log file: $LOG_FILE"
echo "Config: $CONFIG_PATH"
echo "Extra Args: $EXTRA_PYTHON_ARGS"
echo "Temporary script: $TEMP_SCRIPT"
# echo "Temporary script content: $(cat $TEMP_SCRIPT)"
echo "Running command: sbatch $TEMP_SCRIPT"

sbatch "$TEMP_SCRIPT"

# --- 4. Cleanup ---
rm "$TEMP_SCRIPT"