#!/bin/bash
# Usage: ./launch_train_phrase_grounding.sh <path_to_config>
# 
# Examples:
#
# 1. Default (1 GPU, 48G, 1 Day)
#    ./launch_train_phrase_grounding.sh /full/path/to/config.yaml
#
# 2. Override Memory and Time
#    MEM=64G TIME="2-00:00:00" ./launch_train_phrase_grounding.sh /full/path/to/config.yaml
#
# 3. Target a Specific Node
#    TARGET_NODE=ih-loica ./launch_train_phrase_grounding.sh /full/path/to/config.yaml
#
# 4. Everything Custom
#    CONDA_ENV=vlm TARGET_NODE=ih-loica MEM=35G TIME="1-10:00:00" CPUS=4 \
#      ./launch_train_phrase_grounding.sh /full/path/to/config.yaml

# --- 1. Arguments & Defaults ---
if [ -z "$1" ]; then
    echo "ERROR: Missing config path."
    echo "Usage: $0 <path_to_config_yaml>"
    exit 1
fi
CONFIG_PATH=$(realpath $1)

# --- Configurable SLURM Settings (Env Vars with Defaults) ---
TIME=${TIME:-"1-00:00:00"}  # Default 1 day
MEM=${MEM:-"48G"}           # Default 48 GB
CPUS=${CPUS:-6}             # Default 6 CPUs
GPUS=${GPUS:-1}             # Default 1 GPU
TARGET_NODE=${TARGET_NODE:-""} # Default: Any node
CONDA_ENV=${CONDA_ENV:-"py313"} # Default: py313

# Prepare dynamic directives
NODE_DIRECTIVE=""
if [ -n "$TARGET_NODE" ]; then
    NODE_DIRECTIVE="#SBATCH --nodelist=$TARGET_NODE"
fi

JOB_NAME="pg_$(basename "$CONFIG_PATH" .yaml)"
LOG_DIR="/mnt/workspace/$USER/slurm-out"
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
echo "Conda prefix: \$CONDA_PREFIX"
echo "------------------------------------------------"

# Debug GPU
echo "Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Run Script
python /home/pamessina/medvqa/medvqa/scripts/training_scripts/train_phrase_grounding.py \\
    --config_filepath "$CONFIG_PATH" \\
    --save

echo "Job finished at \$(date)"
EOF

# --- 3. Submit Job ---
echo "Submitting job: $JOB_NAME"
echo "Log file: $LOG_FILE"
echo "Config: $CONFIG_PATH"
echo "Temporary script: $TEMP_SCRIPT"
echo "Temporary script content: $(cat $TEMP_SCRIPT)"
echo "Running command: sbatch $TEMP_SCRIPT"

sbatch "$TEMP_SCRIPT"

# --- 4. Cleanup ---
rm "$TEMP_SCRIPT"