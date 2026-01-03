#!/bin/bash
#
# A helper script to launch a Jupyter Notebook in an interactive SLURM session.
# Now supports Batch partition for high-memory requests.
#

# --- Configuration Defaults ---
MEM=${MEM:-"20G"}
CPUS=${CPUS:-"8"}
TIME=${TIME:-"3:00:00"}
PARTITION=${PARTITION:-"interactive"}
QOS=${QOS:-"interactive"}
CONDA_ENV=${CONDA_ENV:-"py313"}

# UPDATED: Use the Full Qualified Domain Name so your laptop can find it
CLUSTER_LOGIN_NODE="ih-condor.ing.puc.cl" 

# Default logic for Node list
NODELIST=${NODELIST:-"ih-condor"} 

# --- Argument Parsing ---
GPU_REQUEST=""
GPU_MESSAGE="No"
JUPYTER_PORT="30019"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpu)
        GPU_REQUEST="--gpus=1"
        GPU_MESSAGE="Yes (1 GPU)"
        shift
        ;;
        --env)
        CONDA_ENV="$2"
        shift; shift
        ;;
        --partition)
        PARTITION="$2"
        if [[ "$PARTITION" == "batch" ]]; then
            if [[ "$QOS" == "interactive" ]]; then QOS="batch"; fi
            if [[ "$NODELIST" == "ih-condor" ]]; then NODELIST=""; fi
        fi
        shift; shift
        ;;
        --qos)
        QOS="$2"
        shift; shift
        ;;
        --mem)
        MEM="$2"
        shift; shift
        ;;
        --time)
        TIME="$2"
        shift; shift
        ;;
        --node)
        NODELIST="$2"
        shift; shift
        ;;
        *)
        JUPYTER_PORT="$1"
        shift
        ;;
    esac
done

# --- Prepare Salloc Directives ---
NODE_ARG=""
if [ -n "$NODELIST" ]; then
    NODE_ARG="--nodelist=$NODELIST"
fi

echo "============================================================"
echo " REQUESTING INTERACTIVE SLURM SESSION..."
echo "============================================================"
echo "Configuration:"
echo "  - Partition:    $PARTITION"
echo "  - Memory:       $MEM"
echo "  - Node:         ${NODELIST:-Any (Scheduler decided)}"
echo "  - GPU Requested:$GPU_MESSAGE"
echo "  - Jupyter Port: $JUPYTER_PORT"
echo "------------------------------------------------------------"

# --- The salloc Command ---
salloc \
    -p "$PARTITION" \
    -q "$QOS" \
    -t "$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    $NODE_ARG \
    $GPU_REQUEST \
    bash -c '
# --- INSIDE SLURM ALLOCATION ---

echo "--> Loading modules and activating conda environment..."
module load conda
conda activate '$CONDA_ENV'

# Get the actual hostname of the compute node
NODE_HOSTNAME=$(srun -N1 -n1 hostname)

echo "============================================================"
echo " SESSION GRANTED ON NODE: $NODE_HOSTNAME"
echo "============================================================"
echo
echo "INSTRUCTIONS:"
echo "1. Wait for the Jupyter server to start below."
echo "2. Open a NEW terminal on your LOCAL (Mac) machine."
echo "3. Run this SSH JUMP command:"
echo
if [[ "$NODE_HOSTNAME" == "ih-condor" ]]; then
    # If on login node, simple forwarding
    echo "   ssh -N -L '$JUPYTER_PORT':localhost:'$JUPYTER_PORT' '${USER}@'$CLUSTER_LOGIN_NODE"
else
    # If on compute node (ih-loica), use Jump Host (-J)
    echo "   ssh -N -L '$JUPYTER_PORT':localhost:'$JUPYTER_PORT' -J '${USER}@${CLUSTER_LOGIN_NODE}' '${USER}@'$NODE_HOSTNAME"
fi
echo
echo "4. Open browser to: http://localhost:'$JUPYTER_PORT'/"
echo "------------------------------------------------------------"

# Start Jupyter
exec srun python -m jupyter notebook --no-browser --port "'$JUPYTER_PORT'" --notebook-dir="$HOME" --ip=0.0.0.0
'