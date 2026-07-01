#!/bin/bash
# run_job.sh
# ----------
# SLURM job script for a single OpenML task.
# Called by submit_all.sh — do not run directly.
#
# Required environment variable:
#   TASK_ID   : OpenML task ID to run
#
# Optional environment variables (override defaults):
#   METHOD    : no_finetuning | full_finetuning  (default: no_finetuning)
#   CONFIG    : config module name               (default: config_1)

#SBATCH --partition=ext_vwl_norm
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/task_%x_%j.out
#SBATCH --error=logs/task_%x_%j.err

set -e

METHOD=${METHOD:-no_finetuning}
CONFIG=${CONFIG:-config_1}

echo "=============================="
echo "Task ID   : $TASK_ID"
echo "Method    : $METHOD"
echo "Config    : $CONFIG"
echo "Node      : $(hostname)"
echo "=============================="

# activate environment
source /work/smmxabba/bachelor/tabpfn_env/bin/activate

# load git module (needed on LiDO3)
module load git 2>/dev/null || true

cd /work/smmxabba/bachelor/finetuningTabPFN

# TabPFN license token (set your token here or export it before submitting)
export TABPFN_TOKEN="${TABPFN_TOKEN}"

python main.py \
    --task_id "$TASK_ID" \
    --finetuning_method "$METHOD" \
    --config "$CONFIG"
