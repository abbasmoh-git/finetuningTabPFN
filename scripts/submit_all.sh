#!/bin/bash
# submit_all.sh
# -------------
# Submit one SLURM job per classification task in the TabArena suite.
#
# Usage:
#   bash scripts/submit_all.sh
#   bash scripts/submit_all.sh no_finetuning
#   bash scripts/submit_all.sh full_finetuning

set -e

METHOD=${1:-no_finetuning}
CONFIG=${CONFIG:-config_1}

# make sure logs directory exists
mkdir -p logs

echo "Submitting jobs for method: $METHOD"
echo "Config: $CONFIG"
echo "------------------------------"

# activate environment to run list_task_ids.py
source /work/smmxabba/bachelor/tabpfn_env/bin/activate
cd /work/smmxabba/bachelor/finetuningTabPFN

# get all classification task IDs
TASK_IDS=$(python scripts/list_task_ids.py)

COUNT=0
for TASK_ID in $TASK_IDS; do
    JOB_NAME="task_${TASK_ID}_${METHOD}"
    sbatch \
        --job-name="$JOB_NAME" \
        --export=ALL,TASK_ID="$TASK_ID",METHOD="$METHOD",CONFIG="$CONFIG" \
        scripts/run_job.sh
    echo "  submitted: task $TASK_ID"
    COUNT=$((COUNT + 1))
done

echo "------------------------------"
echo "Total jobs submitted: $COUNT"
echo "Monitor with: squeue -u smmxabba"
