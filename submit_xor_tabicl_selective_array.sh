#!/bin/bash
#SBATCH --job-name=xor_tabicl_selective
#SBATCH --partition=ext_vwl_norm
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-5
#SBATCH --output=logs/xor_tabicl_selective_%A_%a.out
#SBATCH --error=logs/xor_tabicl_selective_%A_%a.err

cd /work/smmxabba/bachelor/finetuningTabPFN
source /work/smmxabba/bachelor/tabpfn_env/bin/activate

# One array task per seed (SLURM_ARRAY_TASK_ID = 1..5). Each task runs all
# 7 selective strategies for its own seed only and writes a single partial
# result file -- run aggregate_xor_tabicl_extended_results.py after all 5
# array tasks have finished.
python run_xor_tabicl_selective_seed_task.py --seed ${SLURM_ARRAY_TASK_ID}
