#!/bin/bash
#SBATCH --job-name=xor_seed_sweep_tabpfn
#SBATCH --partition=ext_vwl_norm
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/xor_seed_sweep_tabpfn_%j.out
#SBATCH --error=logs/xor_seed_sweep_tabpfn_%j.err

cd /work/smmxabba/bachelor/finetuningTabPFN
source /work/smmxabba/bachelor/tabpfn_env/bin/activate
python run_xor_seed_sweep_tabpfn.py
