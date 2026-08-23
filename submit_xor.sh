#!/bin/bash
#SBATCH --job-name=xor_sanity_check
#SBATCH --partition=ext_vwl_norm
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/xor_sanity_check_%j.out
#SBATCH --error=logs/xor_sanity_check_%j.err

cd /work/smmxabba/bachelor/finetuningTabPFN
python run_xor_sanity_check.py
