#!/bin/bash
#SBATCH --job-name=xor_thesis_figures
#SBATCH --partition=ext_vwl_norm
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/xor_thesis_figures_%j.out
#SBATCH --error=logs/xor_thesis_figures_%j.err

cd /work/smmxabba/bachelor/finetuningTabPFN
source /work/smmxabba/bachelor/tabpfn_env/bin/activate

# Figures 1+2: pure plotting from existing results.pkl files, no GPU needed.
python generate_xor_thesis_figures_summary.py

# Figure 3: refits vanilla + fine-tuned TabPFN/TabICL on seed=3 data purely
# for the decision-boundary plot (see script docstring) -- needs GPU.
python -c "import tabicl" || { echo "tabicl is not installed in this environment -- aborting figure 3."; exit 1; }
python generate_xor_thesis_figures_boundaries.py
