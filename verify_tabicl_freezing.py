"""
verify_tabicl_freezing.py
----------------------------
Freezing-verification mode for the TabICL v2 selective fine-tuning
strategies -- run this BEFORE submitting the full 5-task SLURM array
(submit_xor_tabicl_selective_array.sh).

What it does
---------------
Loads TabICL v2 once (via a small, fixed XOR sample -- seed/size are
irrelevant here, this is purely to build a real, weight-loaded nn.Module),
then for each of the 7 selective strategies:
  1. Instantiates TabICLSelectiveFinetuner(strategy=...)
  2. Calls `_load_pretrained()` + `_apply_freezing()` DIRECTLY -- NOT
     `fit()` -- so NO training epochs, NO optimizer step, NO metrics are
     computed. This is a pure freezing-configuration check.
  3. Prints total parameters, trainable parameters, percentage trainable,
     and the trainable parameter-name prefixes (via
     tabicl_selective_finetuner.verify_freezing).
  4. Asserts the freezing is exactly as intended (see
     tabicl_selective_finetuner.py's verify_freezing for what is checked).

If every strategy prints its verification block and the script reaches
"All strategies verified successfully" without an AssertionError, the
freezing logic is safe to run for real via the SLURM array.

Runs on a single GPU (or CPU, via --device cpu) in well under a minute --
no training loop, no full sweep, no result files are written anywhere.

Usage
-----
    python verify_tabicl_freezing.py
    python verify_tabicl_freezing.py --device cpu
"""

import argparse
import warnings

from run_xor_seed_sweep_tabicl import make_xor
from tabicl_selective_finetuner import STRATEGIES, TabICLSelectiveFinetuner

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Minimal, cheap dataset just to load TabICL's pretrained weights and
    # obtain a real nn.Module to freeze/inspect. Same generation function as
    # the actual sweep (make_xor, unchanged) -- seed/size don't matter for a
    # static freezing check, so fixed small values are used here.
    print("Generating a small XOR sample to load TabICL v2 ...")
    X, y = make_xor(n_samples=200, noise=0.01, n_features=10, random_state=1, gap=0.01)
    n_val = 40
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    results = {}
    for strategy in STRATEGIES:
        print(f"\n{'=' * 70}")
        print(f"Verifying strategy: {strategy}")
        print("=" * 70)

        finetuner = TabICLSelectiveFinetuner(
            strategy=strategy,
            epochs=1,       # irrelevant -- fit() is never called
            device=args.device,
            verbose=False,
        )

        # Deliberately calling the internal methods directly instead of
        # fit(), so no training epoch / optimizer step ever runs here.
        model, clf = finetuner._load_pretrained(X_train, y_train)
        model = model.to(args.device)
        finetuner._apply_freezing(model)  # prints + asserts internally

        results[strategy] = finetuner.verification_info

    print(f"\n{'=' * 70}")
    print("Summary -- trainable parameters per strategy")
    print("=" * 70)
    for strategy, info in results.items():
        print(
            f"{strategy:20s} trainable={info['trainable_params']:>10,} / "
            f"{info['total_params']:>12,}  ({info['pct_trainable']:5.2f}%)"
        )

    print("\nAll strategies verified successfully -- no assertion failures.")
    print("Safe to submit: sbatch submit_xor_tabicl_selective_array.sh")


if __name__ == "__main__":
    main()
