"""Regenerate Stage C results.

    python -m pf.run_all --stage C                 # E1 (predictor test set) + E2-E6, tables and figures
    python -m pf.run_all --stage C --only E2 E4    # a subset
    python -m pf.run_all --stage C --n 10          # quick pass with 10 seeds per scenario

Predictor data/training (M6) must exist (models/predictor_P{1,2}.pt); see PROGRESS.md for the commands.
"""
from __future__ import annotations

import argparse
import os

# one math thread per worker process (episodes already run in parallel)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pf.eval.experiments import EXP_DIR, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["C"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--n", type=int, default=None, help="override episodes per scenario (default from YAML)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--skip-e1", action="store_true")
    args = ap.parse_args()
    ids = args.only or ["E1", "E2", "E3", "E4", "E5", "E6"]
    if "E1" in ids and not args.skip_e1:
        from pf.prediction.train import evaluate
        evaluate(split="test")
    for exp in [i for i in ids if i != "E1"]:
        run_experiment(EXP_DIR / f"{exp}.yaml", procs=args.procs, n_override=args.n)
    from pf.eval.figures import make_all
    make_all(ids)


if __name__ == "__main__":
    main()
