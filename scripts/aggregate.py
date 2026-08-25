#!/usr/bin/env python
"""Aggregate multi-seed inference results into the paper's mean±std tables.

After running the inference configs (see README.md), each run saves a
``saved_metrics/<filename_id>/<filename_id>_arrays_<timestamp>.npz`` file
containing ``predict_hr_fft_all`` and ``gt_hr_fft_all``.  This script groups
those files by task and prints the mean ± std of MAE / RMSE / MAPE / Pearson /
SNR across seeds, matching the numbers reported in the paper.

Usage:
    python scripts/aggregate.py --dir ./saved_metrics
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np

# (task prefix, number of seeds used in the paper, human-readable label)
TASKS = [
    ("PURE_MMPD_DareNet_gated_tf",  3, "PURE  -> MMPD   (DareNet gated)"),
    ("UBFC_MMPD_DareNet_gated_tf",  6, "UBFC  -> MMPD   (DareNet gated)"),
    ("PURE_UBFC_DareNet_gated_tf",  3, "PURE  -> UBFC   (DareNet gated)"),
    ("UBFC_PURE_DareNet_spectral",  3, "UBFC  -> PURE   (DareNet spectral)"),
]


def metrics_from_npz(path):
    d = np.load(path, allow_pickle=True)
    pred = d["predict_hr_fft_all"]
    gt = d["gt_hr_fft_all"]
    snr = d["snr_all"]
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    mape = np.mean(np.abs((pred - gt) / gt)) * 100.0
    pearson = np.corrcoef(pred, gt)[0, 1]
    return {
        "MAE": mae, "RMSE": rmse, "MAPE": mape, "Pearson": pearson, "SNR": float(np.mean(snr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./saved_metrics")
    args = ap.parse_args()

    for prefix, n_seeds, label in TASKS:
        # collect the latest npz per seed
        per_seed = []
        for sdir in sorted(glob.glob(os.path.join(args.dir, prefix + "_seed*"))):
            npz = sorted(glob.glob(os.path.join(sdir, "*_arrays_*.npz")))
            if npz:
                per_seed.append(metrics_from_npz(npz[-1]))
        if not per_seed:
            print(f"[SKIP] {label}: no results found under {prefix}_seed*")
            continue
        print(f"\n{label}  ({len(per_seed)}/{n_seeds} seeds)")
        print(f"{'metric':10s} {'mean':>10s} {'std':>10s}")
        for k in ["MAE", "RMSE", "MAPE", "Pearson", "SNR"]:
            vals = np.array([m[k] for m in per_seed])
            print(f"{k:10s} {vals.mean():10.4f} {vals.std(ddof=1):10.4f}")


if __name__ == "__main__":
    main()
