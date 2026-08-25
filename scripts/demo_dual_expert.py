#!/usr/bin/env python
"""Dual-expert routing demo (paper Sec.3 / Sec.4).

Loads the Complex (spectral) and Gated (gated_tf) experts of DARE-Net and runs
them on the same inputs, then evaluates the routing operators described in the
paper:

    * BestSingle   — always use the per-dataset better expert
    * FixedAvg     — (y_c + y_g) / 2
    * SoftBlend    — alpha * y_c + (1 - alpha) * y_g
    * HardSelect   — per-window binary routing (1{pi=c} y_c + 1{pi=g} y_g)

The HardSelect policy uses the *ground-truth* winner as an oracle upper bound
(corresponding to the paper's "Oracle" row); a deployable selector would
replace it with a learned reliability score (see paper Sec.4).

Example (MMPD intra, both experts from released checkpoints):
    python scripts/demo_dual_expert.py \
        --complex_ckpt pretrained/PreTrainedModels/MMPD_MMPD_DareNet_spectral_seed42_Epoch29.pth \
        --gated_ckpt   pretrained/PreTrainedModels/MMPD_MMPD_DareNet_gated_tf_full_seed42_Epoch29.pth \
        --test_dataset MMPD --train_dataset MMPD
"""
import argparse
import sys

import numpy as np
import torch

# Make the repo root importable when run as `python scripts/...`.
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural_methods.model.DualExpertDareNet import DualExpertDareNet
from evaluation.post_process import get_hr as project_get_hr, _detrend
from scipy.signal import butter, filtfilt


def fast_hr(sig, sr=30):
    """Estimate HR from a waveform with the project's band-pass + Welch FFT."""
    try:
        s = _detrend(np.asarray(sig, dtype=np.float64), 100)
        b, a = butter(1, [0.75 / sr * 2, 2.5 / sr * 2], btype="bandpass")
        s = filtfilt(b, a, s)
        std = np.std(s)
        if std > 1e-10:
            s = (s - np.mean(s)) / std
        return float(project_get_hr(s, sr=sr))
    except Exception:
        return np.nan


def build_test_loader(dataset, cached_path, file_list_path):
    """Minimal MMPD/PURE test loader (160-frame chunks, 128x128, Standardized)."""
    from torch.utils.data import DataLoader

    if dataset == "MMPD":
        from dataset.data_loader.MMPDLoader import MMPDLoader as Loader
        info = dict(LIGHT=[1, 2, 3, 4], MOTION=[1, 2, 3, 4], EXERCISE=[1, 2],
                    SKIN_COLOR=[3, 4, 5, 6], GENDER=[1, 2], GLASSER=[1, 2],
                    HAIR_COVER=[1, 2], MAKEUP=[1, 2])
    else:
        from dataset.data_loader.PURELoader import PURELoader as Loader
        info = {}

    class Cfg:
        def __init__(self):
            self.INFO = type("I", (), info)()
            self.FS = 30
            self.DATASET = dataset
            self.DATA_PATH = ""
            self.CACHED_PATH = cached_path
            self.DO_PREPROCESS = False
            self.DATA_FORMAT = "NDCHW"
            self.BEGIN = 0.0
            self.END = 1.0
            self.EXP_DATA_NAME = ""
            self.FILE_LIST_PATH = file_list_path
            self.PREPROCESS = type("P", (), {
                "DATA_TYPE": ["Standardized"], "LABEL_TYPE": "Standardized",
                "DO_CHUNK": True, "CHUNK_LENGTH": 160,
                "CROP_FACE": type("C", (), {
                    "DO_CROP_FACE": True, "USE_LARGE_FACE_BOX": True,
                    "LARGE_BOX_COEF": 1.5,
                    "DETECTION": type("D", (), {
                        "DO_DYNAMIC_DETECTION": False,
                        "DYNAMIC_DETECTION_FREQUENCY": 30,
                        "USE_MEDIAN_FACE_BOX": False})()})(),
                "RESIZE": type("R", (), {"W": 128, "H": 128})(),
            })()

    loader = Loader(name="test", data_path="", config_data=Cfg())
    gen = torch.Generator().manual_seed(100)
    return DataLoader(loader, num_workers=0, batch_size=4, shuffle=False, generator=gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex_ckpt", required=True)
    ap.add_argument("--gated_ckpt", required=True)
    ap.add_argument("--test_dataset", default="MMPD")
    ap.add_argument("--train_dataset", default="PURE")
    ap.add_argument("--cached_path", default="/home/junjie/Workspace/Data/mini_MMPD")
    ap.add_argument("--file_list", default="", help="explicit file-list csv path")
    ap.add_argument("--max_windows", type=int, default=0,
                    help="cap the number of test windows (0 = all; useful for a quick demo)")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading DualExpertDareNet on {device} ...")
    model = DualExpertDareNet(args.complex_ckpt, args.gated_ckpt,
                              dataset_name=args.train_dataset, device=device)
    print("Both experts loaded (strict).")

    loader = build_test_loader(args.test_dataset, args.cached_path, args.file_list)

    ys_c, ys_g, ys_l = [], [], []
    with torch.no_grad():
        for batch in loader:
            data, labels = batch[0].float(), batch[1].float()
            y_c, y_g, _ = model.predict(data)
            for i in range(data.shape[0]):
                ys_c.append(y_c[i].cpu().numpy().flatten())
                ys_g.append(y_g[i].cpu().numpy().flatten())
                ys_l.append(labels[i].numpy().flatten())
            if args.max_windows and len(ys_c) >= args.max_windows:
                break
    ys_c = ys_c[:args.max_windows or None]
    ys_g = ys_g[:args.max_windows or None]
    ys_l = ys_l[:args.max_windows or None]

    n = len(ys_c)
    hr_c = np.array([fast_hr(s) for s in ys_c])
    hr_g = np.array([fast_hr(s) for s in ys_g])
    hr_l = np.array([fast_hr(s) for s in ys_l])

    err_c = np.abs(hr_c - hr_l)
    err_g = np.abs(hr_g - hr_l)
    valid = ~(np.isnan(err_c) | np.isnan(err_g) | np.isnan(hr_l))

    def mae(e):
        return float(np.mean(e[valid]))

    # routing operators
    hr_fa = np.array([fast_hr((c + g) / 2.0) for c, g in zip(ys_c, ys_g)])
    err_fa = np.abs(hr_fa - hr_l)

    best = "Gated" if mae(err_g) <= mae(err_c) else "Complex"
    err_bs = err_g if best == "Gated" else err_c

    # SoftBlend with a fixed alpha (per-sample blending needs a learned score)
    alpha = 0.5
    hr_blend = np.array([fast_hr(alpha * c + (1 - alpha) * g)
                         for c, g in zip(ys_c, ys_g)])
    err_blend = np.abs(hr_blend - hr_l)

    # Oracle hard-select (ground-truth winner per window)
    err_oracle = np.minimum(err_c, err_g)

    print(f"\n{'policy':<14} {'MAE':>8}")
    print("-" * 24)
    print(f"{'Complex':<14} {mae(err_c):8.3f}")
    print(f"{'Gated':<14} {mae(err_g):8.3f}")
    print(f"{'FixedAvg':<14} {mae(err_fa):8.3f}")
    print(f"{'SoftBlend(0.5)':<14} {mae(err_blend):8.3f}")
    print(f"{'BestSingle':<14} {mae(err_bs):8.3f}   ({best})")
    print(f"{'Oracle':<14} {mae(err_oracle):8.3f}   (upper bound)")


if __name__ == "__main__":
    main()
