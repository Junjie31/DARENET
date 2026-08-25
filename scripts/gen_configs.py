#!/usr/bin/env python
"""Regenerate the train / inference configs under configs/{cross,intra,infer}.

This script is the single source of truth for dataset paths and splits.  Edit
the PATHS / INTRA_SPLIT / CROSS_SPLIT dicts below to point at your local data,
then run:

    python scripts/gen_configs.py

All generated YAMLs follow the rPPG-Toolbox / RhythmMamba config schema and are
the ones used to reproduce the paper results (see README.md for the exact
checkpoint<->table mapping).
"""
import os

# ---------------------------------------------------------------------------
# Local data paths (edit to match your machine).
#   data   : raw dataset folder (used when DO_PREPROCESS=True)
#   cached : preprocessed .npy folder (used when DO_PREPROCESS=False)
# ---------------------------------------------------------------------------
PATHS = {
    "PURE": {"data": "/mnt/d/datasets/PURE_extract",       "cached": "./data/PURE"},
    "UBFC": {"data": "/mnt/d/datasets/UBFC-RPPG/Dataset2", "cached": "./data/UBFC"},
    "MMPD": {"data": "/mnt/d/datasets/mini_MMPD",          "cached": "./data/mini_MMPD"},
}

# Intra-dataset splits (train / valid / test BEGIN-END fractions).
INTRA_SPLIT = {
    "PURE": {"train": (0.0, 0.6), "valid": (0.6, 1.0), "test": (0.6, 1.0)},
    "UBFC": {"train": (0.0, 0.72), "valid": (0.72, 1.0), "test": (0.72, 1.0)},
    "MMPD": {"train": (0.0, 0.7), "valid": (0.7, 0.8), "test": (0.8, 1.0)},
}

# Cross-dataset source split (train / valid).  Test is always the full target.
CROSS_SPLIT = {
    "PURE": {"train": (0.0, 0.8), "valid": (0.8, 1.0)},
    "UBFC": {"train": (0.0, 0.8), "valid": (0.8, 1.0)},
}

MMPD_INFO = """    INFO:
      LIGHT: [1, 2, 3, 4]
      MOTION: [1, 2, 3, 4]
      EXERCISE: [1, 2]
      SKIN_COLOR: [3, 4, 5, 6]
      GENDER: [1, 2]
      GLASSER: [1, 2]
      HAIR_COVER: [1, 2]
      MAKEUP: [1, 2]
"""

# (train_ds, test_ds, ffn_branch, model_name, config_stem, checkpoint_filename, seeds)
# One checkpoint per direction (seed42) is released; the seeds list keeps the
# inference config naming stable but only seed42 is shipped.
CROSS = [
    ("PURE", "UBFC", "gated_tf", "DareNet", "PURE_UBFC_gatedtf",
     "PURE_UBFC_DareNet_gated_tf_seed{seed}_Epoch29.pth", [42]),
    ("UBFC", "PURE", "spectral", "DareNet", "UBFC_PURE_spectral",
     "UBFC_PURE_DareNet_spectral_seed{seed}_Epoch29.pth", [42]),
    ("PURE", "MMPD", "gated_tf", "DareNet", "PURE_MMPD_gatedtf",
     "PURE_MMPD_DareNet_gated_tf_seed{seed}_Epoch29.pth", [42]),
    ("UBFC", "MMPD", "gated_tf", "DareNet", "UBFC_MMPD_gatedtf",
     "UBFC_MMPD_DareNet_gated_tf_seed{seed}_Epoch29.pth", [42]),
]

# (dataset, ffn_branch, model_name, config_stem, checkpoint_filename)
INTRA = [
    ("PURE", "spectral", "DareNet", "PURE_PURE_spectral",
     "PURE_PURE_DareNet_spectral_seed42_Epoch29.pth"),
    ("UBFC", "spectral", "DareNet", "UBFC_UBFC_spectral",
     "UBFC_UBFC_DareNet_spectral_seed42_Epoch29.pth"),
    ("MMPD", "gated_tf", "DareNet", "MMPD_MMPD_gatedtf",
     "MMPD_MMPD_DareNet_gated_tf_seed42_Epoch29.pth"),
]


def dataset_block(name, begin, end, with_info=False):
    p = PATHS[name]
    b = f"""    FS: 30
    DATASET: {name}
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "{p['data']}"
    CACHED_PATH: "{p['cached']}"
    EXP_DATA_NAME: ""
    BEGIN: {begin}
    END: {end}
"""
    if with_info:
        b += MMPD_INFO
    b += """    PREPROCESS:
      DATA_TYPE: ['Standardized']
      LABEL_TYPE: Standardized
      DO_CHUNK: True
      CHUNK_LENGTH: 160
      CROP_FACE:
        DO_CROP_FACE: True
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 128
        W: 128
"""
    return b


def _common_header(mode, model_name, ffn_branch, model_file_name, seed, aug):
    lines = ['BASE: [""]', f'TOOLBOX_MODE: "{mode}"', "TRAIN:"]
    lines += ["  BATCH_SIZE: 16", "  EPOCHS: 30", "  LR: 3e-4",
              f"  MODEL_FILE_NAME: {model_file_name}", f"  SEED: {seed}",
              "  PLOT_LOSSES_AND_LR: False", f"  AUG: {aug}"]
    lines += ["  DATA:"]
    return lines


def _common_tail(model_name, ffn_branch, model_path):
    lines = ["DEVICE: cuda:0", "NUM_OF_GPU_TRAIN: 1", "LOG:", "  PATH: ./logs/",
             "MODEL:", "  DROP_RATE: 0.2", f"  NAME: {model_name}"]
    if ffn_branch:
        lines.append(f"  FFN_BRANCH: {ffn_branch}")
    lines += ["INFERENCE:", "  BATCH_SIZE: 16", '  EVALUATION_METHOD: "FFT"',
              "  EVALUATION_WINDOW:", "    USE_SMALLER_WINDOW: False", "    WINDOW_SIZE: 10",
              f'  MODEL_PATH: "{model_path}"']
    return "\n".join(lines) + "\n"


def train_config(train_ds, test_ds, model_name, ffn_branch, model_file_name, split, test_split):
    lines = _common_header("train_and_test", model_name, ffn_branch, model_file_name, 42, 1)
    lines.append(dataset_block(train_ds, split['train'][0], split['train'][1], with_info=(train_ds == 'MMPD')))
    lines += ["VALID:", "  DATA:"]
    lines.append(dataset_block(train_ds, split['valid'][0], split['valid'][1], with_info=(train_ds == 'MMPD')))
    lines += ["TEST:", "  METRICS: ['MAE','RMSE','MAPE','Pearson','SNR']", "  USE_LAST_EPOCH: True", "  DATA:"]
    lines.append(dataset_block(test_ds, test_split[0], test_split[1], with_info=(test_ds == 'MMPD')))
    lines.append(_common_tail(model_name, ffn_branch, ""))
    return "\n".join(lines)


def infer_config(train_ds, test_ds, model_name, ffn_branch, model_file_name, ckpt, test_split, seed):
    lines = _common_header("only_test", model_name, ffn_branch, model_file_name, seed, 0)
    lines.append(dataset_block(train_ds, 0.0, 0.0, with_info=False))
    lines += ["VALID:", "  DATA:"]
    lines.append(dataset_block(train_ds, 0.0, 0.0, with_info=False))
    lines += ["TEST:", "  METRICS: ['MAE','RMSE','MAPE','Pearson','SNR']", "  USE_LAST_EPOCH: True", "  DATA:"]
    lines.append(dataset_block(test_ds, test_split[0], test_split[1], with_info=(test_ds == 'MMPD')))
    lines.append(_common_tail(model_name, ffn_branch, f"pretrained/PreTrainedModels/{ckpt}"))
    return "\n".join(lines)


os.makedirs("configs/cross", exist_ok=True)
os.makedirs("configs/intra", exist_ok=True)
os.makedirs("configs/infer", exist_ok=True)

for train, test, ffb, mname, stem, ckpt_tpl, seeds in CROSS:
    split = CROSS_SPLIT[train]
    mfn = f"{train}_{test}_{mname}_seed42"
    with open(f"configs/cross/{stem}.yaml", "w") as f:
        f.write(train_config(train, test, mname, ffb, mfn, split, (0.0, 1.0)))
    for s in seeds:
        ckpt = ckpt_tpl.format(seed=s)
        with open(f"configs/infer/{stem}_seed{s}.yaml", "w") as f:
            f.write(infer_config(train, test, mname, ffb, mfn, ckpt, (0.0, 1.0), s))

for ds, ffb, mname, stem, ckpt in INTRA:
    split = INTRA_SPLIT[ds]
    mfn = f"{ds}_{ds}_{mname}_seed42"
    with open(f"configs/intra/{stem}.yaml", "w") as f:
        f.write(train_config(ds, ds, mname, ffb, mfn, split, split['test']))
    with open(f"configs/infer/{stem}.yaml", "w") as f:
        f.write(infer_config(ds, ds, mname, ffb, mfn, ckpt, split['test'], 42))

print("Generated configs under configs/{cross,intra,infer}.")
