#!/usr/bin/env bash
# Reproduce the paper's Sec.4 reliability-policy results (per-window, dual-expert).
#
# Pipeline:
#   1) dump per-window predictions of BOTH experts (Complex/spectral, Gated/gated_tf)
#      on the intra test set (MMPD / PURE);
#   2) run tools/policy_analysis/run_all.py to align, build features, fit selectors
#      (failure-prediction AUC), evaluate routing policies, and compute abstention.
#
# The official per-window labels (err_c/err_g/hr_c/hr_g/winner) are loaded from
# dumps/aligned/aligned_{mmpd,pure}.npz via --use_official_errors, so the failure
# labels follow the paper Eq. (oracle_failure = both>10; failure_c/g = margin DELTA).
#
# Prerequisites: same as reproduce.sh (conda env + preprocessed data + configs).

set -euo pipefail

CKPT=pretrained/PreTrainedModels
MMPD_DATA="/mnt/d/datasets/mini_MMPD"
MMPD_CACHE="/home/junjie/Workspace/Data/mini_MMPD"
PURE_DATA="/mnt/d/datasets/PURE_extract"
PURE_CACHE="/home/junjie/Workspace/Data/PURE"
DUMP_DIR=dumps/policy
OUT_DIR=outputs_policy

mkdir -p "$DUMP_DIR" "$OUT_DIR"

echo "===== Step 1: dump per-window predictions (MMPD) ====="
python scripts/dump_policy_windows.py \
  --ffn_branch spectral --test_dataset MMPD \
  --checkpoint "$CKPT/MMPD_MMPD_DareNet_spectral_seed42_Epoch29.pth" \
  --output "$DUMP_DIR/complex_mmpd.npz" \
  --data_path "$MMPD_DATA" --cached_path "$MMPD_CACHE"

python scripts/dump_policy_windows.py \
  --ffn_branch gated_tf --test_dataset MMPD \
  --checkpoint "$CKPT/MMPD_MMPD_DareNet_gated_tf_full_seed42_Epoch29.pth" \
  --output "$DUMP_DIR/gated_mmpd.npz" \
  --data_path "$MMPD_DATA" --cached_path "$MMPD_CACHE"

echo "===== Step 1b: dump per-window predictions (PURE) ====="
python scripts/dump_policy_windows.py \
  --ffn_branch spectral --test_dataset PURE \
  --checkpoint "$CKPT/PURE_PURE_DareNet_spectral_seed42_Epoch29.pth" \
  --output "$DUMP_DIR/complex_pure.npz" \
  --data_path "$PURE_DATA" --cached_path "$PURE_CACHE"

python scripts/dump_policy_windows.py \
  --ffn_branch gated_tf --test_dataset PURE \
  --checkpoint "$CKPT/PURE_PURE_DareNet_gated_tf_seed42_Epoch29.pth" \
  --output "$DUMP_DIR/gated_pure.npz" \
  --data_path "$PURE_DATA" --cached_path "$PURE_CACHE"

echo "===== Step 2: copy official labels next to dumps ====="
cp dumps/aligned/aligned_mmpd.npz dumps/aligned/aligned_pure.npz "$DUMP_DIR/"

echo "===== Step 3: run policy analysis ====="
python tools/policy_analysis/run_all.py \
  --dump_dir "$DUMP_DIR" \
  --out_dir "$OUT_DIR" \
  --use_official_errors

echo "Done. Tables in $OUT_DIR/tables/, figures in $OUT_DIR/figures/."
