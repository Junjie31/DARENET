#!/usr/bin/env bash
# Reproduce the paper's two core cross-dataset results using the released checkpoints.
#
# Prerequisites:
#   1. conda env with the dependencies in requirements.txt (see setup.sh)
#   2. preprocessed data (PURE / UBFC / MMPD) pointed to by the CACHED_PATH
#      fields in the configs.  Regenerate configs with your local paths via:
#          python scripts/gen_configs.py
#
# Each command runs inference in `only_test` mode and writes per-subject
# metrics to ./saved_metrics/<task>/.
set -euo pipefail

echo "============================================================"
echo "DareNet — core cross-dataset inference (seed42)"
echo "============================================================"

# PURE -> MMPD  (gated expert, MAE ~8.5 bpm)
python main.py --config_file configs/infer/PURE_MMPD_gatedtf_seed42.yaml

# UBFC -> MMPD  (gated expert, MAE ~8.2 bpm)
python main.py --config_file configs/infer/UBFC_MMPD_gatedtf_seed42.yaml
