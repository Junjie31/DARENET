#!/usr/bin/env bash
# Reproduce the paper's core results (Tab.3 cross-dataset + Tab.4 intra-dataset)
# using the released checkpoints under pretrained/PreTrainedModels/.
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
echo "DareNet — cross-dataset inference (Tab.3, 'Ours' rows, seed42)"
echo "============================================================"

python main.py --config_file configs/infer/PURE_UBFC_gatedtf_seed42.yaml
python main.py --config_file configs/infer/UBFC_PURE_spectral_seed42.yaml
python main.py --config_file configs/infer/PURE_MMPD_gatedtf_seed42.yaml
python main.py --config_file configs/infer/UBFC_MMPD_gatedtf_seed42.yaml

echo "============================================================"
echo "DareNet — intra-dataset inference (Tab.4, 'Ours' rows)"
echo "============================================================"
python main.py --config_file configs/infer/PURE_PURE_spectral.yaml
python main.py --config_file configs/infer/UBFC_UBFC_spectral.yaml
python main.py --config_file configs/infer/MMPD_MMPD_gatedtf.yaml
