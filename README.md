# DareNet

**DareNet: Domain-Aware Remote Physiological Measurement**

DareNet is a dual-expert backbone for cross-domain remote photoplethysmography (rPPG)
estimation, built on [RhythmMamba](https://arxiv.org/abs/2404.06483). It introduces two
key modules:

1. **Stability-Aware Fusion Stem (SAFS)** — stabilises feature energy with RMS
   normalisation and adaptively fuses the raw-frame and temporal-difference branches via a
   lightweight `SafuGate`, mitigating feature-scale drift across domains.
2. **Domain-Aware Robust Expert (DARE)** — replaces the single FFN with two complementary
   experts:
   - **Complex (spectral) expert**: global spectral analysis in the complex frequency
     domain, preserving phase coherence; suited to clean, periodic signals.
   - **Gated (gated_tf) expert**: gated time–frequency fusion; suited to motion-corrupted
     signals.

Both experts share the stem and Mamba backbone architecture; the `ffn_branch` in
`Frequencydomain_FFN` selects one of `{spectral, gated_tf}`, so a single weight structure
loads either checkpoint.

> This codebase builds on [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) and
> [RhythmMamba](https://github.com/zizheng-guo/RhythmMamba), following their
> training/evaluation protocol.

---

## Repository structure

```
DareNet/
├── main.py                  # train / test entry point (train_and_test / only_test)
├── config.py                # configuration system (yacs)
├── configs/
│   ├── cross/               # cross-dataset training configs
│   ├── intra/               # intra-dataset training configs
│   └── infer/               # inference configs (only_test)
├── neural_methods/
│   ├── model/DareNet.py            # single-expert backbone (SAFS + switchable FFN expert)
│   ├── model/DualExpertDareNet.py  # dual-expert MoE view (routing operators)
│   ├── model/RhythmMamba.py        # RhythmMamba baseline (code only, no weights)
│   └── trainer/                    # per-model trainers
├── dataset/                 # PURE / UBFC / MMPD data loaders
├── evaluation/              # metrics (MAE/RMSE/MAPE/Pearson/SNR) + HR estimation
├── unsupervised_methods/    # traditional methods (GREEN/ICA/CHROM/LGI/PBV/POS)
├── pretrained/PreTrainedModels/  # released checkpoints (see CHECKSUMS.txt)
├── scripts/
│   ├── gen_configs.py          # regenerate configs with your local data paths
│   ├── aggregate.py            # aggregate multi-seed results into mean±std
│   └── dump_policy_windows.py  # per-window dual-expert dump (Sec.4 policy)
├── tools/policy_analysis/     # reliability-policy analysis pipeline (Sec.4)
├── reproduce.sh             # reproduce the core cross-dataset results
└── setup.sh                 # environment setup (conda / uv)
```

---

## Setup

```bash
bash setup.sh conda        # or: bash setup.sh uv
conda activate DareNet
```

Key dependencies are listed in `requirements.txt`:
`torch==2.1.2+cu121`, `mamba-ssm==2.2.2`, `causal-conv1d==1.4.0`.

---

## Data preparation

Preprocessing follows rPPG-Toolbox. Each dataset is chunked into 160-frame clips, resized to
128×128 after face cropping (large box coefficient 1.5), and standardised at input and label
level (30 Hz). Set `DO_PREPROCESS: True` on first run, then `False` to read the cache.

| Dataset | Source | Notes |
|---------|--------|-------|
| PURE | request from the [authors](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure) | 30 Hz, 160-frame chunks, 128×128 |
| UBFC-rPPG | [link](https://sites.google.com/view/ybenezeth/ubfcrppg) | same as above |
| MMPD | request from the authors | test with all metadata filters open (LIGHT/MOTION/EXERCISE/…) |

Fill in your local paths at the top of `scripts/gen_configs.py` (`PATHS` / `INTRA_SPLIT` /
`CROSS_SPLIT`), then run `python scripts/gen_configs.py` to regenerate the configs.

---

## Reproducing the core cross-dataset results

The two headline cross-dataset results of the paper (trained on PURE or UBFC, tested directly
on MMPD) are reproduced with the released checkpoints:

```bash
# PURE -> MMPD  (gated expert, MAE ~8.5 bpm)
python main.py --config_file configs/infer/PURE_MMPD_gatedtf_seed42.yaml

# UBFC -> MMPD  (gated expert, MAE ~8.2 bpm)
python main.py --config_file configs/infer/UBFC_MMPD_gatedtf_seed42.yaml
```

Or run both at once:

```bash
bash reproduce.sh
```

Each run evaluates in `only_test` mode and writes per-subject metrics to
`./saved_metrics/<task>/`.

### Released checkpoints

This repository ships the two core cross-dataset checkpoints (seed 42), both using the
**Gated (gated_tf)** expert:

| Setting | Checkpoint |
|---------|-----------|
| PURE → MMPD | `pretrained/PreTrainedModels/PURE_MMPD_DareNet_gated_tf_seed42_Epoch29.pth` |
| UBFC → MMPD | `pretrained/PreTrainedModels/UBFC_MMPD_DareNet_gated_tf_seed42_Epoch29.pth` |

Their integrity is listed in `pretrained/PreTrainedModels/CHECKSUMS.txt`.

---

## Acknowledgement

Thanks to the authors of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) and
[RhythmMamba](https://github.com/zizheng-guo/RhythmMamba).

## Citation

```
@inproceedings{darenet2026,
  title={DARE to Generalize: Domain-Aware Robust Experts and Reliability-Aware Decisions for Cross-Domain rPPG},
  author={Hou, Junjie and Chow, Chong Tong and He, Yudong and Odinaev, Ismoil and Wong, Kwan Long and Chin, Jing Wei and So, Richard H. Y.},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2026}
}
```
