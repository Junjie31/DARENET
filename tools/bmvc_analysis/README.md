# BMVC 2026 Failure-Aware Analysis Toolkit

## Quick Start

```bash
cd /home/junjie/Workspace/RhythmMamba_HR_darenet
conda activate RhythmMamba
python tools/bmvc_analysis/run_all.py --dump_dir dumps_20260514 --out_dir outputs_bmvc
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `io_utils.py` | Load/align/save npz dumps and feature tables |
| `feature_engineering.py` | Compute label-free spectral/temporal features |
| `failure_metrics.py` | Tail-risk metrics (MAE, FailureRate@10, etc.) |
| `failure_selectors.py` | Winner/failure prediction with subject-disjoint CV |
| `abstention.py` | Reliability-driven selective prediction curves |
| `plots.py` | Publication-quality figures (PDF + PNG) |
| `run_all.py` | One-click orchestrator |

## Output Structure

```
outputs_bmvc/
  tables/
    features_{mmpd,pure}.csv        # All per-window features
    tail_metrics_{mmpd,pure}.csv    # Failure metrics by method
    selector_auc_{mmpd,pure}.csv    # AUC results for all selector configs
    abstention_{mmpd,pure}.csv      # Abstention curve data
  figures/
    error_cdf_{mmpd,pure}.pdf/png           # Figure X: Error CDF
    disagreement_vs_error_{mmpd,pure}.pdf/png  # Figure X: Disagreement analysis
    fingerprint_kde.pdf/png                  # Figure X: Pre-FFN feature space
    subject_scatter_{mmpd,pure}.pdf/png      # Figure X: Subject-level MAE
    abstention_curves.pdf/png                # Figure X: Abstention curves
    failure_auc_bar.pdf/png                  # Figure X: Failure prediction AUC
  logs/
    summary.json                     # Key numbers for paper
```

## Paper Figure/Table Mapping

| Output | Paper Section | Description |
|--------|--------------|-------------|
| `tail_metrics_{ds}.csv` | Results Table 1 | Per-method MAE, FailureRate@10 |
| `selector_auc_{ds}.csv` | Results Table 2 | Failure prediction AUC comparison |
| `error_cdf_{ds}.pdf` | Fig X(a) | Error CDF: Complex vs Gated vs Oracle |
| `disagreement_vs_error_{ds}.pdf` | Fig X(b) | Expert disagreement vs oracle error |
| `fingerprint_kde.pdf` | Motivational Study | Pre-FFN feature-space visualization |
| `subject_scatter_{ds}.pdf` | Supplementary | Per-subject Complex vs Gated MAE |
| `abstention_curves.pdf` | Fig X(c) | Abstention / selective prediction |
| `failure_auc_bar.pdf` | Fig X(d) | Feature set × model comparison |

## Key Design Decisions

1. **Subject-disjoint CV**: Splits are by `subject_id`, not by window index. This prevents data leakage that would inflate AUC.
2. **Label-free features only in FEATURE_SETS**: `hr_label`, `err_c`, `err_g`, `delta`, `winner`, `oracle_err` are supervision targets only, never selector inputs.
3. **Inference-time reliability**: `r_simple = -hr_diff` and `r_learned = 1 - p(failure)` are computable without ground truth labels.

## Dependencies

- numpy, scipy, pandas
- scikit-learn
- matplotlib, seaborn
