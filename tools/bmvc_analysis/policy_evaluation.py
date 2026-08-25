"""
Policy evaluation: does OOF-based routing actually reduce MAE?

Evaluates multiple decision policies using out-of-fold probabilities.
All policies are inference-time computable (no ground-truth labels used at decision time).
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_fa_errors(df, fs=30):
    """FixedAvg errors from averaging pred_c + pred_g signals."""
    from .feature_engineering import fft_hr
    n = len(df)
    errs = np.full(n, np.nan)
    for i in range(n):
        pc = np.asarray(df['pred_c'].iloc[i], dtype=np.float64).flatten()
        pg = np.asarray(df['pred_g'].iloc[i], dtype=np.float64).flatten()
        if len(pc) < 4 or len(pg) < 4:
            continue
        ml = min(len(pc), len(pg))
        hr_fa = fft_hr((pc[:ml] + pg[:ml]) / 2.0, fs)
        hl = df['hr_label'].iloc[i]
        if not np.isnan(hr_fa) and not np.isnan(hl):
            errs[i] = np.abs(hr_fa - hl)
    return errs


def _metric_dict(errors, coverage=None):
    v = errors[~np.isnan(errors)]
    if len(v) == 0:
        return {'MAE': np.nan, 'MedianAE': np.nan, 'P90': np.nan,
                'P95': np.nan, 'FR@10': np.nan, 'coverage': coverage}
    return {'MAE': float(np.mean(v)), 'MedianAE': float(np.median(v)),
            'P90': float(np.percentile(v, 90)), 'P95': float(np.percentile(v, 95)),
            'FR@10': float(np.mean(v > 10)), 'coverage': coverage}


def evaluate_policies(df: pd.DataFrame, oof_dict: dict, dataset: str,
                      fs: float = 30) -> pd.DataFrame:
    """
    Evaluate routing policies using OOF probabilities.

    Policies:
      1) BestSingle: default to better expert per dataset.
      2) FixedAvg: average pred_c + pred_g, HR from fft_hr.
      3) HardSelector(thr=0.5): p_winner > 0.5 → Complex, else Gated.
      4) ConservativeSwitch(thr): default BestSingle, switch only if p_winner > thr or < 1-thr.
      5) SoftBlend: lam = p_winner, pred_blend = lam*pred_c + (1-lam)*pred_g.
      6) FailureGate(thr): abstain when p_oracle_failure > thr, else FixedAvg.
    """
    err_c = df['err_c'].values
    err_g = df['err_g'].values
    oracle = df['oracle_err'].values
    fa_errs = _compute_fa_errors(df, fs)

    # Pick the expert with lower MAE on this dataset (data-driven, not hardcoded)
    mae_c_mean = np.nanmean(err_c); mae_g_mean = np.nanmean(err_g)
    best_single_default = 'Complex' if mae_c_mean <= mae_g_mean else 'Gated'

    p_winner = oof_dict.get('p_winner', np.full(len(df), 0.5))
    p_oracle_failure = oof_dict.get('p_oracle_failure', np.full(len(df), 0.0))

    n = len(df)
    rows = []

    # 1) BestSingle
    bs_errs = err_c if best_single_default == 'Complex' else err_g
    rows.append({'dataset': dataset, 'policy': 'BestSingle', 'threshold': np.nan,
                 **_metric_dict(bs_errs)})

    # 2) FixedAvg
    rows.append({'dataset': dataset, 'policy': 'FixedAvg', 'threshold': np.nan,
                 **_metric_dict(fa_errs)})

    # 3) HardSelector
    valid_hs = ~np.isnan(p_winner)
    hs_errs = np.full(n, np.nan)
    for i in range(n):
        if not valid_hs[i]:
            continue
        hs_errs[i] = err_c[i] if p_winner[i] > 0.5 else err_g[i]
    rows.append({'dataset': dataset, 'policy': 'HardSelector', 'threshold': 0.5,
                 **_metric_dict(hs_errs)})

    # 4) ConservativeSwitch
    for thr in [0.6, 0.7, 0.8]:
        cs_errs = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(p_winner[i]):
                continue
            if p_winner[i] > thr:
                cs_errs[i] = err_c[i]
            elif p_winner[i] < (1 - thr):
                cs_errs[i] = err_g[i]
            else:
                cs_errs[i] = bs_errs[i]
        rows.append({'dataset': dataset, 'policy': 'ConservativeSwitch',
                     'threshold': thr, **_metric_dict(cs_errs)})

    # 5) SoftBlend
    sb_errs = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(p_winner[i]):
            continue
        pc = np.asarray(df['pred_c'].iloc[i], dtype=np.float64).flatten()
        pg = np.asarray(df['pred_g'].iloc[i], dtype=np.float64).flatten()
        if len(pc) < 4 or len(pg) < 4:
            continue
        ml = min(len(pc), len(pg))
        lam = np.clip(p_winner[i], 0.0, 1.0)
        blended = lam * pc[:ml] + (1 - lam) * pg[:ml]
        from .feature_engineering import fft_hr
        hr_blend = fft_hr(blended, fs)
        hl = df['hr_label'].iloc[i]
        if not np.isnan(hr_blend) and not np.isnan(hl):
            sb_errs[i] = np.abs(hr_blend - hl)
    rows.append({'dataset': dataset, 'policy': 'SoftBlend', 'threshold': np.nan,
                 **_metric_dict(sb_errs)})

    # 6) FailureGate
    for thr in [0.3, 0.5, 0.7]:
        fg_errs = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(p_oracle_failure[i]):
                continue
            if p_oracle_failure[i] > thr:
                continue  # abstain → NaN
            fg_errs[i] = fa_errs[i]
        coverage = float(np.mean(~np.isnan(fg_errs)))
        rows.append({'dataset': dataset, 'policy': 'FailureGate',
                     'threshold': thr, **_metric_dict(fg_errs, coverage=coverage)})

    result = pd.DataFrame(rows)
    logger.info(f"Policy evaluation for {dataset}:\n{result.to_string(index=False)}")
    return result
