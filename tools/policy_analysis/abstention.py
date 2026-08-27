"""
Reliability-driven abstention (selective prediction) with 4 reliability types.

Reliability scores (all inference-time computable):
  r_simple    = -hr_diff
  r_learned   = -p_oracle_failure  (OOF prob)
  r_perexpert = -max(p_failure_c, p_failure_g)
  r_oracle    = -oracle_err  (upper bound, label-derived, reference only)
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _tail_dict(errors):
    v = errors[~np.isnan(errors)]
    if len(v) == 0:
        return {'kept_MAE': np.nan, 'kept_MedianAE': np.nan,
                'kept_P90': np.nan, 'kept_P95': np.nan,
                'kept_FR@10': np.nan, 'kept_FR@15': np.nan}
    return {'kept_MAE': float(np.mean(v)),
            'kept_MedianAE': float(np.median(v)),
            'kept_P90': float(np.percentile(v, 90)),
            'kept_P95': float(np.percentile(v, 95)),
            'kept_FR@10': float(np.mean(v > 10)),
            'kept_FR@15': float(np.mean(v > 15))}


def abstention_curve_full(errors, reliability, fractions=None):
    """Full abstention curve with multiple metrics per fraction."""
    if fractions is None:
        fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    valid = ~(np.isnan(errors) | np.isnan(reliability))
    err, rel = errors[valid], reliability[valid]
    if len(err) == 0:
        return [{'fraction': f, 'coverage': np.nan, 'kept_MAE': np.nan,
                 'kept_MedianAE': np.nan, 'kept_P90': np.nan, 'kept_P95': np.nan,
                 'kept_FR@10': np.nan, 'kept_FR@15': np.nan} for f in fractions]
    order = np.argsort(rel)
    err_s, n = err[order], len(err)
    result = []
    for frac in fractions:
        n_drop = int(np.floor(n * frac))
        if n_drop >= n:
            result.append({'fraction': frac, 'coverage': 0.0, **_tail_dict(np.array([]))})
        else:
            kept = err_s[n_drop:]
            coverage = len(kept) / n
            result.append({'fraction': frac, 'coverage': coverage, **_tail_dict(kept)})
    return result


def compute_abstention_table(df: pd.DataFrame, dataset: str,
                              oof_dict: dict = None,
                              fs: float = 30) -> pd.DataFrame:
    """
    Compute abstention curves for 4 reliability types × 3 methods.
    """
    from .feature_engineering import fft_hr
    if oof_dict is None:
        oof_dict = {}

    err_c, err_g = df['err_c'].values, df['err_g'].values
    oracle = df['oracle_err'].values

    # Pick the expert with lower MAE on this dataset (data-driven, not hardcoded)
    mae_c_mean = np.nanmean(err_c); mae_g_mean = np.nanmean(err_g)
    best_single_default = 'Complex' if mae_c_mean <= mae_g_mean else 'Gated'
    bs_errs = err_c if best_single_default == 'Complex' else err_g

    # FixedAvg
    n = len(df)
    fa_errs = np.full(n, np.nan)
    for i in range(n):
        pc = np.asarray(df['pred_c'].iloc[i], dtype=np.float64).flatten()
        pg = np.asarray(df['pred_g'].iloc[i], dtype=np.float64).flatten()
        if len(pc) < 4 or len(pg) < 4:
            continue
        ml = min(len(pc), len(pg))
        hr_fa = fft_hr((pc[:ml] + pg[:ml]) / 2.0, fs)
        hl = df['hr_label'].iloc[i]
        if not np.isnan(hr_fa) and not np.isnan(hl):
            fa_errs[i] = np.abs(hr_fa - hl)

    # Reliability scores
    r_simple = -df['hr_diff'].values  # higher = less disagreement
    p_of = oof_dict.get('p_oracle_failure', np.full(n, np.nan))
    p_fc = oof_dict.get('p_failure_c', np.full(n, np.nan))
    p_fg = oof_dict.get('p_failure_g', np.full(n, np.nan))
    r_learned = -p_of
    r_perexpert = -np.maximum(p_fc, p_fg)
    r_oracle = -oracle  # upper bound

    reliability_map = {
        'r_simple': r_simple,
        'r_learned': r_learned,
        'r_perexpert': r_perexpert,
        'r_oracle': r_oracle,
    }

    fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    rows = []

    for rname, rvals in reliability_map.items():
        for mname, errs in [('best_single', bs_errs), ('fixed_avg', fa_errs), ('oracle', oracle)]:
            curve = abstention_curve_full(errs, rvals, fractions)
            for pt in curve:
                rows.append({'dataset': dataset, 'method': mname,
                             'reliability_type': rname,
                             'fraction': pt['fraction'], 'coverage': pt['coverage'],
                             'kept_MAE': pt['kept_MAE'],
                             'kept_MedianAE': pt['kept_MedianAE'],
                             'kept_P90': pt['kept_P90'],
                             'kept_P95': pt['kept_P95'],
                             'kept_FR@10': pt['kept_FR@10'],
                             'kept_FR@15': pt['kept_FR@15']})

    result = pd.DataFrame(rows)
    logger.info(f"Abstention table for {dataset}: {len(result)} rows")
    return result


def make_abstention_summary(abst_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Pivot abstention table to summary format for paper tables."""
    sub = abst_df[abst_df['method'] == 'best_single'].copy()
    fracs = [0, 10, 20, 30, 50]
    rows = []
    for rtype in ['r_simple', 'r_learned', 'r_perexpert', 'r_oracle']:
        row = {'dataset': dataset, 'reliability': rtype, 'base_method': 'best_single'}
        for f_pct in fracs:
            frac = f_pct / 100.0
            match = sub[(sub['reliability_type'] == rtype) &
                        (np.abs(sub['fraction'] - frac) < 0.001)]
            if len(match) > 0:
                row[f'abstain@{f_pct}_kept_MAE'] = float(match['kept_MAE'].iloc[0])
            else:
                row[f'abstain@{f_pct}_kept_MAE'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
