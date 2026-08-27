"""
Failure / tail-risk metrics for the policy analysis.

Computes per-expert and per-method MAE, quantiles, and failure rates.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_tail_metrics(errors: np.ndarray, name: str = "") -> dict:
    """
    Compute tail-risk summary for an array of absolute errors.

    Returns dict with keys:
      MAE, MedianAE, P75, P90, P95, FailureRate@5, FailureRate@10, FailureRate@15
    """
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return {k: np.nan for k in ['MAE', 'MedianAE', 'P75', 'P90', 'P95',
                                     'FR@5', 'FR@10', 'FR@15']}

    mae = float(np.mean(valid))
    med = float(np.median(valid))
    p75 = float(np.percentile(valid, 75))
    p90 = float(np.percentile(valid, 90))
    p95 = float(np.percentile(valid, 95))
    fr5 = float(np.mean(valid > 5))
    fr10 = float(np.mean(valid > 10))
    fr15 = float(np.mean(valid > 15))

    return {
        'MAE': mae, 'MedianAE': med,
        'P75': p75, 'P90': p90, 'P95': p95,
        'FR@5': fr5, 'FR@10': fr10, 'FR@15': fr15,
    }


def compute_dataset_table(df: pd.DataFrame, dataset_name: str,
                           fs: float = 30) -> pd.DataFrame:
    """
    Build a tail-metrics table comparing Complex, Gated, FixedAvg, Oracle.

    FixedAvg: HR is estimated from the average of pred_c and pred_g signals.
    Oracle: min(err_c, err_g) per window.
    """
    from .feature_engineering import fft_hr as _fft_hr

    err_c = df['err_c'].values
    err_g = df['err_g'].values
    oracle = df['oracle_err'].values

    # Fixed average: compute HR from averaged signal
    n = len(df)
    fa_errors = np.full(n, np.nan)
    for i in range(n):
        pc = np.asarray(df['pred_c'].iloc[i], dtype=np.float64).flatten()
        pg = np.asarray(df['pred_g'].iloc[i], dtype=np.float64).flatten()
        if len(pc) < 4 or len(pg) < 4:
            continue
        min_len = min(len(pc), len(pg))
        avg_sig = (pc[:min_len] + pg[:min_len]) / 2.0
        hr_fa = _fft_hr(avg_sig, fs)
        hr_lab = df['hr_label'].iloc[i]
        if not np.isnan(hr_fa) and not np.isnan(hr_lab):
            fa_errors[i] = np.abs(hr_fa - hr_lab)

    rows = []
    for method, errs in [('Complex', err_c), ('Gated', err_g),
                          ('FixedAvg', fa_errors), ('Oracle', oracle)]:
        metrics = compute_tail_metrics(errs, method)
        metrics['Method'] = method
        metrics['Dataset'] = dataset_name
        metrics['N_windows'] = int(np.sum(~np.isnan(errs)))
        rows.append(metrics)

    result = pd.DataFrame(rows)
    result = result[['Dataset', 'Method', 'N_windows',
                      'MAE', 'MedianAE', 'P75', 'P90', 'P95',
                      'FR@5', 'FR@10', 'FR@15']]
    logger.info(f"Tail metrics for {dataset_name}:\n{result.to_string(index=False)}")
    return result
