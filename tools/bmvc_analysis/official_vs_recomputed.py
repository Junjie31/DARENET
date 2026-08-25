"""
Compare official (aligned npz) metrics vs recomputed-from-pred metrics.

This validates that per-window HR recomputation matches the official
evaluation pipeline's numbers.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Recognised synonym keys for aligned npz fields
_KEY_MAP = {
    'err_complex': ['err_complex', 'complex_err', 'err_c', 'err_r'],
    'err_gated':   ['err_gated', 'gated_err', 'err_g', 'err_p'],
    'oracle_err':  ['oracle_err', 'oracle_errs'],
    'hr_complex':  ['hr_complex', 'hr_c', 'hr_pred_r'],
    'hr_gated':    ['hr_gated', 'hr_g', 'hr_pred_p'],
    'hr_label':    ['hr_label', 'label_hr', 'gt_hr'],
    'winner':      ['winner'],
    'energy':      ['energy'],
    'entropy':     ['entropy'],
    'subject_id':  ['subject_id', 'subject_ids', 'subj_ids'],
}


def _resolve_key(data: dict, canonical: str):
    """Return the first matching key in `data` from the synonym list."""
    for k in _KEY_MAP.get(canonical, [canonical]):
        if k in data:
            return data[k]
    return None


def load_official_metrics(aligned_path: str) -> dict:
    """
    Load per-window metrics from a pre-aligned npz.

    Returns dict with keys:
      err_c, err_g, oracle_err, hr_label, winner, energy, entropy, subject_id
    Returns {} if file missing or key data unavailable.
    """
    import os
    if not os.path.exists(aligned_path):
        logger.warning(f"Official aligned npz not found: {aligned_path}")
        return {}

    try:
        raw = np.load(aligned_path, allow_pickle=True)
    except Exception as e:
        logger.warning(f"Failed to load {aligned_path}: {e}")
        return {}

    out = {}
    for canon in ['err_complex', 'err_gated', 'hr_label',
                  'winner', 'energy', 'entropy', 'subject_id']:
        val = _resolve_key(raw, canon)
        if val is None:
            logger.warning(f"Missing key for '{canon}' in {aligned_path}")
            return {}
        # subject_id is a string array; everything else is numeric.
        if canon == 'subject_id':
            arr = np.asarray(val).flatten()
        else:
            arr = np.asarray(val, dtype=np.float64).flatten()
        out[canon if canon != 'err_complex' else 'err_c'] = arr
        if canon == 'err_complex':
            out['err_c'] = arr
        elif canon == 'err_gated':
            out['err_g'] = arr

    # Post-process: map canonical names
    out['err_c'] = _resolve_key(raw, 'err_complex')
    out['err_g'] = _resolve_key(raw, 'err_gated')
    # Official per-window HR of the two experts (paper disagreement features
    # are computed from these, NOT recomputed from the raw waveform).
    out['hr_c'] = _resolve_key(raw, 'hr_complex')
    out['hr_g'] = _resolve_key(raw, 'hr_gated')
    # oracle_err is not stored in the aligned npz; derive it from official errs.
    out['oracle_err'] = _resolve_key(raw, 'oracle_err')
    if out['oracle_err'] is None and out['err_c'] is not None and out['err_g'] is not None:
        out['oracle_err'] = np.minimum(out['err_c'], out['err_g'])
    out['hr_label'] = _resolve_key(raw, 'hr_label')
    out['winner'] = _resolve_key(raw, 'winner')
    out['energy'] = _resolve_key(raw, 'energy')
    out['entropy'] = _resolve_key(raw, 'entropy')
    out['subject_ids'] = _resolve_key(raw, 'subject_id')

    # Ensure all are flat float arrays
    for k in ['err_c', 'err_g', 'oracle_err', 'hr_c', 'hr_g', 'hr_label', 'energy', 'entropy']:
        if out.get(k) is not None:
            out[k] = np.asarray(out[k], dtype=np.float64).flatten()
    if out.get('winner') is not None:
        out['winner'] = np.asarray(out['winner'], dtype=int).flatten()
    if out.get('subject_ids') is not None:
        out['subject_ids'] = np.asarray(out['subject_ids']).flatten()

    n = len(out.get('err_c', []))
    logger.info(f"Loaded official metrics from {aligned_path}: {n} windows")
    return out


def _tail_metrics(errors: np.ndarray) -> dict:
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return {k: np.nan for k in ['MAE', 'MedianAE', 'P75', 'P90', 'P95',
                                     'FR@5', 'FR@10', 'FR@15', 'n_windows']}
    return {
        'MAE': float(np.mean(valid)),
        'MedianAE': float(np.median(valid)),
        'P75': float(np.percentile(valid, 75)),
        'P90': float(np.percentile(valid, 90)),
        'P95': float(np.percentile(valid, 95)),
        'FR@5': float(np.mean(valid > 5)),
        'FR@10': float(np.mean(valid > 10)),
        'FR@15': float(np.mean(valid > 15)),
        'n_windows': len(valid),
    }


def compare_official_recomputed(official: dict, feat_df: pd.DataFrame,
                                 dataset: str) -> pd.DataFrame:
    """
    Compare official metrics from aligned npz vs recomputed from pred signals.

    Returns long-format DataFrame with columns:
      dataset, source, method, MAE, MedianAE, P75, P90, P95, FR@5, FR@10, FR@15, n_windows
    """
    rows = []

    # Official: use err_c/err_g/oracle_err from official dict
    for method, col in [('Complex', 'err_c'), ('Gated', 'err_g')]:
        if col in official:
            m = _tail_metrics(official[col])
            m.update({'dataset': dataset, 'source': 'official_aligned', 'method': method})
            rows.append(m)

    if 'oracle_err' in official:
        m = _tail_metrics(official['oracle_err'])
        m.update({'dataset': dataset, 'source': 'official_aligned', 'method': 'Oracle'})
        rows.append(m)

    # Recompute FixedAvg from official signals if available, else use feat_df
    from .feature_engineering import fft_hr as _fft_hr

    # Official FixedAvg: use official hr_label
    if 'hr_label' in official and 'hr_c' in official and 'hr_g' in official:
        # Actually, official aligned npz may not have pred signals.
        # Use feat_df for FixedAvg computation.
        pass

    # Recomputed from feat_df
    for method, col in [('Complex', 'err_c'), ('Gated', 'err_g'), ('Oracle', 'oracle_err')]:
        if col in feat_df.columns:
            m = _tail_metrics(feat_df[col].values)
            m.update({'dataset': dataset, 'source': 'recomputed_from_pred', 'method': method})
            rows.append(m)

    # FixedAvg from feat_df
    n = len(feat_df)
    fa_errs = np.full(n, np.nan)
    for i in range(n):
        pc = np.asarray(feat_df['pred_c'].iloc[i], dtype=np.float64).flatten()
        pg = np.asarray(feat_df['pred_g'].iloc[i], dtype=np.float64).flatten()
        if len(pc) < 4 or len(pg) < 4:
            continue
        ml = min(len(pc), len(pg))
        avg = (pc[:ml] + pg[:ml]) / 2.0
        hr_fa = _fft_hr(avg, 30)
        hl = feat_df['hr_label'].iloc[i]
        if not np.isnan(hr_fa) and not np.isnan(hl):
            fa_errs[i] = np.abs(hr_fa - hl)
    m = _tail_metrics(fa_errs)
    m.update({'dataset': dataset, 'source': 'recomputed_from_pred', 'method': 'FixedAvg'})
    rows.append(m)

    result = pd.DataFrame(rows)
    logger.info(f"Official vs recomputed for {dataset}:\n{result.to_string(index=False)}")
    return result
