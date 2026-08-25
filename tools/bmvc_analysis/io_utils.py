"""
I/O utilities for loading, aligning, and saving analysis data.

All functions operate on numpy arrays and pandas DataFrames.
No model inference happens here.
"""
import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_npz(path: str) -> dict:
    """Load a .npz file and return all arrays as a dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path, allow_pickle=True)
    out = {}
    for key in data.files:
        val = data[key]
        # Convert 0-d arrays to scalars where appropriate
        if isinstance(val, np.ndarray) and val.ndim == 0:
            out[key] = val.item()
        else:
            out[key] = val
    logger.info(f"Loaded {path}: {len(data.files)} keys, {len(out.get('subject_ids', []))} entries")
    return out


def align_pair(complex_npz: dict, gated_npz: dict) -> pd.DataFrame:
    """
    Align two dumps by (subject_id, sort_index) and return a DataFrame.

    Each row is one window with columns:
      subject_id, sort_index,
      pred_c (1D array), pred_g (1D array), label (1D array),
      energy, entropy
    """
    # Build lookups
    c_lookup = {}
    for i in range(len(complex_npz['subject_ids'])):
        key = (str(complex_npz['subject_ids'][i]), int(complex_npz['sort_indices'][i]))
        c_lookup[key] = i

    g_lookup = {}
    for i in range(len(gated_npz['subject_ids'])):
        key = (str(gated_npz['subject_ids'][i]), int(gated_npz['sort_indices'][i]))
        g_lookup[key] = i

    common_keys = sorted(set(c_lookup.keys()) & set(g_lookup.keys()))
    logger.info(f"Align: complex={len(c_lookup)}, gated={len(g_lookup)}, common={len(common_keys)}")

    if len(common_keys) == 0:
        raise ValueError("No common (subject_id, sort_index) pairs found between the two dumps!")

    rows = []
    for key in common_keys:
        ci = c_lookup[key]
        gi = g_lookup[key]
        rows.append({
            'subject_id': key[0],
            'sort_index': key[1],
            'pred_c': np.asarray(complex_npz['pred_ppg'][ci], dtype=np.float64).flatten(),
            'pred_g': np.asarray(gated_npz['pred_ppg'][gi], dtype=np.float64).flatten(),
            'label': np.asarray(complex_npz['label_ppg'][ci], dtype=np.float64).flatten(),
            'energy': float(complex_npz['energy'][ci]),
            'entropy': float(complex_npz['entropy'][ci]),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Aligned DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_aligned_npz(path: str) -> pd.DataFrame:
    """
    Load a pre-aligned .npz file (from rf_ces_analysis.py output).
    Contains err_r, err_p, delta, winner, hr_pred_r, hr_pred_p, hr_label etc.
    Returns a DataFrame ready for feature engineering.
    """
    data = load_npz(path)
    n = len(data['subject_ids'])

    rows = []
    for i in range(n):
        rows.append({
            'subject_id': str(data['subject_ids'][i]),
            'sort_index': int(data['sort_indices'][i]),
            'pred_c': np.asarray(data.get('pred_ppg_r', data.get('pred_ppg', [[]]))[i], dtype=np.float64).flatten() if 'pred_ppg_r' in data else np.array([]),
            'pred_g': np.asarray(data.get('pred_ppg_p', data.get('pred_ppg', [[]]))[i], dtype=np.float64).flatten() if 'pred_ppg_p' in data else np.array([]),
            'label': np.asarray(data.get('label_ppg', data.get('label', [[]]))[i], dtype=np.float64).flatten() if 'label_ppg' in data else np.array([]),
            'energy': float(data['energy'][i]),
            'entropy': float(data['entropy'][i]),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Loaded aligned npz: {len(df)} rows")
    return df


def save_table(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as UTF-8 CSV with header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8', float_format='%.4f')
    logger.info(f"Saved table: {path} ({len(df)} rows × {len(df.columns)} cols)")


def save_json(obj: dict, path: str) -> None:
    """Save a dict as JSON with indent=2."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=_convert, ensure_ascii=False)
    logger.info(f"Saved JSON: {path}")
