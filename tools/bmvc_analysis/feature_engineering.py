"""
Feature engineering for failure-aware analysis.

All functions operate on 1D numpy signals.
No label-derived features are included in FEATURE_SETS.
Label-derived columns (hr_label, err_c, err_g, delta, winner, oracle_err)
are computed only as supervision targets, never as selector inputs.

Uses the project's own HR estimation (get_hr from evaluation/post_process.py)
for consistency with the main training/evaluation pipeline.
"""
import logging
import numpy as np
from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.stats import pearsonr
from scipy.sparse import spdiags
import pandas as pd
import sys, os

# Use project's own get_hr for consistency
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from evaluation.post_process import get_hr as _project_get_hr, _detrend
    _HR_MODE = "project_get_hr"
except ImportError:
    _HR_MODE = "welch_fallback"
    _project_get_hr = None
    from scipy.signal import welch as _fallback_welch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature sets for selector experiments (label-free only!)
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "A_basic":     ["energy", "entropy"],
    "B_disagree":  ["hr_diff", "waveform_corr", "waveform_l2"],
    "C_intrinsic": ["snr_c", "snr_g", "peak_prom_c", "peak_prom_g",
                    "inband_c", "inband_g", "spec_ent_c", "spec_ent_g",
                    "smooth_c", "smooth_g"],
    "D_full":      ["energy", "entropy",
                    "hr_diff", "waveform_corr", "waveform_l2",
                    "snr_c", "snr_g", "peak_prom_c", "peak_prom_g",
                    "inband_c", "inband_g", "spec_ent_c", "spec_ent_g",
                    "smooth_c", "smooth_g"],
}

_FREQ_LOW = 0.7
_FREQ_HIGH = 4.0
_FS_DEFAULT = 30


# ---------------------------------------------------------------------------
# Signal preprocessing helpers (called internally by each spectral function)
# ---------------------------------------------------------------------------
def _preprocess(sig: np.ndarray, fs: float = _FS_DEFAULT) -> np.ndarray:
    """Detrend, bandpass [0.75, 2.5] Hz (matching project), zscore."""
    if len(sig) < 4:
        return sig
    s = _detrend(sig.astype(np.float64), 100)
    try:
        b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        s = filtfilt(b, a, s)
    except Exception:
        pass
    std = np.std(s)
    if std > 1e-10:
        s = (s - np.mean(s)) / std
    return s


def _zscore(sig: np.ndarray) -> np.ndarray:
    std = np.std(sig)
    if std < 1e-10:
        return sig - np.mean(sig)
    return (sig - np.mean(sig)) / std


# ---------------------------------------------------------------------------
# Core spectral / temporal feature functions
#   Each function accepts RAW signal and calls _preprocess INTERNALLY.
# ---------------------------------------------------------------------------
def fft_hr(sig: np.ndarray, fs: float = _FS_DEFAULT) -> float:
    """Estimate HR (BPM) using the project's own get_hr (Welch, 45-150 BPM)."""
    if len(sig) < 4:
        return np.nan
    try:
        s = _preprocess(sig, fs)
        if _project_get_hr is not None:
            return float(_project_get_hr(s, sr=fs))
        # fallback
        f, pxx = welch(s, fs, nperseg=min(len(s), 256))
        mask = (f >= 0.75) & (f <= 2.5)
        if not np.any(mask):
            return np.nan
        return f[mask][np.argmax(pxx[mask])] * 60.0
    except Exception:
        return np.nan


def spectral_snr(sig: np.ndarray, fs: float = _FS_DEFAULT) -> float:
    """SNR (dB) of dominant peak relative to in-band background."""
    s = _preprocess(sig, fs)
    if len(s) < 4:
        return np.nan
    try:
        f, pxx = welch(s, fs, nperseg=min(len(s), 256))
        mask = (f >= _FREQ_LOW) & (f <= _FREQ_HIGH)
        if not np.any(mask):
            return np.nan
        f_band, p_band = f[mask], pxx[mask]
        peak_idx = np.argmax(p_band)
        peak_f = f_band[peak_idx]
        peak_mask = (f_band >= peak_f - 0.1) & (f_band <= peak_f + 0.1)
        sig_pwr = np.sum(p_band[peak_mask])
        noise_pwr = np.sum(p_band[~peak_mask]) + 1e-12
        return 10.0 * np.log10(sig_pwr / noise_pwr)
    except Exception:
        return np.nan


def peak_prominence(sig: np.ndarray, fs: float = _FS_DEFAULT) -> float:
    """Peak prominence in Welch PSD within [0.7, 4.0] Hz band."""
    s = _preprocess(sig, fs)
    if len(s) < 4:
        return np.nan
    try:
        f, pxx = welch(s, fs, nperseg=min(len(s), 256))
        mask = (f >= _FREQ_LOW) & (f <= _FREQ_HIGH)
        if not np.any(mask):
            return np.nan
        p_band = pxx[mask]
        peak_val = np.max(p_band)
        band_std = np.std(p_band)
        if band_std < 1e-12:
            return 0.0
        return (peak_val - np.mean(p_band)) / band_std
    except Exception:
        return np.nan


def inband_power_ratio(sig: np.ndarray, fs: float = _FS_DEFAULT) -> float:
    """Fraction of total PSD power in [0.7, 4.0] Hz band."""
    s = _preprocess(sig, fs)
    if len(s) < 4:
        return np.nan
    try:
        f, pxx = welch(s, fs, nperseg=min(len(s), 256))
        mask = (f >= _FREQ_LOW) & (f <= _FREQ_HIGH)
        inband = np.sum(pxx[mask])
        total = np.sum(pxx)
        if total < 1e-12:
            return np.nan
        return inband / total
    except Exception:
        return np.nan


def spectral_entropy(sig: np.ndarray, fs: float = _FS_DEFAULT) -> float:
    """Normalised spectral entropy of the signal's PSD."""
    s = _preprocess(sig, fs)
    if len(s) < 4:
        return np.nan
    try:
        f, pxx = welch(s, fs, nperseg=min(len(s), 256))
        psd_norm = pxx / (np.sum(pxx) + 1e-12)
        return float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
    except Exception:
        return np.nan


def temporal_smoothness(sig: np.ndarray) -> float:
    """Mean squared first-order difference (lower = smoother)."""
    if len(sig) < 2:
        return np.nan
    return float(np.mean(np.diff(sig) ** 2))


# ---------------------------------------------------------------------------
# Feature table builder
# ---------------------------------------------------------------------------
def build_feature_table(df: pd.DataFrame, fs: float = _FS_DEFAULT,
                        use_official_errors: bool = False,
                        official: dict = None) -> pd.DataFrame:
    """
    Extend a base DataFrame with per-window features.

    Args:
        df: Base DataFrame with pred_c, pred_g, label, energy, entropy,
            subject_id, sort_index.
        fs: Sampling rate.
        use_official_errors: If True and official dict has err_c/err_g/oracle_err,
            use those for error columns instead of recomputing from signals.
        official: Dict from load_official_metrics().

    Returns:
        DataFrame with all feature columns added.
    """
    logger.info(f"HR estimation mode: {_HR_MODE}")

    n = len(df)
    logger.info(f"Building features for {n} windows @ fs={fs} Hz")

    # --- Intrinsic features per expert (raw signal → internal preprocess) ---
    def _compute_intrinsic(preds_col: str):
        snr_list = np.full(n, np.nan)
        prom_list = np.full(n, np.nan)
        inband_list = np.full(n, np.nan)
        spec_ent_list = np.full(n, np.nan)
        smooth_list = np.full(n, np.nan)
        hr_list = np.full(n, np.nan)

        for i in range(n):
            sig = np.asarray(df[preds_col].iloc[i], dtype=np.float64).flatten()
            if len(sig) < 4:
                continue
            hr_list[i] = fft_hr(sig, fs)
            snr_list[i] = spectral_snr(sig, fs)
            prom_list[i] = peak_prominence(sig, fs)
            inband_list[i] = inband_power_ratio(sig, fs)
            spec_ent_list[i] = spectral_entropy(sig, fs)
            smooth_list[i] = temporal_smoothness(sig)

        return hr_list, snr_list, prom_list, inband_list, spec_ent_list, smooth_list

    logger.info("  Computing intrinsic features for Complex expert ...")
    hr_c, snr_c, prom_c, inband_c, spec_ent_c, smooth_c = _compute_intrinsic('pred_c')
    logger.info("  Computing intrinsic features for Gated expert ...")
    hr_g, snr_g, prom_g, inband_g, spec_ent_g, smooth_g = _compute_intrinsic('pred_g')

    # --- Inter-expert disagreement ---
    hr_diff = np.abs(hr_c - hr_g)
    waveform_corr = np.full(n, np.nan)
    waveform_l2 = np.full(n, np.nan)

    for i in range(n):
        pc = np.asarray(df['pred_c'].iloc[i], dtype=np.float64).flatten()
        pg = np.asarray(df['pred_g'].iloc[i], dtype=np.float64).flatten()
        if len(pc) < 2 or len(pg) < 2:
            continue
        min_len = min(len(pc), len(pg))
        pc_a, pg_a = pc[:min_len], pg[:min_len]
        try:
            corr, _ = pearsonr(pc_a, pg_a)
            waveform_corr[i] = corr if not np.isnan(corr) else 0.0
        except Exception:
            waveform_corr[i] = 0.0
        waveform_l2[i] = np.sqrt(np.mean((pc_a - pg_a) ** 2))

    # --- Label-derived HR ---
    hr_label = np.full(n, np.nan)
    for i in range(n):
        lab = np.asarray(df['label'].iloc[i], dtype=np.float64).flatten()
        if len(lab) >= 4:
            hr_label[i] = fft_hr(lab, fs)

    # --- Error computation ---
    err_c = np.abs(hr_c - hr_label)
    err_g = np.abs(hr_g - hr_label)
    oracle_err = np.minimum(err_c, err_g)
    delta = err_g - err_c
    winner = (err_c < err_g).astype(int)

    # --- Official error override ---
    official_err_c = np.full(n, np.nan)
    official_err_g = np.full(n, np.nan)
    official_oracle_err = np.full(n, np.nan)

    if use_official_errors and official is not None and len(official) > 0:
        off_err_c = official.get('err_c')
        off_err_g = official.get('err_g')
        off_oracle = official.get('oracle_err')
        if off_err_c is not None and off_err_g is not None:
            # Align by length (official should have same n)
            official_err_c[:len(off_err_c)] = off_err_c[:n]
            official_err_g[:len(off_err_g)] = off_err_g[:n]
            if off_oracle is not None:
                official_oracle_err[:len(off_oracle)] = off_oracle[:n]
            # Override error columns with official values
            err_c = official_err_c.copy()
            err_g = official_err_g.copy()
            oracle_err = np.minimum(err_c, err_g)
            delta = err_g - err_c
            winner = (err_c < err_g).astype(int)
            logger.info("  Using official errors (overriding recomputed).")
            # Disagreement feature is defined on official per-window HR, not on
            # the recomputed HR (they differ in general; this is the paper's
            # B_disagree feature source).
            off_hr_c = official.get('hr_c')
            off_hr_g = official.get('hr_g')
            if off_hr_c is not None and off_hr_g is not None:
                hr_c = np.full(n, np.nan)
                hr_g = np.full(n, np.nan)
                hr_c[:len(off_hr_c)] = off_hr_c[:n]
                hr_g[:len(off_hr_g)] = off_hr_g[:n]
                hr_diff = np.abs(hr_c - hr_g)
                logger.info("  Using official per-window HR for hr_diff.")
        else:
            logger.warning("  use_official_errors=True but official dict missing err_c/err_g. "
                           "Falling back to recomputed errors.")
    elif use_official_errors:
        logger.warning("  use_official_errors=True but no official data provided. "
                       "Falling back to recomputed errors.")

    # --- Assemble ---
    out = df.copy()
    out['snr_c'] = snr_c
    out['snr_g'] = snr_g
    out['peak_prom_c'] = prom_c
    out['peak_prom_g'] = prom_g
    out['inband_c'] = inband_c
    out['inband_g'] = inband_g
    out['spec_ent_c'] = spec_ent_c
    out['spec_ent_g'] = spec_ent_g
    out['smooth_c'] = smooth_c
    out['smooth_g'] = smooth_g
    out['hr_c'] = hr_c
    out['hr_g'] = hr_g
    out['hr_diff'] = hr_diff
    out['waveform_corr'] = waveform_corr
    out['waveform_l2'] = waveform_l2
    # Label-derived (supervision targets only)
    out['hr_label'] = hr_label
    out['err_c'] = err_c
    out['err_g'] = err_g
    out['delta'] = delta
    out['winner'] = winner
    out['oracle_err'] = oracle_err
    # Official audit columns
    out['official_err_c'] = official_err_c
    out['official_err_g'] = official_err_g
    out['official_oracle_err'] = official_oracle_err

    n_valid = np.sum(~np.isnan(hr_label))
    logger.info(f"Feature table built: {len(out)} rows, {len(out.columns)} cols. "
                f"Valid HR labels: {n_valid}/{n}")

    return out
