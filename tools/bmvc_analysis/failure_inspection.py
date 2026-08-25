"""
Failure inspection: export top-k worst windows and plot waveform/PSD panels.

Useful for sanity-checking what "failure" looks like in the raw signals.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

logger = logging.getLogger(__name__)


def export_top_failures(df: pd.DataFrame, dataset: str, oof_dict: dict = None,
                         k: int = 50) -> pd.DataFrame:
    """
    Export top-k worst windows for each failure type (oracle, gated, complex).

    Returns concatenated DataFrame sorted by error descending.
    """
    if oof_dict is None:
        oof_dict = {}
    p_of = oof_dict.get('p_oracle_failure', np.full(len(df), np.nan))

    pieces = []
    for ftype, col in [('oracle', 'oracle_err'), ('gated', 'err_g'), ('complex', 'err_c')]:
        sub = df.nlargest(k, col).copy()
        sub['rank'] = range(1, len(sub) + 1)
        sub['failure_type'] = ftype
        sub['p_oracle_failure'] = p_of[sub.index]
        cols = ['dataset', 'rank', 'failure_type', 'subject_id', 'sort_index',
                'hr_label', 'hr_c', 'hr_g', 'err_c', 'err_g', 'oracle_err', 'winner',
                'hr_diff', 'waveform_corr', 'waveform_l2', 'energy', 'entropy',
                'snr_c', 'snr_g', 'peak_prom_c', 'peak_prom_g',
                'inband_c', 'inband_g', 'spec_ent_c', 'spec_ent_g',
                'p_oracle_failure']
        available = [c for c in cols if c in sub.columns]
        pieces.append(sub[available])
    result = pd.concat(pieces, ignore_index=True)
    logger.info(f"Top failures for {dataset}: {len(result)} rows")
    return result


def plot_top_failures(df: pd.DataFrame, dataset: str, oof_dict: dict = None,
                       n_panels: int = 12, save_path: str = None):
    """Plot worst oracle-error windows: waveform + PSD panels."""
    if oof_dict is None:
        oof_dict = {}
    p_of = oof_dict.get('p_oracle_failure', np.full(len(df), np.nan))

    top_idx = df['oracle_err'].nlargest(n_panels).index
    fs = 30

    fig, axes = plt.subplots(n_panels, 2, figsize=(12, 2.5 * n_panels))
    if n_panels == 1:
        axes = axes.reshape(1, 2)

    for pi, idx in enumerate(top_idx):
        row = df.loc[idx]
        pc = np.asarray(row['pred_c'], dtype=np.float64).flatten()
        pg = np.asarray(row['pred_g'], dtype=np.float64).flatten()
        lb = np.asarray(row['label'], dtype=np.float64).flatten()
        min_len = min(len(pc), len(pg), len(lb))
        t = np.arange(min_len) / fs
        pc_a, pg_a, lb_a = pc[:min_len], pg[:min_len], lb[:min_len]

        # Left: waveforms
        ax = axes[pi, 0]
        ax.plot(t, lb_a, 'k-', linewidth=0.8, alpha=0.7, label='Label')
        ax.plot(t, pc_a, 'b-', linewidth=0.6, alpha=0.8, label='Complex')
        ax.plot(t, pg_a, 'r-', linewidth=0.6, alpha=0.8, label='Gated')
        ax.set_ylabel('Amp')
        if pi == 0:
            ax.legend(fontsize=7)
        err_c = row.get('err_c', np.nan)
        err_g = row.get('err_g', np.nan)
        pf = p_of[idx] if idx < len(p_of) else np.nan
        ax.set_title(f'subj={row["subject_id"]}, idx={row["sort_index"]}, '
                     f'err_c={err_c:.1f}, err_g={err_g:.1f}, p_fail={pf:.2f}',
                     fontsize=8)

        # Right: PSD
        ax = axes[pi, 1]
        for sig, name, color in [(lb_a, 'Label', 'black'),
                                  (pc_a, 'Complex', 'blue'),
                                  (pg_a, 'Gated', 'red')]:
            if len(sig) < 4:
                continue
            f, pxx = welch(sig, fs, nperseg=min(len(sig), 256))
            mask = (f >= 0.7) & (f <= 4.0)
            ax.plot(f[mask], 10 * np.log10(pxx[mask] + 1e-12),
                    color=color, linewidth=0.8, alpha=0.8, label=name)
        ax.set_xlabel('Hz')
        ax.set_ylabel('dB')
        if pi == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f'Top {n_panels} Oracle Failures — {dataset}', fontsize=12)
    fig.tight_layout()

    if save_path:
        base = save_path.rsplit('.', 1)[0]
        fig.savefig(f'{base}.pdf', bbox_inches='tight')
        fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
        logger.info(f'Saved: {base}.pdf / .png')
    plt.close(fig)
