"""
Publication-quality plotting for BMVC 2026 failure-aware analysis.
All figures saved as both .pdf and .png (dpi=300).
"""
import os, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def set_style():
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12,
                         'legend.fontsize': 10, 'axes.titlesize': 12,
                         'figure.dpi': 150, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight'})


def _save_fig(fig, path):
    base = os.path.splitext(path)[0]
    fig.savefig(f"{base}.pdf", bbox_inches='tight')
    fig.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {base}.pdf / .png")


# ------------------------------------------------------------------ Plot 1
def plot_error_cdf(df, dataset, save_path):
    set_style()
    from .feature_engineering import fft_hr
    fig, ax = plt.subplots(figsize=(5, 4))
    methods = {'Complex': df['err_c'].values, 'Gated': df['err_g'].values,
               'Oracle': df['oracle_err'].values}
    n = len(df)
    fa = np.full(n, np.nan)
    for i in range(n):
        pc = np.asarray(df['pred_c'].iloc[i], dtype=np.float64).flatten()
        pg = np.asarray(df['pred_g'].iloc[i], dtype=np.float64).flatten()
        if len(pc) < 4 or len(pg) < 4: continue
        ml = min(len(pc), len(pg))
        h = fft_hr((pc[:ml] + pg[:ml]) / 2.0, 30)
        hl = df['hr_label'].iloc[i]
        if not np.isnan(h) and not np.isnan(hl): fa[i] = np.abs(h - hl)
    methods['FixedAvg'] = fa
    colors = {'Complex': '#1f77b4', 'Gated': '#ff7f0e', 'FixedAvg': '#2ca02c', 'Oracle': '#d62728'}
    styles = {'Complex': '-', 'Gated': '-', 'FixedAvg': '--', 'Oracle': ':'}
    for name, errs in methods.items():
        v = errs[~np.isnan(errs)]
        if len(v) == 0: continue
        s = np.sort(v); cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, cdf, color=colors.get(name, 'gray'),
                linestyle=styles.get(name, '-'), label=name, linewidth=1.5)
    ax.set_xlim(0, 30); ax.set_xlabel('Absolute HR Error (BPM)')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title(f'Error CDF — {dataset}'); ax.legend(loc='lower right')
    _save_fig(fig, save_path)


# ------------------------------------------------------------------ Plot 2
def plot_disagreement_vs_error(df, dataset, save_path):
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = df['hr_diff'].values; y = df['oracle_err'].values
    v = ~(np.isnan(x) | np.isnan(y)); x, y = x[v], y[v]
    ax1.scatter(x, y, s=4, alpha=0.3, color='steelblue', edgecolors='none')
    ax1.set_xlabel('|HR_c - HR_g| (BPM)'); ax1.set_ylabel('Oracle Error (BPM)')
    ax1.set_title(f'Disagreement vs Error — {dataset}')
    bins = [0, 2, 5, 10, 20, 100]; labels = ['0-2', '2-5', '5-10', '10-20', '>20']
    bi = np.digitize(x, bins[1:-1])
    bd, bl = [], []
    for i in range(len(labels)):
        m = bi == i
        if np.sum(m) > 5: bd.append(y[m]); bl.append(f'{labels[i]}\n(n={np.sum(m)})')
    bp = ax2.boxplot(bd, labels=bl, patch_artist=True)
    for p in bp['boxes']: p.set_facecolor('lightblue')
    ax2.set_xlabel('|HR_c - HR_g| (BPM)'); ax2.set_ylabel('Oracle Error (BPM)')
    ax2.set_title(f'Binned — {dataset}')
    _save_fig(fig, save_path)


# ------------------------------------------------------------------ Plot 3
def plot_fingerprint_kde(df_concat, save_path):
    """2D scatter + KDE contours with MMPD/PURE in distinct hues."""
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    datasets = df_concat['_dataset'].unique() if '_dataset' in df_concat.columns else ['unknown']
    colors = {'MMPD': '#e74c3c', 'PURE': '#3498db'}
    markers = {'MMPD': 'o', 'PURE': 's'}
    for ds in datasets:
        sub = df_concat[df_concat['_dataset'] == ds]
        if len(sub) > 5000: sub = sub.sample(5000, random_state=42)
        ax.scatter(sub['energy'], sub['entropy'],
                   c=colors.get(ds, 'gray'), marker=markers.get(ds, 'o'),
                   s=4, alpha=0.4, label=ds, edgecolors='none')
    for ds in datasets:
        sub = df_concat[df_concat['_dataset'] == ds]
        if len(sub) < 10: continue
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([sub['energy'].values, sub['entropy'].values])
            kde = gaussian_kde(xy)
            xg = np.linspace(sub['energy'].min(), sub['energy'].max(), 50)
            yg = np.linspace(sub['entropy'].min(), sub['entropy'].max(), 50)
            X, Y = np.meshgrid(xg, yg)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=5, colors=colors.get(ds, 'gray'),
                       alpha=0.6, linewidths=0.8)
        except Exception: pass
    ax.set_xlabel('Energy'); ax.set_ylabel('Entropy')
    ax.set_title('Pre-FFN Feature-Space Fingerprint')
    ax.legend(loc='upper right')
    _save_fig(fig, save_path)


# ------------------------------------------------------------------ Plot 4
def plot_subject_scatter(df, dataset, save_path):
    set_style()
    fig, ax = plt.subplots(figsize=(5, 5))
    sm = df.groupby('subject_id').agg(mae_c=('err_c', 'mean'),
                                       mae_g=('err_g', 'mean'),
                                       n=('err_c', 'count')).reset_index()
    mx = max(sm['mae_c'].max(), sm['mae_g'].max()) * 1.1
    ax.plot([0, mx], [0, mx], 'k--', linewidth=0.8, label='y=x')
    ax.scatter(sm['mae_c'], sm['mae_g'], s=sm['n'] / 10, alpha=0.6,
               color='steelblue', edgecolors='none')
    ax.set_xlabel('Complex MAE (BPM)'); ax.set_ylabel('Gated MAE (BPM)')
    ax.set_title(f'Subject-Level MAE — {dataset}')
    ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.legend(loc='upper left')
    _save_fig(fig, save_path)


# ------------------------------------------------------------------ Plot 5: Selective prediction curves
def plot_selective_prediction(abst_df, dataset, save_path):
    """Best_single curves for 4 reliability types. x = coverage (%)."""
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sub = abst_df[abst_df['method'] == 'best_single']
    colors = {'r_simple': '#1f77b4', 'r_learned': '#ff7f0e',
              'r_perexpert': '#2ca02c', 'r_oracle': '#d62728'}
    for rtype in ['r_simple', 'r_learned', 'r_perexpert', 'r_oracle']:
        s = sub[sub['reliability_type'] == rtype].sort_values('fraction')
        if len(s) == 0: continue
        ax.plot(s['coverage'] * 100, s['kept_MAE'], 'o-',
                color=colors.get(rtype, 'gray'), markersize=4,
                linewidth=1.5, label=rtype)
    ax.set_xlabel('Coverage (%)'); ax.set_ylabel('Kept MAE (BPM)')
    ax.set_title(f'Selective Prediction — {dataset}')
    ax.legend(fontsize=9); ax.invert_xaxis()
    _save_fig(fig, save_path)


def plot_selective_prediction_all(abst_dfs, save_path):
    """Multi-dataset selective prediction overview."""
    set_style()
    datasets = list(abst_dfs.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1: axes = [axes]
    colors = {'r_simple': '#1f77b4', 'r_learned': '#ff7f0e',
              'r_perexpert': '#2ca02c', 'r_oracle': '#d62728'}
    for ax, ds in zip(axes, datasets):
        sub = abst_dfs[ds]
        bs = sub[sub['method'] == 'best_single']
        for rtype in ['r_simple', 'r_learned', 'r_perexpert', 'r_oracle']:
            s = bs[bs['reliability_type'] == rtype].sort_values('fraction')
            if len(s) == 0: continue
            ax.plot(s['coverage'] * 100, s['kept_MAE'], 'o-',
                    color=colors.get(rtype, 'gray'), markersize=4,
                    linewidth=1.5, label=rtype)
        ax.set_xlabel('Coverage (%)'); ax.set_ylabel('Kept MAE (BPM)')
        ax.set_title(f'{ds}'); ax.legend(fontsize=8); ax.invert_xaxis()
    fig.suptitle('Selective Prediction', fontsize=13)
    _save_fig(fig, save_path)


# ------------------------------------------------------------------ Plot 6: Failure AUC bar
def plot_failure_auc_bar(selector_results, save_path):
    """Horizontal bar: feature_set x model for oracle_failure target, sorted by AUC."""
    set_style()
    datasets = list(selector_results.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 4))
    if n_ds == 1: axes = [axes]
    for ax, ds in zip(axes, datasets):
        sdf = selector_results[ds]
        sub = sdf[sdf['target'] == 'oracle_failure'].copy()
        if len(sub) == 0: continue
        sub['label'] = sub['feature_set'] + ' / ' + sub['model']
        sub = sub.sort_values('auc_mean', ascending=True)
        colors = ['steelblue' if 'logistic' in l else 'darkorange' for l in sub['label']]
        ax.barh(sub['label'], sub['auc_mean'], xerr=sub['auc_std'],
                color=colors, capsize=3)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, label='chance')
        ax.set_xlabel('AUC'); ax.set_title(f'Failure Prediction AUC — {ds}')
    fig.suptitle('Oracle Failure (both err > 10 BPM) Prediction', fontsize=13)
    _save_fig(fig, save_path)


# ------------------------------------------------------------------ Utility
def concat_with_label(dfs):
    pieces = []
    for ds, df in dfs.items():
        c = df.copy(); c['_dataset'] = ds; pieces.append(c)
    return pd.concat(pieces, ignore_index=True)
