#!/usr/bin/env python3
"""
BMVC 2026 Failure-Aware Analysis — Run-All Orchestrator (v2).

Usage:
    python tools/bmvc_analysis/run_all.py \
        --dump_dir dumps_20260514 \
        --out_dir outputs_bmvc_v2 \
        --use_official_errors
"""
import os, sys, json, logging, argparse
import numpy as np
import pandas as pd

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJ_ROOT)

from tools.bmvc_analysis.io_utils import (load_npz, align_pair, save_table, save_json)
from tools.bmvc_analysis.feature_engineering import build_feature_table
from tools.bmvc_analysis.failure_metrics import compute_dataset_table
from tools.bmvc_analysis.failure_selectors import run_all_selectors
from tools.bmvc_analysis.abstention import (compute_abstention_table,
                                             make_abstention_summary)
from tools.bmvc_analysis.policy_evaluation import evaluate_policies
from tools.bmvc_analysis.failure_inspection import (export_top_failures,
                                                      plot_top_failures)
from tools.bmvc_analysis.official_vs_recomputed import (load_official_metrics,
                                                          compare_official_recomputed)
from tools.bmvc_analysis.plots import (
    plot_error_cdf, plot_disagreement_vs_error, plot_fingerprint_kde,
    plot_subject_scatter, plot_failure_auc_bar, concat_with_label,
    plot_selective_prediction, plot_selective_prediction_all,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('bmvc')


def main():
    p = argparse.ArgumentParser(description='BMVC 2026 Failure-Aware Analysis v2')
    p.add_argument('--dump_dir', default='dumps_20260514')
    p.add_argument('--out_dir', default='outputs_bmvc_v2')
    p.add_argument('--use_official_errors', action='store_true', default=False)
    args = p.parse_args()

    dump_dir = args.dump_dir
    out_dir = args.out_dir
    td = os.path.join(out_dir, 'tables')
    fd = os.path.join(out_dir, 'figures')
    ld = os.path.join(out_dir, 'logs')
    for d in [td, fd, ld]: os.makedirs(d, exist_ok=True)

    summary = {}
    feature_dfs = {}
    selector_results = {}
    abstention_dfs = {}
    oof_dicts = {}
    sanity_warnings = []

    for dataset in ['mmpd', 'pure']:
        logger.info(f"{'='*60}\nProcessing {dataset.upper()}\n{'='*60}")

        cp = os.path.join(dump_dir, f'complex_{dataset}.npz')
        gp = os.path.join(dump_dir, f'gated_{dataset}.npz')
        ap = os.path.join(dump_dir, f'aligned_{dataset}.npz')

        if not (os.path.exists(cp) and os.path.exists(gp)):
            logger.error(f"Missing dumps for {dataset}")
            continue

        # Step 1: Align
        logger.info("Loading & aligning dumps ...")
        c_data, g_data = load_npz(cp), load_npz(gp)
        df = align_pair(c_data, g_data)

        # Step 1b: Load official metrics
        official = {}
        if args.use_official_errors and os.path.exists(ap):
            official = load_official_metrics(ap)
            if official:
                logger.info(f"Official metrics loaded: {len(official)} fields")
            else:
                logger.warning("Official metrics unavailable, will recompute.")

        # Step 2: Build features
        logger.info("Building feature table ...")
        df_feat = build_feature_table(df, fs=30,
                                       use_official_errors=args.use_official_errors,
                                       official=official if official else None)
        feature_dfs[dataset] = df_feat

        # Save features (numeric + metadata only)
        num_cols = [c for c in df_feat.columns
                    if df_feat[c].dtype != 'object' or c in ('subject_id',)]
        save_table(df_feat[num_cols], os.path.join(td, f'features_{dataset}.csv'))

        # Step 3: Tail metrics
        logger.info("Computing tail metrics ...")
        tail_df = compute_dataset_table(df_feat, dataset, fs=30)
        save_table(tail_df, os.path.join(td, f'tail_metrics_{dataset}.csv'))

        mae_c = float(np.nanmean(df_feat['err_c']))
        mae_g = float(np.nanmean(df_feat['err_g']))
        oracle_mae = float(np.nanmean(df_feat['oracle_err']))
        mae_best = min(mae_c, mae_g)

        # Step 4: Selectors
        logger.info("Running selectors (subject-disjoint CV) ...")
        sel_df, oof_dict = run_all_selectors(df_feat, dataset, n_splits=5, seed=0)
        save_table(sel_df, os.path.join(td, f'selector_auc_{dataset}.csv'))
        selector_results[dataset] = sel_df
        oof_dicts[dataset] = oof_dict

        # Append OOF probs to feature table for persistence
        for key, arr in oof_dict.items():
            df_feat[key] = arr
        save_table(df_feat[num_cols + list(oof_dict.keys())],
                   os.path.join(td, f'features_{dataset}.csv'))

        # Extract key AUCs
        def _best_auc(target):
            rows = sel_df[(sel_df['target'] == target) &
                          (sel_df['feature_set'] == 'D_full') &
                          (sel_df['model'] == 'gbm')]
            return float(rows['auc_mean'].iloc[0]) if len(rows) > 0 else np.nan

        auc_of = _best_auc('oracle_failure')
        auc_w = _best_auc('winner')
        auc_fc = _best_auc('failure_c')
        auc_fg = _best_auc('failure_g')
        n_pos = int(sel_df[sel_df['target'] == 'oracle_failure']['n_pos'].max())

        # Step 5: Policy evaluation
        logger.info("Evaluating routing policies ...")
        pol_df = evaluate_policies(df_feat, oof_dict, dataset)
        save_table(pol_df, os.path.join(td, f'policy_eval_{dataset}.csv'))
        best_pol = pol_df.loc[pol_df['MAE'].idxmin()] if len(pol_df) > 0 else {}

        # Step 6: Abstention
        logger.info("Computing abstention curves ...")
        abst_df = compute_abstention_table(df_feat, dataset, oof_dict=oof_dict)
        save_table(abst_df, os.path.join(td, f'abstention_{dataset}.csv'))
        abstention_dfs[dataset] = abst_df

        abst_sum = make_abstention_summary(abst_df, dataset)

        # Step 7: Top failures
        logger.info("Exporting top failures ...")
        top_df = export_top_failures(df_feat, dataset, oof_dict=oof_dict)
        save_table(top_df, os.path.join(td, f'top_failures_{dataset}.csv'))

        # Step 8: Official vs recomputed
        logger.info("Comparing official vs recomputed ...")
        ovr_df = compare_official_recomputed(official, df_feat, dataset)
        save_table(ovr_df, os.path.join(td, f'official_vs_recomputed_{dataset}.csv'))

        # Sanity check: official vs recomputed MAE discrepancy
        for _, row in ovr_df.iterrows():
            if row['source'] == 'official_aligned':
                rc_row = ovr_df[(ovr_df['method'] == row['method']) &
                                (ovr_df['source'] == 'recomputed_from_pred')]
                if len(rc_row) > 0:
                    diff = abs(row['MAE'] - rc_row['MAE'].iloc[0])
                    if diff > 1.0:
                        w = (f"[{dataset}] {row['method']}: "
                             f"|official_MAE - recomputed_MAE| = {diff:.2f} BPM")
                        sanity_warnings.append(w)
                        logger.warning(w)

        # Build summary for this dataset
        ds_summary = {
            'mae_complex': round(mae_c, 4), 'mae_gated': round(mae_g, 4),
            'mae_oracle': round(oracle_mae, 4),
            'oracle_gap': round(mae_best - oracle_mae, 4),
            'winner_auc_dfull_gbm': round(auc_w, 4),
            'oracle_failure_auc_dfull_gbm': round(auc_of, 4),
            'failure_c_auc_dfull_gbm': round(auc_fc, 4),
            'failure_g_auc_dfull_gbm': round(auc_fg, 4),
            'best_single_mae': round(mae_best, 4),
            'best_policy_name': str(best_pol.get('policy', 'N/A')),
            'best_policy_mae': round(float(best_pol.get('MAE', np.nan)), 4),
            'n_windows': len(df_feat),
            'n_subjects': int(df_feat['subject_id'].nunique()),
            'n_positive_oracle_failure': n_pos,
            'positive_rate_oracle_failure': round(
                n_pos / len(df_feat) if len(df_feat) > 0 else np.nan, 4),
        }

        # FixedAvg MAE from tail metrics
        fa_row = tail_df[tail_df['Method'] == 'FixedAvg']
        ds_summary['fixed_avg_mae'] = round(float(fa_row['MAE'].iloc[0]), 4) if len(fa_row) > 0 else np.nan

        # Abstention key points
        for rtype in ['r_simple', 'r_learned']:
            for pct in [10, 20, 30]:
                sub = abst_df[(abst_df['method'] == 'best_single') &
                              (abst_df['reliability_type'] == rtype) &
                              (np.abs(abst_df['fraction'] - pct / 100) < 0.001)]
                val = float(sub['kept_MAE'].iloc[0]) if len(sub) > 0 else np.nan
                ds_summary[f'abstention_best_single_{rtype}_{pct}'] = round(val, 4)

        # Official MAE from aligned npz
        off_c = ovr_df[(ovr_df['source'] == 'official_aligned') & (ovr_df['method'] == 'Complex')]
        off_g = ovr_df[(ovr_df['source'] == 'official_aligned') & (ovr_df['method'] == 'Gated')]
        off_o = ovr_df[(ovr_df['source'] == 'official_aligned') & (ovr_df['method'] == 'Oracle')]
        ds_summary['official_mae_complex'] = round(float(off_c['MAE'].iloc[0]), 4) if len(off_c) > 0 else np.nan
        ds_summary['official_mae_gated'] = round(float(off_g['MAE'].iloc[0]), 4) if len(off_g) > 0 else np.nan
        ds_summary['official_mae_oracle'] = round(float(off_o['MAE'].iloc[0]), 4) if len(off_o) > 0 else np.nan
        ds_summary['recomputed_mae_complex'] = round(mae_c, 4)
        ds_summary['recomputed_mae_gated'] = round(mae_g, 4)
        ds_summary['recomputed_mae_oracle'] = round(oracle_mae, 4)

        summary[dataset] = ds_summary

        # Acceptance checks
        if dataset == 'mmpd':
            cov70_mae = ds_summary.get('abstention_best_single_r_simple_30', 999)
            if cov70_mae > 12.5:
                sanity_warnings.append(
                    f"[MMPD] abstention r_simple coverage=70% kept_MAE={cov70_mae:.2f} > 12.5 threshold")
            if auc_of < 0.65:
                sanity_warnings.append(
                    f"[MMPD] oracle_failure AUC={auc_of:.4f} < 0.65 threshold")
        if dataset == 'pure':
            cov90_mae = ds_summary.get('abstention_best_single_r_learned_10', 999)
            cov80_mae = ds_summary.get('abstention_best_single_r_learned_20', 999)
            if cov90_mae > 1.5:
                sanity_warnings.append(
                    f"[PURE] abstention r_learned coverage=90% kept_MAE={cov90_mae:.2f} > 1.5")
            if cov80_mae > 1.0:
                sanity_warnings.append(
                    f"[PURE] abstention r_learned coverage=80% kept_MAE={cov80_mae:.2f} > 1.0")
            if auc_of < 0.90:
                sanity_warnings.append(
                    f"[PURE] oracle_failure AUC={auc_of:.4f} < 0.90 threshold")

    # =====================================================================
    # Cross-dataset outputs
    # =====================================================================
    # Abstention summary
    all_abst_sum = []
    for ds in ['mmpd', 'pure']:
        if ds in abstention_dfs:
            all_abst_sum.append(make_abstention_summary(abstention_dfs[ds], ds))
    if all_abst_sum:
        save_table(pd.concat(all_abst_sum, ignore_index=True),
                   os.path.join(td, 'abstention_summary.csv'))

    # Sanity warnings
    if sanity_warnings:
        with open(os.path.join(ld, 'sanity_warnings.txt'), 'w') as f:
            for w in sanity_warnings:
                f.write(w + '\n')
        logger.warning(f"{len(sanity_warnings)} sanity warnings written to sanity_warnings.txt")
    else:
        logger.info("All acceptance thresholds passed.")

    # =====================================================================
    # Plots
    # =====================================================================
    logger.info("\nGenerating plots ...")
    for dataset in ['mmpd', 'pure']:
        if dataset not in feature_dfs:
            continue
        df_feat = feature_dfs[dataset]
        ds_up = dataset.upper()
        plot_error_cdf(df_feat, ds_up, os.path.join(fd, f'error_cdf_{dataset}'))
        plot_disagreement_vs_error(df_feat, ds_up,
                                   os.path.join(fd, f'disagreement_vs_error_{dataset}'))
        plot_subject_scatter(df_feat, ds_up,
                             os.path.join(fd, f'subject_scatter_{dataset}'))
        # Selective prediction
        if dataset in abstention_dfs:
            plot_selective_prediction(abstention_dfs[dataset], ds_up,
                                      os.path.join(fd, f'selective_prediction_{dataset}'))
        # Top failures plot
        if dataset in oof_dicts:
            plot_top_failures(df_feat, ds_up, oof_dict=oof_dicts[dataset], n_panels=12,
                              save_path=os.path.join(fd, f'top_failures_{dataset}'))

    if len(feature_dfs) >= 2:
        plot_fingerprint_kde(concat_with_label(feature_dfs),
                             os.path.join(fd, 'fingerprint_kde'))
    if abstention_dfs:
        plot_selective_prediction_all(abstention_dfs,
                                      os.path.join(fd, 'selective_prediction_all'))
    if selector_results:
        plot_failure_auc_bar(selector_results, os.path.join(fd, 'failure_auc_bar'))

    # =====================================================================
    # Summary
    # =====================================================================
    save_json(summary, os.path.join(ld, 'summary.json'))

    # =====================================================================
    # Terminal summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("  BMVC 2026 FAILURE-AWARE ANALYSIS — SUMMARY v2")
    print("=" * 70)
    for ds in ['mmpd', 'pure']:
        if ds not in summary: continue
        s = summary[ds]
        print(f"\n[{ds.upper()}]")
        print(f"  Best Single MAE:     {s['best_single_mae']:.2f} BPM")
        print(f"  Oracle MAE:          {s['mae_oracle']:.2f} BPM")
        print(f"  Complementarity Gap: {s['oracle_gap']:.2f} BPM")
        print(f"  Winner AUC (D_full/gbm):        {s['winner_auc_dfull_gbm']:.4f}")
        print(f"  Oracle Failure AUC (D_full/gbm):{s['oracle_failure_auc_dfull_gbm']:.4f}")
        print(f"  Failure-C AUC (D_full/gbm):     {s['failure_c_auc_dfull_gbm']:.4f}")
        print(f"  Failure-G AUC (D_full/gbm):     {s['failure_g_auc_dfull_gbm']:.4f}")
        print(f"  Best Policy:         {s['best_policy_name']} (MAE={s['best_policy_mae']:.2f})")
        for pct in [10, 20, 30]:
            k_s = f'abstention_best_single_r_simple_{pct}'
            k_l = f'abstention_best_single_r_learned_{pct}'
            if k_s in s:
                print(f"  Selective@{100-pct}% kept (r_simple/r_learned): "
                      f"{s[k_s]:.2f} / {s.get(k_l, np.nan):.2f} BPM")
        print(f"  Windows: {s['n_windows']}, Subjects: {s['n_subjects']}")
        print(f"  Oracle Failure: n_pos={s['n_positive_oracle_failure']}, "
              f"rate={s['positive_rate_oracle_failure']:.3f}")

    if sanity_warnings:
        print(f"\n  *** {len(sanity_warnings)} SANITY WARNINGS (see logs/sanity_warnings.txt) ***")
    print(f"\nOutputs: {out_dir}/")


if __name__ == '__main__':
    main()
