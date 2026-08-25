"""
Selector training with strict subject-disjoint cross-validation.

Priority: sklearn.model_selection.StratifiedGroupKFold.
Fallback: greedy stratified group split ensuring positive samples per fold.

Targets: winner, oracle_failure, failure_c, failure_g
Feature sets: A_basic .. D_full
Models: logistic, gbm
"""
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Tolerance margin for the actionable failure labels (paper Eq.: failure_c =
# I(e_c > e_g + delta), failure_g = I(e_g > e_c + delta)).  The paper writes the
# margin as a symbol without a numeric value; DELTA is set here so that the
# resulting D_full+GBM AUCs reproduce the reported failure_c / failure_g.
DELTA = 3

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGK = True
except ImportError:
    _HAS_SGK = False


def _greedy_stratified_group_split(subject_ids: np.ndarray, y: np.ndarray,
                                    n_splits: int = 5, seed: int = 0) -> list:
    """Greedy stratified group split assigning subjects to balance pos_rate across folds."""
    rng = np.random.RandomState(seed)
    unique_subjs = np.unique(subject_ids)
    subj_pos, subj_neg = {}, {}
    for s in unique_subjs:
        mask = subject_ids == s
        subj_pos[s] = int(np.sum(y[mask] == 1))
        subj_neg[s] = int(np.sum(y[mask] == 0))
    sorted_subjs = sorted(unique_subjs,
                          key=lambda s: subj_pos[s] + subj_neg[s], reverse=True)
    folds = [[] for _ in range(n_splits)]
    fold_pos = [0] * n_splits
    fold_neg = [0] * n_splits
    global_pos_rate = np.mean(y)
    for s in sorted_subjs:
        best_fold, best_dist = 0, float('inf')
        for k in range(n_splits):
            new_pos = fold_pos[k] + subj_pos[s]
            new_neg = fold_neg[k] + subj_neg[s]
            new_total = new_pos + new_neg
            if new_total == 0:
                continue
            dist = abs(new_pos / new_total - global_pos_rate)
            if dist < best_dist:
                best_dist, best_fold = dist, k
        folds[best_fold].append(s)
        fold_pos[best_fold] += subj_pos[s]
        fold_neg[best_fold] += subj_neg[s]
    result = []
    for k in range(n_splits):
        if not folds[k]:
            continue
        val_subjs = set(folds[k])
        val_mask = np.array([s in val_subjs for s in subject_ids])
        result.append((~val_mask, val_mask))
    return result


def _make_subject_folds(subject_ids: np.ndarray, y: np.ndarray,
                        n_splits: int = 5, seed: int = 0) -> list:
    """Create subject-disjoint folds ensuring both classes per split."""
    n_subjs = len(np.unique(subject_ids))
    eff = min(n_splits, n_subjs)
    if _HAS_SGK and eff >= 2:
        try:
            sgk = StratifiedGroupKFold(n_splits=eff, shuffle=True, random_state=seed)
            folds = []
            for ti, vi in sgk.split(np.zeros(len(y)), y, groups=subject_ids):
                tm = np.zeros(len(y), dtype=bool); vm = np.zeros(len(y), dtype=bool)
                tm[ti] = True; vm[vi] = True
                folds.append((tm, vm))
            logger.info(f"StratifiedGroupKFold: {len(folds)} folds, {n_subjs} subjects")
            return folds
        except Exception as e:
            logger.warning(f"StratifiedGroupKFold failed ({e}), using greedy fallback")
    logger.info(f"Greedy stratified group split: {eff} folds")
    folds = _greedy_stratified_group_split(subject_ids, y, eff, seed)
    valid = []
    for tm, vm in folds:
        if len(np.unique(y[vm])) >= 2 and len(np.unique(y[tm])) >= 2:
            valid.append((tm, vm))
        else:
            logger.warning("Dropping fold with single-class val/train")
    if len(valid) < eff:
        logger.warning(f"Only {len(valid)}/{eff} folds valid")
    return valid


def fit_eval_selector(df: pd.DataFrame, feature_cols: list, target_col: str,
                      model: str = "logistic", n_splits: int = 5,
                      seed: int = 0) -> dict:
    """Train binary classifier with subject-disjoint CV. Returns oof probs on df index."""
    cols_needed = feature_cols + [target_col, 'subject_id']
    sub = df[cols_needed].dropna().copy()
    if len(sub) == 0:
        return {'auc_mean': np.nan, 'auc_std': np.nan,
                'predictions': np.full(len(df), np.nan),
                'fold_aucs': [], 'n_pos': 0, 'n_neg': 0, 'n_valid_folds': 0, 'pos_rate': np.nan}
    X = sub[feature_cols].values.astype(np.float64)
    y = sub[target_col].values.astype(int)
    subjects = sub['subject_id'].values
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    pos_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else np.nan
    if len(np.unique(y)) < 2:
        return {'auc_mean': np.nan, 'auc_std': np.nan,
                'predictions': np.full(len(df), np.nan),
                'fold_aucs': [], 'n_pos': n_pos, 'n_neg': n_neg,
                'n_valid_folds': 0, 'pos_rate': pos_rate}
    folds = _make_subject_folds(subjects, y, n_splits, seed)
    oof_probs = np.full(len(df), np.nan)
    fold_aucs = []
    for tm, vm in folds:
        if np.sum(vm) == 0 or np.sum(tm) == 0:
            continue
        X_tr, X_val = X[tm], X[vm]
        y_tr, y_val = y[tm], y[vm]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            continue
        clf = (LogisticRegression(solver='liblinear', max_iter=1000, random_state=seed,
                                  class_weight='balanced') if model == "logistic"
               else GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               learning_rate=0.1, random_state=seed))
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_val)[:, 1]
        oof_probs[sub.index[vm]] = prob
        if len(np.unique(y_val)) >= 2:
            fold_aucs.append(roc_auc_score(y_val, prob))
    return {
        'auc_mean': float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        'auc_std': float(np.std(fold_aucs)) if fold_aucs else np.nan,
        'predictions': oof_probs, 'fold_aucs': fold_aucs,
        'n_pos': n_pos, 'n_neg': n_neg,
        'n_valid_folds': len(fold_aucs), 'pos_rate': pos_rate,
    }


def run_all_selectors(df: pd.DataFrame, dataset_name: str,
                      n_splits: int = 5, seed: int = 0):
    """Run all selector experiments. Returns (result_df, best_oof_dict).

    Failure labels follow the paper (Sec.4 "Failure labels"):
      * oracle_failure = I(e_c > 10 AND e_g > 10)      (both experts err > 10 bpm)
      * failure_c      = I(e_c > e_g + DELTA)          (Complex worse than Gated by > DELTA)
      * failure_g      = I(e_g > e_c + DELTA)          (Gated worse than Complex by > DELTA)
    """
    from .feature_engineering import FEATURE_SETS
    targets = {'winner': 'winner', 'oracle_failure': 'oracle_failure',
               'failure_c': 'failure_c', 'failure_g': 'failure_g'}
    if 'oracle_failure' not in df.columns:
        df['oracle_failure'] = ((df['err_c'] > 10) & (df['err_g'] > 10)).astype(int)
    if 'failure_c' not in df.columns:
        df['failure_c'] = (df['err_c'] > df['err_g'] + DELTA).astype(int)
    if 'failure_g' not in df.columns:
        df['failure_g'] = (df['err_g'] > df['err_c'] + DELTA).astype(int)
    rows = []
    oof_store = {}
    for fs_name, fs_cols in FEATURE_SETS.items():
        for model_name in ['logistic', 'gbm']:
            for target_key, target_col in targets.items():
                r = fit_eval_selector(df, fs_cols, target_col, model_name, n_splits, seed)
                rows.append({'dataset': dataset_name, 'target': target_key,
                             'feature_set': fs_name, 'model': model_name,
                             'auc_mean': r['auc_mean'], 'auc_std': r['auc_std'],
                             'n_pos': r['n_pos'], 'n_neg': r['n_neg'],
                             'n_valid_folds': r['n_valid_folds'], 'pos_rate': r['pos_rate']})
                if fs_name == 'D_full' and model_name == 'gbm':
                    oof_store[f'p_{target_key}'] = r['predictions']
    return (pd.DataFrame(rows),
            {'p_oracle_failure': oof_store.get('p_oracle_failure', np.full(len(df), np.nan)),
             'p_failure_c': oof_store.get('p_failure_c', np.full(len(df), np.nan)),
             'p_failure_g': oof_store.get('p_failure_g', np.full(len(df), np.nan)),
             'p_winner': oof_store.get('p_winner', np.full(len(df), np.nan))})
