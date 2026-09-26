"""
matching_model.py
------------------
The pairwise matching classifier: given engineered features for an
(S1, candidate) pair, predict P(same real-world business entity).

Model choice: LightGBM gradient-boosted trees.
  - Handles tabular similarity features very well out of the box
  - Fast to train/iterate on within a 72h hackathon window
  - Naturally handles class imbalance via `scale_pos_weight` / `is_unbalance`
  - Fully open-source (MIT licensed for the LightGBM library itself); the
    resulting model file has zero "foundation model" parameter count
    concerns -- it trivially satisfies the <=8B-parameter / MIT-Apache
    license constraint in the problem statement.

We frame this as binary classification (match / no-match) and rely on
threshold tuning (see evaluate.py) -- not the default 0.5 cutoff -- to hit
the precision-heavy F_0.5 objective.
"""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.features import FEATURE_COLUMNS


def build_training_examples(candidate_pairs: dict, ground_truth: dict):
    """
    Turn (candidate_pairs, ground_truth) into a labeled pair list for training.

    Args:
        candidate_pairs: dict s1_id -> list of candidate ids (from blocking)
        ground_truth: dict s1_id -> set of TRUE matching ids

    Returns:
        list of (s1_id, cand_id, label) where label=1 if truly matching,
        else 0. Only candidates that blocking actually retrieved are used
        as negatives (this mirrors real inference-time conditions -- the
        model never sees pairs it wouldn't see at test time).

        NOTE: any true positive missing from candidate_pairs (a blocking
        recall miss) is logged as a separate return value so you can
        monitor blocking quality -- it can never become a training example
        since it was never retrieved as a candidate.
    """
    examples = []
    missed_positives = 0
    total_positives = 0

    for s1_id, cands in candidate_pairs.items():
        true_matches = ground_truth.get(s1_id, set())
        total_positives += len(true_matches)
        cand_set = set(cands)

        for cand_id in cand_set:
            label = 1 if cand_id in true_matches else 0
            examples.append((s1_id, cand_id, label))

        missed_positives += len(true_matches - cand_set)

    return examples, missed_positives, total_positives


def train_matching_model(features_df: pd.DataFrame, labels: pd.Series,
                          groups: pd.Series = None, n_splits=5,
                          params: dict = None, random_state=42):
    """
    Train a LightGBM classifier with grouped cross-validation (grouped by
    source1_entity_id so pairs from the same S1 entity never straddle the
    train/val split -- prevents leakage and gives an honest estimate of
    generalization to unseen entities).

    Returns:
        final_model: LightGBM model trained on ALL data (for production use)
        cv_metrics: list of per-fold metric dicts (precision/recall/F1/F0.5/accuracy)
        oof_predictions: out-of-fold predicted probabilities, aligned to features_df index
                          (use these -- not train-set predictions -- for honest
                          threshold tuning in evaluate.py)
    """
    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "is_unbalance": True,  # class imbalance: true matches are rare vs candidates
        "random_state": random_state,
        "verbosity": -1,
    }
    if params:
        default_params.update(params)

    X = features_df[FEATURE_COLUMNS].values
    y = labels.values

    oof_pred = np.zeros(len(y), dtype=float)
    cv_metrics = []

    if groups is not None:
        n_groups = groups.nunique() if hasattr(groups, "nunique") else len(set(groups))
        effective_splits = max(2, min(n_splits, n_groups))
        if effective_splits < n_splits:
            print(f"    [matching_model] Only {n_groups} unique groups available -- "
                  f"reducing CV folds from {n_splits} to {effective_splits}.")
        gkf = GroupKFold(n_splits=effective_splits)
        splits = gkf.split(X, y, groups=groups)
    else:
        from sklearn.model_selection import StratifiedKFold
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        effective_splits = max(2, min(n_splits, n_pos, n_neg)) if min(n_pos, n_neg) > 0 else 2
        skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        splits = skf.split(X, y)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        model = lgb.LGBMClassifier(**default_params)
        model.fit(
            X[train_idx], y[train_idx],
            eval_X=X[val_idx], eval_y=y[val_idx],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        val_pred = model.predict_proba(X[val_idx])[:, 1]
        oof_pred[val_idx] = val_pred

        metrics = _classification_metrics(y[val_idx], (val_pred >= 0.5).astype(int))
        metrics["fold"] = fold_i
        cv_metrics.append(metrics)

    # Final model trained on all available data for production inference
    final_model = lgb.LGBMClassifier(**default_params)
    final_model.fit(X, y)

    return final_model, cv_metrics, oof_pred


def _classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
    }


def predict_proba(model, features_df: pd.DataFrame) -> np.ndarray:
    X = features_df[FEATURE_COLUMNS].values
    return model.predict_proba(X)[:, 1]


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
