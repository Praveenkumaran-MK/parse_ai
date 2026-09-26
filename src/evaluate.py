"""
evaluate.py
-----------
Implements the EXACT competition scoring formula so local validation scores
are predictive of leaderboard scores:

    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed PER Source-1 entity, then macro-averaged across all Source 1
entities in the evaluation set. Per the spec:
  - A Source 1 entity with no true matches scores 1.0 if you predict an
    empty list, 0.0 if you predict ANY match for it.
  - This is NOT the standard micro-averaged F-beta over all pairs -- it is
    a per-entity macro average, so getting large-match entities wrong hurts
    the same amount as getting a singleton wrong. Optimize accordingly.
"""

from typing import Dict, Set

import numpy as np


def _per_entity_f_beta(predicted: Set[str], truth: Set[str], beta: float = 0.5) -> float:
    """
    F_beta for one Source 1 entity's predicted vs true match sets.
    Handles the singleton edge case exactly as specified.
    """
    if not truth:
        # No true matches: full credit for correctly predicting empty,
        # zero credit for predicting anything.
        return 1.0 if not predicted else 0.0

    if not predicted:
        # True matches exist but we predicted none: recall = 0 -> F = 0
        return 0.0

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f_beta(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]],
                  all_source1_ids, beta: float = 0.5) -> float:
    """
    Macro-averaged F_beta across every Source 1 entity in `all_source1_ids`.

    Args:
        predictions: dict s1_id -> predicted set of matched ids
                     (missing keys are treated as "predicted empty")
        ground_truth: dict s1_id -> true set of matched ids
                      (missing keys are treated as "true empty" / singleton)
        all_source1_ids: iterable of every Source 1 id that MUST be scored
                          (ensures entities with no predictions still count)
        beta: 0.5 to match the competition metric exactly

    Returns:
        float macro-averaged F_beta score
    """
    scores = []
    for s1_id in all_source1_ids:
        pred = predictions.get(s1_id, set())
        truth = ground_truth.get(s1_id, set())
        scores.append(_per_entity_f_beta(pred, truth, beta=beta))
    return sum(scores) / len(scores) if scores else 0.0


def per_entity_scores(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]],
                       all_source1_ids, beta: float = 0.5):
    """Same as macro_f_beta but returns the raw per-entity list -- useful for
    error analysis (which entities are dragging the score down)."""
    rows = []
    for s1_id in all_source1_ids:
        pred = predictions.get(s1_id, set())
        truth = ground_truth.get(s1_id, set())
        score = _per_entity_f_beta(pred, truth, beta=beta)
        rows.append({
            "source1_entity_id": s1_id,
            "predicted": sorted(pred),
            "truth": sorted(truth),
            "f_beta": score,
        })
    return rows


def summary_metrics(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]],
                     all_source1_ids):
    """
    Micro-averaged precision/recall/F1/accuracy across all pairs (pooled),
    PLUS the official macro F_0.5. The micro numbers are for your own
    diagnostic/documentation purposes (the PDF asks you to report model
    quality) -- the macro F_0.5 is what's actually scored on the leaderboard.
    """
    tp = fp = fn = 0
    correct_singletons = 0
    total_singletons = 0
    exact_match_entities = 0

    for s1_id in all_source1_ids:
        pred = predictions.get(s1_id, set())
        truth = ground_truth.get(s1_id, set())

        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)

        if not truth:
            total_singletons += 1
            if not pred:
                correct_singletons += 1

        if pred == truth:
            exact_match_entities += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    f0_5 = ((1.25 * precision * recall) / (0.25 * precision + recall)) if (0.25 * precision + recall) > 0 else 0.0

    n_entities = len(list(all_source1_ids)) if not isinstance(all_source1_ids, list) else len(all_source1_ids)

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "micro_f0.5": f0_5,
        "macro_f0.5": macro_f_beta(predictions, ground_truth, all_source1_ids, beta=0.5),
        "singleton_accuracy": correct_singletons / total_singletons if total_singletons else float("nan"),
        "exact_match_entity_rate": exact_match_entities / n_entities if n_entities else float("nan"),
    }


def tune_threshold(scored_pairs_df, ground_truth: Dict[str, Set[str]], all_source1_ids,
                    thresholds=None, id_col_s1="source1_entity_id",
                    id_col_cand="candidate_entity_id", score_col="match_probability"):
    """
    Sweep decision thresholds on a (typically out-of-fold / validation)
    scored-pairs dataframe and return the threshold that maximizes macro
    F_0.5 -- this is the single highest-leverage tuning step for this
    precision-heavy metric. Do NOT just use 0.5.

    Returns:
        best_threshold: float
        results: list of {threshold, macro_f0.5} for the full sweep (for plotting/debugging)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in list(np.arange(0.30, 0.96, 0.02))]

    results = []
    best_threshold, best_score = 0.5, -1.0

    for t in thresholds:
        preds = {}
        subset = scored_pairs_df[scored_pairs_df[score_col] >= t]
        for s1_id, group in subset.groupby(id_col_s1):
            preds[s1_id] = set(group[id_col_cand])

        score = macro_f_beta(preds, ground_truth, all_source1_ids, beta=0.5)
        results.append({"threshold": t, "macro_f0.5": score})
        if score > best_score:
            best_score, best_threshold = score, t

    return best_threshold, results
