"""
train.py
--------
End-to-end training orchestration:

  1. Load train_source{1,2,3}.tsv + train_ground_truth.tsv
  2. Split Source-1 entities into train/validation (grouped split -- an
     entity's candidates never leak across the split)
  3. Run blocking on the TRAIN split, measure blocking recall
  4. Build labeled pairwise features from candidates + ground truth
  5. Train LightGBM matcher with grouped CV, get out-of-fold predictions
  6. Tune the decision threshold on the VALIDATION split against the exact
     competition F_0.5 macro metric
  7. Save: trained model, fitted TF-IDF vectorizers, chosen threshold,
     embedding model name/config -- everything inference.py needs to
     reproduce results on the test set

Run from the `business_entity_resolution/` directory:
    python -m src.train --data-dir dataset/train --output-dir artifacts
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidates
from src.features import compute_features_batch, FEATURE_COLUMNS
from src.matching_model import build_training_examples, train_matching_model, save_model
from src.evaluate import macro_f_beta, summary_metrics, tune_threshold


def load_ground_truth(path: str) -> dict:
    """Parse train_ground_truth.tsv into dict: s1_id -> set(matched ids)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_raw = row.get("matched_entity_ids", "") or ""
        matched = {m.strip() for m in matched_raw.split(",") if m.strip()}
        gt[s1_id] = matched
    return gt


def split_source1_entities(source1_df, ground_truth, val_fraction=0.2, random_state=42):
    """
    Grouped split at the Source-1-entity level. Stratify loosely on
    "has any true match" so singletons aren't disproportionately dumped
    into one split (singletons behave very differently under F_0.5).
    """
    ids = source1_df["entity_id"].tolist()
    has_match = [1 if ground_truth.get(i) else 0 for i in ids]

    # Stratification requires >=2 members per class; fall back to a plain
    # random split on tiny datasets (or datasets with almost no singletons/
    # almost no matched entities) where that condition can't be met.
    class_counts = pd.Series(has_match).value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    train_ids, val_ids = train_test_split(
        ids, test_size=val_fraction, random_state=random_state,
        stratify=has_match if can_stratify else None,
    )
    train_df = source1_df[source1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_df = source1_df[source1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def run(data_dir: str, output_dir: str, use_embeddings: bool = True,
        embedding_device: str = None, max_candidates_per_entity: int = 30,
        val_fraction: float = 0.2, random_state: int = 42,
        max_token_doc_freq: int = None, max_postal_bucket_size: int = None,
        max_candidates_per_token_query: int = 200):
    os.makedirs(output_dir, exist_ok=True)

    print("[1/7] Loading training data...")
    source1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str, keep_default_na=False)
    source2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str, keep_default_na=False)
    source3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", dtype=str, keep_default_na=False)
    ground_truth = load_ground_truth(os.path.join(data_dir, "train_ground_truth.tsv"))

    print(f"    source1={len(source1)} source2={len(source2)} source3={len(source3)} "
          f"labeled_s1_entities={len(ground_truth)}")

    print("[2/7] Splitting Source-1 entities into train/validation...")
    s1_train, s1_val = split_source1_entities(source1, ground_truth, val_fraction, random_state)
    print(f"    train S1 entities={len(s1_train)}  val S1 entities={len(s1_val)}")

    # --- Blocking on TRAIN split (source2/source3 pool is shared/full, since
    #     those aren't split -- only S1 "queries" are split) ---
    print("[3/7] Running blocking (candidate generation) on train split...")
    train_candidates, train_embed_scores = generate_candidates(
        s1_train, source2, source3,
        use_embeddings=use_embeddings, embedding_device=embedding_device,
        max_candidates_per_entity=max_candidates_per_entity,
        max_token_doc_freq=max_token_doc_freq,
        max_postal_bucket_size=max_postal_bucket_size,
        max_candidates_per_token_query=max_candidates_per_token_query,
    )

    # Blocking recall diagnostic -- THE most important number to check before
    # trusting anything downstream.
    _, missed_positives, total_positives = build_training_examples(train_candidates, ground_truth)
    blocking_recall = 1 - (missed_positives / total_positives) if total_positives else float("nan")
    print(f"    Blocking recall on train split: {blocking_recall:.4f} "
          f"({missed_positives} missed / {total_positives} total true matches)")
    if blocking_recall < 0.85:
        print("    WARNING: blocking recall is low -- consider increasing "
              "embedding_top_k / max_candidates_per_entity, or add blocking strategies.")

    print("[4/7] Building pairwise training features...")
    examples, _, _ = build_training_examples(train_candidates, ground_truth)
    pair_list = [(s1_id, cand_id) for s1_id, cand_id, _ in examples]
    labels = pd.Series([lbl for _, _, lbl in examples], name="label")

    s1_train_prep = preprocess_dataframe(s1_train)
    pool_prep = preprocess_dataframe(pd.concat([source2, source3], ignore_index=True))

    features_df, name_vectorizer, addr_vectorizer = compute_features_batch(
        s1_train_prep, pool_prep, pair_list, embedding_scores=train_embed_scores,
    )
    # align labels to whatever rows survived compute_features_batch's defensive filtering
    features_df = features_df.reset_index(drop=True)
    labels = labels.iloc[: len(features_df)].reset_index(drop=True)
    groups = features_df["source1_entity_id"]

    print(f"    {len(features_df)} training pairs "
          f"({int(labels.sum())} positive / {len(labels) - int(labels.sum())} negative)")

    print("[5/7] Training LightGBM matcher with grouped 5-fold CV...")
    model, cv_metrics, oof_pred = train_matching_model(features_df, labels, groups=groups)
    avg_metrics = {k: np.mean([m[k] for m in cv_metrics]) for k in
                   ["accuracy", "precision", "recall", "f1", "f0.5"]}
    print("    CV metrics (threshold=0.5, for reference only -- real threshold tuned next):")
    for k, v in avg_metrics.items():
        print(f"      {k}: {v:.4f}")

    print("[6/7] Blocking + scoring VALIDATION split, tuning decision threshold...")
    val_candidates, val_embed_scores = generate_candidates(
        s1_val, source2, source3,
        use_embeddings=use_embeddings, embedding_device=embedding_device,
        max_candidates_per_entity=max_candidates_per_entity,
        max_token_doc_freq=max_token_doc_freq,
        max_postal_bucket_size=max_postal_bucket_size,
        max_candidates_per_token_query=max_candidates_per_token_query,
    )
    val_pair_list = [(s1_id, cand_id) for s1_id, cands in val_candidates.items() for cand_id in cands]
    s1_val_prep = preprocess_dataframe(s1_val)
    val_features_df, _, _ = compute_features_batch(
        s1_val_prep, pool_prep, val_pair_list, embedding_scores=val_embed_scores,
        name_vectorizer=name_vectorizer, addr_vectorizer=addr_vectorizer,
    )
    val_features_df["match_probability"] = model.predict_proba(val_features_df[FEATURE_COLUMNS].values)[:, 1]

    best_threshold, sweep_results = tune_threshold(
        val_features_df, ground_truth, all_source1_ids=s1_val["entity_id"].tolist(),
    )
    print(f"    Best threshold on validation: {best_threshold} "
          f"(macro F_0.5 = {max(r['macro_f0.5'] for r in sweep_results):.4f})")

    # Final validation report at the chosen threshold
    final_preds = {}
    subset = val_features_df[val_features_df["match_probability"] >= best_threshold]
    for s1_id, group in subset.groupby("source1_entity_id"):
        final_preds[s1_id] = set(group["candidate_entity_id"])
    val_summary = summary_metrics(final_preds, ground_truth, s1_val["entity_id"].tolist())
    print("    Validation summary at chosen threshold:")
    for k, v in val_summary.items():
        print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")

    print("[7/7] Saving artifacts...")
    save_model(model, os.path.join(output_dir, "matching_model.joblib"))
    joblib.dump(name_vectorizer, os.path.join(output_dir, "name_vectorizer.joblib"))
    joblib.dump(addr_vectorizer, os.path.join(output_dir, "addr_vectorizer.joblib"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({
            "threshold": best_threshold,
            "use_embeddings": use_embeddings,
            "max_candidates_per_entity": max_candidates_per_entity,
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "cv_metrics": avg_metrics,
            "validation_summary": val_summary,
            "blocking_recall_train_split": blocking_recall,
        }, f, indent=2)

    print(f"Done. Artifacts saved to {output_dir}/")
    return model, best_threshold, val_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--no-embeddings", action="store_true",
                         help="Disable embedding-based blocking (faster, CPU-only, lower recall).")
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or None for auto-detect.")
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-token-doc-freq", type=int, default=None,
                         help="Drop name tokens appearing in more than this many records "
                              "from the blocking index. Default: max(50, 1%% of corpus). "
                              "Lower this if you hit MemoryError during blocking.")
    parser.add_argument("--max-postal-bucket-size", type=int, default=None,
                         help="Cap entities sharing one postal code used for blocking. "
                              "Default: max(200, 2%% of corpus).")
    parser.add_argument("--max-candidates-per-token-query", type=int, default=200,
                         help="Safety cap on token-blocking candidates per S1 entity "
                              "before ranking/truncation.")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_embeddings=not args.no_embeddings,
        embedding_device=args.device,
        max_candidates_per_entity=args.max_candidates,
        val_fraction=args.val_fraction,
        max_token_doc_freq=args.max_token_doc_freq,
        max_postal_bucket_size=args.max_postal_bucket_size,
        max_candidates_per_token_query=args.max_candidates_per_token_query,
    )
