"""
inference.py
------------
Runs the full pipeline on the TEST set and produces the two required
submission files:

  output/candidate_pairs.tsv   -- blocking stage output (unscored, but must
                                    be a superset of matching_results.tsv)
  output/matching_results.tsv  -- final matches (THIS is what's scored)

Run from the `business_entity_resolution/` directory:
    python -m src.inference --data-dir dataset/test --artifacts-dir artifacts \
        --output-dir output
"""

import argparse
import json
import os

import joblib
import pandas as pd

from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidates
from src.features import compute_features_batch, FEATURE_COLUMNS
from src.matching_model import load_model, predict_proba


def write_id_list_tsv(path: str, id_col: str, list_col_name: str, data: dict):
    """
    Write a {s1_id: [ids...]} dict to TSV in the exact required format:
    one row per key, comma-separated list (no quoting), empty string if none.
    """
    rows = []
    for s1_id, ids in data.items():
        rows.append({id_col: s1_id, list_col_name: ",".join(sorted(ids)) if ids else ""})
    df = pd.DataFrame(rows, columns=[id_col, list_col_name])
    df.to_csv(path, sep="\t", index=False)


def run(data_dir: str, artifacts_dir: str, output_dir: str,
        use_embeddings: bool = True, embedding_device: str = None,
        max_candidates_per_entity: int = 30, threshold: float = None,
        max_token_doc_freq: int = None, max_postal_bucket_size: int = None,
        max_candidates_per_token_query: int = 200):
    os.makedirs(output_dir, exist_ok=True)

    print("[1/5] Loading test data...")
    source1 = pd.read_csv(os.path.join(data_dir, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False)
    source2 = pd.read_csv(os.path.join(data_dir, "test_source2.tsv"), sep="\t", dtype=str, keep_default_na=False)
    source3 = pd.read_csv(os.path.join(data_dir, "test_source3.tsv"), sep="\t", dtype=str, keep_default_na=False)
    print(f"    test source1={len(source1)} source2={len(source2)} source3={len(source3)}")

    print("[2/5] Loading trained artifacts...")
    model = load_model(os.path.join(artifacts_dir, "matching_model.joblib"))
    name_vectorizer = joblib.load(os.path.join(artifacts_dir, "name_vectorizer.joblib"))
    addr_vectorizer = joblib.load(os.path.join(artifacts_dir, "addr_vectorizer.joblib"))
    with open(os.path.join(artifacts_dir, "config.json")) as f:
        config = json.load(f)
    chosen_threshold = threshold if threshold is not None else config["threshold"]
    print(f"    Using decision threshold = {chosen_threshold}")

    print("[3/5] Running blocking (candidate generation) on test set...")
    candidate_pairs, embedding_scores = generate_candidates(
        source1, source2, source3,
        use_embeddings=use_embeddings, embedding_device=embedding_device,
        max_candidates_per_entity=max_candidates_per_entity,
        max_token_doc_freq=max_token_doc_freq,
        max_postal_bucket_size=max_postal_bucket_size,
        max_candidates_per_token_query=max_candidates_per_token_query,
    )
    n_with_candidates = sum(1 for v in candidate_pairs.values() if v)
    print(f"    {n_with_candidates}/{len(candidate_pairs)} Source-1 entities have >=1 candidate")

    print("[4/5] Scoring candidates with matching model...")
    pool_prep = preprocess_dataframe(pd.concat([source2, source3], ignore_index=True))
    s1_prep = preprocess_dataframe(source1)
    pair_list = [(s1_id, cand_id) for s1_id, cands in candidate_pairs.items() for cand_id in cands]

    features_df, _, _ = compute_features_batch(
        s1_prep, pool_prep, pair_list, embedding_scores=embedding_scores,
        name_vectorizer=name_vectorizer, addr_vectorizer=addr_vectorizer,
    )

    matching_results = {eid: set() for eid in source1["entity_id"]}  # every S1 id must appear, default empty
    if len(features_df) > 0:
        features_df["match_probability"] = predict_proba(model, features_df)
        confident = features_df[features_df["match_probability"] >= chosen_threshold]
        for s1_id, group in confident.groupby("source1_entity_id"):
            matching_results[s1_id] = set(group["candidate_entity_id"])

    print("[5/5] Writing output files...")
    # candidate_pairs.tsv: must include every candidate actually fed to the model
    # (i.e. exactly `candidate_pairs` from blocking -- the final pre-inference set)
    write_id_list_tsv(
        os.path.join(output_dir, "candidate_pairs.tsv"),
        "source1_entity_id", "candidate_entity_ids", candidate_pairs,
    )
    write_id_list_tsv(
        os.path.join(output_dir, "matching_results.tsv"),
        "source1_entity_id", "matched_entity_ids", matching_results,
    )

    n_matched = sum(1 for v in matching_results.values() if v)
    n_singleton_pred = sum(1 for v in matching_results.values() if not v)
    print(f"    Done. {n_matched} entities matched, {n_singleton_pred} predicted as singletons.")
    print(f"    Files written to {output_dir}/matching_results.tsv and {output_dir}/candidate_pairs.tsv")

    # Sanity check: every match must be a subset of that entity's candidates.
    # (The organizers' validator checks this too -- catch it here first.)
    violations = 0
    for s1_id, matches in matching_results.items():
        if not matches.issubset(set(candidate_pairs.get(s1_id, []))):
            violations += 1
    if violations:
        print(f"    WARNING: {violations} entities have matches NOT present in their "
              f"candidate set -- this will fail the organizers' validator. Investigate before submitting.")
    else:
        print("    Consistency check passed: all matches are subsets of their candidates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/test")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override the threshold saved in artifacts/config.json")
    parser.add_argument("--max-token-doc-freq", type=int, default=None)
    parser.add_argument("--max-postal-bucket-size", type=int, default=None)
    parser.add_argument("--max-candidates-per-token-query", type=int, default=200)
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        use_embeddings=not args.no_embeddings,
        embedding_device=args.device,
        max_candidates_per_entity=args.max_candidates,
        threshold=args.threshold,
        max_token_doc_freq=args.max_token_doc_freq,
        max_postal_bucket_size=args.max_postal_bucket_size,
        max_candidates_per_token_query=args.max_candidates_per_token_query,
    )
