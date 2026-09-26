"""
smoke_test.py
-------------
End-to-end smoke test on a 5,000-S1-entity subsample drawn from the REAL
training data. This is NOT make_synthetic_data.py — it uses actual data
so the difficulty (name noise, Devanagari, missing addresses) is realistic.

What it does:
  1. Sample 5,000 S1 entities (stratified: ~5.6% singletons)
  2. Find all S2/S3 records that are true matches for those S1 entities
  3. Add ~3x the matched-pool size of *random noise* S2/S3 records so the
     candidate pool is challenging (not just the positives)
  4. Write a mini-dataset to dataset/smoke/ (train + test sub-dirs)
  5. Run train.py --no-embeddings end-to-end
  6. Run inference.py --no-embeddings on the same mini test set
  7. Run validate_submission.py
  8. Print the final macro F_0.5 from artifacts_smoke/config.json

Usage:
    python smoke_test.py [--n-s1 5000] [--noise-factor 3]
"""

import argparse
import json
import os
import random
import subprocess
import sys

import pandas as pd

random.seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-s1", type=int, default=5000,
                        help="Number of S1 entities to sample (default 5000)")
    parser.add_argument("--noise-factor", type=int, default=3,
                        help="Add this many noise pool records per true-match record")
    parser.add_argument("--data-dir", default="dataset/train",
                        help="Path to real training data dir")
    parser.add_argument("--out-dir", default="dataset/smoke",
                        help="Where to write the mini dataset")
    parser.add_argument("--artifacts-dir", default="artifacts_smoke",
                        help="Where to save model artifacts")
    parser.add_argument("--output-dir", default="output_smoke",
                        help="Where to save inference output")
    args = parser.parse_args()

    smoke_train = os.path.join(args.out_dir, "train")
    smoke_test = os.path.join(args.out_dir, "test")
    os.makedirs(smoke_train, exist_ok=True)
    os.makedirs(smoke_test, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load real data (headers only first to verify schema)
    # -----------------------------------------------------------------------
    print(f"[1/7] Loading real training data from {args.data_dir} ...")
    s1_full = pd.read_csv(f"{args.data_dir}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2_full = pd.read_csv(f"{args.data_dir}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3_full = pd.read_csv(f"{args.data_dir}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_full = pd.read_csv(f"{args.data_dir}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_full["matched_entity_ids"] = gt_full["matched_entity_ids"].fillna("")

    print(f"    loaded: s1={len(s1_full):,}  s2={len(s2_full):,}  "
          f"s3={len(s3_full):,}  gt={len(gt_full):,}")

    # -----------------------------------------------------------------------
    # 2. Sample S1 entities (stratified by has_match)
    # -----------------------------------------------------------------------
    print(f"[2/7] Sampling {args.n_s1:,} S1 entities (stratified by singleton/match) ...")
    gt_map = dict(zip(gt_full["source1_entity_id"], gt_full["matched_entity_ids"]))

    has_match = s1_full["entity_id"].map(lambda eid: bool(gt_map.get(eid, "")))
    matched_pool = s1_full[has_match].sample(
        min(int(args.n_s1 * 0.944), has_match.sum()), random_state=42)
    singleton_pool = s1_full[~has_match].sample(
        min(args.n_s1 - len(matched_pool), (~has_match).sum()), random_state=42)
    n_matched_sampled = len(matched_pool)
    n_singleton_sampled = len(singleton_pool)
    s1_sample = pd.concat([matched_pool, singleton_pool]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"    sampled: {len(s1_sample):,} S1  "
          f"({n_singleton_sampled} singletons, {n_matched_sampled} with matches)")

    # -----------------------------------------------------------------------
    # 3. Collect true-match pool records + noise
    # -----------------------------------------------------------------------
    print(f"[3/7] Collecting true-match S2/S3 records + noise ...")
    sampled_ids = set(s1_sample["entity_id"])
    s2_idx = s2_full.set_index("entity_id")
    s3_idx = s3_full.set_index("entity_id")

    true_s2_ids: set[str] = set()
    true_s3_ids: set[str] = set()
    gt_rows = []
    for s1_id in sampled_ids:
        raw = gt_map.get(s1_id, "")
        gt_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": raw})
        if raw:
            for mid in raw.split(","):
                mid = mid.strip()
                if mid.startswith("S2-"):
                    true_s2_ids.add(mid)
                elif mid.startswith("S3-"):
                    true_s3_ids.add(mid)

    n_true = len(true_s2_ids) + len(true_s3_ids)
    n_noise = args.noise_factor * n_true

    # Random noise from the rest of S2/S3 (not in true-match set)
    noise_s2_candidates = s2_full[~s2_full["entity_id"].isin(true_s2_ids)]
    noise_s3_candidates = s3_full[~s3_full["entity_id"].isin(true_s3_ids)]
    n_noise_s2 = min(n_noise // 2, len(noise_s2_candidates))
    n_noise_s3 = min(n_noise - n_noise_s2, len(noise_s3_candidates))

    noise_s2 = noise_s2_candidates.sample(n_noise_s2, random_state=42)
    noise_s3 = noise_s3_candidates.sample(n_noise_s3, random_state=42)

    s2_smoke = pd.concat([
        s2_full[s2_full["entity_id"].isin(true_s2_ids)],
        noise_s2,
    ]).reset_index(drop=True)
    s3_smoke = pd.concat([
        s3_full[s3_full["entity_id"].isin(true_s3_ids)],
        noise_s3,
    ]).reset_index(drop=True)
    gt_smoke = pd.DataFrame(gt_rows)

    print(f"    s2_smoke={len(s2_smoke):,}  s3_smoke={len(s3_smoke):,}  "
          f"(true_s2={len(true_s2_ids):,} true_s3={len(true_s3_ids):,} "
          f"noise_s2={n_noise_s2:,} noise_s3={n_noise_s3:,})")

    # -----------------------------------------------------------------------
    # 4. Write train split (use 80% of sampled S1 as train, 20% as val)
    # -----------------------------------------------------------------------
    print("[4/7] Writing smoke dataset files ...")
    s1_sample.to_csv(f"{smoke_train}/train_source1.tsv", sep="\t", index=False)
    s2_smoke.to_csv(f"{smoke_train}/train_source2.tsv", sep="\t", index=False)
    s3_smoke.to_csv(f"{smoke_train}/train_source3.tsv", sep="\t", index=False)
    gt_smoke.to_csv(f"{smoke_train}/train_ground_truth.tsv", sep="\t", index=False)

    # For test split: use same S1 entities (simulates inference on seen data —
    # good enough for a smoke-test; replace with a fresh holdout for real eval).
    s1_sample.to_csv(f"{smoke_test}/test_source1.tsv", sep="\t", index=False)
    s2_smoke.to_csv(f"{smoke_test}/test_source2.tsv", sep="\t", index=False)
    s3_smoke.to_csv(f"{smoke_test}/test_source3.tsv", sep="\t", index=False)

    print(f"    smoke data written to {args.out_dir}/")

    # -----------------------------------------------------------------------
    # 5. Run train
    # -----------------------------------------------------------------------
    print("[5/7] Running train (--no-embeddings) ...")
    ret = subprocess.run([
        sys.executable, "-m", "src.train",
        "--data-dir", smoke_train,
        "--output-dir", args.artifacts_dir,
        "--no-embeddings",
        "--max-candidates", "30",
        "--val-fraction", "0.2",
    ], check=False)
    if ret.returncode != 0:
        print(f"\n[FAIL] train.py exited with code {ret.returncode}")
        sys.exit(ret.returncode)

    # -----------------------------------------------------------------------
    # 6. Run inference
    # -----------------------------------------------------------------------
    print("[6/7] Running inference (--no-embeddings) ...")
    ret = subprocess.run([
        sys.executable, "-m", "src.inference",
        "--data-dir", smoke_test,
        "--artifacts-dir", args.artifacts_dir,
        "--output-dir", args.output_dir,
        "--no-embeddings",
    ], check=False)
    if ret.returncode != 0:
        print(f"\n[FAIL] inference.py exited with code {ret.returncode}")
        sys.exit(ret.returncode)

    # -----------------------------------------------------------------------
    # 7. Validate submission
    # -----------------------------------------------------------------------
    print("[7/7] Validating submission output ...")
    ret = subprocess.run([
        sys.executable, "utils/validate_submission.py",
        "--matching", f"{args.output_dir}/matching_results.tsv",
        "--candidate", f"{args.output_dir}/candidate_pairs.tsv",
        "--test-dir", smoke_test,
    ], check=False)
    if ret.returncode != 0:
        print(f"\n[FAIL] Validation FAILED (code {ret.returncode})")
        sys.exit(ret.returncode)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    config_path = os.path.join(args.artifacts_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print("\n=== Smoke Test Summary ===")
        print(f"  Best threshold:       {config.get('threshold')}")
        print(f"  Validation macro F0.5:{config.get('validation_summary', {}).get('macro_f0.5', 'n/a'):.4f}")
        print(f"  Blocking recall:      {config.get('blocking_recall_train_split', 'n/a'):.4f}")
    print("\n[PASS] Smoke test PASSED end-to-end\n")


if __name__ == "__main__":
    main()
