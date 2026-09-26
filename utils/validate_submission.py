"""
validate_submission.py
----------------------
Pre-submission validator that checks all hard constraints specified in the
competition rules. Run this BEFORE uploading to the leaderboard.

Checks:
  1. Every S1 entity from the test set appears exactly once in matching_results.tsv
  2. matched_entity_ids only references IDs present in test_source2/3 (no S1 IDs,
     no IDs from outside the test set)
  3. No duplicate entity IDs within a single matched_entity_ids cell
  4. No duplicate source1_entity_id rows in matching_results.tsv
  5. Empty string (not null/NaN) for singleton predictions
  6. Every match in matching_results.tsv also appears in candidate_pairs.tsv
     (candidates must be a strict superset of final matches)
  7. No S1 IDs cross-matched to each other (matches must be S2 or S3 only)

Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

import argparse
import sys

import pandas as pd


def err(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def validate(matching_path: str, candidate_path: str, test_dir: str) -> bool:
    print("\n=== Submission Validator ===\n")
    passed = True

    # --- Load test entity ID universes ---
    s1 = pd.read_csv(f"{test_dir}/test_source1.tsv", sep="\t", dtype=str,
                     keep_default_na=False, usecols=["entity_id"])
    s2 = pd.read_csv(f"{test_dir}/test_source2.tsv", sep="\t", dtype=str,
                     keep_default_na=False, usecols=["entity_id"])
    s3 = pd.read_csv(f"{test_dir}/test_source3.tsv", sep="\t", dtype=str,
                     keep_default_na=False, usecols=["entity_id"])
    s1_ids = set(s1["entity_id"].tolist())
    valid_match_ids = set(s2["entity_id"].tolist()) | set(s3["entity_id"].tolist())
    print(f"Test universe: {len(s1_ids):,} S1 entities, "
          f"{len(valid_match_ids):,} valid match IDs (S2+S3)")

    # --- Load submission files ---
    try:
        matching_df = pd.read_csv(matching_path, sep="\t", dtype=str,
                                   keep_default_na=False)
    except FileNotFoundError:
        err(f"matching_results file not found: {matching_path}")
        return False

    try:
        candidate_df = pd.read_csv(candidate_path, sep="\t", dtype=str,
                                    keep_default_na=False)
    except FileNotFoundError:
        err(f"candidate_pairs file not found: {candidate_path}")
        return False

    # Normalise column names
    matching_df.columns = [c.strip() for c in matching_df.columns]
    candidate_df.columns = [c.strip() for c in candidate_df.columns]

    # -----------------------------------------------------------------------
    # Check 1: required columns present
    # -----------------------------------------------------------------------
    for col in ["source1_entity_id", "matched_entity_ids"]:
        if col not in matching_df.columns:
            err(f"matching_results.tsv is missing column '{col}'")
            passed = False
    for col in ["source1_entity_id", "candidate_entity_ids"]:
        if col not in candidate_df.columns:
            err(f"candidate_pairs.tsv is missing column '{col}'")
            passed = False
    if not passed:
        return False
    ok("Required columns present in both files")

    # -----------------------------------------------------------------------
    # Check 2: no duplicate source1_entity_id rows in matching_results
    # -----------------------------------------------------------------------
    dupe_s1 = matching_df["source1_entity_id"].duplicated()
    if dupe_s1.any():
        err(f"{dupe_s1.sum()} duplicate source1_entity_id rows in matching_results.tsv "
            f"(first: {matching_df.loc[dupe_s1, 'source1_entity_id'].iloc[0]})")
        passed = False
    else:
        ok("No duplicate source1_entity_id rows in matching_results.tsv")

    # -----------------------------------------------------------------------
    # Check 3: every S1 entity appears exactly once
    # -----------------------------------------------------------------------
    result_ids = set(matching_df["source1_entity_id"].tolist())
    missing = s1_ids - result_ids
    extra = result_ids - s1_ids
    if missing:
        err(f"{len(missing):,} S1 entities from test set are MISSING from matching_results.tsv "
            f"(e.g. {sorted(missing)[:3]})")
        passed = False
    else:
        ok(f"All {len(s1_ids):,} S1 entities present in matching_results.tsv")
    if extra:
        err(f"{len(extra):,} entity IDs in matching_results.tsv are NOT in the test S1 set "
            f"(e.g. {sorted(extra)[:3]})")
        passed = False

    # -----------------------------------------------------------------------
    # Check 4: empty string (not NaN) for singletons + no duplicates within
    # a matched_entity_ids cell + all matched IDs in valid universe
    # -----------------------------------------------------------------------
    bad_null = matching_df["matched_entity_ids"].isna().sum()
    if bad_null:
        err(f"{bad_null} rows have null/NaN matched_entity_ids — use empty string '' for singletons")
        passed = False
    else:
        ok("No null matched_entity_ids (empty string used for singletons)")

    n_invalid_ids = 0
    n_s1_self_matches = 0
    n_dupes_within = 0
    for _, row in matching_df.iterrows():
        raw = row["matched_entity_ids"]
        if not raw or not raw.strip():
            continue
        ids = [x.strip() for x in raw.split(",") if x.strip()]
        # duplicates within the cell
        if len(ids) != len(set(ids)):
            n_dupes_within += 1
        for mid in ids:
            if mid in s1_ids:
                n_s1_self_matches += 1
            elif mid not in valid_match_ids:
                n_invalid_ids += 1

    if n_dupes_within:
        err(f"{n_dupes_within} rows contain duplicate IDs within a single matched_entity_ids cell")
        passed = False
    else:
        ok("No duplicate IDs within any matched_entity_ids cell")

    if n_s1_self_matches:
        err(f"{n_s1_self_matches} S1 entity IDs appear as match targets — self-matches are forbidden")
        passed = False
    else:
        ok("No S1 IDs used as match targets")

    if n_invalid_ids:
        err(f"{n_invalid_ids} matched IDs are not present in test_source2 or test_source3")
        passed = False
    else:
        ok("All matched IDs exist in test S2/S3 universe")

    # -----------------------------------------------------------------------
    # Check 5: candidate_pairs is a superset of matching_results
    # -----------------------------------------------------------------------
    # Build candidate set: s1_id -> set of candidate ids
    cand_map: dict[str, set[str]] = {}
    for _, row in candidate_df.iterrows():
        s1id = row["source1_entity_id"]
        raw = row["candidate_entity_ids"]
        ids_set: set[str] = set()
        if raw and raw.strip():
            ids_set = {x.strip() for x in raw.split(",") if x.strip()}
        cand_map[s1id] = ids_set

    n_violations = 0
    for _, row in matching_df.iterrows():
        s1id = row["source1_entity_id"]
        raw = row["matched_entity_ids"]
        if not raw or not raw.strip():
            continue
        matched = {x.strip() for x in raw.split(",") if x.strip()}
        candidates = cand_map.get(s1id, set())
        not_in_cands = matched - candidates
        if not_in_cands:
            n_violations += 1

    if n_violations:
        err(f"{n_violations} S1 entities have matched IDs that don't appear in their "
            f"candidate_pairs entry — candidates must be a superset of matches")
        passed = False
    else:
        ok("candidate_pairs is a strict superset of matching_results (consistency check passed)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_matched = (matching_df["matched_entity_ids"].fillna("") != "").sum()
    n_singleton = (matching_df["matched_entity_ids"].fillna("") == "").sum()
    print(f"\nSummary: {n_matched:,} entities with predictions, "
          f"{n_singleton:,} singletons (empty predictions)")
    if passed:
        print("\n[PASS] All checks PASSED -- safe to submit.\n")
    else:
        print("\n[FAIL] Some checks FAILED -- fix before submitting.\n")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Directory containing test_source{1,2,3}.tsv")
    args = parser.parse_args()
    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
