# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline: multi-strategy blocking (candidate generation) →
pairwise feature engineering → LightGBM matching classifier → F_0.5-tuned
decision threshold → submission files.

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

GPU note: if you have CUDA available, `sentence-transformers` will use it
automatically; pass `--device cuda` to `train.py` / `inference.py` to be
explicit, or `--device cpu` to force CPU (useful for a quick low-recall
first pass while you iterate on other parts of the pipeline).

## 2. Directory layout expected

```
business_entity_resolution/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── src/            (this pipeline)
├── output/         (created automatically)
└── artifacts/      (created automatically — trained model + vectorizers)
```

## 3. Train

```bash
python -m src.train --data-dir dataset/train --output-dir artifacts
```

This will:
1. Load the training sources + ground truth
2. Grouped-split Source-1 entities into train/validation (stratified by
   "has any true match" so singletons are represented in both splits)
3. Run blocking on the train split and **print blocking recall** — check
   this number first. If it's below ~0.85, increase `--max-candidates` or
   `embedding_top_k` in `src/blocking.py` before trusting anything downstream.
4. Build pairwise features + train a LightGBM classifier with grouped 5-fold CV
5. Score the validation split and **sweep decision thresholds** to maximize
   the exact competition F_0.5 macro metric (not accuracy, not F1 — F_0.5
   at 0.5 threshold is a red herring for this task)
6. Save `matching_model.joblib`, `name_vectorizer.joblib`,
   `addr_vectorizer.joblib`, and `config.json` (chosen threshold + metrics)
   to `artifacts/`

Useful flags:
- `--no-embeddings` — skip the sentence-transformer/FAISS blocking stage
  (faster iteration, lower recall; good for a first smoke-test pass)
- `--max-candidates N` — cap candidates per Source-1 entity (default 30)
- `--val-fraction 0.2` — fraction of Source-1 entities held out for
  threshold tuning
- `--max-token-doc-freq N` — drop name tokens appearing in more than N
  records from the blocking index. Defaults to `max(50, 1% of corpus)`.
  **Lower this if you hit `MemoryError` during blocking** — it means a
  common word in your business names (not caught by the static suffix
  list) is exploding the candidate-counting step. See "Known failure
  mode" below.
- `--max-postal-bucket-size N` — same idea for postal codes shared by a
  large number of businesses (e.g. a dense business district).
- `--max-candidates-per-token-query N` — hard safety cap per entity
  regardless of pruning (default 200).

## 4. Run inference on the test set

```bash
python -m src.inference --data-dir dataset/test --artifacts-dir artifacts --output-dir output
```

Produces:
- `output/matching_results.tsv` — **upload this to the leaderboard**
- `output/candidate_pairs.tsv` — your blocking stage's final candidate set

The script runs a self-check confirming every predicted match is a subset
of that entity's candidate set (a pipeline-consistency requirement the
organizers' validator also checks) before you submit.

**Always run the organizers' validator before uploading:**
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## 5. Smoke-testing without real data

`make_synthetic_data.py` generates a small synthetic dataset matching the
exact schema (for verifying your environment/pipeline runs end-to-end
before touching real data). It is NOT representative of real-world
matching difficulty — do not use its metrics for anything except "did the
code crash."

```bash
python3 make_synthetic_data.py
python -m src.train --data-dir dataset/train --output-dir artifacts --no-embeddings
python -m src.inference --data-dir dataset/test --artifacts-dir artifacts --output-dir output --no-embeddings
```

## 6. Known failure mode: MemoryError during blocking

If `train.py` crashes with `MemoryError` inside `token_overlap_candidates`,
it means a business-name word that isn't in the static legal-suffix list
(e.g. "trading", "solutions", "exports") is common enough across your real
dataset to blow up the candidate-counting step for every entity containing
it — a token index will happily build a postings list of thousands of
records for a word that carries zero actual matching signal.

The pipeline already prunes tokens above a document-frequency threshold
(`build_token_index` in `src/blocking.py`, default `max(50, 1% of corpus)`)
and caps postal-code bucket sizes the same way, plus a hard per-entity
safety cap (`max_candidates_per_token_query`, default 200). If you still
hit it on a very large dataset, pass a lower `--max-token-doc-freq` (e.g.
20) or `--max-postal-bucket-size` explicitly. `stress_test_blocking.py`
reproduces this failure mode synthetically if you want to confirm your
tuning holds before running against the real data again.

## 7. Where to focus your time (in priority order)

1. **Blocking recall** — this is your ceiling. Look at `src/blocking.py`.
   If real data reveals noise patterns the static abbreviation maps in
   `src/preprocessing.py` don't cover, add them there first — it's the
   cheapest lever available.
2. **Threshold, not model complexity** — F_0.5 punishes false positives
   2× harder than false negatives. Re-run `tune_threshold` after any
   feature/model change; don't assume 0.5 or your last threshold still holds.
2. **Country generalization** — the test set adds France, unseen in
   training. `extract_postal_code` already falls back gracefully for
   unknown countries, but check France address patterns once test data is
   available and extend `ADDRESS_ABBREV_MAP` / postal regex if needed.
3. **Singleton precision** — a false match on a true singleton costs you
   1.0 on that entity's score. If validation shows singleton accuracy
   dropping, raise the threshold or add a "no confident candidate above
   margin X" abstention rule.
4. **Ensembling (time permitting)** — blend the LightGBM score with a
   small cross-encoder re-ranking model on the top-K candidates per entity
   for a further precision boost.

## Fair-play compliance

- No external APIs, geocoding services, or business-registry lookups
  anywhere in this codebase — confirm this stays true if you extend it.
- Default embedding model (`all-MiniLM-L6-v2`) is Apache-2.0, ~22M
  parameters — comfortably within the MIT/Apache + ≤8B constraint. If you
  swap in a different model, verify its license and parameter count before
  using it in your final submission.
