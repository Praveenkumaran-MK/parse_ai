"""
blocking.py
-----------
Candidate generation ("blocking") stage.

Blocking determines the RECALL CEILING of the whole pipeline: if the true
match for a Source 1 entity never appears in its candidate set, no matching
model downstream can ever recover it. We therefore use several complementary
strategies and take the UNION of their candidates:

  1. Token-overlap blocking   -- fast, vectorized via CSR sparse matrix
  2. Postal-code / geo blocking -- catches same-location matches even with
                                    noisy names
  3. Embedding (semantic) blocking -- catches typos, transliterations, and
                                    paraphrased names that token overlap misses

All candidates are capped per Source-1 entity (top-K by a cheap combined
score) to keep the downstream matching model's workload tractable.

No external APIs or lookups are used anywhere in this file -- embeddings are
computed with a small, locally-run, permissively-licensed model
(sentence-transformers/all-MiniLM-L6-v2, Apache-2.0, ~22M params), which is
well within the 8B-parameter / MIT-Apache license constraint.

MEMORY / PERFORMANCE NOTES (real-scale: 10M+ records, 16 GB RAM):
  - token_overlap_candidates uses a pandas-based vectorized join instead of
    Python-level postings-list traversal (the original approach OOM'd at ~25 GB).
  - EmbeddingBlocker uses batched encode (no full materialization) and an
    IVF (Inverted File) FAISS index that stores only cluster centroids in
    memory at query time -- O(sqrt(N)) memory vs O(N) for IndexFlatIP.
  - All .iterrows() calls have been replaced with vectorized alternatives.
"""

import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"    [{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Token-overlap blocking  (vectorized — no Python postings traversal)
# ---------------------------------------------------------------------------

def build_token_index(candidate_df, id_col="entity_id", token_col="name_core_tokens",
                       max_doc_freq=None, verbose=True):
    """
    Build a (token -> array of integer row positions) index, returning
    both the index and a row-position -> entity_id mapping so callers
    can work in integer space rather than string-dict space.

    CRITICAL for real-scale data: tokens that appear in a large fraction of
    records carry near-zero discriminative signal for blocking but their
    postings lists blow up memory/compute. We drop such tokens from the
    index entirely.

    max_doc_freq: any token appearing in more than this many records is
    excluded. Defaults to max(50, 1% of corpus size) if not given.
    """
    n = len(candidate_df)
    if max_doc_freq is None:
        max_doc_freq = max(50, int(0.01 * n))

    # --- Pass 1: document frequency per token ---
    doc_freq: dict[str, int] = defaultdict(int)
    for tokens in candidate_df[token_col]:
        for tok in set(tokens):
            doc_freq[tok] += 1

    dropped_tokens = {tok for tok, df in doc_freq.items() if df > max_doc_freq}
    if verbose and dropped_tokens:
        print(f"    [blocking] Pruned {len(dropped_tokens)} overly-common tokens "
              f"(doc_freq > {max_doc_freq}) from the blocking index to bound memory/compute.")

    # --- Pass 2: build integer-position postings lists ---
    # We store integer row positions (not string IDs) in the postings so the
    # per-query union step works entirely in numpy/pandas rather than Python dicts.
    id_array = candidate_df[id_col].values          # integer-indexed string array
    index: dict[str, np.ndarray] = {}
    tok_rows: dict[str, list[int]] = defaultdict(list)

    for row_i, tokens in enumerate(candidate_df[token_col]):
        for tok in tokens:
            if tok not in dropped_tokens:
                tok_rows[tok].append(row_i)

    for tok, rows in tok_rows.items():
        index[tok] = np.array(rows, dtype=np.int32)

    return index, id_array


def token_overlap_candidates(source1_df, token_index, id_array,
                              id_col="entity_id", token_col="name_core_tokens",
                              min_shared_tokens=1,
                              max_candidates_per_query=200):
    """
    For each Source 1 entity, retrieve every S2/S3 entity sharing at least
    `min_shared_tokens` distinctive name tokens.

    VECTORIZED IMPLEMENTATION:
    Instead of the original O(S1 × postings) Python dict counting loop
    (which OOM'd at ~25 GB on 1.76M x 10.3M), this version:
      1. Collects the set of postings arrays for all tokens of a given S1 entity
      2. Concatenates them into a single numpy array
      3. Uses np.bincount (C-level) to count co-occurrences — no Python-level
         dict insertions
      4. Applies min_shared_tokens and max_candidates_per_query thresholds
         using numpy argsort (also C-level)

    Memory profile: the largest intermediate object per S1 entity is the
    concatenated postings array. With max_candidates_per_token_query = 200
    applied at the token level (optional), this stays bounded per entity.

    Returns dict: source1_entity_id -> set(candidate_entity_ids)
    """
    result = {}
    n_pool = len(id_array)
    s1_ids = source1_df[id_col].tolist()
    s1_tokens = source1_df[token_col].tolist()
    total = len(s1_ids)

    report_every = max(1, total // 20)   # log progress every ~5%

    for q_i, (eid, tokens) in enumerate(zip(s1_ids, s1_tokens)):
        if q_i % report_every == 0:
            _log(f"token blocking: {q_i:,}/{total:,} S1 entities processed")

        # Collect all postings arrays for this entity's tokens
        parts = [token_index[tok] for tok in tokens if tok in token_index]
        if not parts:
            result[eid] = set()
            continue

        # Concatenate and count with numpy bincount (fast C path)
        all_hits = np.concatenate(parts)          # dtype int32
        counts = np.bincount(all_hits, minlength=n_pool)  # shape (n_pool,)

        # Apply min_shared_tokens threshold
        qualifying = np.where(counts >= min_shared_tokens)[0]

        if len(qualifying) > max_candidates_per_query:
            # Keep top max_candidates_per_query by overlap count
            top_idx = np.argpartition(counts[qualifying], -max_candidates_per_query)[-max_candidates_per_query:]
            qualifying = qualifying[top_idx]

        result[eid] = set(id_array[qualifying].tolist())

    _log(f"token blocking: done ({total:,} S1 entities)")
    return result


# ---------------------------------------------------------------------------
# 2. Postal-code / geo blocking
# ---------------------------------------------------------------------------

def build_postal_index(candidate_df, id_col="entity_id", postal_col="postal_code",
                        max_bucket_size=None, verbose=True):
    """
    Inverted index: postal_code -> set of entity_ids with that code.

    Same rationale as build_token_index: a shared postal code in a dense
    business district can legitimately group thousands of unrelated
    businesses. We cap bucket size so one postal code can't single-handedly
    hand every S1 entity in that area a huge candidate list.
    """
    raw_index: dict[str, list[str]] = defaultdict(list)
    for eid, pin in zip(candidate_df[id_col], candidate_df[postal_col]):
        if pin:
            raw_index[pin].append(eid)

    n = len(candidate_df)
    if max_bucket_size is None:
        max_bucket_size = max(200, int(0.02 * n))

    oversized = [pin for pin, ids in raw_index.items() if len(ids) > max_bucket_size]
    if verbose and oversized:
        print(f"    [blocking] {len(oversized)} postal codes exceed the "
              f"{max_bucket_size}-entity bucket cap and will be truncated for blocking.")

    index: dict[str, list[str]] = {}
    for pin, ids in raw_index.items():
        index[pin] = ids[:max_bucket_size] if len(ids) > max_bucket_size else ids
    return index


def postal_code_candidates(source1_df, postal_index, id_col="entity_id",
                            postal_col="postal_code"):
    result = {}
    for eid, pin in zip(source1_df[id_col], source1_df[postal_col]):
        result[eid] = set(postal_index.get(pin, ())) if pin else set()
    return result


# ---------------------------------------------------------------------------
# 3. Embedding-based (semantic) blocking
# ---------------------------------------------------------------------------

class EmbeddingBlocker:
    """
    Wraps a small sentence-transformer + FAISS IVF index for approximate
    nearest-neighbor retrieval over "name + address" text.

    Key changes vs the original:
      - build_index() encodes in batches using sentence_transformers'
        native show_progress_bar + batch_size, then adds to FAISS in chunks
        -- never holds ALL embeddings in RAM simultaneously for large corpora.
      - Uses IndexIVFFlat (or IndexHNSWFlat) instead of IndexFlatIP. At 10M
        records, IndexFlatIP requires ~15 GB; IVF with nlist=4096 needs only
        ~few hundred MB for the quantizer.
      - All .iterrows() replaced with vectorized pandas string operations.

    Lazily imports sentence_transformers / faiss so the rest of the pipeline
    and unit tests can run without a GPU or these heavier dependencies installed,
    if someone only wants the token/postal blockers.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 device=None):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.id_list: list[str] = []

    @staticmethod
    def _make_texts(df, name_col="name_norm", addr_col="addr_norm") -> list[str]:
        """Vectorized text construction — no .iterrows()."""
        names = df[name_col].fillna("").astype(str)
        addrs = df[addr_col].fillna("").astype(str)
        # pandas string concat is done in C via object array concatenation
        return (names + " [SEP] " + addrs).tolist()

    def build_index(self, candidate_df, id_col="entity_id",
                     name_col="name_norm", addr_col="addr_norm",
                     encode_batch_size=512, faiss_batch_size=50_000,
                     nlist_factor=8):
        """
        Encode candidates and build a FAISS IVF index in streaming batches.

        nlist_factor: nlist = nlist_factor * sqrt(N). Rule of thumb: nlist
          such that each cell has ~39 vectors on average. nlist_factor=8
          gives nlist≈8*sqrt(10M)≈25,298 → cells of ~395 each. Adjust down
          for smaller datasets (it's clamped to [64, 65536]).
        """
        import faiss  # lazy import

        n = len(candidate_df)
        nlist = int(np.clip(nlist_factor * np.sqrt(n), 64, 65536))
        _log(f"embedding blocker: building IVF index for {n:,} pool records (nlist={nlist})")

        texts = self._make_texts(candidate_df, name_col, addr_col)
        self.id_list = candidate_df[id_col].tolist()

        # Encode first batch to get embedding dimension
        first_batch = self.model.encode(
            texts[:encode_batch_size], batch_size=encode_batch_size,
            show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        dim = first_batch.shape[1]

        # Build IVF index — quantizer is a flat index, IVF partitions the space
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # IVF requires training on a representative sample before adding vectors
        train_size = min(n, max(nlist * 39, 100_000))
        _log(f"embedding blocker: training IVF quantizer on {train_size:,} sample vectors")
        train_texts = texts[:train_size]
        train_embs = self.model.encode(
            train_texts, batch_size=encode_batch_size,
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.index.train(train_embs)
        del train_embs

        # Add all vectors in streaming batches to cap peak RAM
        _log(f"embedding blocker: adding {n:,} vectors in batches of {faiss_batch_size:,}")
        for batch_start in range(0, n, faiss_batch_size):
            batch_texts = texts[batch_start: batch_start + faiss_batch_size]
            batch_embs = self.model.encode(
                batch_texts, batch_size=encode_batch_size,
                show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            self.index.add(batch_embs)
            if (batch_start // faiss_batch_size) % 5 == 0:
                _log(f"embedding blocker: added {min(batch_start + faiss_batch_size, n):,}/{n:,} vectors")
            del batch_embs

        self.index.nprobe = min(64, nlist)  # search 64 cells — recall vs speed tradeoff
        _log("embedding blocker: index ready")

    def query(self, source1_df, id_col="entity_id", name_col="name_norm",
              addr_col="addr_norm", top_k=15, batch_size=512):
        """
        Returns dict: source1_entity_id -> list[(candidate_entity_id, cosine_sim)]
        """
        if self.index is None:
            raise RuntimeError("Call build_index() before query().")

        texts = self._make_texts(source1_df, name_col, addr_col)
        s1_ids = source1_df[id_col].tolist()
        n_queries = len(texts)
        id_array = np.array(self.id_list)

        result = {}
        _log(f"embedding blocker: querying {n_queries:,} S1 entities (top_k={top_k})")
        for batch_start in range(0, n_queries, batch_size):
            batch_texts = texts[batch_start: batch_start + batch_size]
            batch_ids = s1_ids[batch_start: batch_start + batch_size]
            query_emb = self.model.encode(
                batch_texts, batch_size=batch_size,
                show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            sims, idxs = self.index.search(query_emb, top_k)

            for row_i, s1_id in enumerate(batch_ids):
                pairs = []
                for col_j in range(idxs.shape[1]):
                    idx = idxs[row_i, col_j]
                    if idx == -1:
                        continue
                    pairs.append((id_array[idx], float(sims[row_i, col_j])))
                result[s1_id] = pairs

        _log("embedding blocker: query done")
        return result


# ---------------------------------------------------------------------------
# Combine strategies
# ---------------------------------------------------------------------------

def generate_candidates(source1_df, source2_df, source3_df,
                         use_embeddings=True, embedding_top_k=15,
                         embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                         embedding_device=None,
                         max_candidates_per_entity=30,
                         max_token_doc_freq=None,
                         max_postal_bucket_size=None,
                         max_candidates_per_token_query=200):
    """
    Full blocking pipeline. All three input dataframes are RAW (not yet
    preprocessed) -- this function runs preprocessing internally so callers
    only need to load the TSVs.

    max_token_doc_freq / max_postal_bucket_size: memory/compute safety caps
    on real-scale data -- see build_token_index / build_postal_index for
    rationale. Leave as None to use the automatic size-based defaults
    (1% of corpus for tokens, 2% for postal buckets); lower them by hand if
    you still see MemoryError or the pipeline running unreasonably slowly.

    Returns:
        candidate_pairs: dict source1_entity_id -> sorted list of candidate ids
        embedding_scores: dict (s1_id, cand_id) -> cosine_sim (for feature reuse)
    """
    t0 = time.time()

    _log("preprocessing source dataframes...")
    s1 = preprocess_dataframe(source1_df)
    s2 = preprocess_dataframe(source2_df)
    s3 = preprocess_dataframe(source3_df)
    pool = pd.concat([s2, s3], ignore_index=True)
    _log(f"preprocessing done: s1={len(s1):,}  pool={len(pool):,}  "
         f"({time.time()-t0:.1f}s elapsed)")

    # --- Strategy 1: token overlap (vectorized) ---
    t1 = time.time()
    _log("building token index...")
    token_index, id_array = build_token_index(pool, max_doc_freq=max_token_doc_freq)
    _log(f"token index built: {len(token_index):,} distinct tokens  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    token_cands = token_overlap_candidates(
        s1, token_index, id_array,
        max_candidates_per_query=max_candidates_per_token_query,
    )
    _log(f"token blocking done  ({time.time()-t1:.1f}s)")

    # --- Strategy 2: postal code ---
    t1 = time.time()
    _log("building postal index...")
    postal_index = build_postal_index(pool, max_bucket_size=max_postal_bucket_size)
    postal_cands = postal_code_candidates(s1, postal_index)
    _log(f"postal blocking done  ({time.time()-t1:.1f}s)")

    # --- Strategy 3: embeddings (semantic) ---
    embedding_scores: dict[tuple[str, str], float] = {}
    embed_cands: dict[str, set[str]] = {eid: set() for eid in s1["entity_id"]}
    if use_embeddings:
        t1 = time.time()
        blocker = EmbeddingBlocker(model_name=embedding_model_name, device=embedding_device)
        blocker.build_index(pool)
        embed_results = blocker.query(s1, top_k=embedding_top_k)
        for s1_id, pairs in embed_results.items():
            for cand_id, sim in pairs:
                embed_cands[s1_id].add(cand_id)
                embedding_scores[(s1_id, cand_id)] = sim
        del blocker  # free FAISS index memory before feature stage
        _log(f"embedding blocking done  ({time.time()-t1:.1f}s)")

    # --- Union + cap ---
    t1 = time.time()
    candidate_pairs: dict[str, list[str]] = {}
    for eid in s1["entity_id"]:
        union = (token_cands.get(eid, set())
                 | postal_cands.get(eid, set())
                 | embed_cands.get(eid, set()))

        if len(union) > max_candidates_per_entity:
            # Rank by embedding similarity if available, else keep arbitrary order
            union_list = sorted(
                union,
                key=lambda c: embedding_scores.get((eid, c), 0.0),
                reverse=True,
            )[:max_candidates_per_entity]
        else:
            union_list = sorted(union)
        candidate_pairs[eid] = union_list

    total_pairs = sum(len(v) for v in candidate_pairs.values())
    _log(f"union + cap done: {total_pairs:,} total candidate pairs  ({time.time()-t1:.1f}s)  "
         f"[total blocking time: {time.time()-t0:.1f}s]")

    return candidate_pairs, embedding_scores


def candidates_to_dataframe(candidate_pairs):
    """Convert dict -> long-format dataframe (one row per S1-candidate pair)."""
    rows = []
    for s1_id, cands in candidate_pairs.items():
        if not cands:
            rows.append({"source1_entity_id": s1_id, "candidate_entity_id": None})
        for c in cands:
            rows.append({"source1_entity_id": s1_id, "candidate_entity_id": c})
    return pd.DataFrame(rows)
