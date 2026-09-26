"""
features.py
-----------
Pairwise similarity feature engineering.

For every (source1_entity, candidate_entity) pair we compute a rich set of
string-similarity, structural, and (optionally) embedding-similarity
features. These feed the matching classifier in matching_model.py.

Uses `rapidfuzz` for fast Levenshtein / Jaro-Winkler / token-sort ratios
(pure-Python + C++ backend, MIT licensed, no external calls).
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_dataframe


FEATURE_COLUMNS = [
    "name_levenshtein_ratio",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard_tokens",
    "name_tfidf_cosine",
    "addr_levenshtein_ratio",
    "addr_jaro_winkler",
    "addr_token_sort_ratio",
    "addr_jaccard_tokens",
    "addr_tfidf_cosine",          # was computed but missing from this list — fixed
    "postal_code_match",
    "postal_code_both_present",
    "country_match",
    "name_len_diff_ratio",
    "addr_len_diff_ratio",
    "embedding_cosine_sim",
]


def _jaccard(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def _safe_len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if max(la, lb) == 0:
        return 1.0
    return 1.0 - (abs(la - lb) / max(la, lb))


def build_tfidf_vectorizer(*dataframes, text_col="name_norm", ngram_range=(1, 2)):
    """
    Fit a single shared char/word TF-IDF vectorizer across all provided
    dataframes' text column, so cosine similarity is comparable across
    all sources. Character n-grams handle typos better than pure word n-grams.
    """
    corpus = pd.concat([df[text_col] for df in dataframes], ignore_index=True)
    corpus = corpus.fillna("")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=1
    )
    vectorizer.fit(corpus)
    return vectorizer


def compute_pair_features(s1_row, cand_row, name_vectorizer, addr_vectorizer,
                           embedding_sim=0.0):
    """Compute the full feature vector for a single (S1, candidate) pair."""
    name_a, name_b = s1_row["name_norm"], cand_row["name_norm"]
    addr_a, addr_b = s1_row["addr_norm"], cand_row["addr_norm"]

    name_tfidf_cos = cosine_similarity(
        name_vectorizer.transform([name_a]), name_vectorizer.transform([name_b])
    )[0, 0]
    # (Note: for large datasets, precompute all TF-IDF vectors once and index
    # into them rather than re-transforming per pair -- see
    # `compute_features_batch` below, which does exactly that for speed.)

    postal_a, postal_b = s1_row["postal_code"], cand_row["postal_code"]

    features = {
        "name_levenshtein_ratio": fuzz.ratio(name_a, name_b) / 100.0,
        "name_jaro_winkler": JaroWinkler.similarity(name_a, name_b),
        "name_token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(name_a, name_b) / 100.0,
        "name_jaccard_tokens": _jaccard(s1_row["name_core_tokens"], cand_row["name_core_tokens"]),
        "name_tfidf_cosine": float(name_tfidf_cos),
        "addr_levenshtein_ratio": fuzz.ratio(addr_a, addr_b) / 100.0,
        "addr_jaro_winkler": JaroWinkler.similarity(addr_a, addr_b),
        "addr_token_sort_ratio": fuzz.token_sort_ratio(addr_a, addr_b) / 100.0,
        "addr_jaccard_tokens": _jaccard(set(addr_a.split()), set(addr_b.split())),
        "postal_code_match": float(bool(postal_a) and bool(postal_b) and postal_a == postal_b),
        "postal_code_both_present": float(bool(postal_a) and bool(postal_b)),
        "country_match": float(s1_row["country"] == cand_row["country"]),
        "name_len_diff_ratio": _safe_len_ratio(name_a, name_b),
        "addr_len_diff_ratio": _safe_len_ratio(addr_a, addr_b),
        "embedding_cosine_sim": float(embedding_sim),
    }
    return features


def compute_features_batch(source1_df, pool_df, pair_list, embedding_scores=None,
                            name_vectorizer=None, addr_vectorizer=None):
    """
    Vectorized-ish feature computation for a list of (s1_id, cand_id) pairs.

    Args:
        source1_df, pool_df: preprocessed dataframes (see preprocessing.preprocess_dataframe)
                              pool_df = concat of source2 + source3, preprocessed
        pair_list: list of (s1_entity_id, candidate_entity_id) tuples
        embedding_scores: dict (s1_id, cand_id) -> cosine_sim, from blocking stage
        name_vectorizer, addr_vectorizer: fitted TfidfVectorizer; fit fresh if None

    Returns:
        pd.DataFrame with columns ["source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS]
    """
    embedding_scores = embedding_scores or {}

    s1_idx = source1_df.set_index("entity_id")
    pool_idx = pool_df.set_index("entity_id")

    if name_vectorizer is None:
        name_vectorizer = build_tfidf_vectorizer(source1_df, pool_df, text_col="name_norm")
    if addr_vectorizer is None:
        addr_vectorizer = build_tfidf_vectorizer(source1_df, pool_df, text_col="addr_norm")

    # Precompute TF-IDF matrices once (fast path) instead of per-pair transform.
    name_matrix_s1 = name_vectorizer.transform(s1_idx["name_norm"])
    name_matrix_pool = name_vectorizer.transform(pool_idx["name_norm"])
    addr_matrix_s1 = addr_vectorizer.transform(s1_idx["addr_norm"])
    addr_matrix_pool = addr_vectorizer.transform(pool_idx["addr_norm"])

    s1_pos = {eid: i for i, eid in enumerate(s1_idx.index)}
    pool_pos = {eid: i for i, eid in enumerate(pool_idx.index)}

    rows = []
    for s1_id, cand_id in pair_list:
        if s1_id not in s1_pos or cand_id not in pool_pos:
            continue  # defensive: skip malformed pairs rather than crash a 72hr run

        s1_row = s1_idx.loc[s1_id]
        cand_row = pool_idx.loc[cand_id]

        i, j = s1_pos[s1_id], pool_pos[cand_id]
        name_tfidf_cos = cosine_similarity(name_matrix_s1[i], name_matrix_pool[j])[0, 0]
        addr_tfidf_cos = cosine_similarity(addr_matrix_s1[i], addr_matrix_pool[j])[0, 0]

        name_a, name_b = s1_row["name_norm"], cand_row["name_norm"]
        addr_a, addr_b = s1_row["addr_norm"], cand_row["addr_norm"]
        postal_a, postal_b = s1_row["postal_code"], cand_row["postal_code"]

        feat = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "name_levenshtein_ratio": fuzz.ratio(name_a, name_b) / 100.0,
            "name_jaro_winkler": JaroWinkler.similarity(name_a, name_b),
            "name_token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b) / 100.0,
            "name_token_set_ratio": fuzz.token_set_ratio(name_a, name_b) / 100.0,
            "name_jaccard_tokens": _jaccard(s1_row["name_core_tokens"], cand_row["name_core_tokens"]),
            "name_tfidf_cosine": float(name_tfidf_cos),
            "addr_levenshtein_ratio": fuzz.ratio(addr_a, addr_b) / 100.0,
            "addr_jaro_winkler": JaroWinkler.similarity(addr_a, addr_b),
            "addr_token_sort_ratio": fuzz.token_sort_ratio(addr_a, addr_b) / 100.0,
            "addr_jaccard_tokens": _jaccard(set(addr_a.split()), set(addr_b.split())),
            "addr_tfidf_cosine": float(addr_tfidf_cos),
            "postal_code_match": float(bool(postal_a) and bool(postal_b) and postal_a == postal_b),
            "postal_code_both_present": float(bool(postal_a) and bool(postal_b)),
            "country_match": float(s1_row["country"] == cand_row["country"]),
            "name_len_diff_ratio": _safe_len_ratio(name_a, name_b),
            "addr_len_diff_ratio": _safe_len_ratio(addr_a, addr_b),
            "embedding_cosine_sim": float(embedding_scores.get((s1_id, cand_id), 0.0)),
        }
        rows.append(feat)

    return pd.DataFrame(rows), name_vectorizer, addr_vectorizer
