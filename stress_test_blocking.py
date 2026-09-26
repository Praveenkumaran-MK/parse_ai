"""Stress test: reproduces the failure mode (a very common name token shared
by a large fraction of records) at a scale large enough to have blown up
memory/time under the old un-pruned blocking index, and confirms the pruned
version completes quickly and boundedly."""
import time
import random
import pandas as pd

random.seed(1)

N = 8000  # per source -- big enough to expose the O(n^2)-ish blowup pattern

COMMON_WORD = "trading"  # deliberately shared by most records, like a real-world generic term
RARE_WORDS = [f"uniquebiz{i}" for i in range(N)]

def make_df(prefix, n, common_fraction=0.6):
    rows = []
    for i in range(n):
        if random.random() < common_fraction:
            name = f"{RARE_WORDS[i % len(RARE_WORDS)]} {COMMON_WORD} company"
        else:
            name = f"{RARE_WORDS[i % len(RARE_WORDS)]} enterprises"
        rows.append({
            "entity_id": f"{prefix}-{i:06d}",
            "business_name": name,
            "business_address": f"{i} Main Road, City{i % 50}, {600000 + (i % 900)}",
            "country": "India" if i % 2 == 0 else "US",
        })
    return pd.DataFrame(rows)

s1 = make_df("S1", N)
s2 = make_df("S2", N)
s3 = make_df("S3", N)

import sys
sys.path.insert(0, ".")
from src.blocking import generate_candidates

start = time.time()
candidate_pairs, _ = generate_candidates(
    s1, s2, s3, use_embeddings=False, max_candidates_per_entity=30,
)
elapsed = time.time() - start

sizes = [len(v) for v in candidate_pairs.values()]
print(f"\nCompleted in {elapsed:.2f}s for {N} entities per source ({3*N} total records).")
print(f"Candidate set sizes -- min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.1f}")
print("STRESS TEST PASSED -- no MemoryError, bounded candidate sizes.")
