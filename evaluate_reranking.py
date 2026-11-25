"""
STEP 7 — Offline Evaluation: LLM Re-Ranking vs Raw ALS Order

Compares Hit@10 / MRR@10 between the raw ALS candidate ordering and the
LLM-reranked ordering, on the SAME candidate pool, restricted to test users
whose held-out item actually appears in that pool (a fair-comparison
constraint -- the LLM can't recover items ALS never retrieved).

Requires Ollama running locally with a pulled model (same as Step 6).

Inputs:
    train_matrix.npz, test_holdout.pkl, user_to_idx.pkl, idx_to_user.pkl,
    idx_to_track.pkl, track_lookup.csv                       (Step 1)
    als_model_user_factors.npy, item_faiss_index.bin         (Step 2/3)
    track_text_lookup.csv                                    (Step 5, optional)

Output: printed comparison table, skip/coverage stats
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import pickle
import random
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import load_npz
import ollama

OLLAMA_MODEL = "llama3.1"
ALS_EVAL_POOL = 50       # larger pool than Step 6's default, gives held-out items more room to appear
SAMPLE_USERS = 20        # how many test users to attempt (many will be skipped for no overlap)
TOP_K = 10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. LOAD ARTIFACTS
# ---------------------------------------------------------------------------
train_matrix = load_npz("train_matrix.npz").tocsr()
user_factors = np.load("als_model_user_factors.npy").astype(np.float32)

with open("test_holdout.pkl", "rb") as f:
    test_holdout = pickle.load(f)  # list of (user_idx, item_idx)
with open("user_to_idx.pkl", "rb") as f:
    user_to_idx = pickle.load(f)
with open("idx_to_user.pkl", "rb") as f:
    idx_to_user = pickle.load(f)
with open("idx_to_track.pkl", "rb") as f:
    idx_to_track = pickle.load(f)

track_lookup = pd.read_csv("track_lookup.csv").set_index("track_key")
als_index = faiss.read_index("item_faiss_index.bin")

try:
    track_text_lookup = pd.read_csv("track_text_lookup.csv").set_index("track_key")
except FileNotFoundError:
    track_text_lookup = pd.DataFrame(columns=["combined_text"])

print("Artifacts loaded.")

# ---------------------------------------------------------------------------
# 2. BUILD user_idx -> set(held_out_item_idx) LOOKUP
# ---------------------------------------------------------------------------
holdout_by_user = {}
for user_idx, item_idx in test_holdout:
    # test_holdout entries can come back as numpy.float64 from the Step 1
    # pandas .map().values call -- cast to plain int here so every downstream
    # use (dict keys, array indexing, set membership vs FAISS int results)
    # is consistent and numpy's strict fancy-indexing doesn't reject them.
    user_idx = int(user_idx)
    item_idx = int(item_idx)
    holdout_by_user.setdefault(user_idx, set()).add(item_idx)

print(f"Sample test_holdout entry type check: {type(test_holdout[0][0])} -> cast to int")

print(f"Test users with at least one held-out item: {len(holdout_by_user)}")

# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------
def track_display_name(track_key):
    if track_key in track_lookup.index:
        row = track_lookup.loc[track_key]
        return row["artist_name"], row["track_name"]
    return "?", track_key

def get_tags(track_key):
    if track_key in track_text_lookup.index:
        return track_text_lookup.loc[track_key, "combined_text"]
    return None

def get_als_pool(user_idx, pool_size=ALS_EVAL_POOL):
    """Returns item indices in similarity-sorted order (excludes train-seen items)."""
    user_idx = int(user_idx)  # defensive cast, see note above
    user_vector = user_factors[user_idx].reshape(1, -1)
    seen_items = set(train_matrix.getrow(user_idx).indices)
    n_items = train_matrix.shape[1]
    search_size = pool_size
    while True:
        scores, item_indices = als_index.search(user_vector, search_size)
        item_indices = item_indices[0]
        ordered = [idx for idx in item_indices if idx != -1 and idx not in seen_items]
        if len(ordered) >= pool_size or search_size >= n_items:
            return ordered[:pool_size]
        search_size = min(search_size * 4, n_items)

def llm_rerank_pool(pool_item_indices, top_n=TOP_K):
    """Re-ranks a GIVEN pool (list of item indices) via the local LLM.
    Returns an ordered list of item indices (subset of the pool)."""
    candidate_lines = []
    key_by_line = {}
    for item_idx in pool_item_indices:
        track_key = idx_to_track[item_idx]
        artist, name = track_display_name(track_key)
        tags = get_tags(track_key) or "no tags available"
        candidate_lines.append(f"- track_key: {track_key} | {artist} - {name} | tags: {tags}")
        key_by_line[track_key] = item_idx

    prompt = f"""Select and rank the best {top_n} tracks from this candidate list for a general
listener with eclectic taste. You MUST only use track_key values that appear EXACTLY in
the list below -- never invent one.

Candidates:
{chr(10).join(candidate_lines)}

Respond with ONLY valid JSON, no other text:
{{"recommendations": [{{"track_key": "<exact match>", "rank": 1}}, ...]}}
Return exactly {top_n} entries, ranked 1 to {top_n}."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.3},
    )
    try:
        parsed = json.loads(response["message"]["content"])
        ranked = sorted(parsed["recommendations"], key=lambda x: x.get("rank", 999))
        result_indices = []
        for r in ranked:
            tk = r.get("track_key")
            if tk in key_by_line:
                result_indices.append(key_by_line[tk])
        return result_indices
    except (json.JSONDecodeError, KeyError, TypeError):
        return []  # treat parse failures as "no valid ranking produced"

def reciprocal_rank(item_idx, ordered_list, top_k=TOP_K):
    if item_idx in ordered_list[:top_k]:
        return 1.0 / (ordered_list.index(item_idx) + 1)
    return 0.0

# ---------------------------------------------------------------------------
# 4. SAMPLE USERS, FILTER TO THOSE WITH POOL OVERLAP, EVALUATE
# ---------------------------------------------------------------------------
candidate_user_indices = list(holdout_by_user.keys())
random.shuffle(candidate_user_indices)
sample = candidate_user_indices[:SAMPLE_USERS]

raw_hits, llm_hits = [], []
raw_rrs, llm_rrs = [], []
n_skipped_no_overlap = 0
n_evaluated = 0

for user_idx in sample:
    held_out_items = holdout_by_user[user_idx]
    pool = get_als_pool(user_idx)

    overlap = held_out_items.intersection(set(pool))
    if not overlap:
        n_skipped_no_overlap += 1
        continue

    target_item = next(iter(overlap))  # if multiple, just evaluate against one
    n_evaluated += 1

    raw_hits.append(target_item in pool[:TOP_K])
    raw_rrs.append(reciprocal_rank(target_item, pool))

    llm_ordered = llm_rerank_pool(pool, top_n=TOP_K)
    llm_hits.append(target_item in llm_ordered[:TOP_K])
    llm_rrs.append(reciprocal_rank(target_item, llm_ordered))

    print(f"  user {idx_to_user[user_idx]}: raw_rank="
          f"{pool.index(target_item)+1 if target_item in pool else 'NA'}, "
          f"llm_rank={llm_ordered.index(target_item)+1 if target_item in llm_ordered else 'not in top 10'}")

# ---------------------------------------------------------------------------
# 5. RESULTS
# ---------------------------------------------------------------------------
print(f"\nSampled users: {len(sample)}")
print(f"Skipped (held-out item not in {ALS_EVAL_POOL}-item pool): {n_skipped_no_overlap}")
print(f"Evaluated (fair comparison possible): {n_evaluated}")

if n_evaluated > 0:
    print(f"\n{'Method':<12} {'Hit@10':<10} {'MRR@10':<10}")
    print(f"{'Raw ALS':<12} {np.mean(raw_hits):<10.4f} {np.mean(raw_rrs):<10.4f}")
    print(f"{'LLM rerank':<12} {np.mean(llm_hits):<10.4f} {np.mean(llm_rrs):<10.4f}")
else:
    print("\nNo evaluable users -- try increasing ALS_EVAL_POOL or SAMPLE_USERS.")

print("\nStep 7 complete. This tells you whether LLM re-ranking is adding real")
print("value over raw ALS ordering, on the cases where a fair comparison is possible.")