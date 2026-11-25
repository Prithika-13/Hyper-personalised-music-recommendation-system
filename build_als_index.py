"""
STEP 3 — Candidate Generation Service (ANN search over ALS embeddings)

Inputs (from Step 1 & 2):
    train_matrix.npz, idx_to_user.pkl, user_to_idx.pkl,
    idx_to_track.pkl, track_lookup.csv,
    als_model_user_factors.npy, als_model_item_factors.npy

Outputs:
    item_faiss_index.bin           (reusable ANN index, load in future services)
    (printed) sample candidate lists for a few real users
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import load_npz

TOP_N = 20

# ---------------------------------------------------------------------------
# 1. LOAD ARTIFACTS
# ---------------------------------------------------------------------------
user_factors = np.load("als_model_user_factors.npy").astype(np.float32)
item_factors = np.load("als_model_item_factors.npy").astype(np.float32)
train_matrix = load_npz("train_matrix.npz").tocsr()

with open("user_to_idx.pkl", "rb") as f:
    user_to_idx = pickle.load(f)
with open("idx_to_user.pkl", "rb") as f:
    idx_to_user = pickle.load(f)
with open("idx_to_track.pkl", "rb") as f:
    idx_to_track = pickle.load(f)

track_lookup = pd.read_csv("track_lookup.csv").set_index("track_key")

n_users, n_items = user_factors.shape[0], item_factors.shape[0]
print(f"User factors: {user_factors.shape}  Item factors: {item_factors.shape}")

# ---------------------------------------------------------------------------
# 2. BUILD FAISS INDEX (exact inner-product search over item vectors)
#    IndexFlatIP = brute-force exact search using dot product, matching
#    exactly how ALS scores user-item affinity. Swap to IndexHNSWFlat later
#    if the catalog grows large enough that exact search becomes too slow.
# ---------------------------------------------------------------------------
dimension = item_factors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(item_factors)
print(f"FAISS index built: {index.ntotal} items indexed, dim={dimension}")

faiss.write_index(index, "item_faiss_index.bin")
print("Saved item_faiss_index.bin")

# ---------------------------------------------------------------------------
# 3. CANDIDATE GENERATION FUNCTION
# ---------------------------------------------------------------------------
def get_candidates(user_id, top_n=TOP_N, initial_pool=200, max_pool=None):
    """
    Given a raw user_id (the original Last.fm user string), return the
    top_n unseen tracks ranked by predicted affinity.

    The search pool starts at initial_pool and DOUBLES until either top_n
    unseen candidates are found or the whole catalog has been searched.
    This matters for active users: their nearest neighbors in embedding
    space are disproportionately tracks they've ALREADY played (that's
    what the ALS objective optimized for), so a small fixed pool can come
    back entirely filtered out.
    """
    if user_id not in user_to_idx:
        raise ValueError(f"Unknown user_id: {user_id}")

    user_idx = user_to_idx[user_id]
    user_vector = user_factors[user_idx].reshape(1, -1)
    seen_items = set(train_matrix.getrow(user_idx).indices)

    max_pool = max_pool or n_items
    search_pool = min(initial_pool, max_pool)

    while True:
        scores, item_indices = index.search(user_vector, search_pool)
        scores, item_indices = scores[0], item_indices[0]

        results = []
        for score, item_idx in zip(scores, item_indices):
            if item_idx == -1 or item_idx in seen_items:
                continue
            track_key = idx_to_track[item_idx]
            if track_key in track_lookup.index:
                row = track_lookup.loc[track_key]
                artist = row["artist_name"] if not isinstance(row, pd.DataFrame) else row["artist_name"].iloc[0]
                name = row["track_name"] if not isinstance(row, pd.DataFrame) else row["track_name"].iloc[0]
            else:
                artist, name = "?", track_key
            results.append({"track_key": track_key, "artist": artist, "track_name": name, "score": float(score)})
            if len(results) >= top_n:
                return results

        if search_pool >= max_pool:
            # exhausted the whole catalog -- return whatever we found
            return results
        search_pool = min(search_pool * 4, max_pool)

# ---------------------------------------------------------------------------
# 4. TEST ON A FEW REAL USERS
# ---------------------------------------------------------------------------
active_user_indices = np.argsort(-train_matrix.getnnz(axis=1))[:3]  # 3 most active users
sample_user_ids = [idx_to_user[i] for i in active_user_indices]

for user_id in sample_user_ids:
    n_seen = train_matrix.getrow(user_to_idx[user_id]).nnz
    print(f"\n=== Candidates for {user_id}  ({n_seen} tracks already in train history) ===")
    candidates = get_candidates(user_id, top_n=10)
    for rank, c in enumerate(candidates, start=1):
        print(f"  {rank:>2}. {c['artist']} - {c['track_name']}  (score={c['score']:.3f})")

print("\nStep 3 complete. get_candidates(user_id) is now your reusable retrieval")
print("function -- Step 4 (RAG layer) and Step 5 (LLM re-ranking) will call this")
print("to get a candidate pool, not search the full catalog themselves.")