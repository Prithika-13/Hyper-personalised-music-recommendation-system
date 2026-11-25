"""
STEP 2 — Train ALS Baseline and Evaluate  (fixed version)

Fixes applied vs. the first version:
    1. OPENBLAS_NUM_THREADS=1 set before numpy/implicit import (avoids
       thread contention between OpenBLAS and implicit's own threading,
       which can slow ALS training down significantly).
    2. Sparse matrices cast to the dtypes implicit's Cython code expects
       (int32 indices/indptr, float32 data) -- fixes the
       "Buffer dtype mismatch: expected 'int' but got 'long long'" error
       that scipy's default int64 indices trigger on Windows.

Inputs (from Step 1):
    train_matrix.npz, test_holdout.pkl,
    user_to_idx.pkl, idx_to_user.pkl, track_to_idx.pkl, idx_to_track.pkl,
    track_lookup.csv

Outputs:
    als_model_user_factors.npy
    als_model_item_factors.npy
    (printed) Precision@K, NDCG@K on held-out test set
"""

# ---------------------------------------------------------------------------
# 0. MUST BE SET BEFORE numpy / implicit ARE IMPORTED
# ---------------------------------------------------------------------------
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, ndcg_at_k

K = 10
FACTORS = 64
REGULARIZATION = 0.01
ITERATIONS = 15
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD ARTIFACTS FROM STEP 1
# ---------------------------------------------------------------------------
train_matrix = load_npz("train_matrix.npz")  # users x items, confidence-weighted

with open("test_holdout.pkl", "rb") as f:
    test_holdout = pickle.load(f)  # list of (user_idx, item_idx)
with open("user_to_idx.pkl", "rb") as f:
    user_to_idx = pickle.load(f)
with open("idx_to_user.pkl", "rb") as f:
    idx_to_user = pickle.load(f)
with open("track_to_idx.pkl", "rb") as f:
    track_to_idx = pickle.load(f)
with open("idx_to_track.pkl", "rb") as f:
    idx_to_track = pickle.load(f)

track_lookup = pd.read_csv("track_lookup.csv").set_index("track_key")

n_users, n_items = train_matrix.shape
print(f"Train matrix: {train_matrix.shape}, nnz = {train_matrix.nnz}")

# ---------------------------------------------------------------------------
# 2. REBUILD TEST MATRIX (same shape, binary relevance is fine for eval)
# ---------------------------------------------------------------------------
test_rows = [pair[0] for pair in test_holdout]
test_cols = [pair[1] for pair in test_holdout]
test_vals = np.ones(len(test_holdout))
test_matrix = csr_matrix((test_vals, (test_rows, test_cols)), shape=(n_users, n_items))
print(f"Test matrix: {test_matrix.shape}, nnz = {test_matrix.nnz}")

# ---------------------------------------------------------------------------
# 2b. FIX DTYPES FOR IMPLICIT'S CYTHON CODE
#     implicit expects int32 indices/indptr and float32 data internally;
#     scipy defaults to int64 when building from Python lists on most
#     platforms, causing a "Buffer dtype mismatch" error in the compiled
#     evaluation functions.
# ---------------------------------------------------------------------------
def to_implicit_dtypes(matrix):
    matrix = matrix.tocsr()
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    matrix.data = matrix.data.astype(np.float32)
    return matrix

train_matrix = to_implicit_dtypes(train_matrix)
test_matrix = to_implicit_dtypes(test_matrix)

# ---------------------------------------------------------------------------
# 3. TRAIN ALS
#    Note: implicit's ALS expects a user_items matrix (rows=users) with
#    confidence values already applied -- exactly what train_matrix is.
# ---------------------------------------------------------------------------
model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    random_state=RANDOM_STATE,
    use_gpu=False,
)

print("\nTraining ALS...")
model.fit(train_matrix, show_progress=True)
print("Training complete.")

# ---------------------------------------------------------------------------
# 4. EVALUATE — Precision@K and NDCG@K on held-out test interactions
#    (implicit's eval functions automatically exclude items already seen
#    in train_matrix when generating recommendations for scoring)
# ---------------------------------------------------------------------------
print("\nEvaluating...")
prec = precision_at_k(model, train_matrix, test_matrix, K=K, show_progress=True)
ndcg = ndcg_at_k(model, train_matrix, test_matrix, K=K, show_progress=True)

print(f"\nPrecision@{K}: {prec:.4f}")
print(f"NDCG@{K}: {ndcg:.4f}")

# ---------------------------------------------------------------------------
# 4b. QUICK POPULARITY-BASELINE SANITY CHECK
#     ALS should clearly beat "just recommend whatever's globally popular" --
#     if it doesn't, something upstream (confidence formula, hyperparameters)
#     needs a second look before trusting the ALS numbers.
# ---------------------------------------------------------------------------
popularity = np.asarray(train_matrix.sum(axis=0)).flatten()
top_popular_items = np.argsort(-popularity)[:K]

hits = 0
total = 0
for user_idx in range(n_users):
    true_items = set(test_matrix.getrow(user_idx).indices)
    if not true_items:
        continue
    hits += len(true_items.intersection(top_popular_items))
    total += min(K, len(true_items))

pop_precision = hits / total if total > 0 else 0.0
print(f"Popularity-baseline Precision@{K} (rough): {pop_precision:.4f}")
print("(ALS should score meaningfully above this)")

# ---------------------------------------------------------------------------
# 5. SAVE MODEL FACTORS (for reuse in Phase 3 embedding layer / candidate gen)
# ---------------------------------------------------------------------------
np.save("als_model_user_factors.npy", model.user_factors)
np.save("als_model_item_factors.npy", model.item_factors)
print("\nSaved als_model_user_factors.npy and als_model_item_factors.npy")

# ---------------------------------------------------------------------------
# 6. QUALITATIVE SANITY CHECK — pick a real user, compare history vs recs
# ---------------------------------------------------------------------------
def track_name(track_key):
    if track_key in track_lookup.index:
        row = track_lookup.loc[track_key]
        artist = row["artist_name"] if not isinstance(row, pd.DataFrame) else row["artist_name"].iloc[0]
        name = row["track_name"] if not isinstance(row, pd.DataFrame) else row["track_name"].iloc[0]
        return f"{artist} - {name}"
    return track_key

sample_user_idx = int(train_matrix.getnnz(axis=1).argmax())  # a very active user, easy to inspect
sample_user_id = idx_to_user[sample_user_idx]

user_row = train_matrix.getrow(sample_user_idx)
top_known_items = user_row.indices[np.argsort(-user_row.data)][:5]

print(f"\nSample user: {sample_user_id}")
print("Top 5 known (train) tracks:")
for item_idx in top_known_items:
    print("  -", track_name(idx_to_track[item_idx]))

recommended = model.recommend(
    sample_user_idx, train_matrix[sample_user_idx], N=5, filter_already_liked_items=True
)
print("Top 5 recommended tracks:")
for item_idx, score in zip(*recommended):
    print(f"  - {track_name(idx_to_track[item_idx])}  (score={score:.3f})")

print("\nStep 2 complete. This Precision@K / NDCG@K is your baseline to beat")
print("with embeddings, RAG, and the LLM re-ranking layer in later phases.")