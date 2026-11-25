"""
STEP 1 (Last.fm 1K dataset) — Session Segmentation, Skip/Replay Inference,
and Recency-Weighted Implicit Confidence Scoring

Input file expected: 'userid-timestamp-artid-artname-traid-traname.tsv'
(tab-separated, no header)
Columns: user_id, timestamp, artist_id, artist_name, track_id, track_name

Output artifacts (used by Step 2 — ALS baseline training):
    - interaction_matrix.npz
    - train_matrix.npz / test_holdout.pkl   (TIME-BASED split, not random)
    - user_to_idx.pkl / idx_to_user.pkl
    - track_to_idx.pkl / idx_to_track.pkl
    - track_lookup.csv   (track_key -> artist_name, track_name, for later use)
"""

import csv

import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix, save_npz

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SESSION_GAP_MINUTES = 20        # gap beyond this starts a new session
SKIP_THRESHOLD_SECONDS = 30     # gap to next track under this = inferred skip
RECENCY_HALF_LIFE_DAYS = 60     # confidence halves every N days of inactivity
MIN_INTERACTIONS_PER_USER = 5
MIN_INTERACTIONS_PER_ITEM = 5
ALPHA = 10                      # weight on log(1+play_count)
BETA = 3                        # penalty weight for skip ratio
GAMMA = 5                       # bonus weight for replays
TEST_FRACTION_PER_USER = 0.2    # most recent 20% of each user's interactions -> test

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
col_names = ["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"]

df = pd.read_csv(
    "lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv",
    sep="\t",
    header=None,
    names=col_names,
    dtype={"user_id": "category", "artist_id": "string", "artist_name": "string",
           "track_id": "string", "track_name": "string"},
    parse_dates=["timestamp"],
    engine="python",
    quoting=csv.QUOTE_NONE,
    on_bad_lines="warn",   # in case a few rows are still genuinely malformed
)

print("Raw shape:", df.shape)
print("Missing track_id:", df["track_id"].isna().sum(), f"({df['track_id'].isna().mean():.1%})")
print("Missing track_name:", df["track_name"].isna().sum())
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())

# ---------------------------------------------------------------------------
# 2. BUILD A STABLE TRACK KEY (fallback to artist+track name when mbid missing)
# ---------------------------------------------------------------------------
df = df.dropna(subset=["track_name", "artist_name"])  # can't identify these at all

df["track_key"] = np.where(
    df["track_id"].notna(),
    "mbid:" + df["track_id"].astype(str),
    "name:" + df["artist_name"].str.lower().str.strip() + "|" + df["track_name"].str.lower().str.strip(),
)

# save a lookup so you can map track_key back to readable names later
track_lookup = (
    df[["track_key", "artist_name", "track_name"]]
    .drop_duplicates(subset="track_key")
    .reset_index(drop=True)
)
track_lookup.to_csv("track_lookup.csv", index=False)
print("Unique track_keys:", df["track_key"].nunique())

# ---------------------------------------------------------------------------
# 3. SORT & COMPUTE INTER-EVENT GAPS PER USER
# ---------------------------------------------------------------------------
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

df["prev_timestamp"] = df.groupby("user_id", observed=True)["timestamp"].shift(1)
df["gap_seconds"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()

# gap to the NEXT event (used for skip inference — did they bail quickly?)
df["next_timestamp"] = df.groupby("user_id", observed=True)["timestamp"].shift(-1)
df["gap_to_next_seconds"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds()

# ---------------------------------------------------------------------------
# 4. SESSION SEGMENTATION
# ---------------------------------------------------------------------------
new_session = (df["gap_seconds"].isna()) | (df["gap_seconds"] > SESSION_GAP_MINUTES * 60)
df["session_num"] = new_session.groupby(df["user_id"], observed=True).cumsum()
df["session_id"] = df["user_id"].astype(str) + "_s" + df["session_num"].astype(str)

print("\nTotal sessions:", df["session_id"].nunique())
print("Avg events per session:", df.groupby("session_id").size().mean().round(2))

# ---------------------------------------------------------------------------
# 5. SKIP INFERENCE
#    A play is treated as skipped if the user moved to the next event within
#    SKIP_THRESHOLD_SECONDS. Last event of a session/user has no "next" -> NaN,
#    treated as not-skipped (can't tell, don't penalize).
# ---------------------------------------------------------------------------
df["is_skip"] = df["gap_to_next_seconds"].fillna(np.inf) < SKIP_THRESHOLD_SECONDS

print("\nOverall skip rate (inferred):", df["is_skip"].mean().round(3))

# ---------------------------------------------------------------------------
# 6. REPLAY DETECTION (same track more than once within the same session)
# ---------------------------------------------------------------------------
df["replay_in_session"] = df.duplicated(subset=["session_id", "track_key"], keep=False)

# ---------------------------------------------------------------------------
# 7. AGGREGATE PER (user, track)
# ---------------------------------------------------------------------------
agg = df.groupby(["user_id", "track_key"], observed=True).agg(
    play_count=("track_key", "size"),
    skip_count=("is_skip", "sum"),
    replay_count=("replay_in_session", "sum"),
    last_played=("timestamp", "max"),
).reset_index()

agg["skip_ratio"] = agg["skip_count"] / agg["play_count"]
agg["non_skip_plays"] = agg["play_count"] - agg["skip_count"]

# ---------------------------------------------------------------------------
# 8. RECENCY-WEIGHTED CONFIDENCE SCORE
#    "now" = the most recent timestamp in the whole dataset (this is a static
#    historical dataset, so we can't use wall-clock "today")
# ---------------------------------------------------------------------------
NOW = df["timestamp"].max()
agg["days_since_last_play"] = (NOW - agg["last_played"]).dt.total_seconds() / 86400
LAMBDA = np.log(2) / RECENCY_HALF_LIFE_DAYS
agg["recency_weight"] = np.exp(-LAMBDA * agg["days_since_last_play"])

agg["confidence"] = (
    1
    + ALPHA * np.log1p(agg["non_skip_plays"])
    + GAMMA * np.log1p(agg["replay_count"])
    - BETA * agg["skip_ratio"]
) * agg["recency_weight"]

agg["confidence"] = agg["confidence"].clip(lower=0.01)  # keep strictly positive

print("\nConfidence score distribution:\n", agg["confidence"].describe())

# ---------------------------------------------------------------------------
# 9. K-CORE FILTERING
# ---------------------------------------------------------------------------
def k_core_filter(data, min_user=5, min_item=5):
    while True:
        user_counts = data["user_id"].value_counts()
        item_counts = data["track_key"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        valid_items = item_counts[item_counts >= min_item].index
        new_data = data[data["user_id"].isin(valid_users) & data["track_key"].isin(valid_items)]
        if len(new_data) == len(data):
            return new_data
        data = new_data

agg = k_core_filter(agg, MIN_INTERACTIONS_PER_USER, MIN_INTERACTIONS_PER_ITEM)
print(f"\nAfter k-core filtering: {agg.shape[0]} (user,track) pairs, "
      f"{agg['user_id'].nunique()} users, {agg['track_key'].nunique()} tracks")

# ---------------------------------------------------------------------------
# 10. ID <-> INDEX MAPPINGS
# ---------------------------------------------------------------------------
unique_users = agg["user_id"].unique()
unique_tracks = agg["track_key"].unique()

user_to_idx = {u: i for i, u in enumerate(unique_users)}
idx_to_user = {i: u for u, i in user_to_idx.items()}
track_to_idx = {t: i for i, t in enumerate(unique_tracks)}
idx_to_track = {i: t for t, i in track_to_idx.items()}

n_users, n_items = len(user_to_idx), len(track_to_idx)

# ---------------------------------------------------------------------------
# 11. TIME-BASED TRAIN/TEST SPLIT
#     For each user, hold out their most-recent TEST_FRACTION_PER_USER of
#     interactions (by last_played) as test set. More realistic than random
#     leave-one-out since we're predicting future behavior from past behavior.
# ---------------------------------------------------------------------------
def time_based_split(data, test_frac=0.2):
    train_parts, test_parts = [], []
    for _, group in data.groupby("user_id"):
        group = group.sort_values("last_played")
        n_test = max(1, int(len(group) * test_frac)) if len(group) >= 5 else 0
        if n_test == 0:
            train_parts.append(group)
            continue
        test_parts.append(group.iloc[-n_test:])
        train_parts.append(group.iloc[:-n_test])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)

train_df, test_df = time_based_split(agg, TEST_FRACTION_PER_USER)
print(f"\nTrain: {len(train_df)} pairs, Test (held-out, most recent): {len(test_df)} pairs")

# ---------------------------------------------------------------------------
# 12. BUILD SPARSE MATRICES
# ---------------------------------------------------------------------------
def build_matrix(data):
    rows = data["user_id"].map(user_to_idx).values
    cols = data["track_key"].map(track_to_idx).values
    vals = data["confidence"].values
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

full_matrix = build_matrix(agg)
train_matrix = build_matrix(train_df)
test_holdout = list(zip(
    test_df["user_id"].map(user_to_idx).values,
    test_df["track_key"].map(track_to_idx).values,
))

print("\nFull matrix:", full_matrix.shape, "nnz =", full_matrix.nnz)
print("Density:", full_matrix.nnz / (n_users * n_items))

# ---------------------------------------------------------------------------
# 13. PERSIST
# ---------------------------------------------------------------------------
save_npz("interaction_matrix.npz", full_matrix)
save_npz("train_matrix.npz", train_matrix)

for name, obj in [
    ("test_holdout.pkl", test_holdout),
    ("user_to_idx.pkl", user_to_idx),
    ("idx_to_user.pkl", idx_to_user),
    ("track_to_idx.pkl", track_to_idx),
    ("idx_to_track.pkl", idx_to_track),
]:
    with open(name, "wb") as f:
        pickle.dump(obj, f)

print("\nSaved all artifacts. Step 1 complete.")
print("Next: Step 2 — train ALS baseline on train_matrix.npz using the `implicit` library,")
print("evaluate Recall@K / NDCG@K against test_holdout.pkl.")