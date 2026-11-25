"""
STEP 10 — Evaluation Suite: Traditional Baselines vs. Hyper-Personalized Pipeline

Compares THREE complete systems head-to-head:
    1. Popularity baseline    (recommend globally popular tracks to everyone)
    2. Pure ALS               (Steps 2/3 collaborative filtering alone)
    3. Full pipeline          (ALS candidates + semantic RAG + LLM re-ranking)

Reports:
    - Accuracy: Recall@10, NDCG@10, Hit@10 (all evaluated over the SAME users,
      no restriction to "pool overlap" -- a miss counts against a system
      regardless of which stage caused it)
    - Beyond-accuracy: personalization, diversity, catalog coverage
    - Statistical significance: paired Wilcoxon signed-rank test (ALS vs Full)
    - A query-sensitivity demonstration (a capability traditional systems lack)

Requires Ollama running locally (same as Steps 6/7).
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
from scipy.stats import wilcoxon
import ollama

OLLAMA_MODEL = "llama3.1"
SAMPLE_USERS = 30      # kept small because the full pipeline calls a local LLM per user
K = 10
ALS_POOL_FOR_FULL_PIPELINE = 30
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. LOAD ARTIFACTS
# ---------------------------------------------------------------------------
train_matrix = load_npz("train_matrix.npz").tocsr()
user_factors = np.load("als_model_user_factors.npy").astype(np.float32)

with open("test_holdout.pkl", "rb") as f:
    test_holdout_raw = pickle.load(f)
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

n_users, n_items = train_matrix.shape

# int-cast fix (same issue as Step 7 -- test_holdout can contain numpy.float64)
holdout_by_user = {}
for u, i in test_holdout_raw:
    holdout_by_user.setdefault(int(u), set()).add(int(i))

print(f"Loaded. {n_users} users, {n_items} items. Test users available: {len(holdout_by_user)}")

# ---------------------------------------------------------------------------
# 2. HELPERS
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

# ---------------------------------------------------------------------------
# 3. SYSTEM 1 — POPULARITY BASELINE
# ---------------------------------------------------------------------------
item_popularity = np.asarray(train_matrix.sum(axis=0)).flatten()
GLOBAL_POPULAR_ORDER = np.argsort(-item_popularity)  # item indices, most popular first

def get_popularity_topk(user_idx, k=K):
    seen = set(train_matrix.getrow(user_idx).indices)
    result = []
    for item_idx in GLOBAL_POPULAR_ORDER:
        if item_idx in seen:
            continue
        result.append(int(item_idx))
        if len(result) >= k:
            break
    return result

# ---------------------------------------------------------------------------
# 4. SYSTEM 2 — PURE ALS
# ---------------------------------------------------------------------------
def get_als_topk(user_idx, k=K, initial_pool=200):
    user_vector = user_factors[user_idx].reshape(1, -1)
    seen = set(train_matrix.getrow(user_idx).indices)
    search_pool = min(initial_pool, n_items)
    while True:
        _, item_indices = als_index.search(user_vector, search_pool)
        item_indices = item_indices[0]
        result = [int(idx) for idx in item_indices if idx != -1 and idx not in seen]
        if len(result) >= k or search_pool >= n_items:
            return result[:k]
        search_pool = min(search_pool * 4, n_items)

# ---------------------------------------------------------------------------
# 5. SYSTEM 3 — FULL PIPELINE (ALS candidates -> LLM re-rank)
#    No query used here -- this is the fairest head-to-head against baselines
#    that also have no query mechanism. Query capability is demonstrated
#    separately in section 8, since it isn't something a metric can score
#    against a baseline that structurally can't do it at all.
# ---------------------------------------------------------------------------
def get_full_pipeline_topk(user_idx, k=K):
    pool_items = get_als_topk(user_idx, k=ALS_POOL_FOR_FULL_PIPELINE)
    if not pool_items:
        return []

    candidate_lines, key_by_line = [], {}
    for item_idx in pool_items:
        track_key = idx_to_track[item_idx]
        artist, name = track_display_name(track_key)
        tags = get_tags(track_key) or "no tags available"
        candidate_lines.append(f"- track_key: {track_key} | {artist} - {name} | tags: {tags}")
        key_by_line[track_key] = item_idx

    prompt = f"""Select and rank the best {k} tracks from this candidate list for a listener
with the taste implied by these candidates. You MUST only use track_key values that
appear EXACTLY in the list below.

Candidates:
{chr(10).join(candidate_lines)}

Respond with ONLY valid JSON: {{"recommendations": [{{"track_key": "<exact match>", "rank": 1}}, ...]}}
Return exactly {k} entries, ranked 1 to {k}."""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.3},
        )
        parsed = json.loads(response["message"]["content"])
        ranked = sorted(parsed["recommendations"], key=lambda x: x.get("rank", 999))
        result = [key_by_line[r["track_key"]] for r in ranked if r.get("track_key") in key_by_line]
        return result[:k]
    except Exception as e:
        print(f"    [LLM call failed for user {idx_to_user.get(user_idx, '?')}: {type(e).__name__}: {e}]")
        return pool_items[:k]  # fall back to raw ALS order if the LLM call fails

# ---------------------------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------------------------
def recall_at_k(recommended, relevant, k=K):
    if not relevant:
        return None
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)

def hit_at_k(recommended, relevant, k=K):
    if not relevant:
        return None
    return 1.0 if set(recommended[:k]) & relevant else 0.0

def ndcg_at_k(recommended, relevant, k=K):
    if not relevant:
        return None
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def personalization(all_user_lists):
    """Average pairwise dissimilarity (1 - Jaccard) between users' rec lists.
    0 = everyone gets the same list. 1 = no overlap at all between any two users."""
    lists = [set(l) for l in all_user_lists if l]
    if len(lists) < 2:
        return None
    dissimilarities = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            union = lists[i] | lists[j]
            if not union:
                continue
            jaccard = len(lists[i] & lists[j]) / len(union)
            dissimilarities.append(1 - jaccard)
    return float(np.mean(dissimilarities)) if dissimilarities else None

def artist_diversity(recommended_items):
    """Unique artists / list length. 1.0 = every track a different artist."""
    if not recommended_items:
        return None
    artists = [track_display_name(idx_to_track[i])[0] for i in recommended_items]
    return len(set(artists)) / len(artists)

def catalog_coverage(all_user_lists, n_items):
    unique_items = set()
    for l in all_user_lists:
        unique_items.update(l)
    return len(unique_items) / n_items

# ---------------------------------------------------------------------------
# 7. RUN EVALUATION ACROSS SAMPLED USERS
# ---------------------------------------------------------------------------
eligible_users = [u for u in holdout_by_user.keys()]
random.shuffle(eligible_users)
sample = eligible_users[:SAMPLE_USERS]

results = {"popularity": {"recall": [], "ndcg": [], "hit": [], "lists": []},
           "als": {"recall": [], "ndcg": [], "hit": [], "lists": []},
           "full": {"recall": [], "ndcg": [], "hit": [], "lists": []}}

print(f"\nEvaluating {len(sample)} users across 3 systems (this calls the LLM once per user for the full pipeline)...")
for n, user_idx in enumerate(sample):
    relevant = holdout_by_user[user_idx]

    pop_list = get_popularity_topk(user_idx)
    als_list = get_als_topk(user_idx)
    full_list = get_full_pipeline_topk(user_idx)

    for name, rec_list in [("popularity", pop_list), ("als", als_list), ("full", full_list)]:
        results[name]["recall"].append(recall_at_k(rec_list, relevant))
        results[name]["ndcg"].append(ndcg_at_k(rec_list, relevant))
        results[name]["hit"].append(hit_at_k(rec_list, relevant))
        results[name]["lists"].append(rec_list)

    print(f"  [{n+1}/{len(sample)}] user {idx_to_user[user_idx]} done")

# ---------------------------------------------------------------------------
# 8. ACCURACY RESULTS TABLE
# ---------------------------------------------------------------------------
def safe_mean(values):
    clean = [v for v in values if v is not None]
    return np.mean(clean) if clean else float("nan")

print("\n" + "=" * 60)
print("ACCURACY  (higher is better)")
print("=" * 60)
print(f"{'System':<14}{'Recall@10':<14}{'NDCG@10':<14}{'Hit@10':<14}")
for name in ["popularity", "als", "full"]:
    r = results[name]
    print(f"{name:<14}{safe_mean(r['recall']):<14.4f}{safe_mean(r['ndcg']):<14.4f}{safe_mean(r['hit']):<14.4f}")

# ---------------------------------------------------------------------------
# 9. STATISTICAL SIGNIFICANCE (paired Wilcoxon on NDCG@10, ALS vs Full)
# ---------------------------------------------------------------------------
als_ndcg = [v for v in results["als"]["ndcg"] if v is not None]
full_ndcg = [v for v in results["full"]["ndcg"] if v is not None]
paired = [(a, f) for a, f in zip(results["als"]["ndcg"], results["full"]["ndcg"]) if a is not None and f is not None]

print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank, ALS vs Full, NDCG@10)")
print("=" * 60)
if len(paired) >= 5 and any(a != f for a, f in paired):
    a_vals, f_vals = zip(*paired)
    stat, p_value = wilcoxon(a_vals, f_vals)
    print(f"n = {len(paired)} paired users, statistic = {stat:.3f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("-> Statistically significant difference at the 0.05 level.")
    else:
        print("-> NOT statistically significant at the 0.05 level -- treat any observed")
        print("   difference as inconclusive at this sample size.")
else:
    print("Not enough paired, non-identical observations to run the test meaningfully.")
    print("(This is common at small sample sizes -- consider raising SAMPLE_USERS.)")

# ---------------------------------------------------------------------------
# 10. BEYOND-ACCURACY METRICS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BEYOND-ACCURACY METRICS")
print("=" * 60)
print(f"{'System':<14}{'Personalization':<18}{'Avg Artist Diversity':<22}{'Catalog Coverage':<18}")
for name in ["popularity", "als", "full"]:
    lists = results[name]["lists"]
    p = personalization(lists)
    d = safe_mean([artist_diversity(l) for l in lists])
    c = catalog_coverage(lists, n_items)
    print(f"{name:<14}{p:<18.4f}{d:<22.4f}{c:<18.5f}")

print("\nPersonalization: 0 = everyone gets an identical list, 1 = no two users share anything.")
print("Catalog coverage: fraction of the full catalog that appears across ALL sampled users' lists.")

# ---------------------------------------------------------------------------
# 11. CAPABILITY DEMONSTRATION — QUERY SENSITIVITY
#     Traditional ALS/popularity have NO mechanism to respond to this at all.
#     This isn't a number to compare against them -- it's a capability gap.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CAPABILITY DEMONSTRATION: query sensitivity (full pipeline only)")
print("=" * 60)
demo_user_idx = sample[0]
demo_user_id = idx_to_user[demo_user_idx]

# reuse the pool but pass a query through a lightweight local re-implementation
def full_pipeline_with_query(user_idx, query, k=K):
    pool_items = get_als_topk(user_idx, k=ALS_POOL_FOR_FULL_PIPELINE)
    candidate_lines, key_by_line = [], {}
    for item_idx in pool_items:
        track_key = idx_to_track[item_idx]
        artist, name = track_display_name(track_key)
        tags = get_tags(track_key) or "no tags available"
        candidate_lines.append(f"- track_key: {track_key} | {artist} - {name} | tags: {tags}")
        key_by_line[track_key] = item_idx

    prompt = f"""The user asked for: "{query}". Select and rank the best {k} tracks from this
list that fit that request. Only use track_key values that appear EXACTLY below.

Candidates:
{chr(10).join(candidate_lines)}

Respond with ONLY valid JSON: {{"recommendations": [{{"track_key": "<exact match>", "rank": 1}}]}}
Return exactly {k} entries."""
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], format="json", options={"temperature": 0.3})
        parsed = json.loads(response["message"]["content"])
        ranked = sorted(parsed["recommendations"], key=lambda x: x.get("rank", 999))
        return [key_by_line[r["track_key"]] for r in ranked if r.get("track_key") in key_by_line][:k]
    except Exception as e:
        print(f"    [LLM call failed: {type(e).__name__}: {e}]")
        return pool_items[:k]

list_a = full_pipeline_with_query(demo_user_idx, "something energetic for a workout")
list_b = full_pipeline_with_query(demo_user_idx, "something mellow for late night")

overlap = len(set(list_a) & set(list_b)) / K
print(f"User: {demo_user_id}")
print(f"Overlap between 'workout' and 'late night' recommendation lists: {overlap:.1%}")
print("(Low overlap = the system meaningfully changes its output based on context.")
print(" A popularity or pure-ALS baseline has NO query input at all -- this comparison")
print(" isn't a 'higher score is better' metric, it's a capability that only exists here.)")

print("\nStep 10 complete.")