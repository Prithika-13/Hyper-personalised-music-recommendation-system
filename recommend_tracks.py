"""
STEP 6 (Ollama version, no API key) — LLM Re-Ranking Layer

Runs entirely locally via Ollama -- no API key, no cost, no signup.
Requires Ollama installed and running (https://ollama.com/download) with a
model pulled, e.g.: ollama pull llama3.2:3b

Same guardrail as the Claude version: the model can ONLY select track_keys
that exist in the candidate pool we hand it -- anything else is dropped.

Inputs:
    train_matrix.npz, user_to_idx.pkl, idx_to_user.pkl, idx_to_track.pkl,
    track_lookup.csv                                        (Step 1)
    als_model_user_factors.npy, item_faiss_index.bin         (Step 2/3)
    text_faiss_index.bin, text_idx_to_track.pkl,
    track_text_lookup.csv                                    (Step 5)

Output: printed final recommendations with LLM-generated explanations
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
import ollama

OLLAMA_MODEL = "llama3.1"   # change to whatever you pulled, e.g. "llama3.1:8b"
ALS_CANDIDATE_POOL = 20
SEMANTIC_CANDIDATE_POOL = 15
FINAL_RECOMMENDATIONS = 10

# ---------------------------------------------------------------------------
# 1. LOAD ARTIFACTS
# ---------------------------------------------------------------------------
train_matrix = load_npz("train_matrix.npz").tocsr()
user_factors = np.load("als_model_user_factors.npy").astype(np.float32)

with open("user_to_idx.pkl", "rb") as f:
    user_to_idx = pickle.load(f)
with open("idx_to_user.pkl", "rb") as f:
    idx_to_user = pickle.load(f)
with open("idx_to_track.pkl", "rb") as f:
    idx_to_track = pickle.load(f)

track_lookup = pd.read_csv("track_lookup.csv").set_index("track_key")

als_index = faiss.read_index("item_faiss_index.bin")

text_index = faiss.read_index("text_faiss_index.bin")
with open("text_idx_to_track.pkl", "rb") as f:
    text_idx_to_track = pickle.load(f)
track_text_lookup = pd.read_csv("track_text_lookup.csv").set_index("track_key")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("All artifacts loaded.")

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
# 3. ALS CANDIDATE GENERATION (same adaptive-pool logic as Step 3)
# ---------------------------------------------------------------------------
def get_als_candidates(user_id, top_n=ALS_CANDIDATE_POOL, initial_pool=200, max_pool=None):
    user_idx = user_to_idx[user_id]
    user_vector = user_factors[user_idx].reshape(1, -1)
    seen_items = set(train_matrix.getrow(user_idx).indices)
    n_items = train_matrix.shape[1]
    max_pool = max_pool or n_items
    search_pool = min(initial_pool, max_pool)

    while True:
        scores, item_indices = als_index.search(user_vector, search_pool)
        scores, item_indices = scores[0], item_indices[0]
        results = []
        for score, item_idx in zip(scores, item_indices):
            if item_idx == -1 or item_idx in seen_items:
                continue
            track_key = idx_to_track[item_idx]
            results.append({"track_key": track_key, "score": float(score), "source": "collaborative"})
            if len(results) >= top_n:
                return results
        if search_pool >= max_pool:
            return results
        search_pool = min(search_pool * 4, max_pool)

# ---------------------------------------------------------------------------
# 4. SEMANTIC CANDIDATE GENERATION (Step 5's index, only used if a query is given)
# ---------------------------------------------------------------------------
def get_semantic_candidates(query, top_n=SEMANTIC_CANDIDATE_POOL):
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = text_index.search(query_vec, top_n)
    scores, indices = scores[0], indices[0]
    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        track_key = text_idx_to_track[idx]
        results.append({"track_key": track_key, "score": float(score), "source": "semantic"})
    return results

# ---------------------------------------------------------------------------
# 5. USER TASTE SUMMARY
# ---------------------------------------------------------------------------
def build_taste_summary(user_id, top_n=15):
    user_idx = user_to_idx[user_id]
    user_row = train_matrix.getrow(user_idx)
    top_item_indices = user_row.indices[np.argsort(-user_row.data)][:top_n]

    artists, tag_bag = [], []
    for item_idx in top_item_indices:
        track_key = idx_to_track[item_idx]
        artist, _ = track_display_name(track_key)
        artists.append(artist)
        tags = get_tags(track_key)
        if tags:
            tag_bag.extend(tags.split())

    top_artists = pd.Series(artists).value_counts().head(8).index.tolist()
    top_tags = pd.Series(tag_bag).value_counts().head(10).index.tolist() if tag_bag else []

    summary = f"Top artists this user listens to: {', '.join(top_artists)}."
    if top_tags:
        summary += f" Frequently associated genres/tags: {', '.join(top_tags)}."
    return summary

# ---------------------------------------------------------------------------
# 6. BUILD CANDIDATE POOL
# ---------------------------------------------------------------------------
def build_candidate_pool(user_id, query=None):
    candidates = get_als_candidates(user_id)
    if query:
        candidates += get_semantic_candidates(query)

    seen_keys, pool = set(), []
    for c in candidates:
        if c["track_key"] in seen_keys:
            continue
        seen_keys.add(c["track_key"])
        artist, name = track_display_name(c["track_key"])
        tags = get_tags(c["track_key"]) or "no tags available"
        pool.append({
            "track_key": c["track_key"],
            "artist": artist,
            "track_name": name,
            "tags": tags,
            "source": c["source"],
        })
    return pool

# ---------------------------------------------------------------------------
# 7. LLM RE-RANKING CALL (Ollama, JSON-mode, constrained to pool)
# ---------------------------------------------------------------------------
def llm_rerank(user_id, query=None, top_n=FINAL_RECOMMENDATIONS):
    pool = build_candidate_pool(user_id, query=query)
    taste_summary = build_taste_summary(user_id)

    candidate_lines = "\n".join(
        f"- track_key: {c['track_key']} | {c['artist']} - {c['track_name']} | tags: {c['tags']} | source: {c['source']}"
        for c in pool
    )

    query_line = f'\nThe user explicitly asked for: "{query}"' if query else ""

    prompt = f"""You are a music recommendation assistant. Select and rank the best {top_n} tracks
for this user from the candidate list below. You MUST only choose track_key values
that appear EXACTLY (character for character) in the candidate list -- never invent
a track or modify a track_key.

User taste profile: {taste_summary}{query_line}

Candidates:
{candidate_lines}

Respond with ONLY valid JSON in exactly this shape, no other text:
{{
  "recommendations": [
    {{"track_key": "<must match a candidate exactly>", "rank": 1, "explanation": "<one sentence>"}},
    ...
  ]
}}
Return exactly {top_n} recommendations, ranked 1 to {top_n}."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",   # forces Ollama to return valid JSON
        options={"temperature": 0.3},
    )

    raw_content = response["message"]["content"]
    try:
        parsed = json.loads(raw_content)
        result = parsed["recommendations"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [error] Failed to parse LLM output as expected JSON: {e}")
        print(f"  Raw output was: {raw_content[:500]}")
        return []

    valid_keys = {c["track_key"] for c in pool}
    pool_lookup = {c["track_key"]: c for c in pool}

    final = []
    for r in result:
        if r.get("track_key") not in valid_keys:
            print(f"  [warning] LLM returned a track_key not in the pool, dropping: {r.get('track_key')}")
            continue
        c = pool_lookup[r["track_key"]]
        final.append({**r, "artist": c["artist"], "track_name": c["track_name"]})

    return sorted(final, key=lambda x: x.get("rank", 999))

# ---------------------------------------------------------------------------
# 8. TEST
# ---------------------------------------------------------------------------
sample_user_idx = int(np.argsort(train_matrix.getnnz(axis=1))[len(idx_to_user) // 2])  # median-activity user
sample_user_id = idx_to_user[sample_user_idx]

print(f"\n=== LLM re-ranked recommendations for {sample_user_id} (no explicit query) ===")
for rec in llm_rerank(sample_user_id):
    print(f"  {rec['rank']}. {rec['artist']} - {rec['track_name']}")
    print(f"     -> {rec['explanation']}")

print(f"\n=== LLM re-ranked recommendations for {sample_user_id} (query: 'something upbeat for a workout') ===")
for rec in llm_rerank(sample_user_id, query="something upbeat for a workout"):
    print(f"  {rec['rank']}. {rec['artist']} - {rec['track_name']}")
    print(f"     -> {rec['explanation']}")

print("\nStep 6 complete (fully local, no API key). Full pipeline: implicit scoring")
print("-> ALS -> candidate generation -> semantic RAG -> local LLM re-ranking.")