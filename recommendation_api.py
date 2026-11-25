"""
STEP 8 — FastAPI Recommendation Service

Wraps Steps 1, 3, 5, and 6 into a single running service. Loads all
artifacts once at startup, then serves recommendations over HTTP.

Run with:
    uvicorn recommendation_api:app --reload

Requires Ollama running locally (same as Step 6/7).

Inputs (loaded at startup):
    train_matrix.npz, user_to_idx.pkl, idx_to_user.pkl, idx_to_track.pkl,
    track_lookup.csv, als_model_user_factors.npy, item_faiss_index.bin,
    text_faiss_index.bin, text_idx_to_track.pkl, track_text_lookup.csv
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import pickle
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import faiss
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
import ollama

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_MODEL = "llama3.1"
ALS_CANDIDATE_POOL = 20
SEMANTIC_CANDIDATE_POOL = 15

# ---------------------------------------------------------------------------
# GLOBAL STATE (populated at startup, read by request handlers)
# ---------------------------------------------------------------------------
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading artifacts...")
    state["train_matrix"] = load_npz("train_matrix.npz").tocsr()
    state["user_factors"] = np.load("als_model_user_factors.npy").astype(np.float32)

    with open("user_to_idx.pkl", "rb") as f:
        state["user_to_idx"] = pickle.load(f)
    with open("idx_to_user.pkl", "rb") as f:
        state["idx_to_user"] = pickle.load(f)
    with open("idx_to_track.pkl", "rb") as f:
        state["idx_to_track"] = pickle.load(f)

    state["track_lookup"] = pd.read_csv("track_lookup.csv").set_index("track_key")
    state["als_index"] = faiss.read_index("item_faiss_index.bin")

    state["text_index"] = faiss.read_index("text_faiss_index.bin")
    with open("text_idx_to_track.pkl", "rb") as f:
        state["text_idx_to_track"] = pickle.load(f)
    state["track_text_lookup"] = pd.read_csv("track_text_lookup.csv").set_index("track_key")

    state["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")

    print("Warming up Ollama (loading model into memory)...")
    try:
        ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "hello"}])
        print("Ollama warm-up complete.")
    except Exception as e:
        print(f"Ollama warm-up failed (service may not be running): {e}")

    print("All artifacts loaded. Service ready.")
    yield
    state.clear()
    print("Shutting down, state cleared.")

app = FastAPI(title="Music Recommendation Service", lifespan=lifespan)

# ---------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    user_id: str
    query: str | None = Field(default=None, description="Optional natural-language mood/context query")
    top_n: int = Field(default=10, ge=1, le=25)

class Recommendation(BaseModel):
    rank: int
    track_key: str
    artist: str
    track_name: str
    explanation: str

class RecommendResponse(BaseModel):
    user_id: str
    query: str | None
    recommendations: list[Recommendation]

# ---------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS (same logic as Steps 3/5/6, using `state` instead
# of module-level globals)
# ---------------------------------------------------------------------------
def track_display_name(track_key):
    lookup = state["track_lookup"]
    if track_key in lookup.index:
        row = lookup.loc[track_key]
        return row["artist_name"], row["track_name"]
    return "?", track_key

def get_tags(track_key):
    lookup = state["track_text_lookup"]
    if track_key in lookup.index:
        return lookup.loc[track_key, "combined_text"]
    return None

def get_als_candidates(user_id, top_n=ALS_CANDIDATE_POOL, initial_pool=200):
    user_to_idx = state["user_to_idx"]
    if user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {user_id}")

    user_idx = user_to_idx[user_id]
    train_matrix = state["train_matrix"]
    user_vector = state["user_factors"][user_idx].reshape(1, -1)
    seen_items = set(train_matrix.getrow(user_idx).indices)
    n_items = train_matrix.shape[1]
    search_pool = min(initial_pool, n_items)

    while True:
        scores, item_indices = state["als_index"].search(user_vector, search_pool)
        item_indices = item_indices[0]
        results = []
        for item_idx in item_indices:
            if item_idx == -1 or item_idx in seen_items:
                continue
            track_key = state["idx_to_track"][item_idx]
            results.append({"track_key": track_key, "source": "collaborative"})
            if len(results) >= top_n:
                return results
        if search_pool >= n_items:
            return results
        search_pool = min(search_pool * 4, n_items)

def get_semantic_candidates(query, top_n=SEMANTIC_CANDIDATE_POOL):
    query_vec = state["embed_model"].encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = state["text_index"].search(query_vec, top_n)
    indices = indices[0]
    results = []
    for idx in indices:
        if idx == -1:
            continue
        track_key = state["text_idx_to_track"][idx]
        results.append({"track_key": track_key, "source": "semantic"})
    return results

def build_taste_summary(user_id, top_n=15):
    user_idx = state["user_to_idx"][user_id]
    train_matrix = state["train_matrix"]
    user_row = train_matrix.getrow(user_idx)
    top_item_indices = user_row.indices[np.argsort(-user_row.data)][:top_n]

    artists, tag_bag = [], []
    for item_idx in top_item_indices:
        track_key = state["idx_to_track"][item_idx]
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
            "track_key": c["track_key"], "artist": artist, "track_name": name,
            "tags": tags, "source": c["source"],
        })
    return pool

def llm_rerank(user_id, query, top_n):
    pool = build_candidate_pool(user_id, query=query)
    if not pool:
        return []

    taste_summary = build_taste_summary(user_id)
    candidate_lines = "\n".join(
        f"- track_key: {c['track_key']} | {c['artist']} - {c['track_name']} | tags: {c['tags']} | source: {c['source']}"
        for c in pool
    )
    query_line = f'\nThe user explicitly asked for: "{query}"' if query else ""

    prompt = f"""You are a music recommendation assistant. Select and rank the best {top_n} tracks
for this user from the candidate list below. You MUST only choose track_key values
that appear EXACTLY in the candidate list -- never invent a track.

User taste profile: {taste_summary}{query_line}

Candidates:
{candidate_lines}

Respond with ONLY valid JSON, no other text:
{{"recommendations": [{{"track_key": "<exact match>", "rank": 1, "explanation": "<one sentence>"}}, ...]}}
Return exactly {top_n} entries, ranked 1 to {top_n}."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.3},
    )

    try:
        parsed = json.loads(response["message"]["content"])
        raw_recs = parsed["recommendations"]
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=502, detail="LLM returned malformed output; try again.")

    valid_keys = {c["track_key"] for c in pool}
    pool_lookup = {c["track_key"]: c for c in pool}

    final = []
    for r in raw_recs:
        if r.get("track_key") not in valid_keys:
            continue  # silently drop hallucinated/invalid keys, same guardrail as Step 6
        c = pool_lookup[r["track_key"]]
        final.append({
            "rank": r.get("rank", 999),
            "track_key": r["track_key"],
            "artist": c["artist"],
            "track_name": c["track_name"],
            "explanation": r.get("explanation", ""),
        })

    return sorted(final, key=lambda x: x["rank"])[:top_n]

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "artifacts_loaded": len(state) > 0}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recs = llm_rerank(req.user_id, req.query, req.top_n)
    return RecommendResponse(user_id=req.user_id, query=req.query, recommendations=recs)