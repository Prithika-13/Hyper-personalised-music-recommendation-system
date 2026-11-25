"""
STEP 5 — Embed Text Corpus & Build Semantic (RAG) Retrieval Index

Inputs (from Step 4):
    matched_track_metadata.csv

Outputs:
    text_faiss_index.bin        -- semantic search index (cosine similarity)
    text_idx_to_track.pkl        -- maps index position -> track_key
    track_text_lookup.csv        -- track_key -> artist/track/combined_text
                                     (for readable output)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# 1. LOAD METADATA, KEEP ONLY TRACKS WITH REAL TEXT
# ---------------------------------------------------------------------------
metadata = pd.read_csv("matched_track_metadata.csv")
text_df = metadata[metadata["has_metadata"] == True].reset_index(drop=True)

print(f"Total tracks in metadata file: {len(metadata)}")
print(f"Tracks with usable text (has_metadata=True): {len(text_df)}")

if len(text_df) == 0:
    raise ValueError("No tracks with metadata found -- check Step 4 output before continuing.")

# ---------------------------------------------------------------------------
# 2. LOAD EMBEDDING MODEL
# ---------------------------------------------------------------------------
print(f"\nLoading embedding model: {EMBEDDING_MODEL} (first run downloads ~80MB)")
model = SentenceTransformer(EMBEDDING_MODEL)

# ---------------------------------------------------------------------------
# 3. ENCODE COMBINED TEXT
# ---------------------------------------------------------------------------
texts = text_df["combined_text"].tolist()
print(f"\nEncoding {len(texts)} texts...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)
embeddings = embeddings.astype(np.float32)
print(f"Embeddings shape: {embeddings.shape}")

# ---------------------------------------------------------------------------
# 4. L2-NORMALIZE (so IndexFlatIP == cosine similarity)
# ---------------------------------------------------------------------------
faiss.normalize_L2(embeddings)

# ---------------------------------------------------------------------------
# 5. BUILD FAISS INDEX
# ---------------------------------------------------------------------------
dimension = embeddings.shape[1]
text_index = faiss.IndexFlatIP(dimension)
text_index.add(embeddings)
print(f"\nText FAISS index built: {text_index.ntotal} tracks, dim={dimension}")

faiss.write_index(text_index, "text_faiss_index.bin")

text_idx_to_track = {i: track_key for i, track_key in enumerate(text_df["track_key"])}
with open("text_idx_to_track.pkl", "wb") as f:
    pickle.dump(text_idx_to_track, f)

text_df[["track_key", "artist_name", "track_name", "combined_text"]].to_csv(
    "track_text_lookup.csv", index=False
)

print("Saved text_faiss_index.bin, text_idx_to_track.pkl, track_text_lookup.csv")

# ---------------------------------------------------------------------------
# 6. SEMANTIC SEARCH FUNCTIONS
# ---------------------------------------------------------------------------
track_text_lookup = text_df.set_index("track_key")

def semantic_search(query, top_n=10):
    """Free-text query -> ranked list of semantically similar tracks."""
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    scores, indices = text_index.search(query_vec, top_n)
    scores, indices = scores[0], indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        track_key = text_idx_to_track[idx]
        row = track_text_lookup.loc[track_key]
        results.append({
            "track_key": track_key,
            "artist": row["artist_name"],
            "track_name": row["track_name"],
            "tags": row["combined_text"],
            "score": float(score),
        })
    return results

def find_similar_tracks(track_key, top_n=10):
    """Given a track already in the text index, find semantically similar tracks."""
    if track_key not in track_text_lookup.index:
        raise ValueError(f"{track_key} has no text metadata -- not in the semantic index.")

    idx_lookup = {v: k for k, v in text_idx_to_track.items()}
    source_idx = idx_lookup[track_key]
    query_vec = embeddings[source_idx].reshape(1, -1)

    scores, indices = text_index.search(query_vec, top_n + 1)  # +1 to skip itself
    scores, indices = scores[0], indices[0]

    results = []
    for score, idx in zip(scores, indices):
        candidate_key = text_idx_to_track[idx]
        if candidate_key == track_key:
            continue
        row = track_text_lookup.loc[candidate_key]
        results.append({
            "track_key": candidate_key,
            "artist": row["artist_name"],
            "track_name": row["track_name"],
            "tags": row["combined_text"],
            "score": float(score),
        })
        if len(results) >= top_n:
            break
    return results

# ---------------------------------------------------------------------------
# 7. TEST
# ---------------------------------------------------------------------------
print("\n=== Semantic search test: 'moody atmospheric electronic' ===")
for r in semantic_search("moody atmospheric electronic", top_n=5):
    print(f"  {r['artist']} - {r['track_name']}  [{r['tags']}]  (score={r['score']:.3f})")

print("\n=== Semantic search test: 'upbeat feel good pop' ===")
for r in semantic_search("upbeat feel good pop", top_n=5):
    print(f"  {r['artist']} - {r['track_name']}  [{r['tags']}]  (score={r['score']:.3f})")

sample_track_key = text_df.iloc[0]["track_key"]
sample_artist = text_df.iloc[0]["artist_name"]
sample_name = text_df.iloc[0]["track_name"]
print(f"\n=== Tracks similar to: {sample_artist} - {sample_name} ===")
for r in find_similar_tracks(sample_track_key, top_n=5):
    print(f"  {r['artist']} - {r['track_name']}  [{r['tags']}]  (score={r['score']:.3f})")

print("\nStep 5 complete. semantic_search() and find_similar_tracks() are now")
print("available -- Step 6 will combine THIS with Step 3's get_candidates()")
print("output before the LLM re-ranking layer.")