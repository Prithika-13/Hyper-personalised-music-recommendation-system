"""
STEP 4 (no API key) — Build RAG Text Corpus by Joining Local Datasets

Uses data you already have on disk, no external API calls:
    - track_lookup.csv        (from Step 1: your Last.fm 1K filtered catalog)
    - Music Info.csv          (from your original Kaggle download: has
                                genre, tags, and audio features per track)

Matches tracks across the two datasets by normalized artist + track name.
Coverage will be PARTIAL (Music Info.csv has ~50K tracks vs. your ~175K
filtered Last.fm catalog) -- unmatched tracks simply won't have RAG text
and will keep relying on pure collaborative filtering from Steps 1-3.

Output:
    matched_track_metadata.csv  -- track_key, artist, track_name, genre,
                                    tags, combined_text, matched audio features
"""

import re
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- adjust this path to wherever you saved the original Kaggle download
# ---------------------------------------------------------------------------
MUSIC_INFO_PATH = "Music Info.csv"

# ---------------------------------------------------------------------------
# 1. LOAD BOTH DATASETS
# ---------------------------------------------------------------------------
track_lookup = pd.read_csv("track_lookup.csv")  # track_key, artist_name, track_name
music_info = pd.read_csv(MUSIC_INFO_PATH)

print("Last.fm catalog (track_lookup):", track_lookup.shape)
print("Music Info.csv:", music_info.shape)
print("Music Info columns:", music_info.columns.tolist())

# ---------------------------------------------------------------------------
# 2. NORMALIZE ARTIST + TRACK NAMES FOR MATCHING
#    Lowercase, strip whitespace, drop punctuation -- reduces mismatches from
#    things like "Don't Stop" vs "Dont Stop" or trailing "(Remastered)".
# ---------------------------------------------------------------------------
def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)      # collapse repeated whitespace
    return text

track_lookup["match_key"] = (
    track_lookup["artist_name"].apply(normalize) + "|" + track_lookup["track_name"].apply(normalize)
)

# Music Info.csv column names: adjust here if your file uses different
# capitalization (check the printed columns list above if this errors)
music_info["match_key"] = (
    music_info["artist"].apply(normalize) + "|" + music_info["name"].apply(normalize)
)

# ---------------------------------------------------------------------------
# 3. JOIN
# ---------------------------------------------------------------------------
metadata_cols = ["match_key", "genre", "tags"]
audio_feature_cols = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "liveness", "speechiness", "loudness", "key", "mode",
]
# only keep columns that actually exist in this copy of the file
metadata_cols = [c for c in metadata_cols if c in music_info.columns]
audio_feature_cols = [c for c in audio_feature_cols if c in music_info.columns]

merged = track_lookup.merge(
    music_info[metadata_cols + audio_feature_cols],
    on="match_key",
    how="left",
)

matched = merged.dropna(subset=["genre"]) if "genre" in merged.columns else merged.dropna(subset=metadata_cols[1:])
match_rate = len(matched) / len(merged)

print(f"\nMatched {len(matched)} / {len(merged)} tracks ({match_rate:.1%} coverage)")

# ---------------------------------------------------------------------------
# 4. BUILD COMBINED TEXT FIELD FOR EMBEDDING (Step 5)
# ---------------------------------------------------------------------------
def combine_text(row):
    parts = []
    if "genre" in row and pd.notna(row.get("genre")):
        parts.append(str(row["genre"]))
    if "tags" in row and pd.notna(row.get("tags")):
        # tags column in this dataset is often a comma or pipe separated string
        parts.append(str(row["tags"]).replace(",", " "))
    return " ".join(parts).strip()

merged["combined_text"] = merged.apply(combine_text, axis=1)
merged["has_metadata"] = merged["combined_text"].str.len() > 0

print(f"Tracks with non-empty combined_text: {merged['has_metadata'].sum()}")

# ---------------------------------------------------------------------------
# 5. SAVE
# ---------------------------------------------------------------------------
output_cols = ["track_key", "artist_name", "track_name", "combined_text", "has_metadata"] + audio_feature_cols
merged[output_cols].to_csv("matched_track_metadata.csv", index=False)

print("\nSaved matched_track_metadata.csv")
print("\nSample matched rows:")
print(merged[merged["has_metadata"]][["artist_name", "track_name", "combined_text"]].head(10).to_string(index=False))

print("\nSample UNmatched rows (these fall back to pure CF, no RAG text):")
print(merged[~merged["has_metadata"]][["artist_name", "track_name"]].head(5).to_string(index=False))

print("\nStep 4 complete (no API key used). matched_track_metadata.csv is your")
print("partial-coverage text corpus for Step 5 (embedding + text-based FAISS index).")