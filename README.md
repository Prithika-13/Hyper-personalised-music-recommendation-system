# Music Recommendation System

A research prototype for personalized music discovery built from Last.fm listening history. It combines implicit collaborative filtering with optional text based retrieval and a local large language model (LLM) reranker. The project also includes an offline evaluation suite, a FastAPI service, and a Streamlit interface.

## What it does

Given a known Last.fm user and optionally a natural language mood or activity, the system:

1. Builds confidence weighted user–track interactions from play events.
2. Retrieves personalized candidates from an ALS collaborative filtering model.
3. When a text query is supplied, retrieves additional tracks by semantic similarity to metadata text.
4. Asks a local Ollama model to choose and order tracks from the candidate pool, with a user taste summary as context.
5. Returns track names, artists, ranks, and short explanations.

Recommendations are limited to retrieved candidates; the service checks returned track keys against the candidate pool.

## Data

### Listening history

`lastfm-dataset-1K/` contains the Last.fm 1K dataset. The main input is `userid-timestamp-artid-artname-traid-traname.tsv`, a tab separated file without a header, with user ID, timestamp, artist ID/name, and track ID/name. `userid-profile.tsv` is also present, but the current pipeline does not use it.

The project does not rely on a rating scale. It infers engagement from listening events:

- A gap over 20 minutes starts a new session.
- A play is inferred to be a skip when the next event occurs in under 30 seconds.
- Replays of the same track within a session are counted as replay signals.
- Confidence combines `1 + 10*log1p(non-skip plays) + 5*log1p(replays) - 3*skip_ratio`, multiplied by an exponential recency weight with a 60 day half life. Scores are clipped to at least 0.01.

Track identity uses MusicBrainz track ID when available; otherwise it falls back to normalized artist and track names. Users and tracks with fewer than five aggregated interactions are iteratively filtered out.

### Track metadata and audio samples

`Music Info.csv` provides metadata used by Step 4, including genre/tags and audio features where available. Step 4 joins it to the Last.fm catalog and creates `matched_track_metadata.csv`. This join is partial; only tracks with usable metadata contribute text to semantic retrieval. The matching key and the actual join coverage are reported by Step 4 when run; no separate coverage summary is checked into the repository.

`MP3-Example/` contains sample audio files. The current recommendation and evaluation code does not load or analyze these audio files.

## Pipeline and scripts

Run the numbered scripts in order from the project root. The scripts communicate through files in the same directory.

| Script | Purpose |
| --- | --- |
| `prepare_interactions.py` | Parses Last.fm events, infers sessions/skips/replays, computes confidence scores, applies iterative 5-core filtering, performs a per-user chronological split, and writes sparse matrices, mappings, and a track lookup. |
| `train_als.py` | Trains and evaluates the ALS model; saves user and item latent factors. |
| `build_als_index.py` | Builds a FAISS inner-product index over ALS item factors for approximate/reusable candidate retrieval. |
| `build_track_metadata.py` | Joins track names with `Music Info.csv` and constructs text fields from available metadata. |
| `build_semantic_index.py` | Encodes metadata text with `all-MiniLM-L6-v2` and builds a FAISS semantic search index. |
| `recommend_tracks.py` | Demonstrates the local ALS + semantic retrieval + Ollama reranking pipeline. |
| `evaluate_reranking.py` | Compares raw ALS ranking with LLM reranking on users where held-out items occur in the candidate pool (Hit@10 and MRR@10). |
| `recommendation_api.py` | Implements the FastAPI recommendation service. |
| `recommendation_api_direct.py` | Alternate/direct copy of the FastAPI service implementation. |
| `recommendation_ui.py` | Implements a Streamlit UI that calls the FastAPI service. |
| `evaluate_recommenders.py` | Evaluates popularity, ALS, and the full pipeline on a reproducible sample; includes accuracy, beyond-accuracy, paired significance, and query-sensitivity measures. |
| `evaluate_recommenders_full.py` | Expanded evaluation: all eligible users for popularity/ALS; sampled users for the LLM pipeline; checkpointing and context sensitivity. |

### Training and retrieval details

Step 1 holds out the most recent 20% of each user's aggregated user–track interactions (at least one when a user has at least five interactions). The remainder becomes the training matrix. The split is chronological by `last_played`, so it approximates prediction of future interactions. The held-out list stores binary relevance pairs (user index, track index).

Step 2 trains `implicit`'s CPU `AlternatingLeastSquares` model with 64 factors, regularization 0.01, 15 iterations, and random seed 42. It uses the confidence weighted training matrix. Step 3 indexes the 64 dimensional item factors with FAISS `IndexFlatIP` and retrieves by inner product using a user's factor vector; already seen training items are filtered.

Step 5 uses Sentence Transformers model `all-MiniLM-L6-v2`. Its text index supports mood/context query retrieval and similar-track search. Step 6/8 typically combine up to 20 ALS candidates with up to 15 semantic candidates when a query is provided. A taste summary from a user's higher-confidence tracks is included in the LLM prompt. The configured local model is `llama3.1`, called through Ollama. If an LLM call fails, the evaluation scripts fall back to ALS order; the API reports malformed LLM output as an error.

## Evaluation

The repository implements these measures; each `@K` metric uses a top-10 list unless otherwise noted:

- **Precision@K** and **NDCG@K**: Step 2's `implicit` evaluation of ALS against the held-out matrix.
- **Recall@10**: fraction of a user's held-out tracks recommended in the top 10.
- **Hit@10**: whether at least one held-out track appears in the top 10.
- **NDCG@10**: discounted ranking gain for held-out tracks, normalized by ideal gain.
- **MRR@10**: reciprocal rank of the first held-out track (Step 7).
- **Personalization**: one minus average pairwise Jaccard similarity between recommendation lists.
- **Artist diversity**: fraction of unique artists within a recommendation list.
- **Catalog coverage**: fraction of the full track catalog recommended across evaluated users.
- **Wilcoxon signed-rank test**: paired comparison of ALS and full pipeline NDCG@10.
- **Context sensitivity**: overlap between recommendations under contrasting text queries; lower overlap indicates more change, but is not an accuracy score.

Step 10 samples 30 users (seed 42) for a three-system comparison. Step 11 evaluates popularity and ALS across all eligible test users, and the LLM pipeline over a reproducible sample of up to 50 users. Step 11 writes intermediate recommendation lists to `step11_checkpoint_50.pkl` every five users, so interrupted runs can resume.

### Results in this checkout

There is no checked-in evaluation log, CSV/JSON summary, or populated results table. Therefore this README does not claim numerical Precision, Recall, NDCG, Hit, diversity, or statistical significance values. The presence of `step11_checkpoint_50.pkl` confirms that 50 users have saved full-pipeline and ALS top-10 lists, but those lists alone are not a published results summary. Run `evaluate_recommenders.py` or `evaluate_recommenders_full.py` to produce current metrics in the console; Step 11 is the larger evaluation. Record the resulting output alongside the code/version and Ollama model if reporting numbers.

## Generated artifacts

Important files already present in the checkout:

| Artifact | Description |
| --- | --- |
| `interaction_matrix.npz` | Full confidence weighted user × track matrix. |
| `train_matrix.npz` | Training portion of that matrix. |
| `test_holdout.pkl` | Held-out `(user_index, track_index)` pairs. |
| `user_to_idx.pkl`, `idx_to_user.pkl` | User index mappings. |
| `track_to_idx.pkl`, `idx_to_track.pkl` | Track index mappings. |
| `track_lookup.csv` | Track key to artist and track names. |
| `als_model_user_factors.npy`, `als_model_item_factors.npy` | Trained ALS embeddings. |
| `item_faiss_index.bin` | FAISS index over ALS item factors. |
| `matched_track_metadata.csv` | Track metadata and combined text used by semantic retrieval. |
| `text_faiss_index.bin`, `text_idx_to_track.pkl`, `track_text_lookup.csv` | Semantic FAISS index and its track mappings/text. |
| `step11_checkpoint_50.pkl` | Saved per-user ALS and full-pipeline recommendation lists for resumable Step 11 evaluation. |

The CSVs and model/index binaries are generated artifacts and may be recreated by running their respective steps. The source datasets and generated assets can be large; the listening-history and audio files are not required to serve recommendations if the generated model and lookup artifacts are already available.

## Setup

Python dependencies used by the scripts include:

```text
numpy
pandas
scipy
implicit
faiss-cpu
sentence-transformers
ollama
fastapi
uvicorn
streamlit
requests
```

Install the Python packages in a virtual environment using a Python version supported by those packages. The repository does not include a pinned `requirements.txt`, so exact reproducibility requires recording package versions separately. Step 5 may download the Sentence Transformers model on first use. Install and start Ollama separately, then make the configured model available, for example with `ollama pull llama3.1`.

## Run the pipeline

From the repository root:

```powershell
python prepare_interactions.py
python train_als.py
python build_als_index.py
python build_track_metadata.py
python build_semantic_index.py
python recommend_tracks.py
```

Steps 6 and onward require a running local Ollama service and the configured model. Steps 1–5 do not require an LLM. Step 1 starts from the TSV in `lastfm-dataset-1K/`; if the derived artifacts are already present, model training and serving can start at the relevant later step.

## Run the API and UI

Start the API in one terminal:

```powershell
uvicorn recommendation_api:app --host 127.0.0.1 --port 8000
```

Start the UI in another terminal:

```powershell
streamlit run recommendation_ui.py
```

The UI calls `http://127.0.0.1:8000`. API endpoints:

- `GET /health` — service status and whether artifacts loaded.
- `POST /recommend` — accepts `{"user_id":"<Last.fm user id>","query":"optional mood/context","top_n":10}`. `top_n` is constrained to 1–25.

Example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/recommend `
  -H "Content-Type: application/json" `
  -d '{"user_id":"user_000691","query":"something upbeat for a workout","top_n":10}'
```

## Known limitations

- Skip detection uses time to the next event as a proxy; event timestamps do not directly indicate whether playback was skipped.
- Implicit feedback is constructed from historical plays and heuristics rather than explicit ratings.
- The metadata join is incomplete, so some tracks have no tags/text and cannot be retrieved through the semantic index.
- The offline evaluation uses a single per-user temporal holdout. Ranking metrics on this split do not establish future live performance.
- The LLM reranks a limited candidate pool, so it cannot recommend tracks absent from retrieval. In evaluations, its taste context differs by script: Step 10's full pipeline prompt mainly reflects the candidate list, while the serving API includes the user's taste summary and optionally semantic query candidates.
- Ollama model/runtime version and machine performance affect latency and possibly output; no dependency lockfile or environment manifest is included.
- Evaluation checkpoint contents may be stale relative to modified code or model artifacts; rerun evaluation after changes.

## Reproducibility notes

Random seeds are set to 42 in the ALS training and evaluation scripts. Step 10 samples 30 users; Step 11 samples up to 50 for LLM evaluation and checkpoints every five completions. To reproduce meaningful results, keep the dataset, generated artifacts, package versions, Ollama model, and configuration fixed, and save the complete console output from the evaluation run.
