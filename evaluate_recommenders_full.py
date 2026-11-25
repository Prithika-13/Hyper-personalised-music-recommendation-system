"""
STEP 11 — Efficient Full-System Evaluation

Evaluation strategy
-------------------
TIER 1:
    Popularity + ALS evaluated on ALL test users.
    No LLM calls.

TIER 2:
    Full pipeline evaluated on a reproducible random sample of users.
    Default = 50 users.

Evaluates:
    Accuracy:
        - Recall@10
        - NDCG@10
        - Hit@10

    Beyond accuracy:
        - Personalization
        - Artist diversity
        - Catalog coverage

    Statistical comparison:
        - Full vs ALS NDCG@10
        - Win / Tie / Loss
        - Wilcoxon signed-rank test

    Context-awareness demonstration:
        - Compare recommendations for "workout" vs "late night"
        - Measures recommendation-list overlap

Checkpointing is used so expensive LLM evaluation can resume.
"""

# ============================================================
# 0. CONFIGURATION
# ============================================================

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import pickle
import time

import numpy as np
import pandas as pd
import faiss
import ollama

from scipy.sparse import load_npz
from scipy.stats import wilcoxon


OLLAMA_MODEL = "llama3.1"

K = 10

# Number of ALS candidates passed to LLM
ALS_POOL_FOR_FULL_PIPELINE = 30

# 50 is sufficient for a practical project-level evaluation.
LLM_SAMPLE_USERS = 50

# Separate checkpoint so it does not mix with old Step 11 runs.
CHECKPOINT_FILE = "step11_checkpoint_50.pkl"

CHECKPOINT_EVERY = 5

RANDOM_SEED = 42

# Number of users for context-sensitivity evaluation.
CONTEXT_SAMPLE_USERS = 10

CONTEXT_QUERY_1 = "upbeat energetic music for a workout"
CONTEXT_QUERY_2 = "calm atmospheric music for late night"


# ============================================================
# 1. LOAD ARTIFACTS
# ============================================================

print("=" * 70)
print("STEP 11 — EFFICIENT FULL-SYSTEM EVALUATION")
print("=" * 70)

train_matrix = load_npz(
    "train_matrix.npz"
).tocsr()

user_factors = np.load(
    "als_model_user_factors.npy"
).astype(np.float32)


with open("test_holdout.pkl", "rb") as f:
    test_holdout_raw = pickle.load(f)

with open("user_to_idx.pkl", "rb") as f:
    user_to_idx = pickle.load(f)

with open("idx_to_user.pkl", "rb") as f:
    idx_to_user = pickle.load(f)

with open("idx_to_track.pkl", "rb") as f:
    idx_to_track = pickle.load(f)


track_lookup = (
    pd.read_csv("track_lookup.csv")
    .set_index("track_key")
)

als_index = faiss.read_index(
    "item_faiss_index.bin"
)


try:

    track_text_lookup = (
        pd.read_csv("track_text_lookup.csv")
        .set_index("track_key")
    )

except FileNotFoundError:

    print(
        "WARNING: track_text_lookup.csv not found. "
        "Semantic tags will not be available."
    )

    track_text_lookup = pd.DataFrame(
        columns=["combined_text"]
    )


n_users, n_items = train_matrix.shape


# Build holdout dictionary
holdout_by_user = {}

for u, i in test_holdout_raw:

    holdout_by_user.setdefault(
        int(u),
        set()
    ).add(int(i))


all_users = list(
    holdout_by_user.keys()
)


print(
    f"\nLoaded {n_users:,} users and "
    f"{n_items:,} items."
)

print(
    f"Test users available: "
    f"{len(all_users):,}"
)


# ============================================================
# 2. LOOKUP HELPERS
# ============================================================

def track_display_name(track_key):

    if track_key in track_lookup.index:

        row = track_lookup.loc[track_key]

        return (
            row["artist_name"],
            row["track_name"]
        )

    return "Unknown Artist", str(track_key)


def get_tags(track_key):

    if track_key in track_text_lookup.index:

        value = track_text_lookup.loc[
            track_key,
            "combined_text"
        ]

        if pd.notna(value):
            return str(value)

    return None


# ============================================================
# 3. POPULARITY BASELINE
# ============================================================

item_popularity = np.asarray(
    train_matrix.sum(axis=0)
).flatten()

GLOBAL_POPULAR_ORDER = np.argsort(
    -item_popularity
)


def get_popularity_topk(
    user_idx,
    k=K
):

    seen = set(
        train_matrix
        .getrow(user_idx)
        .indices
    )

    result = []

    for item_idx in GLOBAL_POPULAR_ORDER:

        item_idx = int(item_idx)

        if item_idx in seen:
            continue

        result.append(item_idx)

        if len(result) >= k:
            break

    return result


# ============================================================
# 4. ALS RECOMMENDATION
# ============================================================

def get_als_topk(
    user_idx,
    k=K,
    initial_pool=200
):

    user_vector = (
        user_factors[user_idx]
        .reshape(1, -1)
    )

    seen = set(
        train_matrix
        .getrow(user_idx)
        .indices
    )

    search_pool = min(
        initial_pool,
        n_items
    )

    while True:

        _, item_indices = als_index.search(
            user_vector,
            search_pool
        )

        item_indices = item_indices[0]

        result = [
            int(idx)
            for idx in item_indices
            if idx != -1
            and idx not in seen
        ]

        if (
            len(result) >= k
            or search_pool >= n_items
        ):

            return result[:k]

        search_pool = min(
            search_pool * 4,
            n_items
        )


# ============================================================
# 5. FULL PIPELINE — LLM RE-RANKING
# ============================================================

def get_full_pipeline_topk(
    user_idx,
    k=K,
    context_query=None
):

    pool_items = get_als_topk(
        user_idx,
        k=ALS_POOL_FOR_FULL_PIPELINE
    )

    if not pool_items:
        return []

    candidate_lines = []

    key_by_line = {}

    for item_idx in pool_items:

        track_key = idx_to_track[item_idx]

        artist, name = track_display_name(
            track_key
        )

        tags = (
            get_tags(track_key)
            or "no tags available"
        )

        candidate_lines.append(
            f"- track_key: {track_key} | "
            f"{artist} - {name} | "
            f"tags: {tags}"
        )

        key_by_line[str(track_key)] = item_idx


    if context_query:

        context_instruction = f"""
The listener has also provided this context:

"{context_query}"

Use this context when ranking the tracks.
"""

    else:

        context_instruction = """
Rank the tracks according to the listener's
taste implied by the candidate pool.
"""


    prompt = f"""
You are ranking music recommendations.

Select and rank the best {k} tracks from the
candidate list.

{context_instruction}

IMPORTANT RULES:

1. You MUST only use track_key values that
   appear EXACTLY in the candidate list.

2. Do not invent songs.

3. Return exactly {k} recommendations.

Candidates:

{chr(10).join(candidate_lines)}

Respond ONLY with valid JSON in this format:

{{
    "recommendations": [
        {{
            "track_key": "<exact track_key>",
            "rank": 1
        }}
    ]
}}
"""


    try:

        response = ollama.chat(

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            format="json",

            options={
                "temperature": 0.2
            }
        )


        parsed = json.loads(
            response["message"]["content"]
        )

        recommendations = parsed.get(
            "recommendations",
            []
        )

        ranked = sorted(
            recommendations,
            key=lambda x: x.get(
                "rank",
                999
            )
        )


        result = []

        for rec in ranked:

            track_key = str(
                rec.get(
                    "track_key",
                    ""
                )
            )

            if track_key in key_by_line:

                item_idx = key_by_line[
                    track_key
                ]

                if item_idx not in result:

                    result.append(
                        item_idx
                    )


        # If LLM returns fewer than K valid
        # tracks, fill remaining positions
        # using ALS candidates.

        if len(result) < k:

            for item_idx in pool_items:

                if item_idx not in result:

                    result.append(
                        item_idx
                    )

                if len(result) >= k:
                    break


        return result[:k]


    except Exception as e:

        user_name = idx_to_user.get(
            user_idx,
            str(user_idx)
        )

        print(
            f"\nLLM call failed for "
            f"{user_name}: "
            f"{type(e).__name__}: {e}"
        )

        # Safe fallback
        return pool_items[:k]


# ============================================================
# 6. ACCURACY METRICS
# ============================================================

def recall_at_k(
    recommended,
    relevant,
    k=K
):

    if not relevant:
        return None

    hits = len(
        set(recommended[:k])
        & relevant
    )

    return hits / len(relevant)


def hit_at_k(
    recommended,
    relevant,
    k=K
):

    if not relevant:
        return None

    return (
        1.0
        if set(recommended[:k]) & relevant
        else 0.0
    )


def ndcg_at_k(
    recommended,
    relevant,
    k=K
):

    if not relevant:
        return None

    dcg = sum(

        1.0 / np.log2(i + 2)

        for i, item
        in enumerate(
            recommended[:k]
        )

        if item in relevant
    )


    ideal_hits = min(
        len(relevant),
        k
    )


    idcg = sum(

        1.0 / np.log2(i + 2)

        for i in range(
            ideal_hits
        )
    )


    if idcg == 0:
        return 0.0

    return dcg / idcg


def safe_mean(values):

    clean = [
        v
        for v in values
        if v is not None
    ]

    if not clean:
        return float("nan")

    return float(
        np.mean(clean)
    )


# ============================================================
# 7. BEYOND-ACCURACY METRICS
# ============================================================

def personalization(
    recommendation_lists
):

    """
    Measures how different recommendation
    lists are across users.

    0 -> everyone gets identical lists
    1 -> lists are completely different
    """

    if len(recommendation_lists) < 2:
        return float("nan")

    similarities = []

    for i in range(
        len(recommendation_lists)
    ):

        a = set(
            recommendation_lists[i]
        )

        for j in range(
            i + 1,
            len(recommendation_lists)
        ):

            b = set(
                recommendation_lists[j]
            )

            union = a | b

            if not union:
                continue

            jaccard = (
                len(a & b)
                / len(union)
            )

            similarities.append(
                jaccard
            )


    if not similarities:
        return float("nan")

    return 1.0 - float(
        np.mean(similarities)
    )


def artist_diversity(
    recommendation_lists
):

    """
    Average fraction of unique artists
    inside each user's Top-K list.
    """

    scores = []

    for recs in recommendation_lists:

        if not recs:
            continue

        artists = []

        for item_idx in recs:

            track_key = idx_to_track[
                item_idx
            ]

            artist, _ = (
                track_display_name(
                    track_key
                )
            )

            artists.append(
                str(artist)
            )


        unique_artists = len(
            set(artists)
        )

        scores.append(
            unique_artists
            / len(artists)
        )


    return safe_mean(
        scores
    )


def catalog_coverage(
    recommendation_lists
):

    """
    Fraction of entire item catalog that
    appears in recommendation lists.
    """

    recommended_items = set()

    for recs in recommendation_lists:

        recommended_items.update(
            recs
        )

    if n_items == 0:
        return 0.0

    return (
        len(recommended_items)
        / n_items
    )


# ============================================================
# 8. TIER 1 — ALL USERS
# ============================================================

print("\n" + "=" * 70)
print(
    "TIER 1 — POPULARITY + ALS "
    "ON ALL TEST USERS"
)
print("=" * 70)


pop_metrics = {
    "recall": [],
    "ndcg": [],
    "hit": []
}

als_metrics = {
    "recall": [],
    "ndcg": [],
    "hit": []
}


pop_lists = []
als_all_lists = []


start = time.time()


for count, user_idx in enumerate(
    all_users,
    start=1
):

    relevant = holdout_by_user[
        user_idx
    ]

    pop_list = (
        get_popularity_topk(
            user_idx
        )
    )

    als_list = (
        get_als_topk(
            user_idx
        )
    )


    pop_lists.append(
        pop_list
    )

    als_all_lists.append(
        als_list
    )


    pop_metrics["recall"].append(
        recall_at_k(
            pop_list,
            relevant
        )
    )

    pop_metrics["ndcg"].append(
        ndcg_at_k(
            pop_list,
            relevant
        )
    )

    pop_metrics["hit"].append(
        hit_at_k(
            pop_list,
            relevant
        )
    )


    als_metrics["recall"].append(
        recall_at_k(
            als_list,
            relevant
        )
    )

    als_metrics["ndcg"].append(
        ndcg_at_k(
            als_list,
            relevant
        )
    )

    als_metrics["hit"].append(
        hit_at_k(
            als_list,
            relevant
        )
    )


elapsed = time.time() - start


print(
    f"\nEvaluated "
    f"{len(all_users):,} users "
    f"in {elapsed:.1f}s."
)


print(
    f"\n{'System':<15}"
    f"{'Recall@10':<15}"
    f"{'NDCG@10':<15}"
    f"{'Hit@10':<15}"
)


print(
    f"{'Popularity':<15}"
    f"{safe_mean(pop_metrics['recall']):<15.4f}"
    f"{safe_mean(pop_metrics['ndcg']):<15.4f}"
    f"{safe_mean(pop_metrics['hit']):<15.4f}"
)


print(
    f"{'ALS':<15}"
    f"{safe_mean(als_metrics['recall']):<15.4f}"
    f"{safe_mean(als_metrics['ndcg']):<15.4f}"
    f"{safe_mean(als_metrics['hit']):<15.4f}"
)


# ============================================================
# 9. SELECT RANDOM USERS FOR FULL PIPELINE
# ============================================================

print("\n" + "=" * 70)
print(
    "TIER 2 — FULL PIPELINE"
)
print("=" * 70)


rng = np.random.default_rng(
    RANDOM_SEED
)


sample_size = min(
    LLM_SAMPLE_USERS,
    len(all_users)
)


sampled_users = list(
    rng.choice(
        all_users,
        size=sample_size,
        replace=False
    )
)


print(
    f"\nRandomly selected "
    f"{len(sampled_users)} users "
    f"(seed={RANDOM_SEED})."
)


# ============================================================
# 10. LOAD CHECKPOINT
# ============================================================

if os.path.exists(
    CHECKPOINT_FILE
):

    with open(
        CHECKPOINT_FILE,
        "rb"
    ) as f:

        checkpoint = pickle.load(
            f
        )


    # Only keep users belonging to
    # current random sample.

    checkpoint = {

        int(u): value

        for u, value
        in checkpoint.items()

        if int(u) in sampled_users
    }


    print(
        f"Checkpoint found: "
        f"{len(checkpoint)} / "
        f"{len(sampled_users)} "
        f"users already evaluated."
    )

else:

    checkpoint = {}

    print(
        "No checkpoint found. "
        "Starting fresh."
    )


remaining_users = [

    u
    for u in sampled_users
    if u not in checkpoint
]


print(
    f"Remaining users: "
    f"{len(remaining_users)}"
)


# ============================================================
# 11. RUN FULL PIPELINE
# ============================================================

if remaining_users:

    print(
        "\nStarting LLM evaluation..."
    )


    start_time = time.time()


    for n, user_idx in enumerate(
        remaining_users,
        start=1
    ):

        user_start = time.time()


        full_list = (
            get_full_pipeline_topk(
                user_idx
            )
        )


        als_list = (
            get_als_topk(
                user_idx
            )
        )


        checkpoint[
            user_idx
        ] = {

            "full": full_list,

            "als": als_list
        }


        user_elapsed = (
            time.time()
            - user_start
        )


        print(
            f"[{n}/{len(remaining_users)}] "
            f"{idx_to_user.get(user_idx, user_idx)} "
            f"done "
            f"({user_elapsed:.1f}s)"
        )


        if (
            n % CHECKPOINT_EVERY == 0
            or n == len(
                remaining_users
            )
        ):

            with open(
                CHECKPOINT_FILE,
                "wb"
            ) as f:

                pickle.dump(
                    checkpoint,
                    f
                )


            elapsed = (
                time.time()
                - start_time
            )

            avg_time = (
                elapsed / n
            )

            remaining_count = (
                len(remaining_users)
                - n
            )

            eta = (
                avg_time
                * remaining_count
                / 60
            )


            print(
                f"   Checkpoint saved | "
                f"Avg {avg_time:.1f}s/user | "
                f"ETA {eta:.1f} min"
            )


else:

    print(
        "\nAll sampled users already "
        "exist in checkpoint."
    )


# ============================================================
# 12. AGGREGATE FULL PIPELINE RESULTS
# ============================================================

full_metrics = {
    "recall": [],
    "ndcg": [],
    "hit": []
}

als_subset_metrics = {
    "recall": [],
    "ndcg": [],
    "hit": []
}


full_lists = []
als_subset_lists = []

paired_ndcg = []


for user_idx in sampled_users:

    if user_idx not in checkpoint:
        continue

    relevant = holdout_by_user[
        user_idx
    ]

    entry = checkpoint[
        user_idx
    ]

    full_list = entry[
        "full"
    ]

    als_list = entry[
        "als"
    ]


    full_lists.append(
        full_list
    )

    als_subset_lists.append(
        als_list
    )


    full_metrics["recall"].append(
        recall_at_k(
            full_list,
            relevant
        )
    )

    full_metrics["ndcg"].append(
        ndcg_at_k(
            full_list,
            relevant
        )
    )

    full_metrics["hit"].append(
        hit_at_k(
            full_list,
            relevant
        )
    )


    als_subset_metrics[
        "recall"
    ].append(

        recall_at_k(
            als_list,
            relevant
        )
    )


    als_subset_metrics[
        "ndcg"
    ].append(

        ndcg_at_k(
            als_list,
            relevant
        )
    )


    als_subset_metrics[
        "hit"
    ].append(

        hit_at_k(
            als_list,
            relevant
        )
    )


    a_ndcg = ndcg_at_k(
        als_list,
        relevant
    )

    f_ndcg = ndcg_at_k(
        full_list,
        relevant
    )


    if (
        a_ndcg is not None
        and f_ndcg is not None
    ):

        paired_ndcg.append(
            (
                a_ndcg,
                f_ndcg
            )
        )


evaluated_n = len(
    full_lists
)


print("\n" + "=" * 70)
print(
    f"TIER 2 RESULTS "
    f"(n={evaluated_n})"
)
print("=" * 70)


print(
    f"\n{'System':<15}"
    f"{'Recall@10':<15}"
    f"{'NDCG@10':<15}"
    f"{'Hit@10':<15}"
)


print(
    f"{'ALS':<15}"
    f"{safe_mean(als_subset_metrics['recall']):<15.4f}"
    f"{safe_mean(als_subset_metrics['ndcg']):<15.4f}"
    f"{safe_mean(als_subset_metrics['hit']):<15.4f}"
)


print(
    f"{'Full':<15}"
    f"{safe_mean(full_metrics['recall']):<15.4f}"
    f"{safe_mean(full_metrics['ndcg']):<15.4f}"
    f"{safe_mean(full_metrics['hit']):<15.4f}"
)


# ============================================================
# 13. BEYOND-ACCURACY EVALUATION
# ============================================================

print("\n" + "=" * 70)
print(
    "BEYOND-ACCURACY METRICS"
)
print("=" * 70)


pop_subset_lists = [

    get_popularity_topk(u)

    for u in sampled_users
    if u in checkpoint
]


systems = {

    "Popularity":
        pop_subset_lists,

    "ALS":
        als_subset_lists,

    "Full":
        full_lists
}


print(
    f"\n{'System':<15}"
    f"{'Personalization':<20}"
    f"{'Artist Diversity':<20}"
    f"{'Catalog Coverage':<20}"
)


for name, lists in systems.items():

    p = personalization(
        lists
    )

    d = artist_diversity(
        lists
    )

    c = catalog_coverage(
        lists
    )


    print(
        f"{name:<15}"
        f"{p:<20.4f}"
        f"{d:<20.4f}"
        f"{c:<20.5f}"
    )


# ============================================================
# 14. FULL VS ALS — WIN / TIE / LOSS
# ============================================================

wins = sum(

    1
    for als_score, full_score
    in paired_ndcg

    if full_score > als_score
)


losses = sum(

    1
    for als_score, full_score
    in paired_ndcg

    if full_score < als_score
)


ties = sum(

    1
    for als_score, full_score
    in paired_ndcg

    if full_score == als_score
)


print("\n" + "=" * 70)
print(
    "FULL PIPELINE VS ALS"
)
print("=" * 70)


print(
    f"\nNDCG@10 comparison:"
)

print(
    f"Improved : {wins}"
)

print(
    f"Worse    : {losses}"
)

print(
    f"Tied     : {ties}"
)


# ============================================================
# 15. STATISTICAL SIGNIFICANCE
# ============================================================

if (
    len(paired_ndcg) >= 5
    and (wins + losses) > 0
):

    als_values = [
        x[0]
        for x in paired_ndcg
    ]

    full_values = [
        x[1]
        for x in paired_ndcg
    ]


    try:

        statistic, p_value = (
            wilcoxon(
                als_values,
                full_values
            )
        )


        print(
            "\nPaired Wilcoxon "
            "signed-rank test:"
        )

        print(
            f"Statistic = "
            f"{statistic:.3f}"
        )

        print(
            f"p-value   = "
            f"{p_value:.4f}"
        )


        if p_value < 0.05:

            print(
                "Result: statistically "
                "significant at p < 0.05."
            )

        else:

            print(
                "Result: not statistically "
                "significant at p < 0.05."
            )


    except ValueError as e:

        print(
            f"\nWilcoxon test could "
            f"not be computed: {e}"
        )


# ============================================================
# 16. CONTEXT-SENSITIVITY EVALUATION
# ============================================================

print("\n" + "=" * 70)
print(
    "CONTEXT-SENSITIVITY DEMONSTRATION"
)
print("=" * 70)


context_users = sampled_users[
    :min(
        CONTEXT_SAMPLE_USERS,
        len(sampled_users)
    )
]


context_overlaps = []


print(
    f"\nComparing:\n"
    f'  Query A: "{CONTEXT_QUERY_1}"\n'
    f'  Query B: "{CONTEXT_QUERY_2}"\n'
)


for n, user_idx in enumerate(
    context_users,
    start=1
):

    print(
        f"[{n}/{len(context_users)}] "
        f"{idx_to_user.get(user_idx, user_idx)}"
    )


    workout_list = (
        get_full_pipeline_topk(
            user_idx,
            context_query=CONTEXT_QUERY_1
        )
    )


    night_list = (
        get_full_pipeline_topk(
            user_idx,
            context_query=CONTEXT_QUERY_2
        )
    )


    set_a = set(
        workout_list
    )

    set_b = set(
        night_list
    )


    # Top-K overlap
    overlap = (
        len(set_a & set_b)
        / K
    )


    context_overlaps.append(
        overlap
    )


    print(
        f"   Recommendation overlap: "
        f"{overlap * 100:.1f}%"
    )


mean_context_overlap = safe_mean(
    context_overlaps
)


print(
    "\nAverage overlap between "
    "contrasting contexts:"
)

print(
    f"{mean_context_overlap * 100:.1f}%"
)


print(
    "Average recommendation change:"
)

print(
    f"{(1 - mean_context_overlap) * 100:.1f}%"
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(
    "FINAL EVALUATION SUMMARY"
)
print("=" * 70)


print(
    f"\nBaseline population: "
    f"{len(all_users):,} users"
)

print(
    f"Full-pipeline sample: "
    f"{evaluated_n} users"
)


print(
    "\nFULL PIPELINE ACCURACY"
)

print(
    f"Recall@10 : "
    f"{safe_mean(full_metrics['recall']):.4f}"
)

print(
    f"NDCG@10   : "
    f"{safe_mean(full_metrics['ndcg']):.4f}"
)

print(
    f"Hit@10    : "
    f"{safe_mean(full_metrics['hit']):.4f}"
)


if full_lists:

    print(
        "\nFULL PIPELINE "
        "BEYOND-ACCURACY"
    )

    print(
        f"Personalization  : "
        f"{personalization(full_lists):.4f}"
    )

    print(
        f"Artist diversity : "
        f"{artist_diversity(full_lists):.4f}"
    )

    print(
        f"Catalog coverage : "
        f"{catalog_coverage(full_lists):.5f}"
    )


if context_overlaps:

    print(
        "\nCONTEXT AWARENESS"
    )

    print(
        f"Average list overlap : "
        f"{mean_context_overlap * 100:.1f}%"
    )

    print(
        f"Average list change  : "
        f"{(1 - mean_context_overlap) * 100:.1f}%"
    )


print(
    "\nEvaluation complete."
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT_FILE}"
)