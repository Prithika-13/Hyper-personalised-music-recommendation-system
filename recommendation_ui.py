"""
STEP 9 — Streamlit UI

A frontend for the Step 8 FastAPI service. Run this ALONGSIDE the FastAPI
service (in a separate terminal) -- this does not replace or restart it.

Run with:
    streamlit run recommendation_ui.py

Requires:
    - Step 8's FastAPI service already running on http://127.0.0.1:8000
    - idx_to_user.pkl in the same folder (optional, just for the user dropdown)
"""

import pickle
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")

st.title("🎵 Hyper-Personalized Music Recommender")
st.caption("ALS collaborative filtering + semantic RAG + local LLM re-ranking")

# ---------------------------------------------------------------------------
# SIDEBAR — SERVICE HEALTH CHECK
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Service status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.status_code == 200 and health.json().get("artifacts_loaded"):
            st.success("Connected — artifacts loaded")
        else:
            st.warning("Service reachable but artifacts not fully loaded")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the recommendation service.\n\nMake sure uvicorn is running:\n`uvicorn recommendation_api:app --reload`")

# ---------------------------------------------------------------------------
# LOAD KNOWN USER IDs FOR CONVENIENCE (optional -- falls back to free text)
# ---------------------------------------------------------------------------
@st.cache_data
def load_user_ids():
    try:
        with open("idx_to_user.pkl", "rb") as f:
            idx_to_user = pickle.load(f)
        return sorted(idx_to_user.values())
    except FileNotFoundError:
        return []

user_ids = load_user_ids()

# ---------------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------------
with st.form("recommend_form"):
    if user_ids:
        user_id = st.selectbox("User", options=user_ids, index=0)
    else:
        user_id = st.text_input("User ID", value="user_000691")

    query = st.text_input(
        "Optional mood / context",
        placeholder="e.g. something upbeat for a workout",
    )
    top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

    submitted = st.form_submit_button("Get Recommendations", type="primary")

# ---------------------------------------------------------------------------
# CALL THE API AND DISPLAY RESULTS
# ---------------------------------------------------------------------------
if submitted:
    with st.spinner("Generating recommendations... (local LLM calls can take a bit)"):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={"user_id": user_id, "query": query or None, "top_n": top_n},
                timeout=300,
            )
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the recommendation service. Is uvicorn running on port 8000?")
            st.stop()

    if response.status_code == 404:
        st.error(response.json().get("detail", "User not found."))
    elif response.status_code != 200:
        st.error(f"Error {response.status_code}: {response.text}")
    else:
        data = response.json()
        recs = data["recommendations"]

        if not recs:
            st.warning("No recommendations returned. Try a different user or query.")
        else:
            header = f"Top {len(recs)} picks for **{data['user_id']}**"
            if data.get("query"):
                header += f' — _"{data["query"]}"_'
            st.markdown(header)
            st.divider()

            for rec in recs:
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.markdown(f"### #{rec['rank']}")
                with col2:
                    st.markdown(f"**{rec['artist']}** — {rec['track_name']}")
                    st.caption(rec["explanation"])
                st.divider()