"""
streamlit_demo.py
Interactive demo of the hybrid recommender: personalized recommendations,
similar movies, per-component score breakdown, explanations, and a live
model/strategy comparison — built to demonstrate the system's reasoning,
not just output a list.

Run with:  streamlit run streamlit_demo.py
"""
import streamlit as st
import pandas as pd
import hashlib

import config
from src.data.loader import load_data
from recommender import HybridRecommender

st.set_page_config(page_title="Advanced Hybrid Recommender", layout="wide", page_icon="🎬")

# ---------------------------------------------------------------------------
# Visual helpers: since this is a synthetic dataset with no real poster
# images, each movie gets a deterministic gradient "poster" placeholder
# colored by its primary genre — same movie always renders the same way.
# ---------------------------------------------------------------------------
GENRE_STYLE = {
    "Action":      ("🎬", "#7B2D26", "#2B0F0C"),
    "Comedy":      ("😂", "#B8860B", "#3A2A05"),
    "Drama":       ("🎭", "#3A4E7A", "#101828"),
    "Sci-Fi":      ("🚀", "#1F6F78", "#052B2E"),
    "Romance":     ("💕", "#A63A6B", "#340F22"),
    "Horror":      ("👻", "#5B2333", "#1A0710"),
    "Thriller":    ("😱", "#4A3B7A", "#150F2B"),
    "Animation":   ("🎨", "#2E7D5B", "#0B2A1C"),
    "Documentary": ("📽️", "#556270", "#161C21"),
    "Fantasy":     ("🧙", "#6B4A9E", "#1F1633"),
}
DEFAULT_STYLE = ("🎞️", "#4A4A4A", "#141414")


def poster_html(genres_str, height="150px"):
    primary = genres_str.split("|")[0].strip() if genres_str else ""
    emoji, c1, c2 = GENRE_STYLE.get(primary, DEFAULT_STYLE)
    return (
        f'<div style="height:{height};border-radius:10px;'
        f'background:linear-gradient(135deg,{c1},{c2});'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:2.6rem;margin-bottom:0.5rem;">{emoji}</div>'
    )


def genre_badges(genres_str):
    parts = [g.strip() for g in genres_str.split("|")] if genres_str else []
    spans = "".join(
        f'<span style="background:{GENRE_STYLE.get(g, DEFAULT_STYLE)[1]};'
        f'color:#fff;padding:2px 9px;border-radius:12px;font-size:0.75rem;'
        f'margin-right:5px;">{g}</span>'
        for g in parts
    )
    return spans


@st.cache_resource
def load_and_train(strategy):
    movies, ratings = load_data(config.MOVIES_PATH, config.RATINGS_PATH)
    rec = HybridRecommender(strategy=strategy).fit(ratings, movies)
    return rec, ratings, movies


st.title("Advanced Hybrid Movie Recommender")
st.caption("Content (TF-IDF) + Semantic (embeddings) + Collaborative (SVD) + Neural CF, "
           "fused via a two-stage candidate-generation → hybrid-ranking → MMR-diversity pipeline.")

@st.cache_data
def load_genre_list():
    movies_only = pd.read_csv(config.MOVIES_PATH)
    genres = set()
    for gs in movies_only.genres.dropna():
        genres.update(g.strip() for g in gs.split("|"))
    return sorted(genres)


with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox("Fusion strategy", ["normalized", "weighted", "learned"], index=0)
    apply_diversity = st.checkbox("Apply MMR diversity re-ranking", value=True)
    mmr_lambda = st.slider("MMR λ (1.0 = pure relevance, 0.0 = pure diversity)", 0.0, 1.0, 0.7, 0.05)
    config.DIVERSITY.mmr_lambda = mmr_lambda

    st.divider()
    st.subheader("Filters")
    genre_filter = st.multiselect("Only show genres", load_genre_list())
    min_score_filter = st.slider("Minimum fused score", 0.0, 1.0, 0.0, 0.05)

with st.spinner(f"Training models (strategy={strategy})..."):
    recommender, ratings, movies = load_and_train(strategy)

st.info(f"Semantic backend in use: **{recommender.semantic.backend}** "
        f"({'sentence-transformer embeddings' if recommender.semantic.backend == 'sentence-transformer' else 'TF-IDF + SVD fallback (no network access to model hub in this environment)'})")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎬 Recommendations", "🔍 Similar Movies", "📊 System Overview", "⚖️ Compare Users"]
)

# ---------------- Tab 1: Recommendations ----------------
with tab1:
    col1, col2 = st.columns([1, 3])
    user_ids = sorted(ratings.userId.unique())

    with col1:
        user_id = st.selectbox("Select a user", user_ids)
        top_k = st.slider("Number of recommendations", 5, 20, 10)

        n_ratings = len(ratings[ratings.userId == user_id])
        is_cold = n_ratings < config.HYBRID.cold_start_threshold
        st.metric("User's rating count", n_ratings,
                  delta="Cold-start (content-heavy weights)" if is_cold else "Warm user")

        st.subheader("Rating history")
        hist = ratings[ratings.userId == user_id].merge(movies, on="movieId").sort_values(
            "rating", ascending=False)
        st.dataframe(hist[["title", "genres", "rating"]], height=300, use_container_width=True)

    with col2:
        st.subheader(f"Top {top_k} Recommendations for User {user_id}")
        recs = recommender.recommend(user_id, top_k=top_k, apply_diversity=apply_diversity)

        if genre_filter:
            recs = [r for r in recs if any(g in r["genres"].split("|") for g in genre_filter)]
        if min_score_filter > 0:
            recs = [r for r in recs if r["score"] >= min_score_filter]

        genre_counts = {}
        for r in recs:
            for g in r["genres"].split("|"):
                genre_counts[g] = genre_counts.get(g, 0) + 1
        if genre_counts:
            st.caption(f"Genre spread in this list: {dict(sorted(genre_counts.items(), key=lambda x: -x[1]))}")

        if not recs:
            st.warning("No recommendations match the current filters — try relaxing them in the sidebar.")

        # Render in a responsive grid of cards, 2 per row
        for i in range(0, len(recs), 2):
            row_recs = recs[i:i + 2]
            cols = st.columns(2)
            for col, r in zip(cols, row_recs):
                with col:
                    with st.container(border=True):
                        st.markdown(poster_html(r["genres"]), unsafe_allow_html=True)
                        top_row = st.columns([3, 1])
                        top_row[0].markdown(f"**{r['title']}**")
                        top_row[1].markdown(f"`{r['score']:.3f}`")
                        st.markdown(genre_badges(r["genres"]), unsafe_allow_html=True)
                        st.caption(r["explanation"])
                        comp = r["components"]
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        cc1.progress(min(max(comp["content"], 0) / 5, 1.0), text=f"Content {comp['content']:.2f}")
                        cc2.progress(min(max(comp["semantic"], 0) / 5, 1.0), text=f"Semantic {comp['semantic']:.2f}")
                        cc3.progress(min(comp["collaborative"] / 5, 1.0), text=f"CF {comp['collaborative']:.2f}")
                        cc4.progress(min(comp["neural"] / 5, 1.0), text=f"NCF {comp['neural']:.2f}")

# ---------------- Tab 2: Similar Movies ----------------
with tab2:
    st.subheader("Find movies similar to a given title (content-based item-item similarity)")
    movie_title = st.selectbox("Pick a movie", movies.title.tolist())
    movie_id = movies[movies.title == movie_title].movieId.values[0]
    sims = recommender.content.similar_items(movie_id, top_k=10)
    if sims:
        sim_df = pd.DataFrame([
            {"Title": movies[movies.movieId == mid].title.values[0],
             "Genres": movies[movies.movieId == mid].genres.values[0],
             "Similarity": round(score, 3)}
            for mid, score in sims
        ])
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
    else:
        st.write("No similar movies found.")

# ---------------- Tab 3: System Overview ----------------
with tab3:
    st.subheader("System statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", ratings.userId.nunique())
    c2.metric("Movies", len(movies))
    c3.metric("Ratings", len(ratings))
    c4.metric("Avg Rating", round(ratings.rating.mean(), 2))

    st.subheader("Fusion strategy in use")
    st.write(f"**{strategy}** — " + {
        "weighted": "fixed weights per component (baseline, no per-request normalization).",
        "normalized": "each component's scores are min-max normalized before weighting, "
                       "so components on different scales (e.g. cosine similarity 0-1 vs. "
                       "predicted rating 0.5-5) contribute fairly.",
        "learned": "a logistic regression trained on the training split's own interactions "
                   "predicts P(relevant) from the four component scores.",
    }[strategy])

    if strategy == "learned" and recommender.learned_ranker is not None:
        if recommender.learned_ranker.fitted:
            auc = recommender.learned_ranker.val_auc
            st.metric("Learned ranker validation AUC", f"{auc:.3f}" if auc else "n/a")
        else:
            st.warning("Learned ranker could not be fit (insufficient label diversity in training data) "
                       "— falling back to uniform scores. Try 'normalized' strategy instead.")

    st.caption("Run `python evaluate.py` in a terminal for the full offline evaluation: "
               "RMSE/MAE, Precision/Recall/F1/NDCG/HitRate/MAP@10, catalog & user coverage — "
               "compared across Random, Popularity, Content, Semantic, Collaborative, NeuralCF, "
               "Hybrid, and Hybrid+Diversity.")

# ---------------- Tab 4: Compare Users ----------------
with tab4:
    st.subheader("Compare recommendations for two different users side by side")
    st.caption("Useful for sanity-checking that recommendations are actually personalized, "
               "not the same list regardless of who's asking.")

    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        user_a = st.selectbox("User A", user_ids, index=0, key="cmp_user_a")
    with cmp_col2:
        default_b_index = 1 if len(user_ids) > 1 else 0
        user_b = st.selectbox("User B", user_ids, index=default_b_index, key="cmp_user_b")

    cmp_top_k = st.slider("Recommendations per user", 3, 10, 5, key="cmp_top_k")

    recs_a = recommender.recommend(user_a, top_k=cmp_top_k, apply_diversity=apply_diversity)
    recs_b = recommender.recommend(user_b, top_k=cmp_top_k, apply_diversity=apply_diversity)

    titles_a = {r["movieId"] for r in recs_a}
    titles_b = {r["movieId"] for r in recs_b}
    overlap = titles_a & titles_b

    st.metric("Overlap between the two lists", f"{len(overlap)} / {cmp_top_k}",
              help="Low overlap between users with different tastes is a good sign the "
                   "system is actually personalizing, not defaulting to a popularity list.")

    col_a, col_b = st.columns(2)
    for col, uid, recs in [(col_a, user_a, recs_a), (col_b, user_b, recs_b)]:
        with col:
            st.markdown(f"### User {uid}")
            n_ratings_cmp = len(ratings[ratings.userId == uid])
            st.caption(f"{n_ratings_cmp} ratings in history"
                       f"{' · cold-start' if n_ratings_cmp < config.HYBRID.cold_start_threshold else ' · warm user'}")
            for r in recs:
                highlight = r["movieId"] in overlap
                border_note = " 🔗 *also recommended to the other user*" if highlight else ""
                with st.container(border=True):
                    st.markdown(poster_html(r["genres"], height="90px"), unsafe_allow_html=True)
                    st.markdown(f"**{r['title']}** · `{r['score']:.3f}`{border_note}")
                    st.markdown(genre_badges(r["genres"]), unsafe_allow_html=True)