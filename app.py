"""
app.py
FastAPI backend for the hybrid recommender. Models are trained ONCE at
startup (module import time) and reused across requests — never retrained
per-request.

Run with:  uvicorn app:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive API docs.
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

import config
from src.data.loader import load_data
from recommender import HybridRecommender

app = FastAPI(
    title="Advanced Hybrid Recommendation System",
    description="Content + Semantic + Collaborative Filtering + Neural CF, "
                "fused via configurable hybrid ranking with MMR diversity re-ranking.",
    version="2.0.0",
)

# ---------------- Pydantic schemas ----------------

class RecommendationComponent(BaseModel):
    content: float
    semantic: float
    collaborative: float
    neural: float


class Recommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    score: float
    components: RecommendationComponent
    explanation: str


class RecommendResponse(BaseModel):
    userId: int
    strategy: str
    recommendations: List[Recommendation]


class RecommendRequest(BaseModel):
    userId: int
    top_k: int = Field(default=10, ge=1, le=50)
    apply_diversity: Optional[bool] = None
    strategy: Optional[str] = Field(
        default=None, description="weighted | normalized | learned | xgboost"
    )


class SimilarMovie(BaseModel):
    movieId: int
    title: str
    genres: str
    similarity: float


class HealthResponse(BaseModel):
    status: str
    semantic_backend: str
    n_users: int
    n_movies: int


# ---------------- Startup: train once ----------------

movies, ratings = load_data(config.MOVIES_PATH, config.RATINGS_PATH)
print("Training recommender on startup (this may take a moment)...")
recommender = HybridRecommender(strategy=config.HYBRID.strategy).fit(ratings, movies)
print(f"Model ready. Semantic backend: {recommender.semantic.backend}")


# ---------------- Endpoints ----------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        semantic_backend=recommender.semantic.backend,
        n_users=ratings.userId.nunique(),
        n_movies=len(movies),
    )


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    row = movies[movies.movieId == movie_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    return row.iloc[0].to_dict()


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, top_k: int = 10, apply_diversity: Optional[bool] = None):
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    recs = recommender.recommend(user_id, top_k=top_k, apply_diversity=apply_diversity)
    return RecommendResponse(userId=user_id, strategy=recommender.strategy, recommendations=recs)


@app.post("/recommend", response_model=RecommendResponse)
def recommend_post(req: RecommendRequest):
    rec_engine = recommender
    if req.strategy and req.strategy != recommender.strategy:
        # Build a differently-strategized recommender on demand (cheap: reuses fit models
        # would require refactor; for now, re-fit — acceptable for a demo-scale API)
        rec_engine = HybridRecommender(strategy=req.strategy).fit(ratings, movies)
    recs = rec_engine.recommend(req.userId, top_k=req.top_k, apply_diversity=req.apply_diversity)
    return RecommendResponse(userId=req.userId, strategy=rec_engine.strategy, recommendations=recs)


@app.get("/similar/{movie_id}", response_model=List[SimilarMovie])
def similar_movies(movie_id: int, top_k: int = 10):
    if movie_id not in recommender.content.movie_index:
        raise HTTPException(status_code=404, detail="Movie not found")
    sims = recommender.content.similar_items(movie_id, top_k=top_k)
    results = []
    for mid, score in sims:
        row = movies[movies.movieId == mid].iloc[0]
        results.append(SimilarMovie(movieId=mid, title=row.title, genres=row.genres,
                                     similarity=round(score, 4)))
    return results


@app.get("/explain/{user_id}/{movie_id}")
def explain(user_id: int, movie_id: int):
    if movie_id not in recommender.content.movie_index:
        raise HTTPException(status_code=404, detail="Movie not found")
    candidates = [movie_id]
    c_scores, _ = recommender.content.score_all(recommender.ratings_df, user_id, candidates)
    s_scores, _ = recommender.semantic.score_all(recommender.ratings_df, user_id, candidates)
    cf_scores, _ = recommender.collaborative.score_all(user_id, candidates)
    n_scores, _ = recommender.neural.score_all(user_id, candidates)
    from src.explainability.explain import explain_recommendation
    component_signals = {
        "content": True, "semantic": True,
        "collaborative": not recommender.collaborative.is_cold_start(user_id),
        "neural": user_id in recommender.neural.uid_map,
    }
    component_scores = {"content": c_scores[movie_id], "semantic": s_scores[movie_id],
                         "collaborative": cf_scores[movie_id], "neural": n_scores[movie_id]}
    explanation = explain_recommendation(user_id, movie_id, movies, recommender.ratings_df,
                                          component_scores, component_signals)
    return {"userId": user_id, "movieId": movie_id, "explanation": explanation,
            "components": component_scores}


@app.get("/metrics")
def system_metrics():
    """Lightweight live system stats (not a full offline eval — run evaluate.py for that)."""
    return {
        "n_users": ratings.userId.nunique(),
        "n_movies": len(movies),
        "n_ratings": len(ratings),
        "avg_rating": round(float(ratings.rating.mean()), 3),
        "semantic_backend": recommender.semantic.backend,
        "hybrid_strategy": recommender.strategy,
        "cold_start_threshold": config.HYBRID.cold_start_threshold,
    }


@app.get("/user/{user_id}/history")
def user_history(user_id: int):
    hist = ratings[ratings.userId == user_id].merge(movies, on="movieId")
    hist = hist.sort_values("rating", ascending=False)
    return {"userId": user_id, "ratings_count": len(hist),
            "history": hist[["movieId", "title", "genres", "rating"]].to_dict(orient="records")}


@app.get("/users")
def list_users():
    return {"user_ids": sorted(ratings.userId.unique().tolist())}
