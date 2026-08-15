"""
src/models/popularity.py
Popularity baseline: recommends the highest-rated, most-rated items to
everyone (non-personalized). Also used as the cold-start fallback for
brand-new users with zero ratings.
"""
import pandas as pd


class PopularityRecommender:
    def fit(self, ratings_df: pd.DataFrame, min_ratings: int = 3):
        stats = ratings_df.groupby("movieId").agg(
            avg_rating=("rating", "mean"), n_ratings=("rating", "count")
        ).reset_index()
        stats = stats[stats.n_ratings >= min_ratings]
        # Bayesian-ish popularity score: favors items with both high AND
        # well-supported ratings, avoiding a single 5-star rating topping the list.
        C = stats.n_ratings.mean()
        m = stats.avg_rating.mean()
        stats["score"] = (stats.n_ratings / (stats.n_ratings + C)) * stats.avg_rating + \
                          (C / (stats.n_ratings + C)) * m
        self.scores = dict(zip(stats.movieId, stats.score))
        self.ranked_items = stats.sort_values("score", ascending=False).movieId.tolist()
        return self

    def score_all(self, candidate_movie_ids):
        default = min(self.scores.values()) if self.scores else 0.0
        return {m: self.scores.get(m, default) for m in candidate_movie_ids}

    def top_k(self, k=10, exclude=None):
        exclude = exclude or set()
        return [m for m in self.ranked_items if m not in exclude][:k]
