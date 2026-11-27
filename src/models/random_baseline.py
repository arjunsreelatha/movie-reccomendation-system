"""
src/models/random_baseline.py
Random baseline: the floor every real model must beat. If a fancy model
can't outperform random recommendations on Precision@K, something is wrong.
"""
import numpy as np


class RandomRecommender:
    def fit(self, movies_df):
        self.all_movie_ids = movies_df.movieId.tolist()
        return self

    def score_all(self, candidate_movie_ids, seed=None):
        rng = np.random.default_rng(seed)
        return {m: float(rng.random()) for m in candidate_movie_ids}

    def top_k(self, k=10, exclude=None, seed=None):
        exclude = exclude or set()
        pool = [m for m in self.all_movie_ids if m not in exclude]
        rng = np.random.default_rng(seed)
        rng.shuffle(pool)
        return pool[:k]
