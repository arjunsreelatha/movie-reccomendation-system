"""
src/models/collaborative.py
Matrix-factorization collaborative filtering (Truncated SVD on the
mean-centered user-item matrix), built on a SPARSE matrix so it scales
past small demo catalogs without a dense-matrix memory blow-up.

Explicit vs implicit feedback:
This dataset (and MovieLens-style data generally) contains EXPLICIT
5-point ratings. We model it as explicit feedback throughout. We do NOT
claim an implicit-feedback interpretation (e.g. treating "rated" as a
positive click signal) because that would misrepresent what the data
actually is — rating a movie 1.5 stars is explicit negative feedback, not
implicit positive feedback, and conflating the two would bias recommendations.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class CollaborativeModel:
    def __init__(self, n_factors=20, min_ratings_for_cf=5):
        self.n_factors = n_factors
        self.min_ratings_for_cf = min_ratings_for_cf
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)

    def fit(self, ratings_df: pd.DataFrame):
        self.user_ids = sorted(ratings_df.userId.unique())
        self.movie_ids = sorted(ratings_df.movieId.unique())
        self.uid_map = {u: i for i, u in enumerate(self.user_ids)}
        self.mid_map = {m: i for i, m in enumerate(self.movie_ids)}

        self.rating_counts = ratings_df.groupby("userId").size().to_dict()

        rows = ratings_df.userId.map(self.uid_map).values
        cols = ratings_df.movieId.map(self.mid_map).values
        vals = ratings_df.rating.values
        sparse_matrix = csr_matrix((vals, (rows, cols)),
                                    shape=(len(self.user_ids), len(self.movie_ids)))

        self.global_mean = float(ratings_df.rating.mean())
        counts = np.diff(sparse_matrix.tocsr().indptr)
        sums = np.asarray(sparse_matrix.sum(axis=1)).flatten()
        self.user_means = np.divide(sums, counts, out=np.full(len(sums), self.global_mean),
                                     where=counts != 0)

        # mean-center only the observed (non-zero) entries
        centered = sparse_matrix.tolil()
        for i in range(centered.shape[0]):
            row_idx = centered.rows[i]
            if row_idx:
                centered.data[i] = [v - self.user_means[i] for v in centered.data[i]]
        centered = centered.tocsr()

        n_comp = min(self.n_factors, min(centered.shape) - 1)
        self.svd = TruncatedSVD(n_components=max(n_comp, 2), random_state=42)
        self.user_factors = self.svd.fit_transform(centered)
        self.item_factors = self.svd.components_.T
        return self

    def is_cold_start(self, user_id) -> bool:
        return self.rating_counts.get(user_id, 0) < self.min_ratings_for_cf

    def predict(self, user_id, movie_id) -> float:
        if user_id not in self.uid_map or movie_id not in self.mid_map:
            return self.global_mean
        u, m = self.uid_map[user_id], self.mid_map[movie_id]
        pred = self.user_means[u] + float(np.dot(self.user_factors[u], self.item_factors[m]))
        return float(np.clip(pred, 0.5, 5.0))

    def score_all(self, user_id, candidate_movie_ids):
        """Vectorized batch prediction across all candidates at once."""
        has_signal = user_id in self.uid_map and not self.is_cold_start(user_id)
        if not candidate_movie_ids:
            return {}, has_signal
        if user_id not in self.uid_map:
            return {m: self.global_mean for m in candidate_movie_ids}, has_signal

        u = self.uid_map[user_id]
        known = [m for m in candidate_movie_ids if m in self.mid_map]
        unknown = [m for m in candidate_movie_ids if m not in self.mid_map]
        scores = {m: self.global_mean for m in unknown}
        if known:
            item_idxs = [self.mid_map[m] for m in known]
            preds = self.user_means[u] + self.item_factors[item_idxs] @ self.user_factors[u]
            preds = np.clip(preds, 0.5, 5.0)
            scores.update({m: float(p) for m, p in zip(known, preds)})
        return scores, has_signal
