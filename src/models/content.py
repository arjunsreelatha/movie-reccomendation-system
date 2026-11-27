"""
src/models/content.py
Content-based filtering with proper text preprocessing over multiple
metadata fields (genres, tags, overview), TF-IDF vectorization, and
cosine similarity via sparse matrix operations (no dense blow-up).
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _preprocess(text: str) -> str:
    """Lowercase, strip punctuation/pipes, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[|_/]", " ", text)          # genre separators -> spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)     # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ContentBasedModel:
    def __init__(self, text_fields=("genres", "tags", "overview"), max_features=5000):
        self.text_fields = text_fields
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    def fit(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        combined = self.movies_df.apply(
            lambda row: " ".join(_preprocess(row.get(f, "")) for f in self.text_fields
                                  if f in row), axis=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(combined)
        self.movie_index = {m: i for i, m in enumerate(self.movies_df.movieId)}
        return self

    def build_user_profile(self, ratings_df: pd.DataFrame, user_id: int, like_threshold=3.5):
        liked = ratings_df[(ratings_df.userId == user_id) & (ratings_df.rating >= like_threshold)]
        idxs = [self.movie_index[m] for m in liked.movieId if m in self.movie_index]
        if not idxs:
            return None
        weights = liked.rating.values[:len(idxs)]
        profile = np.asarray(self.tfidf_matrix[idxs].multiply(weights[:, None]).mean(axis=0))
        return profile

    def score_all(self, ratings_df, user_id, candidate_movie_ids):
        profile = self.build_user_profile(ratings_df, user_id)
        if profile is None:
            return {m: 0.0 for m in candidate_movie_ids}, False  # (scores, has_signal)
        idxs = [self.movie_index[m] for m in candidate_movie_ids if m in self.movie_index]
        if not idxs:
            return {m: 0.0 for m in candidate_movie_ids}, False
        sims = cosine_similarity(profile, self.tfidf_matrix[idxs])[0]
        scores = {m: 0.0 for m in candidate_movie_ids}
        for m, s in zip([m for m in candidate_movie_ids if m in self.movie_index], sims):
            scores[m] = float(s) * 5.0
        return scores, True

    def similar_items(self, movie_id, top_k=10):
        """Item-item similarity, used for the /similar/{movie_id} endpoint."""
        if movie_id not in self.movie_index:
            return []
        idx = self.movie_index[movie_id]
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]
        order = np.argsort(-sims)
        results = []
        for i in order:
            if i == idx:
                continue
            mid = self.movies_df.iloc[i].movieId
            results.append((int(mid), float(sims[i])))
            if len(results) >= top_k:
                break
        return results
