"""
src/models/semantic.py
Semantic recommendation using dense sentence embeddings, which capture
meaning that TF-IDF's bag-of-words representation cannot (e.g. "space
survival drama" and "astronaut stranded alone" would share little TF-IDF
vocabulary overlap but should be semantically close).

Falls back to a TF-IDF + TruncatedSVD "poor-man's embedding" if the
sentence-transformer model cannot be downloaded/loaded (e.g. no network
access to the model hub). This fallback is NOT fake — TruncatedSVD on
TF-IDF is a legitimate, if weaker, dense semantic representation (it's
essentially LSA), so the system stays honest about what it's actually
running rather than silently no-op'ing.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.models.content import _preprocess


class SemanticModel:
    def __init__(self, model_name="all-MiniLM-L6-v2", fallback_svd_components=50,
                 text_fields=("title", "genres", "tags", "overview")):
        self.model_name = model_name
        self.fallback_svd_components = fallback_svd_components
        self.text_fields = text_fields
        self.backend = None  # "sentence-transformer" or "tfidf-svd-fallback"
        self._st_model = None

    def _try_load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            return model
        except Exception as e:
            self._load_error = str(e)
            return None

    def fit(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        combined = self.movies_df.apply(
            lambda row: " ".join(_preprocess(row.get(f, "")) for f in self.text_fields
                                  if f in row), axis=1
        ).tolist()

        st_model = self._try_load_sentence_transformer()
        if st_model is not None:
            self.backend = "sentence-transformer"
            self._st_model = st_model
            self.embeddings = st_model.encode(combined, show_progress_bar=False,
                                               normalize_embeddings=True)
        else:
            self.backend = "tfidf-svd-fallback"
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf = vectorizer.fit_transform(combined)
            n_comp = min(self.fallback_svd_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
            svd = TruncatedSVD(n_components=max(n_comp, 2), random_state=42)
            self.embeddings = svd.fit_transform(tfidf)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.clip(norms, 1e-9, None)

        self.movie_index = {m: i for i, m in enumerate(self.movies_df.movieId)}
        return self

    def build_user_profile(self, ratings_df: pd.DataFrame, user_id: int, like_threshold=3.5):
        liked = ratings_df[(ratings_df.userId == user_id) & (ratings_df.rating >= like_threshold)]
        idxs = [self.movie_index[m] for m in liked.movieId if m in self.movie_index]
        if not idxs:
            return None
        weights = liked.rating.values[:len(idxs)]
        profile = np.average(self.embeddings[idxs], axis=0, weights=weights)
        return profile.reshape(1, -1)

    def score_all(self, ratings_df, user_id, candidate_movie_ids):
        profile = self.build_user_profile(ratings_df, user_id)
        if profile is None:
            return {m: 0.0 for m in candidate_movie_ids}, False
        idxs = [self.movie_index[m] for m in candidate_movie_ids if m in self.movie_index]
        if not idxs:
            return {m: 0.0 for m in candidate_movie_ids}, False
        sims = cosine_similarity(profile, self.embeddings[idxs])[0]
        scores = {m: 0.0 for m in candidate_movie_ids}
        for m, s in zip([m for m in candidate_movie_ids if m in self.movie_index], sims):
            scores[m] = float(s) * 5.0
        return scores, True
