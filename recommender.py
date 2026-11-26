"""
recommender.py
Top-level orchestrator: wires together content, semantic, collaborative,
and neural models through candidate generation -> hybrid ranking ->
diversity re-ranking -> explanation, per the target architecture.

Kept at the project root (rather than nested in src/) so `app.py` has
one obvious entry point, matching the original project's layout for
backward compatibility.
"""
import numpy as np

import config
from src.models.content import ContentBasedModel
from src.models.semantic import SemanticModel
from src.models.collaborative import CollaborativeModel
from src.models.neural_cf import NeuralCFModel
from src.models.popularity import PopularityRecommender
from src.ranking.candidate_gen import generate_candidates
from src.ranking.hybrid import weighted_fusion, normalized_fusion, LearnedRanker
from src.ranking.diversity import mmr_rerank
from src.explainability.explain import explain_recommendation


class HybridRecommender:
    def __init__(self, strategy: str = None):
        self.strategy = strategy or config.HYBRID.strategy
        self.content = ContentBasedModel(**vars(config.CONTENT))
        self.semantic = SemanticModel(**vars(config.SEMANTIC))
        self.collaborative = CollaborativeModel(**vars(config.COLLABORATIVE))
        self.neural = NeuralCFModel(**vars(config.NEURAL_CF))
        self.popularity = PopularityRecommender()
        self.learned_ranker = None

    def fit(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df

        self.content.fit(movies_df)
        self.semantic.fit(movies_df)
        self.collaborative.fit(ratings_df)
        self.neural.fit(ratings_df)
        self.popularity.fit(ratings_df)

        self.models = {
            "content": self.content, "semantic": self.semantic,
            "collaborative": self.collaborative, "neural": self.neural,
            "popularity": self.popularity,
        }

        if self.strategy == "learned":
            self._fit_learned_ranker()
        return self

    # ---------------- Learned ranker training ----------------
    def _fit_learned_ranker(self):
        """
        Builds (features, label) pairs from TRAIN interactions only.
        Label: rating >= 3.5. Features: each component's score for that
        (user, item) pair, computed from models fit on train, so nothing
        from a held-out evaluation set ever touches this fitting step.
        """
        feature_names = ["content", "semantic", "collaborative", "neural", "popularity"]
        X, y = [], []
        for user_id, group in self.ratings_df.groupby("userId"):
            item_ids = group.movieId.tolist()
            c_scores, _ = self.content.score_all(self.ratings_df, user_id, item_ids)
            s_scores, _ = self.semantic.score_all(self.ratings_df, user_id, item_ids)
            cf_scores, _ = self.collaborative.score_all(user_id, item_ids)
            n_scores, _ = self.neural.score_all(user_id, item_ids)
            pop_scores = self.popularity.score_all(item_ids)
            for _, row in group.iterrows():
                m = row.movieId
                X.append([c_scores.get(m, 0), s_scores.get(m, 0), cf_scores.get(m, 0),
                          n_scores.get(m, 0), pop_scores.get(m, 0)])
                y.append(1 if row.rating >= 3.5 else 0)

        self.learned_ranker = LearnedRanker(feature_names)
        self.learned_ranker.fit(np.array(X), np.array(y))

    # ---------------- Inference ----------------
    def _weights_for_user(self, user_id):
        n = len(self.ratings_df[self.ratings_df.userId == user_id])
        if n < config.HYBRID.cold_start_threshold:
            return config.HYBRID.cold_start_weights
        return config.HYBRID.weighted_baseline

    def recommend(self, user_id, top_k=10, exclude_seen=True, apply_diversity=None):
        apply_diversity = config.DIVERSITY.enabled if apply_diversity is None else apply_diversity
        seen = set(self.ratings_df[self.ratings_df.userId == user_id].movieId) if exclude_seen else set()

        candidates, component_scores = generate_candidates(
            self.models, self.ratings_df, self.movies_df, user_id,
            per_model_k=config.CANDIDATE_GEN.per_model_k,
            max_candidates=config.CANDIDATE_GEN.max_candidates,
            exclude_seen_ids=seen,
        )
        if not candidates:
            return []

        weights = self._weights_for_user(user_id)
        pop_scores = self.popularity.score_all(candidates)
        scored_components = {**component_scores, "popularity": pop_scores}
        # restrict each component's score dict to just the candidate set for fusion
        restricted = {name: {m: scores.get(m, 0.0) for m in candidates}
                      for name, scores in scored_components.items()}

        if self.strategy == "weighted":
            fused = weighted_fusion(restricted, weights)
        elif self.strategy == "learned" and self.learned_ranker is not None and self.learned_ranker.fitted:
            feature_names = ["content", "semantic", "collaborative", "neural", "popularity"]
            X = np.array([[restricted[f][m] for f in feature_names] for m in candidates])
            probs = self.learned_ranker.score(X)
            fused = dict(zip(candidates, probs))
        else:  # "normalized" default
            fused = normalized_fusion(restricted, weights)

        ranked = sorted(candidates, key=lambda m: fused[m], reverse=True)

        if apply_diversity:
            valid_ids = [m for m in ranked if m in self.content.movie_index]
            if valid_ids:
                idxs = [self.content.movie_index[m] for m in valid_ids]
                dense_block = self.content.tfidf_matrix[idxs].toarray()
                item_vectors = {m: dense_block[i] for i, m in enumerate(valid_ids)}
            else:
                item_vectors = {}
            final_ids = mmr_rerank(ranked, fused, item_vectors, top_k=top_k,
                                    lambda_param=config.DIVERSITY.mmr_lambda)
        else:
            final_ids = ranked[:top_k]

        component_signals = {
            "content": True, "semantic": True,
            "collaborative": not self.collaborative.is_cold_start(user_id),
            "neural": user_id in self.neural.uid_map,
        }

        results = []
        for m in final_ids:
            title = self.movies_df.loc[self.movies_df.movieId == m, "title"].values[0]
            genres = self.movies_df.loc[self.movies_df.movieId == m, "genres"].values[0]
            per_item_components = {name: restricted[name][m] for name in
                                    ["content", "semantic", "collaborative", "neural"]}
            explanation = explain_recommendation(
                user_id, m, self.movies_df, self.ratings_df,
                per_item_components, component_signals
            )
            results.append({
                "movieId": int(m), "title": title, "genres": genres,
                "score": round(float(fused[m]), 4),
                "components": {k: round(float(v), 3) for k, v in per_item_components.items()},
                "explanation": explanation,
            })
        return results
