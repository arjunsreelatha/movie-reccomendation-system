import numpy as np
import pandas as pd

import config
from download_movielens import convert_dataset
from recommender import HybridRecommender
from src.ranking.hybrid import XGBoostRanker


def test_xgboost_ranker_scores_positive_examples_higher():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    ranker = XGBoostRanker(
        ["content", "semantic", "collaborative"],
        n_estimators=20,
        max_depth=2,
        n_jobs=1,
        seed=42,
    )

    ranker.fit(X, y)
    scores = ranker.score(X)

    assert ranker.fitted
    assert scores.shape == (len(y),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_xgboost_ranker_stays_unfitted_without_both_labels():
    ranker = XGBoostRanker(["feature"], n_jobs=1)

    ranker.fit(np.ones((4, 1)), np.ones(4, dtype=int))

    assert not ranker.fitted


def test_movielens_converter_maps_source_fields():
    movie_data = b"1::Toy Story (1995)::Animation|Children's\n"
    rating_data = b"7::1::4.5::978300760\n"

    movies, ratings = convert_dataset(movie_data, rating_data)

    assert movies.loc[0, ["movieId", "title", "genres"]].tolist() == [
        1, "Toy Story (1995)", "Animation|Children's"
    ]
    assert movies.loc[0, "tags"] == ""
    assert ratings.loc[0].tolist() == [7, 1, 4.5, 978300760]


def test_recommender_builds_xgboost_features_per_user(monkeypatch):
    class UserScoreModel:
        def score_all(self, ratings, user_id, item_ids):
            assert ratings.userId.unique().tolist() == [user_id]
            return {item_id: float(item_id % 3) for item_id in item_ids}, True

    class ItemScoreModel:
        def score_all(self, *args):
            item_ids = args[-1]
            return {item_id: float(item_id % 2) for item_id in item_ids}, True

    class PopularityScoreModel:
        def score_all(self, item_ids):
            return {item_id: float(item_id % 2) for item_id in item_ids}

    ratings = pd.DataFrame([
        {"userId": user_id, "movieId": item_id,
         "rating": 5.0 if item_id % 2 else 1.0}
        for user_id in (1, 2)
        for item_id in range(1, 21)
    ])
    recommender = object.__new__(HybridRecommender)
    recommender.strategy = "xgboost"
    recommender.ratings_df = ratings
    recommender.content = UserScoreModel()
    recommender.semantic = UserScoreModel()
    recommender.collaborative = ItemScoreModel()
    recommender.neural = ItemScoreModel()
    recommender.popularity = PopularityScoreModel()
    monkeypatch.setattr(
        config, "XGBOOST_RANKER",
        config.XGBoostRankerConfig(n_estimators=10, max_depth=2, n_jobs=1),
    )

    recommender._fit_learned_ranker()

    assert recommender.learned_ranker.fitted
    assert recommender.learned_ranker.model.n_features_in_ == 5
