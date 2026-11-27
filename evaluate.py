"""
evaluate.py
Evaluates: Random baseline, Popularity baseline, Content-only, Semantic-only,
Collaborative-only, NeuralCF-only, Hybrid (normalized), Hybrid+Diversity.

Uses a CHRONOLOGICAL per-user split (src/data/loader.py) — earliest
interactions train, most recent test — to avoid the leakage a random
split would introduce (see loader.py docstring for the full rationale).

All models are fit on TRAIN ONLY. Test-set items are never seen during
fitting. `exclude_seen=False` is used only so recommend() doesn't filter
out the test items themselves (which live only in `train`'s memory of
"seen" items) — recommend() has no access to test ratings at any point.
"""
import numpy as np
import os
import sys

# Ensure project root and src package are on sys.path so `src` package can
# be imported when running this script directly (silences runtime import
# errors when the repo isn't installed into site-packages).
ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = ROOT
PARENT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")

# Insert paths in order of precedence and avoid duplicates.
for p in reversed((REPO_ROOT, SRC_ROOT, PARENT_ROOT)):
    if p and p not in sys.path:
        sys.path.insert(0, p)
# Invalidate import caches in case the editor/IDE needs refresh.
try:
    import importlib
    importlib.invalidate_caches()
except Exception:
    pass

import config
from src.data.loader import load_data, chronological_split
from src.models.content import ContentBasedModel
from src.models.semantic import SemanticModel
from src.models.collaborative import CollaborativeModel
from src.models.neural_cf import NeuralCFModel
from src.models.popularity import PopularityRecommender
from src.models.random_baseline import RandomRecommender
from src.evaluation import metrics as M
from recommender import HybridRecommender

K = 10
RELEVANCE_THRESHOLD = 3.5


def eval_ranked_list(recommend_fn, test_df, movies_df, k=K):
    """
    recommend_fn(user_id) -> list of recommended movieIds (already ranked)
    Returns a dict of aggregated metrics across all test users, plus raw
    lists needed for MAP@K and coverage.
    """
    precisions, recalls, f1s, hit_rates, ndcgs = [], [], [], [], []
    all_recommended, all_relevant = [], []
    users_covered = 0

    for user_id in test_df.userId.unique():
        user_test = test_df[test_df.userId == user_id]
        relevant_ids = user_test[user_test.rating >= RELEVANCE_THRESHOLD].movieId.tolist()
        if not relevant_ids:
            continue
        relevance_map = dict(zip(user_test.movieId, user_test.rating))

        recommended_ids = recommend_fn(user_id)
        if not recommended_ids:
            continue
        users_covered += 1

        precisions.append(M.precision_at_k(recommended_ids, relevant_ids, k))
        recalls.append(M.recall_at_k(recommended_ids, relevant_ids, k))
        f1s.append(M.f1_at_k(recommended_ids, relevant_ids, k))
        hit_rates.append(M.hit_rate_at_k(recommended_ids, relevant_ids, k))
        ndcgs.append(M.ndcg_at_k(recommended_ids, relevance_map, k))
        all_recommended.append(recommended_ids)
        all_relevant.append(relevant_ids)

    n_test_users = test_df.userId.nunique()
    return {
        "Precision@10": np.mean(precisions) if precisions else 0.0,
        "Recall@10": np.mean(recalls) if recalls else 0.0,
        "F1@10": np.mean(f1s) if f1s else 0.0,
        "HitRate@10": np.mean(hit_rates) if hit_rates else 0.0,
        "NDCG@10": np.mean(ndcgs) if ndcgs else 0.0,
        "MAP@10": M.map_at_k(all_recommended, all_relevant, k),
        "CatalogCoverage": M.catalog_coverage(all_recommended, len(movies_df)),
        "UserCoverage": M.user_coverage(users_covered, n_test_users),
    }


def eval_rating_prediction(predict_fn, test_df):
    preds = [predict_fn(row.userId, row.movieId) for row in test_df.itertuples()]
    return M.rmse(test_df.rating.values, preds), M.mae(test_df.rating.values, preds)


def main():
    movies, ratings = load_data(config.MOVIES_PATH, config.RATINGS_PATH)
    train, test = chronological_split(ratings, test_fraction=0.2)
    print(f"Train interactions: {len(train)} | Test interactions: {len(test)}")
    print(f"Train users: {train.userId.nunique()} | Test users: {test.userId.nunique()}\n")

    results = {}

    # ---- Baselines ----
    print("Evaluating Random baseline...")
    rand = RandomRecommender().fit(movies)
    seen_by_user = train.groupby("userId").movieId.apply(set).to_dict()
    results["Random"] = eval_ranked_list(
        lambda u: rand.top_k(k=K, exclude=seen_by_user.get(u, set()), seed=42),
        test, movies)

    print("Evaluating Popularity baseline...")
    pop = PopularityRecommender().fit(train)
    results["Popularity"] = eval_ranked_list(
        lambda u: pop.top_k(k=K, exclude=seen_by_user.get(u, set())), test, movies)

    # ---- Individual models ----
    print("Evaluating Content-based (standalone)...")
    content = ContentBasedModel(**vars(config.CONTENT)).fit(movies)

    def content_recommend(user_id):
        candidates = [m for m in movies.movieId if m not in seen_by_user.get(user_id, set())]
        scores, has_signal = content.score_all(train, user_id, candidates)
        if not has_signal:
            return pop.top_k(k=K, exclude=seen_by_user.get(user_id, set()))
        return [m for m, _ in sorted(scores.items(), key=lambda x: -x[1])[:K]]

    results["Content-Based"] = eval_ranked_list(content_recommend, test, movies)

    print("Evaluating Semantic (standalone)...")
    semantic = SemanticModel(**vars(config.SEMANTIC)).fit(movies)

    def semantic_recommend(user_id):
        candidates = [m for m in movies.movieId if m not in seen_by_user.get(user_id, set())]
        scores, has_signal = semantic.score_all(train, user_id, candidates)
        if not has_signal:
            return pop.top_k(k=K, exclude=seen_by_user.get(user_id, set()))
        return [m for m, _ in sorted(scores.items(), key=lambda x: -x[1])[:K]]

    results["Semantic"] = eval_ranked_list(semantic_recommend, test, movies)
    print(f"  (semantic backend used: {semantic.backend})")

    print("Evaluating Collaborative Filtering (standalone)...")
    cf = CollaborativeModel(**vars(config.COLLABORATIVE)).fit(train)

    def cf_recommend(user_id):
        candidates = [m for m in movies.movieId if m not in seen_by_user.get(user_id, set())]
        if cf.is_cold_start(user_id):
            return pop.top_k(k=K, exclude=seen_by_user.get(user_id, set()))
        scores, _ = cf.score_all(user_id, candidates)
        return [m for m, _ in sorted(scores.items(), key=lambda x: -x[1])[:K]]

    results["Collaborative"] = eval_ranked_list(cf_recommend, test, movies)
    cf_rmse, cf_mae = eval_rating_prediction(cf.predict, test)

    print("Evaluating NeuralCF (standalone)...")
    ncf = NeuralCFModel(**vars(config.NEURAL_CF)).fit(train)

    def ncf_recommend(user_id):
        candidates = [m for m in movies.movieId if m not in seen_by_user.get(user_id, set())]
        scores, has_signal = ncf.score_all(user_id, candidates)
        if not has_signal:
            return pop.top_k(k=K, exclude=seen_by_user.get(user_id, set()))
        return [m for m, _ in sorted(scores.items(), key=lambda x: -x[1])[:K]]

    results["NeuralCF"] = eval_ranked_list(ncf_recommend, test, movies)
    ncf_rmse, ncf_mae = eval_rating_prediction(ncf.predict, test)

    # ---- Hybrid variants ----
    print(f"Evaluating Hybrid ({config.HYBRID.strategy} strategy)...")
    hybrid = HybridRecommender(strategy=config.HYBRID.strategy).fit(train, movies)
    results["Hybrid"] = eval_ranked_list(
        lambda u: [r["movieId"] for r in hybrid.recommend(u, top_k=K, apply_diversity=False)],
        test, movies)

    print("Evaluating Hybrid + Diversity (MMR re-ranking)...")
    results["Hybrid+Diversity"] = eval_ranked_list(
        lambda u: [r["movieId"] for r in hybrid.recommend(u, top_k=K, apply_diversity=True)],
        test, movies)

    # ---- Report ----
    print("\n" + "=" * 100)
    print("RATING PREDICTION METRICS (models that predict ratings directly)")
    print("=" * 100)
    print(f"{'Model':<20}{'RMSE':<10}{'MAE':<10}")
    print(f"{'Collaborative':<20}{cf_rmse:<10.4f}{cf_mae:<10.4f}")
    print(f"{'NeuralCF':<20}{ncf_rmse:<10.4f}{ncf_mae:<10.4f}")

    print("\n" + "=" * 100)
    print(f"TOP-{K} RECOMMENDATION & SYSTEM-LEVEL METRICS")
    print("=" * 100)
    header = f"{'Model':<20}" + "".join(f"{k:<16}" for k in next(iter(results.values())).keys())
    print(header)
    for name, m in results.items():
        row = f"{name:<20}" + "".join(f"{v:<16.4f}" for v in m.values())
        print(row)

    print(
        f"\nDataset: {len(ratings):,} ratings across "
        f"{ratings.userId.nunique():,} users and {len(movies):,} movies."
    )
    print(f"Hybrid strategy: {hybrid.strategy}.")


if __name__ == "__main__":
    main()
