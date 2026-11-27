"""
src/ranking/candidate_gen.py
Stage 1 of the two-stage architecture: cheaply pull a manageable candidate
set from each source model instead of scoring the FULL catalog against
every model on every request (which is what the prototype did).

Each source contributes its own top-K by its own score, candidates are
deduped, and this smaller pool is what Stage 2 (hybrid ranking) actually
scores in full. This is the standard "retrieval then ranking" pattern
used in production recommenders (retrieval is cheap+approximate, ranking
is precise+expensive, so you only pay ranking cost on a small pool).
"""
import heapq


def generate_candidates(models: dict, ratings_df, movies_df, user_id,
                         per_model_k=50, max_candidates=150, exclude_seen_ids=None):
    """
    models: {"content": ContentBasedModel, "semantic": SemanticModel,
             "collaborative": CollaborativeModel, "neural": NeuralCFModel,
             "popularity": PopularityRecommender}
    Returns: (candidate_ids: list[int], component_scores: {model_name: {id: score}})
    """
    exclude_seen_ids = exclude_seen_ids or set()
    all_ids = [m for m in movies_df.movieId.tolist() if m not in exclude_seen_ids]

    candidate_pool = set()
    component_scores_full = {}

    # Content & semantic: score full catalog (cheap — vectorized cosine sim) then take top-K
    content_scores, content_signal = models["content"].score_all(ratings_df, user_id, all_ids)
    component_scores_full["content"] = content_scores
    candidate_pool.update(_top_n(content_scores, per_model_k))

    semantic_scores, semantic_signal = models["semantic"].score_all(ratings_df, user_id, all_ids)
    component_scores_full["semantic"] = semantic_scores
    candidate_pool.update(_top_n(semantic_scores, per_model_k))

    # CF & NCF: only meaningful for warm users, but cheap enough to still score directly
    cf_scores, cf_signal = models["collaborative"].score_all(user_id, all_ids)
    component_scores_full["collaborative"] = cf_scores
    if cf_signal:
        candidate_pool.update(_top_n(cf_scores, per_model_k))

    ncf_scores, ncf_signal = models["neural"].score_all(user_id, all_ids)
    component_scores_full["neural"] = ncf_scores
    if ncf_signal:
        candidate_pool.update(_top_n(ncf_scores, per_model_k))

    # Popularity fallback — always contributes some candidates, dominant for true cold-start
    pop_candidates = models["popularity"].top_k(k=per_model_k, exclude=exclude_seen_ids)
    candidate_pool.update(pop_candidates)

    if not candidate_pool:
        candidate_pool.update(all_ids[:max_candidates])

    candidates = list(candidate_pool)[:max_candidates]
    return candidates, component_scores_full


def _top_n(score_dict, n):
    return [k for k, _ in heapq.nlargest(n, score_dict.items(), key=lambda x: x[1])]
