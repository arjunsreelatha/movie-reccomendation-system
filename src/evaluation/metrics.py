"""
src/evaluation/metrics.py
Full offline evaluation suite, clearly separated into:
  - Rating prediction metrics (RMSE, MAE)
  - Top-K ranking metrics (Precision/Recall/F1/NDCG/HitRate/MAP @K)
  - System-level metrics (Catalog Coverage, User Coverage, Intra-list
    Diversity, Novelty)
"""
import numpy as np


# ---------------- Rating prediction ----------------

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------- Top-K ranking ----------------

def precision_at_k(recommended_ids, relevant_ids, k):
    if k == 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / k


def recall_at_k(recommended_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def f1_at_k(recommended_ids, relevant_ids, k):
    p = precision_at_k(recommended_ids, relevant_ids, k)
    r = recall_at_k(recommended_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_rate_at_k(recommended_ids, relevant_ids, k):
    top_k = set(recommended_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def ndcg_at_k(recommended_ids, relevance_map: dict, k):
    """relevance_map: {item_id: graded relevance (e.g. rating)}"""
    dcg = 0.0
    for idx, item in enumerate(recommended_ids[:k]):
        rel = relevance_map.get(item, 0)
        dcg += (2 ** rel - 1) / np.log2(idx + 2)
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)
    hits, score = 0, 0.0
    for idx, item in enumerate(recommended_ids[:k]):
        if item in relevant_ids:
            hits += 1
            score += hits / (idx + 1)
    return score / min(len(relevant_ids), k) if relevant_ids else 0.0


def map_at_k(all_recommended: list, all_relevant: list, k):
    """all_recommended / all_relevant: list of per-user lists."""
    aps = [average_precision_at_k(rec, rel, k) for rec, rel in zip(all_recommended, all_relevant)]
    return float(np.mean(aps)) if aps else 0.0


# ---------------- System-level ----------------

def catalog_coverage(all_recommended_ids: list, catalog_size: int):
    unique_recommended = set()
    for ids in all_recommended_ids:
        unique_recommended.update(ids)
    return len(unique_recommended) / catalog_size if catalog_size else 0.0


def user_coverage(users_with_recommendations: int, total_users: int):
    return users_with_recommendations / total_users if total_users else 0.0


def intra_list_diversity(recommended_ids, item_vectors: dict):
    """1 - average pairwise cosine similarity within a single user's rec list."""
    from sklearn.metrics.pairwise import cosine_similarity
    vecs = [item_vectors[i] for i in recommended_ids if i in item_vectors]
    if len(vecs) < 2:
        return 0.0
    sims = []
    for a in range(len(vecs)):
        for b in range(a + 1, len(vecs)):
            sims.append(cosine_similarity(vecs[a].reshape(1, -1), vecs[b].reshape(1, -1))[0][0])
    return float(1 - np.mean(sims)) if sims else 0.0


def novelty(recommended_ids, item_popularity: dict, total_interactions: int):
    """Average self-information: -log2(p(item)). Higher = recommending less obvious items."""
    scores = []
    for item in recommended_ids:
        p = max(item_popularity.get(item, 1), 1) / total_interactions
        scores.append(-np.log2(p))
    return float(np.mean(scores)) if scores else 0.0
