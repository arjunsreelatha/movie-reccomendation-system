"""
src/ranking/diversity.py
Maximum Marginal Relevance (MMR) re-ranking so top-N lists aren't just
the same genre five times over.

MMR = λ * relevance(item) - (1-λ) * max_similarity(item, already_selected)

At each step, pick the candidate maximizing that score, greedily building
the final list. λ close to 1 -> almost pure relevance ranking. λ close to
0 -> prioritizes spreading across the similarity space.
"""
import numpy as np


def mmr_rerank(candidate_ids, relevance_scores: dict, item_vectors: dict,
               top_k=10, lambda_param=0.7):
    """
    candidate_ids: list of item ids to consider (already relevance-ranked upstream)
    relevance_scores: {item_id: float} — the fused hybrid score
    item_vectors: {item_id: np.ndarray} — a feature vector per item used ONLY to
                  measure similarity for diversity (e.g. the content TF-IDF or
                  semantic embedding row), not to compute relevance

    Vectorized: precomputes a single (n_candidates x n_features) normalized
    matrix once, so each greedy MMR step is one matrix-vector product
    instead of O(candidates x selected) individual cosine_similarity calls
    (the latter was the dominant cost during evaluation across many users).
    """
    if not candidate_ids:
        return []

    ids_with_vec = [c for c in candidate_ids if c in item_vectors]
    if not ids_with_vec:
        return list(candidate_ids)[:top_k]

    id_to_pos = {c: i for i, c in enumerate(ids_with_vec)}
    matrix = np.stack([item_vectors[c] for c in ids_with_vec]).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm_matrix = matrix / np.clip(norms, 1e-9, None)

    remaining_ids = list(candidate_ids)
    selected = []

    rel_vals = np.array([relevance_scores.get(c, 0.0) for c in remaining_ids])
    lo, hi = rel_vals.min(), rel_vals.max()
    norm_rel = {c: (relevance_scores.get(c, 0.0) - lo) / (hi - lo + 1e-9) for c in remaining_ids}

    max_sim_to_selected = {c: 0.0 for c in remaining_ids}  # running max similarity

    while remaining_ids and len(selected) < top_k:
        best_item, best_score = None, -1e18
        for c in remaining_ids:
            mmr_score = lambda_param * norm_rel[c] - (1 - lambda_param) * max_sim_to_selected[c]
            if mmr_score > best_score:
                best_score, best_item = mmr_score, c

        selected.append(best_item)
        remaining_ids.remove(best_item)

        # update running max similarity for remaining items in one vectorized pass
        if best_item in id_to_pos and remaining_ids:
            best_vec = norm_matrix[id_to_pos[best_item]]
            for c in remaining_ids:
                if c in id_to_pos:
                    sim = float(norm_matrix[id_to_pos[c]] @ best_vec)
                    if sim > max_sim_to_selected[c]:
                        max_sim_to_selected[c] = sim

    return selected
