"""
src/ranking/hybrid.py
Three hybrid fusion strategies, selectable via config:

A. WEIGHTED BASELINE — fixed weights, kept as the honest baseline this
   whole system upgrades from. Not claimed as "learned" or "AI".

B. NORMALIZED-SCORE HYBRID — each component score is min-max normalized
   PER REQUEST before weighting, so a component whose scale happens to
   run 0-1 (content/semantic cosine sim) doesn't get dominated by one that
   runs 0.5-5 (CF/NCF predicted ratings) just because of units.

C. LEARNED RANKER — a logistic regression trained on TRAIN-SET interactions
   only, predicting P(user will like item | component scores as features).
   Target: binary relevance = rating >= 3.5, built exclusively from the
   training split. Features: content/semantic/collaborative/neural scores,
   popularity, user's average rating (affinity), genre-overlap count.
   This is a real (if simple) learned model — not a manually tuned formula
   — and is evaluated on held-out data like everything else, so it's not
   overstated as more than it is: logistic regression on 4-8 features.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _minmax_normalize(score_dict):
    if not score_dict:
        return score_dict
    vals = np.array(list(score_dict.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-9:
        return {k: 0.5 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def weighted_fusion(component_scores: dict, weights: dict) -> dict:
    """component_scores: {model_name: {movieId: score}}. Simple weighted sum, no normalization."""
    items = next(iter(component_scores.values())).keys()
    return {m: sum(weights.get(name, 0) * component_scores[name].get(m, 0.0)
                    for name in component_scores) for m in items}


def normalized_fusion(component_scores: dict, weights: dict) -> dict:
    """Min-max normalize each component's scores to [0,1] before weighting."""
    normalized = {name: _minmax_normalize(scores) for name, scores in component_scores.items()}
    items = next(iter(component_scores.values())).keys()
    return {m: sum(weights.get(name, 0) * normalized[name].get(m, 0.0)
                    for name in component_scores) for m in items}


class LearnedRanker:
    """
    Logistic regression ranker.

    Training data: (user, item, feature vector, label) tuples built ONLY
    from the training split's interactions. Label = 1 if rating >= 3.5
    else 0. Features are the *training-time* component model outputs —
    since all component models (CF/NCF/content/semantic) are themselves
    fit on train only, there is no leakage of test information into the
    features or the label.

    Validation: a held-out slice of the training interactions (not the
    evaluation test set) is used to check the ranker isn't just memorizing.
    """
    def __init__(self, feature_names, val_fraction=0.15, seed=42):
        self.feature_names = feature_names
        self.val_fraction = val_fraction
        self.seed = seed
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        n_val = int(len(X) * self.val_fraction)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        if len(np.unique(y[train_idx])) < 2:
            # not enough label diversity to fit logistic regression meaningfully
            self.fitted = False
            return self

        self.model.fit(X[train_idx], y[train_idx])
        self.fitted = True

        if n_val > 0 and len(np.unique(y[val_idx])) > 1:
            from sklearn.metrics import roc_auc_score
            val_probs = self.model.predict_proba(X[val_idx])[:, 1]
            self.val_auc = roc_auc_score(y[val_idx], val_probs)
        else:
            self.val_auc = None
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            # fallback: uniform score so callers can still rank (degrades to input order)
            return np.full(len(X), 0.5)
        return self.model.predict_proba(X)[:, 1]
