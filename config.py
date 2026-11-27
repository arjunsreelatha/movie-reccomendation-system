"""
config.py
Single source of truth for hyperparameters, weights, and paths.
No magic numbers should live inside model/ranking code.
"""
from dataclasses import dataclass, field
from typing import Dict

RANDOM_SEED = 42

DATA_DIR = "data"
MODELS_DIR = "models"
MOVIES_PATH = f"{DATA_DIR}/movies.csv"
RATINGS_PATH = f"{DATA_DIR}/ratings.csv"


@dataclass
class ContentConfig:
    text_fields: tuple = ("genres", "tags", "overview")
    max_features: int = 5000


@dataclass
class SemanticConfig:
    model_name: str = "all-MiniLM-L6-v2"   # lightweight sentence-transformer
    fallback_svd_components: int = 50       # used if the transformer can't be loaded
    text_fields: tuple = ("title", "genres", "tags", "overview")


@dataclass
class CollaborativeConfig:
    n_factors: int = 20
    min_ratings_for_cf: int = 5   # below this -> treat user as cold-start


@dataclass
class NeuralCFConfig:
    n_factors: int = 16
    hidden_dims: tuple = (32, 16)
    epochs: int = 60
    batch_size: int = 256
    lr: float = 0.005
    val_fraction: float = 0.15
    early_stopping_patience: int = 6
    weight_decay: float = 1e-5


@dataclass
class HybridConfig:
    # Baseline strategy: fixed weights
    weighted_baseline: Dict[str, float] = field(default_factory=lambda: {
        "content": 0.15, "semantic": 0.15, "collaborative": 0.35, "neural": 0.35
    })
    # Cold-start override (few/no ratings -> trust content/semantic far more)
    cold_start_weights: Dict[str, float] = field(default_factory=lambda: {
        "content": 0.45, "semantic": 0.45, "collaborative": 0.05, "neural": 0.05
    })
    cold_start_threshold: int = 5   # ratings count below which a user is "cold"
    strategy: str = "normalized"    # "weighted" | "normalized" | "learned"


@dataclass
class CandidateGenConfig:
    per_model_k: int = 50           # candidates pulled from EACH source model
    max_candidates: int = 150       # cap after dedup, before ranking


@dataclass
class DiversityConfig:
    mmr_lambda: float = 0.7         # 1.0 = pure relevance, 0.0 = pure diversity
    enabled: bool = True


CONTENT = ContentConfig()
SEMANTIC = SemanticConfig()
COLLABORATIVE = CollaborativeConfig()
NEURAL_CF = NeuralCFConfig()
HYBRID = HybridConfig()
CANDIDATE_GEN = CandidateGenConfig()
DIVERSITY = DiversityConfig()
