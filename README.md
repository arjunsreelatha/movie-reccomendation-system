# Advanced Hybrid Recommendation System (v2)

Built for **Task 3: Recommendation System** — upgraded from a flat
"combine three models" prototype into a modern, two-stage, multi-model
recommendation pipeline with proper offline evaluation, diversity
re-ranking, cold-start handling, and explainability.

## Architecture

```
                                USER
                                  │
                                  ▼
                          User Profile
                (rating history, cold-start detection)
                                  │
        ┌───────────┬────────────┼────────────┬────────────┐
        ▼            ▼            ▼            ▼            ▼
   Content Model  Semantic    Collaborative   NeuralCF    Popularity
   (TF-IDF +      Model       (Truncated SVD, (PyTorch     (Bayesian
   cosine sim)    (sentence-  sparse matrix)  embeddings+  fallback /
                  transformer,                MLP, mini-   cold-start)
                  TF-IDF-SVD                  batch, early
                  fallback)                   stopping)
        │            │            │            │            │
        └────────────┴─────┬──────┴────────────┴────────────┘
                            ▼
                 Stage 1: Candidate Generation
             (per-model top-K, deduped, capped pool)
                            │
                            ▼
                 Stage 2: Hybrid Ranking
        (weighted | normalized | learned-ranker strategy,
         configurable weights, cold-start weight override)
                            │
                            ▼
                Diversity Re-ranking (MMR)
         (λ·relevance − (1−λ)·similarity_to_selected)
                            │
                            ▼
                       Top-N Items
                            │
                            ▼
                   Explanation Engine
        (grounded in actual per-component signal, not templated)
```

## What changed from the v1 prototype

| Area | v1 (prototype) | v2 (this version) |
|---|---|---|
| Train/test split | Random 80/20 | **Chronological per-user split** — no leakage from future ratings into training |
| CF matrix | Dense NumPy matrix | **Sparse (scipy.csr)** matrix factorization |
| Scoring | Full catalog scored per model per request | **Two-stage candidate generation** — cheap retrieval, then ranking on a bounded pool |
| Fusion | Fixed `0.4/0.2/0.4` weighted sum | **Three selectable strategies**: weighted baseline, normalized min-max fusion, and a logistic-regression **learned ranker** trained on train-only interactions |
| Semantic signal | None (TF-IDF only) | **Sentence-transformer embeddings**, with an honest TF-IDF+SVD fallback if the transformer can't load |
| NeuralCF training | Full-batch, fixed 15 epochs, no validation | **Mini-batch, validation split, early stopping, seeded, save/load** |
| Diversity | None | **MMR re-ranking**, configurable λ |
| Cold-start | One hard-coded rule | **Documented strategy** for new users (content/popularity-weighted) and new items (content/semantic only, no CF/NCF signal) |
| Evaluation | RMSE, Precision/Recall/NDCG, coverage | **RMSE, MAE, Precision/Recall/F1/HitRate/NDCG/MAP@K, catalog coverage, user coverage, intra-list diversity, novelty** — computed for 8 models/baselines side-by-side |
| Baselines | None | **Random** and **Popularity** (Bayesian-adjusted) baselines every advanced model is compared against |
| Config | Magic numbers inline | **`config.py`** — single source of truth for every hyperparameter and weight |
| Code layout | One `recommender.py` | **`src/` package**: `data/`, `models/`, `ranking/`, `evaluation/`, `explainability/`, `utils.py` |
| API | 5 basic endpoints | **`/health`, `/movies/{id}`, `/recommend/{id}`, `POST /recommend`, `/similar/{id}`, `/explain/{user}/{movie}`, `/metrics`, `/user/{id}/history`, `/users`** — all Pydantic-typed |
| Model persistence | Retrains every process start | **NeuralCF saves/loads** (`models/neural_cf.pt`) |

## Files

```
recommendation-system/
├── README.md
├── requirements.txt
├── config.py                    # all hyperparameters/weights, single source of truth
├── generate_data.py             # synthetic dataset generator (offline, reproducible)
├── download_movielens.py        # download and convert the official MovieLens 1M dataset
├── recommender.py                # top-level orchestrator (HybridRecommender)
├── evaluate.py                   # model comparison; hybrid strategy is configurable
├── app.py                        # FastAPI backend, models trained once at startup
├── Dockerfile                    # container image for the FastAPI service
├── docker-compose.yml            # local deployment and persistent model cache
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── models/
│   └── neural_cf.pt              # saved NeuralCF weights
└── src/
    ├── data/loader.py            # load + chronological split
    ├── models/
    │   ├── content.py            # TF-IDF content-based
    │   ├── semantic.py           # sentence-transformer (+ TF-IDF-SVD fallback)
    │   ├── collaborative.py      # sparse SVD matrix factorization
    │   ├── neural_cf.py          # PyTorch embeddings + MLP, mini-batch, early stopping
    │   ├── popularity.py         # Bayesian-adjusted popularity baseline
    │   └── random_baseline.py    # random baseline
    ├── ranking/
    │   ├── candidate_gen.py      # Stage 1: per-model top-K retrieval
    │   ├── hybrid.py             # Stage 2: weighted / normalized / logistic / XGBoost fusion
    │   └── diversity.py          # MMR re-ranking
    ├── evaluation/metrics.py     # full offline metric suite
    ├── explainability/explain.py # signal-grounded explanations
    └── utils.py                  # seeding, device handling
```

## How to run

### Run the API with Docker Compose

```bash
docker compose up --build
```

The API is available at `http://localhost:8000` and interactive API docs
at `http://localhost:8000/docs`. The first startup trains the recommender
and may download the sentence-transformer model; a Docker volume caches
the downloaded model between runs. If the model hub is unavailable, the
semantic model uses its TF-IDF/SVD fallback.

Stop the service with:

```bash
docker compose down
```

### Deploy publicly with Render

1. Push this project to a GitHub repository.
2. In Render, choose **New → Blueprint**, connect the repository, and deploy
   the detected `render.yaml` configuration.
3. Wait for the first model download and training to finish. Render's health
   check uses `/health`; the persistent disk keeps the downloaded model cache
   between deployments.
4. Open the public service URL or append `/docs` to try the API.

The blueprint uses a paid `standard` instance because the recommender loads
PyTorch and trains at startup. Check Render's current pricing before deploying.
The deployment exposes the REST API and its interactive docs, not a separate
recommendation website.

### Run locally without Docker

```bash
pip install -r requirements.txt

# 1. Generate the dataset (already generated, but re-runnable / reproducible via seed)
python generate_data.py

# 2. Run the full model evaluation comparison
python evaluate.py

# 3. Run the REST API (trains once at startup)
uvicorn app:app --reload --port 8000
# then open http://localhost:8000/docs
```

## Verified working (this build)

All of the following were actually executed, not just written:
- `evaluate.py` runs end-to-end and produces a comparison table across
  Random, Popularity, Content, Semantic, Collaborative, NeuralCF,
  Hybrid, and Hybrid+Diversity.
- The semantic model correctly detects when it can't reach the
  sentence-transformer model hub and falls back to the TF-IDF+SVD path
  (confirmed via `/health` reporting `semantic_backend`).
- The FastAPI server was started and every endpoint
  (`/health`, `/recommend/{id}`, `/similar/{id}`, `/explain/{user}/{movie}`)
  was hit and returned valid responses.
- NeuralCF was saved to `models/neural_cf.pt` and successfully reloaded
  with matching predictions — persistence works.

## Honest evaluation notes

- The default dataset is synthetic (~11K ratings, 300 users, 500 movies), so
  **absolute metric values are modest** (Precision@10 in the 0.01 range).
  This is expected at this scale and is *not* a bug — what demonstrates
  methodology is the **relative comparison**: collaborative filtering
  and the hybrid clearly outperform the random and popularity baselines
  on Precision/Recall/NDCG, while popularity (unsurprisingly) has poor
  catalog coverage since it recommends the same well-rated items to
  everyone.
- The candidate generation stage still computes content/semantic/CF/NCF
  scores across the full catalog before taking each model's top-K — the
  win versus the v1 prototype is that **hybrid fusion and MMR re-ranking**
  (the more expensive, combinatorial steps) now only run on the
  ~150-item candidate pool, not the full catalog. Making the per-model
  scoring itself sub-linear (e.g. via approximate nearest-neighbor
  search) is a reasonable next step if the catalog grows to real-world
  scale (10K+ items) — noted as a known limitation, not silently glossed
  over.
- The logistic-regression learned ranker remains available for the
  smaller synthetic dataset. XGBoost can be selected for larger datasets
  and is trained/evaluated against the same held-out split.

## Swapping in a real dataset

The project includes an opt-in downloader for the official MovieLens 1M
dataset. It preserves the synthetic files and writes the converted dataset
to the Git-ignored `data/movielens-1m/` directory:

```bash
python download_movielens.py
```

The archive is about 6 MB; the converted ratings CSV is about 25 MB.
To run evaluation or the API using this dataset with XGBoost, select the
data files and ranker through environment variables:

```powershell
$env:MOVIES_PATH = "data/movielens-1m/movies.csv"
$env:RATINGS_PATH = "data/movielens-1m/ratings.csv"
$env:HYBRID_STRATEGY = "xgboost"
python evaluate.py
# Or: uvicorn app:app --reload --port 8000
```

The dataset has the existing `movieId,title,genres,tags,overview` and
`userId,movieId,rating,timestamp` schemas; tags and overview are empty
because MovieLens 1M does not include those fields.

## Possible further extensions

- Approximate nearest-neighbor retrieval (FAISS) for candidate
  generation at real-world catalog scale
- Add implicit feedback (clicks/watch-time) as a second signal alongside
  explicit ratings, modeled separately rather than conflated with them
- A/B testing framework to tune hybrid weights against live engagement
  rather than offline metrics alone
